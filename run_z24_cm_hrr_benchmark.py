"""Leak-free CM-HRR preprocessing ablation on the exported Z24 split.

CM-HRR (condition-matched healthy-reference residualisation) is deliberately
model agnostic.  Every fitted statistic comes from S01/train only.  S01/val is
used only to calibrate a healthy threshold and, for the autoencoder, early
stopping.  S01/test, S08, and every damage record are evaluation-only.

The benchmark compares the same four detectors on (a) energy-normalised
log-Welch PSD and (b) CM-HRR.  This isolates preprocessing from the model.
Reference lookup is hierarchical because held-out setups have no legal S01
training reference: setup+excitation -> excitation -> global healthy.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_z24_strict_psd_scatterad as common


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def robust_location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - center), axis=0)
    floor = max(float(np.median(scale[scale > 0])) * .02 if np.any(scale > 0) else 0., 1e-4)
    return center.astype(np.float32), np.maximum(scale, floor).astype(np.float32)


@dataclass
class Reference:
    center: np.ndarray
    scale: np.ndarray
    records: int
    windows: int


class CMHRR:
    """Train-only hierarchical healthy-reference residualizer."""

    def __init__(self, clip: float = 10.):
        self.clip = clip
        self.references: dict[tuple, Reference] = {}

    def fit(self, rows: pd.DataFrame, raw: dict[str, np.ndarray]) -> "CMHRR":
        if len(rows) == 0 or set(rows.state) != {"01"} or set(rows.label.astype(int)) != {0}:
            raise ValueError("CM-HRR fit requires non-empty S01 healthy rows only")
        groups: dict[tuple, list[tuple[str, np.ndarray]]] = {("global",): []}
        for row in rows.itertuples():
            x = raw[row.record_id]
            excitation = str(row.excitation).upper()
            keys = [("global",), ("excitation", excitation),
                    ("condition", excitation, int(row.setup))]
            for key in keys: groups.setdefault(key, []).append((row.record_id, x))
        for key, parts in groups.items():
            values = np.concatenate([x for _, x in parts])
            center, scale = robust_location_scale(values)
            self.references[key] = Reference(center, scale, len({r for r, _ in parts}), len(values))
        return self

    def key_for(self, excitation: str, setup: int) -> tuple:
        condition = ("condition", str(excitation).upper(), int(setup))
        excitation_key = ("excitation", str(excitation).upper())
        if condition in self.references: return condition
        if excitation_key in self.references: return excitation_key
        return ("global",)

    def transform(self, x: np.ndarray, excitation: str, setup: int) -> tuple[np.ndarray, tuple]:
        key = self.key_for(excitation, setup); ref = self.references[key]
        residual = np.clip((x - ref.center) / ref.scale, -self.clip, self.clip)
        return residual.astype(np.float32), key

    def audit(self) -> list[dict]:
        return [{"key": "|".join(map(str, key)), "records": value.records,
                 "windows": value.windows}
                for key, value in sorted(self.references.items(), key=lambda item: str(item[0]))]


class Detector:
    def fit(self, train: np.ndarray, val: np.ndarray | None = None) -> None: raise NotImplementedError
    def score(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError


class OCSVMDetector(Detector):
    def __init__(self, nu: float): self.scaler = RobustScaler(); self.model = OneClassSVM(nu=nu, gamma="scale")
    def fit(self, train, val=None): self.model.fit(self.scaler.fit_transform(train))
    def score(self, x): return -self.model.decision_function(self.scaler.transform(x))


class PCADetector(Detector):
    def __init__(self, components: int, seed: int):
        self.components = components; self.seed = seed; self.scaler = RobustScaler(); self.model = None
    def fit(self, train, val=None):
        z = self.scaler.fit_transform(train)
        n = max(2, min(self.components, len(z) - 1, z.shape[1]))
        self.model = PCA(n_components=n, random_state=self.seed).fit(z)
    def score(self, x):
        z = self.scaler.transform(x); rec = self.model.inverse_transform(self.model.transform(z))
        return np.mean((z - rec) ** 2, axis=1)


class IsolationForestDetector(Detector):
    def __init__(self, seed: int):
        self.scaler = RobustScaler(); self.model = IsolationForest(
            n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)
    def fit(self, train, val=None): self.model.fit(self.scaler.fit_transform(train))
    def score(self, x): return -self.model.score_samples(self.scaler.transform(x))


class SmallAE(nn.Module):
    def __init__(self, features: int, latent: int):
        super().__init__(); hidden = max(latent * 2, min(64, features // 2))
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ELU(), nn.Linear(hidden, latent),
                                 nn.ELU(), nn.Linear(latent, hidden), nn.ELU(), nn.Linear(hidden, features))
    def forward(self, x): return self.net(x)


class AEDetector(Detector):
    def __init__(self, latent: int, epochs: int, patience: int, batch: int, seed: int, device: torch.device):
        self.latent, self.epochs, self.patience, self.batch = latent, epochs, patience, batch
        self.seed, self.device, self.scaler, self.model = seed, device, RobustScaler(), None

    @staticmethod
    def loader(x, batch, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32))), batch_size=batch,
                          shuffle=shuffle, num_workers=0)

    @torch.no_grad()
    def _loss(self, x):
        self.model.eval(); losses = []
        for (part,) in self.loader(x, self.batch, False):
            part = part.to(self.device); losses.append(torch.mean((self.model(part) - part) ** 2, dim=1).cpu().numpy())
        return np.concatenate(losses)

    def fit(self, train, val=None):
        seed_all(self.seed); z = self.scaler.fit_transform(train).astype(np.float32)
        zv = self.scaler.transform(val).astype(np.float32) if val is not None and len(val) else z
        self.model = SmallAE(z.shape[1], self.latent).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        best, state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self.model.train()
            for (part,) in self.loader(z, self.batch, True):
                part = part.to(self.device); opt.zero_grad(set_to_none=True)
                loss = torch.mean((self.model(part) - part) ** 2); loss.backward(); opt.step()
            current = float(np.mean(self._loss(zv)))
            if current < best - 1e-7: best, state, stale = current, copy.deepcopy(self.model.state_dict()), 0
            else: stale += 1
            if stale >= self.patience: break
        self.model.load_state_dict(state)

    def score(self, x): return self._loss(self.scaler.transform(x).astype(np.float32))


def build_detector(name: str, args, device: torch.device) -> Detector:
    if name == "ocsvm": return OCSVMDetector(args.ocsvm_nu)
    if name == "pca": return PCADetector(args.pca_components, args.seed)
    if name == "iforest": return IsolationForestDetector(args.seed)
    if name == "ae": return AEDetector(args.ae_latent, args.epochs, args.patience,
                                        args.batch_size, args.seed, device)
    raise ValueError(name)


def record_score(detector: Detector, x: np.ndarray, quantile: float) -> float:
    return float(np.quantile(detector.score(x), quantile))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_cm_hrr_benchmark"))
    p.add_argument("--models", default="ocsvm,pca,iforest,ae")
    p.add_argument("--fs", type=float, default=50.); p.add_argument("--raw-window", type=int, default=2048)
    p.add_argument("--raw-stride", type=int, default=1024); p.add_argument("--psd-bins", type=int, default=96)
    p.add_argument("--fmin", type=float, default=3.); p.add_argument("--fmax", type=float, default=30.)
    p.add_argument("--record-quantile", type=float, default=.95); p.add_argument("--cmhrr-clip", type=float, default=10.)
    p.add_argument("--ocsvm-nu", type=float, default=.05); p.add_argument("--pca-components", type=int, default=12)
    p.add_argument("--ae-latent", type=int, default=12); p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = common.resolve_manifest(args.manifest, args.data_root)
    required = {"split", "record_id", "state", "label", "setup", "excitation", "scenario", "path"}
    missing = required - set(frame.columns)
    if missing: raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    train = frame[(frame.split == "train") & (frame.state == "01") & (frame.label.astype(int) == 0)].copy()
    calibration = frame[(frame.split == "val") & (frame.state == "01") & (frame.label.astype(int) == 0)].copy()
    test = frame[frame.split == "test"].copy()
    if len(train) == 0 or len(calibration) == 0 or len(test) == 0: raise ValueError("Need S01 train, S01 val, and test rows")
    if np.any(test.record_id.isin(set(train.record_id) | set(calibration.record_id))): raise ValueError("Record leakage")
    print(f"device={device} S01_train={len(train)} S01_calibration={len(calibration)} test={len(test)}", flush=True)

    raw: dict[str, np.ndarray] = {}
    for i, row in enumerate(frame.itertuples(), 1):
        raw[row.record_id] = common.psd_vectors(row.path, args.fs, args.raw_window, args.raw_stride,
                                                args.psd_bins, args.fmin, args.fmax)
        if i % 24 == 0 or i == len(frame): print(f"features={i}/{len(frame)}", flush=True)
    cmhrr = CMHRR(args.cmhrr_clip).fit(train, raw)
    processed, reference_rows = {}, []
    for row in frame.itertuples():
        value, key = cmhrr.transform(raw[row.record_id], row.excitation, int(row.setup))
        processed[row.record_id] = value
        reference_rows.append({"record_id": row.record_id, "split": row.split, "state": row.state,
                               "reference_key": "|".join(map(str, key))})
    pd.DataFrame(reference_rows).to_csv(args.output / "reference_assignment.csv", index=False)
    pd.DataFrame(cmhrr.audit()).to_csv(args.output / "healthy_reference_audit.csv", index=False)

    variants = {"raw_logpsd": raw, "cm_hrr": processed}; summaries, all_scores = [], []
    model_names = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    started = time.time()
    for variant, cache in variants.items():
        train_x = np.concatenate([cache[x] for x in train.record_id])
        cal_x = np.concatenate([cache[x] for x in calibration.record_id])
        for model_name in model_names:
            detector = build_detector(model_name, args, device); detector.fit(train_x, cal_x)
            calibration_scores = [record_score(detector, cache[x], args.record_quantile) for x in calibration.record_id]
            # Strict healthy-only calibration: no damage and no S08 are used.
            threshold = float(max(calibration_scores))
            rows = []
            for row in test.itertuples():
                score = record_score(detector, cache[row.record_id], args.record_quantile)
                rows.append({"preprocessing": variant, "model": model_name, "record_id": row.record_id,
                             "state": row.state, "scenario": row.scenario, "excitation": row.excitation,
                             "setup": int(row.setup), "label": int(row.label), "score": score,
                             "threshold": threshold, "alarm": score > threshold})
            scores = pd.DataFrame(rows); all_scores.append(scores)
            metrics = common.metrics(scores.label, scores.score, threshold)
            summaries.append({"preprocessing": variant, "model": model_name,
                              "train_records": len(train), "calibration_records": len(calibration),
                              "damage_used_for_fit": False, "s08_used_for_fit": False,
                              "threshold_rule": "max S01/val record q95", **metrics})
            print(f"done preprocessing={variant} model={model_name} AUROC={metrics['auroc']:.4f} "
                  f"BA={metrics['balanced_accuracy']:.4f}", flush=True)

    summary = pd.DataFrame(summaries); score_frame = pd.concat(all_scores, ignore_index=True)
    raw_metrics = summary[summary.preprocessing == "raw_logpsd"].set_index("model")
    deltas = []
    for row in summary[summary.preprocessing == "cm_hrr"].itertuples():
        before = raw_metrics.loc[row.model]
        deltas.append({"model": row.model,
                       "delta_auroc": row.auroc - before.auroc,
                       "delta_auprc": row.auprc - before.auprc,
                       "delta_balanced_accuracy": row.balanced_accuracy - before.balanced_accuracy,
                       "delta_f1": row.f1 - before.f1,
                       "auroc_improved": row.auroc > before.auroc,
                       "balanced_accuracy_improved": row.balanced_accuracy > before.balanced_accuracy})
    delta = pd.DataFrame(deltas)
    summary.to_csv(args.output / "metrics.csv", index=False); delta.to_csv(args.output / "cm_hrr_deltas.csv", index=False)
    score_frame.to_csv(args.output / "locked_test_record_scores.csv", index=False)
    state_scores = score_frame.groupby(["preprocessing", "model", "state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score_median=("score", "median"), alarm_rate=("alarm", "mean"))
    state_scores["alarm"] = state_scores.alarm_rate >= .5
    state_scores.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    metadata = {"method": "CM-HRR model-agnostic preprocessing ablation",
                "protocol": "fit=S01/train; calibration=S01/val; evaluation=split/test",
                "reference_fallback": "setup+excitation -> excitation -> global; train rows only",
                "feature": "energy-normalised log-Welch PSD, 96 bins by default",
                "parameters": vars(args), "elapsed_seconds": time.time() - started}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot = summary.pivot(index="model", columns="preprocessing", values="auroc")
    pivot.plot(kind="bar", ax=axes[0], ylim=(0, 1), title="Locked-test AUROC")
    pivot_ba = summary.pivot(index="model", columns="preprocessing", values="balanced_accuracy")
    pivot_ba.plot(kind="bar", ax=axes[1], ylim=(0, 1), title="Locked-test balanced accuracy")
    for ax in axes: ax.grid(axis="y", alpha=.25); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(args.output / "cm_hrr_model_comparison.png", dpi=190); plt.close(fig)
    print("\nRESULTS\n" + summary.to_string(index=False), flush=True)
    print("\nCM-HRR DELTAS\n" + delta.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__": main()

