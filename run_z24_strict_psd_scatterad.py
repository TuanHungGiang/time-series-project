"""Strict healthy-only Z24 baseline using the official ScatterAD model.

Training and calibration use state 01 only.  State 08 and every damage state
are evaluation-only.  Raw records remain the split unit: setup-grouped
cross-fitting produces out-of-fold healthy scores used to lock the threshold.

ScatterAD is used unchanged after a disclosed adapter converts each vertical
sensor window to an energy-normalised log-Welch-PSD vector.  This is a baseline
adapter, not a reproduction of the paper's original benchmark protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCATTER_ROOT = ROOT / "baselines_repo" / "ScatterAD"
sys.path.insert(0, str(SCATTER_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import detrend, resample_poly, welch
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_manifest(path: Path, root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"state": str})
    frame["state"] = frame.state.str.zfill(2)
    frame = frame[frame.state != "02"].copy()
    if "relative_path" in frame:
        frame["path"] = frame.relative_path.map(lambda x: (root / str(x)).resolve())
    elif "path" in frame:
        frame["path"] = frame.path.map(Path)
    else:
        raise ValueError("Manifest needs relative_path or path")
    missing = frame.loc[~frame.path.map(Path.exists), "path"]
    if len(missing): raise FileNotFoundError(f"Missing {len(missing)} files; first={missing.iloc[0]}")
    train = frame[frame.state == "01"]
    if len(train) != 18 or set(train.label.astype(int)) != {0}:
        raise ValueError(f"Strict protocol expects exactly 18 healthy S01 records, got {len(train)}")
    return frame


def vertical_values(path: Path) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    labels = [str(x).strip() for x in mat["labelshulp"].tolist()]
    use = [i for i, name in enumerate(labels)
           if name.endswith("V") and name[0].isdigit() and not name.startswith(("R", "DP"))]
    values = np.asarray(mat["data"][:, use], dtype=np.float32)
    if not np.isfinite(values).all(): raise ValueError(f"NaN/Inf in {path}")
    return resample_poly(values, 1, 2, axis=0).astype(np.float32)


def psd_vectors(path: Path, fs: float, raw_window: int, raw_stride: int,
                bins: int, fmin: float, fmax: float) -> np.ndarray:
    """Return independent sensor-window PSD shapes; no fitted statistics."""
    values = vertical_values(path)
    grid = np.linspace(fmin, fmax, bins)
    output = []
    for channel in range(values.shape[1]):
        signal = values[:, channel]
        for start in range(0, len(signal) - raw_window + 1, raw_stride):
            part = detrend(signal[start:start + raw_window], type="linear")
            freq, power = welch(part, fs=fs, nperseg=min(512, raw_window),
                                noverlap=min(256, raw_window // 2))
            band = (freq >= fmin) & (freq <= fmax)
            shape = np.interp(grid, freq[band], power[band])
            area = np.trapezoid(shape, grid)
            shape = np.log10(shape / max(float(area), np.finfo(float).tiny)
                             + np.finfo(float).tiny)
            output.append(shape.astype(np.float32))
    if not output: raise ValueError(f"No PSD windows extracted from {path}")
    return np.stack(output)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray): self.x = torch.from_numpy(x.astype(np.float32))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def loader(x: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(ArrayDataset(x), batch_size=batch, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def chain_edges(length: int, device: torch.device) -> torch.Tensor:
    pairs = [(i, i + d) for i in range(length) for d in (-1, 1) if 0 <= i + d < length]
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()


class GraphBatcher:
    def __init__(self, length: int, device: torch.device):
        self.length, self.device = length, device
        self.base = chain_edges(length, device); self.cache = {}
    def convert(self, batch: torch.Tensor):
        batch = batch.to(self.device, non_blocking=True)
        n = len(batch)
        if n not in self.cache:
            offsets = torch.arange(n, device=self.device).view(-1, 1, 1) * self.length
            edge = self.base.view(1, 2, -1) + offsets
            self.cache[n] = edge.permute(1, 0, 2).reshape(2, -1)
        return batch.reshape(n * self.length, 1), self.cache[n]


def fit_scaler(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(parts)
    center = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - center), axis=0)
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def transform(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((x - center) / scale, -10, 10).astype(np.float32)


def train_model(parts: list[np.ndarray], args, device, run_name: str):
    # ScatterAD is an optional dependency for modules that only reuse this
    # file's Z24 loading/metrics helpers (for example the Procrustes baseline).
    # Import it only when a ScatterAD model is actually trained.
    from model.ScatterAD import Model

    center, scale = fit_scaler(parts)
    x = transform(np.concatenate(parts), center, scale)
    model = Model(input_dim=1, hidden_dim=args.hidden, num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batcher = GraphBatcher(args.psd_bins, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in loader(x, args.batch_size, True):
            nodes, edge = batcher.convert(batch)
            optimizer.zero_grad(set_to_none=True)
            pred, target = model.model(nodes, edge)
            loss = model.model.compute_loss(nodes, pred, target, edge)
            loss.backward(); optimizer.step(); model.model.update_target()
            losses.append(float(loss.detach().cpu()))
        value = float(np.mean(losses))
        history.append({"run": run_name, "epoch": epoch, "train_loss": value})
        print(f"{run_name} epoch={epoch}/{args.epochs} loss={value:.6f}", flush=True)
    return model, center, scale, batcher, history


@torch.no_grad()
def vector_scores(model, x: np.ndarray, center: np.ndarray, scale: np.ndarray,
                  batcher: GraphBatcher, batch_size: int) -> np.ndarray:
    model.eval(); result = []
    for batch in loader(transform(x, center, scale), batch_size, False):
        nodes, edge = batcher.convert(batch)
        score = model.model.get_anomaly_scores(nodes, edge)
        result.append(score.reshape(len(batch), batcher.length).mean(1).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def record_score(model, features, center, scale, batcher, args) -> float:
    scores = vector_scores(model, features, center, scale, batcher, args.batch_size)
    return float(np.quantile(scores, args.record_quantile))


def robust_reference(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if scale < 1e-9: scale = float(np.std(values))
    return center, max(scale, 1e-9)


def metrics(y, score, threshold) -> dict:
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    pred = (score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "specificity": float(tn / max(tn + fp, 1)),
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "threshold": float(threshold)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_strict_psd_scatterad"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--raw-window", type=int, default=2048)
    p.add_argument("--raw-stride", type=int, default=2048)
    p.add_argument("--psd-bins", type=int, default=96)
    p.add_argument("--fmin", type=float, default=3.0)
    p.add_argument("--fmax", type=float, default=20.0)
    p.add_argument("--record-quantile", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not 2 <= args.cv_folds <= 9: raise ValueError("--cv-folds must be in [2, 9]")
    args.output.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = resolve_manifest(args.manifest, args.data_root)
    train = frame[frame.state == "01"].sort_values(["setup", "excitation"]).copy()
    test = frame[frame.state != "01"].sort_values(["state", "setup", "excitation"]).copy()
    print(f"device={device} strict_train_S01={len(train)} evaluation={len(test)}", flush=True)

    feature_args = (args.fs, args.raw_window, args.raw_stride, args.psd_bins, args.fmin, args.fmax)
    cache = {}
    for i, row in enumerate(train.itertuples(), 1):
        cache[row.record_id] = psd_vectors(row.path, *feature_args)
        print(f"features S01 {i}/{len(train)} {row.record_id} {cache[row.record_id].shape}", flush=True)

    histories, oof_rows = [], []
    # Setup is the cross-fit group; AVT/FVT for one setup always remain together.
    train["fold"] = (train.setup.astype(int) - 1) % args.cv_folds
    for fold in range(args.cv_folds):
        fit_rows = train[train.fold != fold]; held = train[train.fold == fold]
        model, center, scale, batcher, hist = train_model(
            [cache[x] for x in fit_rows.record_id], args, device, f"cv{fold}")
        histories.extend(hist)
        fit_scores = [record_score(model, cache[x], center, scale, batcher, args)
                      for x in fit_rows.record_id]
        ref_center, ref_scale = robust_reference(fit_scores)
        for row in held.itertuples():
            raw = record_score(model, cache[row.record_id], center, scale, batcher, args)
            oof_rows.append({"record_id": row.record_id, "setup": int(row.setup),
                             "excitation": row.excitation, "fold": fold,
                             "raw_score": raw, "score": (raw - ref_center) / ref_scale})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    oof = pd.DataFrame(oof_rows).sort_values("record_id")
    if len(oof) != 18 or oof.record_id.nunique() != 18:
        raise RuntimeError("Cross-fitting failed to score every S01 record exactly once")
    # Maximum OOF healthy score: zero empirical S01 false alarms, fixed before test.
    threshold = float(oof.score.max())
    oof.to_csv(args.output / "s01_crossfit_scores.csv", index=False)

    final_model, center, scale, batcher, hist = train_model(
        [cache[x] for x in train.record_id], args, device, "final")
    histories.extend(hist)
    final_train_raw = [record_score(final_model, cache[x], center, scale, batcher, args)
                       for x in train.record_id]
    ref_center, ref_scale = robust_reference(final_train_raw)
    pd.DataFrame(histories).to_csv(args.output / "training_history.csv", index=False)
    torch.save({"state_dict": final_model.state_dict(), "feature_center": center,
                "feature_scale": scale, "score_reference": (ref_center, ref_scale),
                "threshold": threshold, "args": vars(args)}, args.output / "model.pt")

    rows = []; started = time.time()
    for i, row in enumerate(test.itertuples(), 1):
        features = psd_vectors(row.path, *feature_args)
        raw = record_score(final_model, features, center, scale, batcher, args)
        score = (raw - ref_center) / ref_scale
        rows.append({"record_id": row.record_id, "state": row.state,
                     "scenario": row.scenario, "excitation": row.excitation,
                     "setup": int(row.setup), "label": int(row.label),
                     "raw_score": raw, "score": score, "alarm": score > threshold})
        if i % 18 == 0 or i == len(test): print(f"evaluated={i}/{len(test)}", flush=True)
    records = pd.DataFrame(rows)
    records.to_csv(args.output / "record_scores.csv", index=False)
    states = records.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("score", "median"))
    states["alarm"] = states.score > threshold
    states.to_csv(args.output / "state_scores.csv", index=False)
    record_result = metrics(records.label, records.score, threshold)
    state_result = metrics(states.label, states.score, threshold)
    result = {"method": "ScatterAD_NeurIPS2025_strict_S01_logPSD_adapter",
              "epochs_per_model": args.epochs, "cv_folds": args.cv_folds,
              "train_records": 18, "damage_used_for_fit": False,
              "s08_used_for_fit": False, "evaluation_seconds": time.time() - started,
              **{f"record_{k}": v for k, v in record_result.items()},
              **{f"state_{k}": v for k, v in state_result.items()}}
    pd.DataFrame([result]).to_csv(args.output / "metrics.csv", index=False)
    metadata = {"official_repository": "https://github.com/jk-sounds/ScatterAD",
                "official_commit": "7572fa954468e46c598badf1c9cfaf6c1835c4cb",
                "paper": "ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection, NeurIPS 2025",
                "disclosed_adapter": "energy-normalised log-Welch-PSD per vertical sensor window",
                "strict_protocol": "S01 only for fitting/cross-fit threshold; S08 and damage evaluation-only",
                "split_unit": "whole MAT record; AVT/FVT paired by setup in cross-fitting",
                "threshold_rule": "maximum standardized out-of-fold S01 record score",
                "parameters": vars(args)}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    history = pd.DataFrame(histories)
    if len(history):
        for name, group in history.groupby("run"):
            axes[0, 0].plot(group.epoch, group.train_loss, marker="o", label=name)
    else:
        axes[0, 0].text(.5, .5, "Smoke run: epochs=0", ha="center", va="center")
    axes[0, 0].set(title="Healthy-only training loss", xlabel="Epoch", ylabel="ScatterAD loss")
    if len(history): axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(alpha=.25)
    fpr, tpr, _ = roc_curve(records.label, records.score)
    axes[0, 1].plot(fpr, tpr, label=f"AUROC={record_result['auroc']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray"); axes[0, 1].legend()
    axes[0, 1].set(title="Record ROC", xlabel="FPR", ylabel="TPR")
    precision, recall, _ = precision_recall_curve(records.label, records.score)
    axes[1, 0].plot(recall, precision, label=f"AUPRC={record_result['auprc']:.3f}")
    axes[1, 0].legend(); axes[1, 0].set(title="Record precision-recall", xlabel="Recall", ylabel="Precision")
    pred = (states.score > threshold).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(states.label, pred, labels=[0, 1]),
                           display_labels=["Healthy", "Damage"]).plot(
                               ax=axes[1, 1], cmap="Greens", colorbar=False)
    axes[1, 1].set_title("State confusion matrix")
    fig.tight_layout(); fig.savefig(args.output / "results.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("\nRESULT\n" + pd.DataFrame([result]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

