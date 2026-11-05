"""Z24 adaptation of the MDFA preprocessing workflow (Information Fusion 2026).

Paper: Mousavi et al., "A data-driven digital twin model for bridge health
monitoring using feature fusion and unsupervised deep learning",
Information Fusion 126 (2026) 103534, DOI 10.1016/j.inffus.2025.103534.

Faithful components:
  * healthy-only training and healthy validation;
  * linear detrending and a Welch-derived low-pass bandwidth;
  * parallel raw, time, frequency and three-level db4 wavelet branches;
  * a separate encoder per branch followed by latent feature fusion;
  * reconstruction error as the damage index.

Disclosed Z24 adaptation:
The paper uses a fixed accelerometer network, whereas Z24 PDT has 15--17
roving vertical channels.  Consequently, each sensor-window is treated as one
sample and record decisions aggregate its sensor-window errors.  This preserves
all measured channels without padding fictitious sensors.  It is an equation /
workflow reproduction, not author code (no public author repository found).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import torch
from scipy.signal import butter, detrend, sosfiltfilt, welch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_z24_strict_psd_scatterad as common


VAL_DAMAGE = {"04", "05", "06", "10", "14", "16", "17"}
TEST_DAMAGE = {"03", "07", "09", "11", "12", "13", "15"}
BRANCHES = ("raw", "time", "frequency", "wavelet")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def bandwidth_90(values: list[np.ndarray], fs: float, minimum: float, maximum: float) -> float:
    """Welch bandwidth containing 90% of healthy training power."""
    spectra = []
    for record in values:
        for sensor in range(record.shape[1]):
            f, p = welch(detrend(record[:, sensor]), fs=fs,
                         nperseg=min(2048, len(record)))
            spectra.append(p)
    power = np.mean(spectra, axis=0)
    cumulative = np.cumsum(power)
    cutoff = float(f[np.searchsorted(cumulative, .90 * cumulative[-1])])
    return float(np.clip(cutoff, minimum, min(maximum, .95 * fs / 2)))


def resample_vector(x: np.ndarray, points: int) -> np.ndarray:
    old = np.linspace(0., 1., len(x)); new = np.linspace(0., 1., points)
    return np.interp(new, old, x).astype(np.float32)


def spectral_moments(f: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    weight = p / max(float(p.sum()), np.finfo(float).tiny)
    centroid = float(np.sum(f * weight))
    spread = float(np.sqrt(np.sum((f - centroid) ** 2 * weight)))
    kurtosis = float(np.sum((f - centroid) ** 4 * weight) / max(spread ** 4, 1e-12))
    return centroid, spread, kurtosis


def shannon_histogram(x: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(x, bins=bins)
    prob = hist[hist > 0] / max(hist.sum(), 1)
    return float(-np.sum(prob * np.log2(prob)))


def extract_record(values: np.ndarray, fs: float, window: int, stride: int,
                   raw_points: int, psd_bins: int, cutoff: float) -> dict[str, np.ndarray]:
    """Return paper branches for every (window, vertical-sensor) pair."""
    values = detrend(values, axis=0, type="linear")
    sos = butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, values, axis=0)
    grid = np.linspace(0., cutoff, psd_bins)
    result = {name: [] for name in BRANCHES}
    for start in range(0, len(values) - window + 1, stride):
        raw_block = values[start:start + window]
        feature_block = filtered[start:start + window]
        for sensor in range(values.shape[1]):
            # The raw branch is detrended but deliberately not low-pass filtered,
            # matching the paper's separation of raw and feature streams.
            raw = raw_block[:, sensor]
            raw = (raw - raw.mean()) / max(float(raw.std()), 1e-8)
            result["raw"].append(resample_vector(raw, raw_points))

            x = feature_block[:, sensor]
            rms = float(np.sqrt(np.mean(x ** 2)))
            crest = float(np.max(np.abs(x)) / max(rms, 1e-12))
            result["time"].append(np.array([rms, crest, shannon_histogram(x)], np.float32))

            f, p = welch(x, fs=fs, nperseg=min(1024, window), noverlap=min(512, window // 2))
            band = f <= cutoff; f, p = f[band], p[band]
            shape = np.interp(grid, f, p)
            shape /= max(float(np.trapezoid(shape, grid)), np.finfo(float).tiny)
            centroid, spread, kurtosis = spectral_moments(grid, shape)
            result["frequency"].append(np.r_[np.log10(shape + 1e-12), centroid, spread, kurtosis].astype(np.float32))

            coeffs = pywt.wavedec(x, "db4", level=3)
            energy = np.asarray([np.sum(c * c) for c in coeffs], dtype=np.float64)
            result["wavelet"].append((energy / max(float(energy.sum()), 1e-12)).astype(np.float32))
    if not result["raw"]: raise ValueError("Record shorter than --window")
    return {name: np.stack(parts) for name, parts in result.items()}


def fit_scalers(records: list[dict[str, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scalers = {}
    for name in BRANCHES:
        x = np.concatenate([r[name] for r in records])
        center = np.median(x, axis=0)
        scale = 1.4826 * np.median(np.abs(x - center), axis=0)
        scale[scale < 1e-7] = 1.
        scalers[name] = (center.astype(np.float32), scale.astype(np.float32))
    return scalers


def transform(record: dict[str, np.ndarray], scalers) -> dict[str, np.ndarray]:
    return {name: np.clip((record[name] - scalers[name][0]) / scalers[name][1], -10, 10).astype(np.float32)
            for name in BRANCHES}


def concatenate(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([r[name] for r in records]) for name in BRANCHES}


class MDFA(nn.Module):
    def __init__(self, dimensions: dict[str, int], hidden: int, latent: int):
        super().__init__()
        self.encoders = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(dimensions[name], hidden), nn.ELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ELU()) for name in BRANCHES})
        self.fusion = nn.Sequential(nn.Linear(hidden * len(BRANCHES), hidden), nn.ELU(),
                                    nn.Linear(hidden, latent))
        self.decoders = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(latent, hidden), nn.ELU(), nn.Linear(hidden, dimensions[name]))
            for name in BRANCHES})

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = torch.cat([self.encoders[name](batch[name]) for name in BRANCHES], dim=-1)
        latent = self.fusion(encoded)
        return {name: self.decoders[name](latent) for name in BRANCHES}


def make_loader(data: dict[str, np.ndarray], batch: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(data[name]) for name in BRANCHES]
    return DataLoader(TensorDataset(*tensors), batch_size=batch, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def unpack(items, device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in zip(BRANCHES, items)}


def loss_vector(prediction, target) -> torch.Tensor:
    # Equal branch contribution is the MDFA fusion principle; otherwise the
    # high-dimensional raw/PSD branches would dominate solely by dimension.
    return torch.stack([torch.mean((prediction[n] - target[n]) ** 2, dim=1)
                        for n in BRANCHES], dim=1).mean(1)


@torch.no_grad()
def sample_scores(model, data, batch, device) -> np.ndarray:
    model.eval(); result = []
    for items in make_loader(data, batch, False):
        target = unpack(items, device); result.append(loss_vector(model(target), target).cpu().numpy())
    return np.concatenate(result)


def train_model(train, validation, args, device):
    dimensions = {name: train[name].shape[1] for name in BRANCHES}
    model = MDFA(dimensions, args.hidden, args.latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for items in make_loader(train, args.batch_size, True):
            target = unpack(items, device); optimizer.zero_grad(set_to_none=True)
            loss = loss_vector(model(target), target).mean(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))
        val_loss = float(sample_scores(model, validation, args.batch_size, device).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "healthy_validation_loss": val_loss})
        if val_loss < best - 1e-7:
            best, state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch}/{args.epochs} train={train_loss:.6f} healthy_val={val_loss:.6f}", flush=True)
        if stale >= args.patience: break
    if state is None: raise RuntimeError("No model checkpoint")
    model.load_state_dict(state)
    return model, pd.DataFrame(history)


def record_score(model, record, args, device) -> float:
    return float(np.quantile(sample_scores(model, record, args.batch_size, device), args.record_quantile))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_mdfa_paper"))
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=32); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--fs", type=float, default=50.)
    p.add_argument("--window", type=int, default=4096); p.add_argument("--stride", type=int, default=2048)
    p.add_argument("--raw-points", type=int, default=128); p.add_argument("--psd-bins", type=int, default=64)
    p.add_argument("--min-cutoff", type=float, default=10.); p.add_argument("--max-cutoff", type=float, default=20.)
    p.add_argument("--record-quantile", type=float, default=.95)
    p.add_argument("--threshold-quantile", type=float, default=.95); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = common.resolve_manifest(args.manifest, args.data_root)
    # Respect the exported record-level split verbatim.  All preprocessing
    # statistics (bandwidth and robust scaling) are fit on healthy train only.
    train_rows = frame[(frame.split == "train") & (frame.label == 0)].sort_values(
        ["state", "setup", "excitation"])
    cal_rows = frame[(frame.split == "val") & (frame.label == 0)].sort_values(
        ["state", "setup", "excitation"])
    validation_rows = frame[(frame.split == "val") & (frame.label == 1)].sort_values(
        ["state", "setup", "excitation"])
    test_rows = frame[frame.split == "test"].sort_values(["state", "setup", "excitation"])
    if (len(train_rows), len(cal_rows), len(validation_rows), len(test_rows)) != (24, 6, 126, 132):
        raise ValueError("Unexpected manifest split; expected train/cal/val-damage/test = 24/6/126/132")
    raw_train = [common.vertical_values(path) for path in train_rows.path]
    cutoff = bandwidth_90(raw_train, args.fs, args.min_cutoff, args.max_cutoff)
    print(f"device={device} train_healthy={len(train_rows)} cal_healthy={len(cal_rows)} "
          f"validation_damage={len(validation_rows)} locked_test={len(test_rows)} cutoff={cutoff:.3f}Hz", flush=True)
    cache = {}
    all_rows = pd.concat([train_rows, cal_rows, validation_rows, test_rows]).drop_duplicates("record_id")
    for i, row in enumerate(all_rows.itertuples(), 1):
        cache[row.record_id] = extract_record(common.vertical_values(row.path), args.fs, args.window,
                                               args.stride, args.raw_points, args.psd_bins, cutoff)
        if i % 18 == 0 or i == len(all_rows): print(f"features={i}/{len(all_rows)}", flush=True)
    scalers = fit_scalers([cache[x] for x in train_rows.record_id])
    scaled = {key: transform(value, scalers) for key, value in cache.items()}
    train = concatenate([scaled[x] for x in train_rows.record_id])
    cal = concatenate([scaled[x] for x in cal_rows.record_id])
    started = time.time(); model, history = train_model(train, cal, args, device)
    history.to_csv(args.output / "training_history.csv", index=False)
    cal_scores = np.asarray([record_score(model, scaled[x], args, device) for x in cal_rows.record_id])
    threshold = float(np.quantile(cal_scores, args.threshold_quantile))

    def evaluate(rows, split):
        output = []
        for row in rows.itertuples():
            score = record_score(model, scaled[row.record_id], args, device)
            output.append({"record_id": row.record_id, "state": row.state, "scenario": row.scenario,
                           "excitation": row.excitation, "setup": int(row.setup), "label": int(row.label),
                           "split": split, "score": score, "alarm": score > threshold})
        return pd.DataFrame(output)

    val_healthy = pd.DataFrame([{"record_id": row.record_id, "state": "01", "scenario": row.scenario,
                                 "excitation": row.excitation, "setup": int(row.setup), "label": 0,
                                 "split": "healthy_validation", "score": score, "alarm": score > threshold}
                                for row, score in zip(cal_rows.itertuples(), cal_scores)])
    validation = pd.concat([val_healthy, evaluate(validation_rows, "damage_validation")], ignore_index=True)
    test = evaluate(test_rows, "locked_test")
    validation.to_csv(args.output / "validation_record_scores.csv", index=False)
    test.to_csv(args.output / "locked_test_record_scores.csv", index=False)
    val_metrics = common.metrics(validation.label, validation.score, threshold)
    test_metrics = common.metrics(test.label, test.score, threshold)
    states = test.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("score", "median"))
    states["alarm"] = states.score > threshold; states.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    state_metrics = common.metrics(states.label, states.score, threshold)
    summary = {"method": "MDFA_InformationFusion2026_workflow_reproduction_Z24_adapter",
               "healthy_train_records": len(train_rows), "healthy_calibration_records": len(cal_rows),
               "damage_validation_records": len(validation_rows), "damage_used_for_gradient_fit": False,
               "damage_used_for_threshold": False, "damage_validation_opened_before_locked_test": True,
               "welch_90pct_cutoff_hz": cutoff, "epochs_run": int(history.epoch.max()),
               "seconds": time.time() - started,
               **{f"validation_{k}": v for k, v in val_metrics.items()},
               **{f"record_{k}": v for k, v in test_metrics.items()},
               **{f"state_{k}": v for k, v in state_metrics.items()}}
    pd.DataFrame([summary]).to_csv(args.output / "metrics.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "scalers": scalers, "cutoff": cutoff,
                "threshold": threshold, "args": vars(args)}, args.output / "model.pt")
    metadata = {"paper": "Mousavi et al., Information Fusion 126 (2026) 103534",
                "doi": "10.1016/j.inffus.2025.103534", "author_code_available": False,
                "reproduced": ["OLS detrend", "Welch 90% bandwidth", "Butterworth low-pass for feature streams",
                               "raw/time/frequency/db4-wavelet branches", "branch encoders and latent fusion",
                               "healthy-only reconstruction training"],
                "z24_adaptation": "sensor-window samples because PDT sensor count varies by setup; no fake padding",
                "split": {"train": "manifest train: 24 healthy S01/S08 records",
                          "calibration": "manifest val: 6 healthy S01/S08 records",
                          "validation": "manifest val: 126 damage records",
                          "locked_test": "manifest test: 6 healthy + 126 damage records"},
                "parameters": vars(args)}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(history.epoch, history.train_loss, label="train healthy")
    axes[0].plot(history.epoch, history.healthy_validation_loss, label="validation healthy")
    axes[0].legend(); axes[0].set_title("MDFA reconstruction loss")
    fpr, tpr, _ = roc_curve(test.label, test.score); axes[1].plot(fpr, tpr, label=f"AUROC={test_metrics['auroc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray"); axes[1].legend(); axes[1].set_title("Locked-test ROC")
    ConfusionMatrixDisplay(confusion_matrix(test.label, test.alarm, labels=[0, 1]),
                           display_labels=["Healthy", "Damage"]).plot(ax=axes[2], cmap="Blues", colorbar=False)
    axes[2].set_title("Locked-test confusion matrix"); fig.tight_layout()
    fig.savefig(args.output / "mdfa_results.png", dpi=190); plt.close(fig)
    print("\nVALIDATION\n" + pd.DataFrame([{f"validation_{k}": v for k, v in val_metrics.items()}]).to_string(index=False))
    print("\nLOCKED TEST\n" + pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__": main()
