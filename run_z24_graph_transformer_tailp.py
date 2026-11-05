"""Strict Z24 Graph-CNN + Transformer + Tail-p paper-method experiment.

Equation-based adaptation of Liu et al., RESS 252 (2024) 110465. Vertical
sensors remain graph nodes; dynamic top-k edges are inferred from each window's
spectral features, and a Transformer models each sensor through time. The model
is a healthy-only reconstruction autoencoder because no public author code was
found. Tail probabilities and the alert threshold use S01 only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import detrend, welch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch import nn

import run_z24_strict_psd_scatterad as base

VAL_DAMAGE = {"04", "05", "06", "10", "14", "16", "17"}
TEST_DAMAGE = {"03", "07", "09", "11", "12", "13", "15"}


def sensor_window_features(path: Path, fs: float, raw_window: int, stride: int,
                           bins: int, fmin: float, fmax: float) -> np.ndarray:
    values = base.vertical_values(path); grid = np.linspace(fmin, fmax, bins)
    starts = range(0, len(values) - raw_window + 1, stride); windows = []
    for start in starts:
        nodes = []
        for sensor in range(values.shape[1]):
            x = detrend(values[start:start + raw_window, sensor], type="linear")
            f, p = welch(x, fs=fs, nperseg=min(1024, raw_window), noverlap=512)
            band = (f >= fmin) & (f <= fmax); shape = np.interp(grid, f[band], p[band])
            energy = max(float(np.trapezoid(shape, grid)), np.finfo(float).tiny)
            log_shape = np.log10(shape / energy + np.finfo(float).tiny)
            nodes.append(np.r_[log_shape, np.log10(energy)].astype(np.float32))
        windows.append(np.stack(nodes))
    if not windows: raise ValueError(f"No windows from {path}")
    return np.stack(windows)  # time, sensors, features


def fit_scaler(records: list[np.ndarray]):
    x = np.concatenate([r.reshape(-1, r.shape[-1]) for r in records])
    center = np.median(x, 0); scale = 1.4826 * np.median(np.abs(x - center), 0)
    scale[scale < 1e-6] = 1.; return center.astype(np.float32), scale.astype(np.float32)


def scale_record(x, center, scale): return np.clip((x - center) / scale, -10, 10).astype(np.float32)


class GraphTemporalAE(nn.Module):
    def __init__(self, features: int, hidden: int, heads: int, layers: int, topk: int):
        super().__init__(); self.topk = topk
        self.input = nn.Linear(features, hidden)
        self.graph = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=.1,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, features))

    def forward(self, x):
        # x: time x sensors x features
        h = self.input(x); unit = nn.functional.normalize(h, dim=-1)
        similarity = torch.einsum("tih,tjh->tij", unit, unit)
        eye = torch.eye(h.shape[1], device=h.device, dtype=torch.bool).unsqueeze(0)
        similarity = similarity.masked_fill(eye, -torch.inf)
        k = min(self.topk, max(h.shape[1] - 1, 1))
        index = similarity.topk(k, dim=-1).indices
        expanded = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        neighbors = torch.gather(expanded, 2, index.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])).mean(2)
        h = self.graph(torch.cat([h, neighbors], -1))
        h = self.temporal(h.permute(1, 0, 2)).permute(1, 0, 2)
        return self.output(h)


def record_error(model, x, device):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, device=device); prediction = model(tensor)
        return torch.mean((prediction - tensor) ** 2, dim=-1).cpu().numpy()


def train_model(train, val, args, device):
    model = GraphTemporalAE(train[0].shape[-1], args.hidden, args.heads,
                            args.layers, args.topk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        order = np.random.permutation(len(train))
        for i in order:
            x = torch.tensor(train[i], device=device); optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - x) ** 2); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = float(np.mean([record_error(model, x, device).mean() for x in val]))
        train_loss = float(np.mean(losses)); history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_healthy_loss": val_loss})
        if val_loss < best - 1e-7: best, state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if stale >= args.patience: break
    model.load_state_dict(state); return model, pd.DataFrame(history)


def empirical_tail(errors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference.reshape(-1)); flat = errors.reshape(-1)
    rank = np.searchsorted(ordered, flat, side="left")
    p = (len(ordered) - rank + 1) / (len(ordered) + 1)
    return p.reshape(errors.shape)


def record_tail_score(errors, reference, time_threshold):
    p = empirical_tail(errors, reference)
    time_scores = np.mean(-np.log10(np.clip(p, np.finfo(float).tiny, 1)), axis=1)
    fraction = float(np.mean(time_scores > time_threshold))
    return fraction, float(np.median(time_scores)), time_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_graph_transformer_tailp"))
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--hidden", type=int, default=64); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--topk", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--fs", type=float, default=50.); p.add_argument("--raw-window", type=int, default=4096)
    p.add_argument("--stride", type=int, default=4096); p.add_argument("--psd-bins", type=int, default=48)
    p.add_argument("--fmin", type=float, default=3.); p.add_argument("--fmax", type=float, default=20.)
    p.add_argument("--tail-quantile", type=float, default=.95)
    p.add_argument("--record-fraction", type=float, default=.20); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paper-split", action="store_true",
                   help="Use mixed damage validation and locked S08+damage test instead of evaluating all states")
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = base.resolve_manifest(args.manifest, args.data_root)
    train_rows = frame[(frame.state == "01") & (frame.setup.astype(int) <= 6)].copy()
    cal_rows = frame[(frame.state == "01") & (frame.setup.astype(int) >= 7)].copy()
    validation_rows = frame[frame.state.isin(VAL_DAMAGE)].copy() if args.paper_split else frame.iloc[0:0].copy()
    evaluation = (frame[frame.state.isin({"08"} | TEST_DAMAGE)].copy() if args.paper_split
                  else frame[~frame.state.isin(["01", "02"])].copy())
    feature_args = (args.fs, args.raw_window, args.stride, args.psd_bins, args.fmin, args.fmax)
    cache = {}
    feature_groups = [("train", train_rows), ("cal", cal_rows)]
    if args.paper_split: feature_groups.append(("damage_validation", validation_rows))
    feature_groups.append(("evaluation", evaluation))
    for group, rows in feature_groups:
        for row in rows.itertuples(): cache[row.record_id] = sensor_window_features(row.path, *feature_args)
        print(f"features {group}={len(rows)}", flush=True)
    train_raw = [cache[x] for x in train_rows.record_id]; center, scale = fit_scaler(train_raw)
    train = [scale_record(x, center, scale) for x in train_raw]
    cal = [scale_record(cache[x], center, scale) for x in cal_rows.record_id]
    started = time.time(); model, history = train_model(train, cal, args, device)
    history.to_csv(args.output / "training_history.csv", index=False)
    train_errors = np.concatenate([record_error(model, x, device).reshape(-1) for x in train])
    cal_time_scores = []
    for x in cal:
        pvals = empirical_tail(record_error(model, x, device), train_errors)
        cal_time_scores.extend(np.mean(-np.log10(np.clip(pvals, np.finfo(float).tiny, 1)), 1))
    time_threshold = float(np.quantile(cal_time_scores, args.tail_quantile))
    cal_fractions = [record_tail_score(record_error(model, x, device), train_errors,
                                       time_threshold)[0] for x in cal]
    print(f"S01 calibration time_threshold={time_threshold:.6f} fractions={cal_fractions}", flush=True)
    def score_rows(source, split):
        rows = []
        for row in source.itertuples():
            x = scale_record(cache[row.record_id], center, scale)
            fraction, median_tail, _ = record_tail_score(
                record_error(model, x, device), train_errors, time_threshold)
            rows.append({"record_id": row.record_id, "state": row.state, "scenario": row.scenario,
                         "excitation": row.excitation, "setup": int(row.setup), "label": int(row.label),
                         "split": split, "tail_fraction": fraction, "median_tail_score": median_tail,
                         "alarm": fraction > args.record_fraction})
        return pd.DataFrame(rows)
    records = score_rows(evaluation, "locked_test" if args.paper_split else "all_evaluation")
    records.to_csv(args.output / "record_scores.csv", index=False)
    validation_metrics = {}
    if args.paper_split:
        damage_validation = score_rows(validation_rows, "damage_validation")
        healthy_validation = []
        for row, fraction in zip(cal_rows.itertuples(), cal_fractions):
            healthy_validation.append({"record_id": row.record_id, "state": "01", "scenario": row.scenario,
                                       "excitation": row.excitation, "setup": int(row.setup), "label": 0,
                                       "split": "healthy_validation", "tail_fraction": fraction,
                                       "median_tail_score": np.nan,
                                       "alarm": fraction > args.record_fraction})
        validation = pd.concat([pd.DataFrame(healthy_validation), damage_validation], ignore_index=True)
        validation.to_csv(args.output / "validation_record_scores.csv", index=False)
        validation_metrics = base.metrics(validation.label, validation.tail_fraction, args.record_fraction)
    result = base.metrics(records.label, records.tail_fraction, args.record_fraction)
    states = records.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("tail_fraction", "median"), alarm_rate=("alarm", "mean"))
    states["alarm"] = states.alarm_rate >= .5; states.to_csv(args.output / "state_scores.csv", index=False)
    sy, sp = states.label.to_numpy(), states.alarm.astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(sy, sp, labels=[0, 1]).ravel()
    summary = {"method": "DynamicGraphCNN_Transformer_TailP_RESS2024_equation_adaptation",
               "damage_used_for_fit": False, "s08_used_for_fit": False,
               "paper_damage_validation_split": bool(args.paper_split),
               "damage_validation_opened_before_locked_test": bool(args.paper_split),
               "epochs_run": int(history.epoch.max()), "seconds": time.time() - started,
               "s01_calibration_alarm_records": int(sum(x > args.record_fraction for x in cal_fractions)),
               **{f"validation_{k}": v for k, v in validation_metrics.items()},
               **{f"record_{k}": v for k, v in result.items()},
               "state_tn": int(tn), "state_fp": int(fp), "state_fn": int(fn), "state_tp": int(tp),
               "state_accuracy": float((sy == sp).mean())}
    pd.DataFrame([summary]).to_csv(args.output / "metrics.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "center": center, "scale": scale,
                "train_errors": train_errors, "time_threshold": time_threshold,
                "args": vars(args)}, args.output / "model.pt")
    meta = {"paper": "Liu et al., Structural damage detection and localization via an unsupervised anomaly detection method, RESS 252 (2024) 110465",
            "doi": "10.1016/j.ress.2024.110465", "implementation_scope": "equation-based healthy reconstruction adaptation; no public author code found",
            "strict_protocol": ("S01 setups 1-6 train, S01 setups 7-9 calibrate, damage validation states "
                                f"{sorted(VAL_DAMAGE)}, locked S08+damage test {sorted(TEST_DAMAGE)}"
                                if args.paper_split else
                                "S01 setups 1-6 train, S01 setups 7-9 calibrate, S08/damage evaluation-only"),
            "parameters": vars(args)}
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8)); axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_healthy_loss, label="S01 cal"); axes[0].legend(); axes[0].set_title("Reconstruction loss")
    fpr, tpr, _ = roc_curve(records.label, records.tail_fraction); axes[1].plot(fpr, tpr, label=f"AUROC={result['auroc']:.3f}")
    axes[1].plot([0,1],[0,1],"--",color="gray"); axes[1].legend(); axes[1].set_title("Record ROC")
    ConfusionMatrixDisplay(confusion_matrix(sy, sp, labels=[0,1]), display_labels=["Healthy","Damage"]).plot(ax=axes[2],cmap="Greens",colorbar=False)
    axes[2].set_title("State confusion matrix"); fig.tight_layout(); fig.savefig(args.output/"graph_tailp_results.png",dpi=190,bbox_inches="tight"); plt.close(fig)
    print("\nRESULT\n" + pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__": main()

