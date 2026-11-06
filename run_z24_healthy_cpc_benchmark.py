"""Healthy-only CPC preprocessing for the exported Z24 split.

Paper idea: Yang et al., Computers in Industry 2023, 103878, healthy-only
self-supervised contrastive predictive representations followed by one-class
fault detection. No author code was available in the repository, therefore
this is a disclosed equation/workflow adaptation to bridge PSD sequences.

Only S01/train fits PSD scaling and CPC. S01/val performs CPC early stopping
and healthy threshold calibration. S08 and damage never fit CPC. The frozen
representation is tested with two existing paper backbones on identical data:
target-to-source mapping + OCSVM and source AE + orthogonal Procrustes.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import detrend, welch
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal
from run_z24_target_source_ocsvm import nearest_source, train_mapper, map_features
from run_z24_procrustes_ae import (
    train_source_ae, encode, fit_rotation, reconstruction_errors,
)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def record_psd_sequence(path: Path, fs: float, window: int, stride: int,
                        bins: int, fmin: float, fmax: float) -> np.ndarray:
    """Permutation-invariant temporal sequence of sensor-median log-PSD shapes."""
    values = common.vertical_values(path)
    grid = np.linspace(fmin, fmax, bins); output = []
    for start in range(0, len(values) - window + 1, stride):
        sensor_psd = []
        for channel in range(values.shape[1]):
            signal = detrend(values[start:start + window, channel], type="linear")
            f, p = welch(signal, fs=fs, nperseg=min(512, window),
                         noverlap=min(256, window // 2))
            keep = (f >= fmin) & (f <= fmax)
            shape = np.interp(grid, f[keep], p[keep])
            shape /= max(float(np.trapezoid(shape, grid)), np.finfo(float).tiny)
            sensor_psd.append(np.log10(shape + np.finfo(float).tiny))
        output.append(np.median(np.stack(sensor_psd), axis=0))
    if len(output) < 8: raise ValueError(f"Too few PSD windows: {path}")
    return np.asarray(output, np.float32)


def fit_robust(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(parts); center = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - center), axis=0)
    positive = scale[scale > 1e-6]
    floor = max(float(np.median(positive)) * .02 if len(positive) else 0., 1e-4)
    return center.astype(np.float32), np.maximum(scale, floor).astype(np.float32)


def scale_sequences(cache, center, scale):
    return {key: np.clip((value - center) / scale, -10., 10.).astype(np.float32)
            for key, value in cache.items()}


def segments(parts: list[np.ndarray], length: int, stride: int) -> np.ndarray:
    result = []
    for value in parts:
        for start in range(0, len(value) - length + 1, stride):
            result.append(value[start:start + length])
    if not result: raise ValueError("No CPC sequences; reduce --context-length")
    return np.stack(result).astype(np.float32)


class HealthyCPC(nn.Module):
    def __init__(self, features: int, latent: int, context: int, prediction_steps: int):
        super().__init__(); self.prediction_steps = prediction_steps
        self.encoder = nn.Sequential(nn.Linear(features, 128), nn.ELU(),
                                     nn.Linear(128, latent), nn.LayerNorm(latent))
        self.context = nn.GRU(latent, context, batch_first=True)
        self.predictors = nn.ModuleList([nn.Linear(context, latent)
                                         for _ in range(prediction_steps)])

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x); c, _ = self.context(z); return c

    def loss(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        z = F.normalize(self.encoder(x), dim=-1); context, _ = self.context(z)
        losses = []
        for index, step in enumerate(range(1, self.prediction_steps + 1)):
            predicted = F.normalize(self.predictors[index](context[:, :-step]), dim=-1)
            target = z[:, step:]
            p = predicted.reshape(-1, predicted.shape[-1]); q = target.reshape(-1, target.shape[-1])
            logits = p @ q.T / temperature
            labels = torch.arange(len(p), device=x.device)
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean()


def physics_safe_view(x: torch.Tensor, noise: float, mask_probability: float) -> torch.Tensor:
    """No time warp/resampling/frequency shift: preserve modal coordinates."""
    view = x + noise * torch.randn_like(x)
    if mask_probability > 0:
        mask = torch.rand(view.shape[0], 1, view.shape[2], device=view.device) < mask_probability
        view = view.masked_fill(mask, 0.)
    return view


@torch.no_grad()
def cpc_loss(model, x, args, device):
    model.eval(); values = []
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=args.batch_size):
        values.append(float(model.loss(batch.to(device), args.temperature).cpu()))
    return float(np.mean(values))


def train_cpc(train, validation, args, device):
    model = HealthyCPC(train.shape[-1], args.cpc_latent, args.cpc_context,
                       args.prediction_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.cpc_lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(train)), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.cpc_epochs + 1):
        model.train(); losses = []
        for (batch,) in loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            # Train both invariant healthy views without treating their matching
            # time positions as cross-view negatives.
            view_a = physics_safe_view(batch, args.aug_noise, args.mask_probability)
            view_b = physics_safe_view(batch, args.aug_noise, args.mask_probability)
            loss = .5 * (model.loss(view_a, args.temperature) +
                         model.loss(view_b, args.temperature))
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = cpc_loss(model, validation, args, device); train_value = float(np.mean(losses))
        history.append({"epoch": epoch, "cpc_train_loss": train_value, "healthy_val_loss": val})
        if val < best - 1e-5: best, state, stale = val, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"cpc epoch={epoch}/{args.cpc_epochs} train={train_value:.6f} val={val:.6f}", flush=True)
        if stale >= args.cpc_patience: break
    if state is None: raise RuntimeError("CPC produced no checkpoint")
    model.load_state_dict(state); return model, pd.DataFrame(history)


@torch.no_grad()
def healthy_center(model, sequences, batch_size, device):
    model.eval(); total, count = None, 0
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(sequences)), batch_size=batch_size):
        context = model.embed(batch.to(device)).reshape(-1, model.context.hidden_size)
        value = context.sum(dim=0); total = value if total is None else total + value; count += len(context)
    center = total / max(count, 1)
    # Deep-SVDD convention: avoid dimensions exactly at zero.
    eps = 1e-3; center[(center.abs() < eps) & (center < 0)] = -eps
    center[(center.abs() < eps) & (center >= 0)] = eps
    return center.detach()


@torch.no_grad()
def compact_val_loss(model, sequences, center, args, device):
    model.eval(); values = []
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(sequences)), batch_size=args.batch_size):
        batch = batch.to(device); context = model.embed(batch)
        compact = torch.mean((context - center.view(1, 1, -1)) ** 2)
        value = args.compact_cpc_weight * model.loss(batch, args.temperature) + args.compact_weight * compact
        values.append(float(value.cpu()))
    return float(np.mean(values))


def train_compact_cpc(model, train, validation, center, args, device):
    """OATC-inspired one-class refinement: CPC + compactness + view consistency."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.compact_lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(train)), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.compact_epochs + 1):
        model.train(); totals, compact_values, consistency_values = [], [], []
        for (batch,) in loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            view_a = physics_safe_view(batch, args.aug_noise, args.mask_probability)
            view_b = physics_safe_view(batch, args.aug_noise, args.mask_probability)
            context_a, context_b = model.embed(view_a), model.embed(view_b)
            compact = .5 * (torch.mean((context_a - center.view(1, 1, -1)) ** 2) +
                            torch.mean((context_b - center.view(1, 1, -1)) ** 2))
            consistency = 1. - F.cosine_similarity(context_a, context_b, dim=-1).mean()
            predictive = .5 * (model.loss(view_a, args.temperature) +
                                model.loss(view_b, args.temperature))
            loss = (args.compact_cpc_weight * predictive + args.compact_weight * compact +
                    args.consistency_weight * consistency)
            loss.backward(); optimizer.step()
            totals.append(float(loss.detach().cpu())); compact_values.append(float(compact.detach().cpu()))
            consistency_values.append(float(consistency.detach().cpu()))
        val = compact_val_loss(model, validation, center, args, device)
        train_value = float(np.mean(totals))
        history.append({"epoch": epoch, "total_train_loss": train_value,
                        "compact_train_loss": float(np.mean(compact_values)),
                        "consistency_train_loss": float(np.mean(consistency_values)),
                        "healthy_val_objective": val})
        if val < best - 1e-5: best, state, stale = val, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"compact epoch={epoch}/{args.compact_epochs} train={train_value:.6f} val={val:.6f}", flush=True)
        if stale >= args.compact_patience: break
    if state is None: raise RuntimeError("Compact CPC produced no checkpoint")
    model.load_state_dict(state); return model, pd.DataFrame(history)


@torch.no_grad()
def embed_records(model, cache, device):
    model.eval(); output = {}
    for key, value in cache.items():
        x = torch.from_numpy(value[None]).to(device)
        output[key] = model.embed(x).squeeze(0).cpu().numpy().astype(np.float32)
    return output


def prepare_variant(cache, source_train):
    center, scale = fit_robust([cache[x] for x in source_train.record_id])
    return scale_sequences(cache, center, scale), center, scale


def score_frame(rows, cache, scorer, threshold, split):
    output = []
    for row in rows.itertuples():
        score = float(np.quantile(scorer(cache[row.record_id], modal.domain_for_state(row.state)), .95))
        output.append({"record_id": row.record_id, "state": row.state, "scenario": row.scenario,
                       "excitation": row.excitation, "setup": int(row.setup), "label": int(row.label),
                       "split": split, "score": score, "alarm": score > threshold})
    return pd.DataFrame(output)


def run_target_source(variant, cache, frame_parts, args, device, output):
    source_train, target_train, source_val, target_val, val_rows, test_rows = frame_parts
    scaled, center, scale = prepare_variant(cache, source_train)
    sx = np.concatenate([scaled[x] for x in source_train.record_id])
    tx = np.concatenate([scaled[x] for x in target_train.record_id])
    tv = np.concatenate([scaled[x] for x in target_val.record_id])
    paired, _ = nearest_source(tx, sx); paired_val, _ = nearest_source(tv, sx)
    mapper_args = SimpleNamespace(hidden=args.mapper_hidden, latent=args.mapper_latent,
                                  lr=args.mapper_lr, weight_decay=args.weight_decay,
                                  epochs=args.downstream_epochs, patience=args.downstream_patience,
                                  batch_size=args.batch_size)
    mapper, history = train_mapper(tx, paired, tv, paired_val, mapper_args, device)
    ocsvm = OneClassSVM(nu=args.ocsvm_nu, gamma="scale").fit(sx); threshold = 0.
    def scorer(x, domain):
        representation = map_features(mapper, x, device) if domain == "target" else x
        return -ocsvm.decision_function(representation)
    validation = score_frame(val_rows, scaled, scorer, threshold, "validation")
    test = score_frame(test_rows, scaled, scorer, threshold, "locked_test")
    metrics = common.metrics(test.label, test.score, threshold)
    return {"backbone": "MSSP2026_target_source_OCSVM", "preprocessing": variant,
            "epochs_run": int(history.epoch.max()), **metrics}, validation, test


def run_procrustes(variant, cache, frame_parts, args, device, output):
    source_train, target_train, source_val, target_val, val_rows, test_rows = frame_parts
    scaled, center, scale = prepare_variant(cache, source_train)
    sx = np.concatenate([scaled[x] for x in source_train.record_id])
    tx = np.concatenate([scaled[x] for x in target_train.record_id])
    sv = np.concatenate([scaled[x] for x in source_val.record_id])
    ae_args = SimpleNamespace(latent=args.ae_latent, dropout=args.dropout, lr=args.ae_lr,
                              weight_decay=args.weight_decay, epochs=args.downstream_epochs,
                              patience=args.downstream_patience, batch_size=args.batch_size)
    model, history = train_source_ae(sx, sv, ae_args, device)
    rotation = fit_rotation(encode(model, sx, device), encode(model, tx, device))
    threshold = float(np.quantile(reconstruction_errors(model, sv, device), .99))
    def scorer(x, domain):
        return reconstruction_errors(model, x, device, rotation if domain == "target" else None)
    validation = score_frame(val_rows, scaled, scorer, threshold, "validation")
    test = score_frame(test_rows, scaled, scorer, threshold, "locked_test")
    metrics = common.metrics(test.label, test.score, threshold)
    return {"backbone": "MSSP2026_AE_OrthogonalProcrustes", "preprocessing": variant,
            "epochs_run": int(history.epoch.max()), **metrics}, validation, test


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_healthy_cpc"))
    p.add_argument("--backbones", default="target_source,procrustes")
    p.add_argument("--fs", type=float, default=50.); p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=1024); p.add_argument("--psd-bins", type=int, default=96)
    p.add_argument("--fmin", type=float, default=3.); p.add_argument("--fmax", type=float, default=30.)
    p.add_argument("--context-length", type=int, default=12); p.add_argument("--segment-stride", type=int, default=4)
    p.add_argument("--prediction-steps", type=int, default=3); p.add_argument("--cpc-latent", type=int, default=32)
    p.add_argument("--cpc-context", type=int, default=32); p.add_argument("--temperature", type=float, default=.1)
    p.add_argument("--aug-noise", type=float, default=.02); p.add_argument("--mask-probability", type=float, default=.03)
    p.add_argument("--cpc-epochs", type=int, default=50); p.add_argument("--cpc-patience", type=int, default=10)
    p.add_argument("--cpc-lr", type=float, default=1e-3)
    p.add_argument("--compact-epochs", type=int, default=30); p.add_argument("--compact-patience", type=int, default=8)
    p.add_argument("--compact-lr", type=float, default=3e-4)
    p.add_argument("--compact-weight", type=float, default=1.0)
    p.add_argument("--consistency-weight", type=float, default=.5)
    p.add_argument("--compact-cpc-weight", type=float, default=.2)
    p.add_argument("--downstream-epochs", type=int, default=100); p.add_argument("--downstream-patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mapper-hidden", type=int, default=32); p.add_argument("--mapper-latent", type=int, default=8)
    p.add_argument("--mapper-lr", type=float, default=1e-3); p.add_argument("--ocsvm-nu", type=float, default=.05)
    p.add_argument("--ae-latent", type=int, default=4); p.add_argument("--ae-lr", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=.2); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = common.resolve_manifest(args.manifest, args.data_root)
    source_train = frame[(frame.split == "train") & (frame.state == "01")].copy()
    target_train = frame[(frame.split == "train") & (frame.state == "08")].copy()
    source_val = frame[(frame.split == "val") & (frame.state == "01")].copy()
    target_val = frame[(frame.split == "val") & (frame.state == "08")].copy()
    val_rows, test_rows = frame[frame.split == "val"].copy(), frame[frame.split == "test"].copy()
    counts = tuple(map(len, (source_train, target_train, source_val, target_val, val_rows, test_rows)))
    if counts != (12, 12, 3, 3, 132, 132): raise ValueError(f"Unexpected split: {counts}")
    print(f"device={device} split={counts} damage_used_for_CPC=False", flush=True)

    raw = {}
    for i, row in enumerate(frame.itertuples(), 1):
        raw[row.record_id] = record_psd_sequence(row.path, args.fs, args.window, args.stride,
                                                 args.psd_bins, args.fmin, args.fmax)
        if i % 24 == 0 or i == len(frame): print(f"PSD sequences={i}/{len(frame)}", flush=True)
    raw_scaled, psd_center, psd_scale = prepare_variant(raw, source_train)
    train_segments = segments([raw_scaled[x] for x in source_train.record_id], args.context_length, args.segment_stride)
    val_segments = segments([raw_scaled[x] for x in source_val.record_id], args.context_length, args.segment_stride)
    started = time.time(); cpc, cpc_history = train_cpc(train_segments, val_segments, args, device)
    cpc_history.to_csv(args.output / "cpc_training_history.csv", index=False)
    cpc_cache = embed_records(cpc, raw_scaled, device)
    cpc_state = copy.deepcopy(cpc.state_dict())
    center = healthy_center(cpc, train_segments, args.batch_size, device)
    cpc, compact_history = train_compact_cpc(cpc, train_segments, val_segments, center, args, device)
    compact_history.to_csv(args.output / "compact_training_history.csv", index=False)
    compact_cache = embed_records(cpc, raw_scaled, device)
    torch.save({"cpc_state_dict": cpc_state, "compact_state_dict": cpc.state_dict(),
                "healthy_center": center.cpu(), "psd_center": psd_center, "psd_scale": psd_scale,
                "args": vars(args)}, args.output / "healthy_cpc.pt")

    frame_parts = (source_train, target_train, source_val, target_val, val_rows, test_rows)
    runners = {"target_source": run_target_source, "procrustes": run_procrustes}
    selected = [x.strip() for x in args.backbones.split(",") if x.strip()]
    unknown = set(selected) - set(runners)
    if unknown: raise ValueError(f"Unknown backbones: {sorted(unknown)}")
    summaries, scores = [], []
    for backbone in selected:
        for variant, cache in (("logpsd", raw_scaled), ("healthy_cpc", cpc_cache),
                               ("compact_cpc", compact_cache)):
            summary, validation, test = runners[backbone](variant, cache, frame_parts, args, device, args.output)
            summaries.append(summary); validation.insert(0, "backbone", backbone); validation.insert(1, "preprocessing", variant)
            test.insert(0, "backbone", backbone); test.insert(1, "preprocessing", variant); scores.append(test)
            print(f"done {backbone}/{variant}: AUROC={summary['auroc']:.4f} BA={summary['balanced_accuracy']:.4f}", flush=True)
    summary = pd.DataFrame(summaries); score_frame = pd.concat(scores, ignore_index=True)
    summary.to_csv(args.output / "metrics.csv", index=False); score_frame.to_csv(args.output / "locked_test_scores.csv", index=False)
    state_scores = score_frame.groupby(
        ["backbone", "preprocessing", "state", "scenario", "label"], as_index=False).agg(
            records=("record_id", "size"), score_median=("score", "median"),
            alarm_rate=("alarm", "mean"))
    state_scores["alarm"] = state_scores.alarm_rate >= .5
    state_scores.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    deltas = []
    for backbone in selected:
        pair = summary.iloc[[i for i, row in summary.iterrows() if row.preprocessing in {"logpsd", "healthy_cpc"}
                             or row.preprocessing == "compact_cpc"]]
        pair = pair[((pair.backbone.str.contains("target_source", case=False) if backbone == "target_source"
                      else pair.backbone.str.contains("procrustes", case=False)))]
        raw_row = pair[pair.preprocessing == "logpsd"].iloc[0]
        for variant in ("healthy_cpc", "compact_cpc"):
            changed = pair[pair.preprocessing == variant].iloc[0]
            deltas.append({"backbone": backbone, "preprocessing": variant,
                           "delta_auroc": changed.auroc - raw_row.auroc,
                           "delta_balanced_accuracy": changed.balanced_accuracy - raw_row.balanced_accuracy,
                           "delta_f1": changed.f1 - raw_row.f1,
                           "delta_specificity": changed.specificity - raw_row.specificity})
    delta = pd.DataFrame(deltas); delta.to_csv(args.output / "cpc_deltas.csv", index=False)
    metadata = {"paper_idea": "Yang et al., Computers in Industry 2023, 103878",
                "doi": "10.1016/j.compind.2023.103878", "author_code_used": False,
                "adaptation": "sensor-median temporal log-PSD; in-batch cross-record CPC negatives; physics-safe views",
                "fit_protocol": "PSD scaler and CPC use S01/train only; early stop S01/val; damage excluded",
                "elapsed_seconds": time.time() - started, "parameters": vars(args)}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary.pivot(index="backbone", columns="preprocessing", values="auroc").plot.bar(ax=axes[0], ylim=(0, 1), title="AUROC")
    summary.pivot(index="backbone", columns="preprocessing", values="balanced_accuracy").plot.bar(ax=axes[1], ylim=(0, 1), title="Balanced accuracy")
    fig.tight_layout(); fig.savefig(args.output / "healthy_cpc_comparison.png", dpi=190); plt.close(fig)
    print("\nRESULTS\n" + summary.to_string(index=False), flush=True)
    print("\nCPC DELTAS\n" + delta.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__": main()

