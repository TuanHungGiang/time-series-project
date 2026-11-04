"""Z24 adaptation of the MSSP 2026 autoencoder + Procrustes framework.

Protocol (record-disjoint and damage-blind):
* AE train: S01 setups 1--6 (12 records)
* healthy calibration: S01 setups 7--9 (6 records)
* target commissioning alignment: all nine S08 setups from one excitation
* evaluation: the opposite-excitation S08 records and damage records

Two cross-fit directions (AVT -> FVT and FVT -> AVT) ensure every evaluated S08
record is scored by a transform that did not use it.  This is not the zero-target
S01-only protocol; it is the paper's target-healthy adaptation setting.  No
public author code was found, so this is an equation-based reproduction.
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
import torch
from scipy.signal import find_peaks, welch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch import nn

import run_z24_strict_psd_scatterad as base


def modal_features(path: Path, fs: float, fmin: float, fmax: float,
                   modes: int, min_separation_hz: float) -> np.ndarray:
    values = base.vertical_values(path)
    spectra = []
    for channel in range(values.shape[1]):
        x = values[:, channel] - np.mean(values[:, channel])
        f, p = welch(x, fs=fs, nperseg=min(8192, len(x)), noverlap=4096)
        spectra.append(p)
    spectrum = np.median(np.stack(spectra), axis=0)
    band = (f >= fmin) & (f <= fmax); fb, pb = f[band], spectrum[band]
    step = float(np.median(np.diff(fb)))
    peaks, _ = find_peaks(np.log10(pb + np.finfo(float).tiny),
                          distance=max(1, int(min_separation_hz / step)), prominence=.05)
    candidates = list(peaks[np.argsort(pb[peaks])[::-1]])
    if len(candidates) < modes:
        candidates.extend(i for i in np.argsort(pb)[::-1] if i not in candidates)
    chosen = sorted(candidates[:modes], key=lambda i: fb[i])
    if len(chosen) != modes: raise ValueError(f"Cannot extract {modes} modes from {path}")
    return fb[chosen].astype(np.float32)


class AE(nn.Module):
    def __init__(self, features: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, 16), nn.LeakyReLU(.01),
                                     nn.Linear(16, 8), nn.LeakyReLU(.01),
                                     nn.Linear(8, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 8), nn.LeakyReLU(.01),
                                     nn.Linear(8, 16), nn.LeakyReLU(.01),
                                     nn.Linear(16, features))
    def forward(self, x): return self.decoder(self.encoder(x))


def standardizer(x: np.ndarray):
    center = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - center), axis=0)
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def tx(x, center, scale): return ((x - center) / scale).astype(np.float32)


def train_ae(train_x, val_x, args, device):
    model = AE(train_x.shape[1], args.latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_t = torch.tensor(train_x, device=device); val_t = torch.tensor(val_x, device=device)
    best, best_state, patience = float("inf"), None, 0; rows = []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(train_t) - train_t) ** 2); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val = torch.mean((model(val_t) - val_t) ** 2)
        rows.append({"epoch": epoch, "train_loss": float(loss.detach()), "val_healthy_loss": float(val)})
        if float(val) < best - 1e-8:
            best, best_state, patience = float(val), copy.deepcopy(model.state_dict()), 0
        else: patience += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch}/{args.epochs} train={float(loss.detach()):.6f} val={float(val):.6f}", flush=True)
        if patience >= args.patience: break
    model.load_state_dict(best_state); return model, pd.DataFrame(rows)


@torch.no_grad()
def encode(model, x, device):
    return model.encoder(torch.tensor(x, device=device)).cpu().numpy()


@torch.no_grad()
def reconstruction_error(model, x, device, transform=None):
    xt = torch.tensor(x, device=device); z = model.encoder(xt)
    if transform is not None:
        rotation, target_mean, source_mean = transform
        z = (z - torch.tensor(target_mean, device=device)) @ torch.tensor(rotation, device=device)
        z = z + torch.tensor(source_mean, device=device)
    prediction = model.decoder(z)
    return torch.mean((prediction - xt) ** 2, dim=1).cpu().numpy()


def fit_procrustes(source_z, target_z):
    source_mean = source_z.mean(0, keepdims=True); target_mean = target_z.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd((target_z - target_mean).T @ (source_z - source_mean), full_matrices=False)
    rotation = (u @ vt).astype(np.float32)
    return rotation, target_mean.astype(np.float32), source_mean.astype(np.float32)


def evaluate(frame, scores, threshold, name):
    part = frame.copy(); part["score"] = scores; part["alarm"] = part.score > threshold
    result = base.metrics(part.label, part.score, threshold)
    states = part.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("score", "median"), alarm_rate=("alarm", "mean"))
    states["alarm"] = states.alarm_rate >= .5
    y, pred = states.label.to_numpy(), states.alarm.astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    state = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
             "accuracy": float((y == pred).mean())}
    return part, states, {"method": name, **{f"record_{k}": v for k, v in result.items()},
                          **{f"state_{k}": v for k, v in state.items()}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_procrustes_ae"))
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--latent", type=int, default=2); p.add_argument("--fs", type=float, default=50.)
    p.add_argument("--fmin", type=float, default=3.); p.add_argument("--fmax", type=float, default=20.)
    p.add_argument("--modes", type=int, default=3); p.add_argument("--min-separation-hz", type=float, default=.3)
    p.add_argument("--calibration-quantile", type=float, default=.98); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = base.resolve_manifest(args.manifest, args.data_root)
    source_train = frame[(frame.state == "01") & (frame.setup.astype(int) <= 6)].copy()
    source_cal = frame[(frame.state == "01") & (frame.setup.astype(int) >= 7)].copy()
    source_all = frame[frame.state == "01"].copy()
    target_all = frame[frame.state == "08"].copy()
    damage_all = frame[~frame.state.isin(["01", "02", "08"])].copy()
    groups = {"source_train": source_train, "source_cal": source_cal,
              "source_alignment": source_all, "target_S08": target_all,
              "damage_evaluation": damage_all}
    cache = {}
    for group, rows in groups.items():
        for i, row in enumerate(rows.itertuples(), 1):
            cache[row.record_id] = modal_features(row.path, args.fs, args.fmin, args.fmax,
                                                   args.modes, args.min_separation_hz)
        print(f"features {group}={len(rows)}", flush=True)
    x_train = np.stack([cache[x] for x in source_train.record_id]); center, scale = standardizer(x_train)
    x_train = tx(x_train, center, scale)
    x_cal = tx(np.stack([cache[x] for x in source_cal.record_id]), center, scale)
    started = time.time(); model, history = train_ae(x_train, x_cal, args, device)
    history.to_csv(args.output / "training_history.csv", index=False)
    cal_errors = reconstruction_error(model, x_cal, device)
    threshold = float(np.quantile(cal_errors, args.calibration_quantile))

    raw_parts, aligned_parts, evaluation_parts, transforms = [], [], [], {}
    # Cross-fit target adaptation: every setup is represented in commissioning,
    # while each evaluated S08 record is scored by a transform fitted only on
    # the opposite excitation. Damage records are never used for alignment.
    for adapt_excitation, eval_excitation in (("avt", "fvt"), ("fvt", "avt")):
        source_align_rows = source_all[source_all.excitation == adapt_excitation].sort_values("setup")
        target_align_rows = target_all[target_all.excitation == adapt_excitation].sort_values("setup")
        evaluation = pd.concat([
            target_all[target_all.excitation == eval_excitation],
            damage_all[damage_all.excitation == eval_excitation],
        ], ignore_index=True).sort_values(["state", "setup"])
        source_align = tx(np.stack([cache[x] for x in source_align_rows.record_id]), center, scale)
        target_align = tx(np.stack([cache[x] for x in target_align_rows.record_id]), center, scale)
        transform = fit_procrustes(encode(model, source_align, device),
                                   encode(model, target_align, device))
        x_eval = tx(np.stack([cache[x] for x in evaluation.record_id]), center, scale)
        raw_parts.append(reconstruction_error(model, x_eval, device))
        aligned_parts.append(reconstruction_error(model, x_eval, device, transform))
        evaluation["protocol_fold"] = f"adapt_{adapt_excitation}_test_{eval_excitation}"
        evaluation_parts.append(evaluation)
        transforms[adapt_excitation] = transform
    evaluation = pd.concat(evaluation_parts, ignore_index=True)
    raw_scores = np.concatenate(raw_parts); aligned_scores = np.concatenate(aligned_parts)
    raw_records, raw_states, raw_result = evaluate(evaluation, raw_scores, threshold, "AE_modal_no_alignment")
    aligned_records, aligned_states, aligned_result = evaluate(
        evaluation, aligned_scores, threshold, "AE_modal_Procrustes_MSSP2026_equation_reproduction")
    raw_records.to_csv(args.output / "record_scores_no_alignment.csv", index=False)
    aligned_records.to_csv(args.output / "record_scores_procrustes.csv", index=False)
    raw_states.to_csv(args.output / "state_scores_no_alignment.csv", index=False)
    aligned_states.to_csv(args.output / "state_scores_procrustes.csv", index=False)
    results = pd.DataFrame([raw_result, aligned_result]); results["seconds"] = time.time() - started
    results.to_csv(args.output / "metrics.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "center": center, "scale": scale,
                "transforms": transforms, "threshold": threshold, "args": vars(args)}, args.output / "model.pt")
    metadata = {"paper": "Nogueira et al., integrated AE-based unsupervised domain adaptation, MSSP 261 (2026) 115013",
                "doi": "10.1016/j.ymssp.2026.115013",
                "implementation_scope": "equation-based PDT adapter; architecture reduced for 12 source records",
                "important_protocol_change": (
                    "two cross-fit target folds: all nine S08 AVT records align the FVT evaluation fold, "
                    "then all nine S08 FVT records align the AVT evaluation fold; no record is evaluated "
                    "by a transform that used it"
                ),
                "damage_used_for_fit": False, "evaluation_records": len(evaluation), "parameters": vars(args)}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(history.epoch, history.train_loss, label="S01 train")
    axes[0].plot(history.epoch, history.val_healthy_loss, label="S01 calibration")
    axes[0].set(title="Autoencoder loss", xlabel="Epoch", ylabel="MSE"); axes[0].legend(); axes[0].grid(alpha=.2)
    for scores, label in [(raw_scores, "No alignment"), (aligned_scores, "Procrustes")]:
        fpr, tpr, _ = roc_curve(evaluation.label, scores); axes[1].plot(fpr, tpr, label=label)
    axes[1].plot([0, 1], [0, 1], "--", color="gray"); axes[1].set(title="Record ROC", xlabel="FPR", ylabel="TPR"); axes[1].legend()
    pred = (aligned_scores > threshold).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(evaluation.label, pred, labels=[0, 1]),
                           display_labels=["Healthy", "Damage"]).plot(ax=axes[2], cmap="Greens", colorbar=False)
    axes[2].set_title("Procrustes record confusion")
    fig.tight_layout(); fig.savefig(args.output / "procrustes_results.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("\nRESULTS\n" + results.to_string(index=False), flush=True)
    print("\nPROCRUSTES STATE SCORES\n" + aligned_states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__": main()

