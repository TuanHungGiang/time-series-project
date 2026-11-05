"""Natural-frequency AE + orthogonal Procrustes on the exported Z24 split.

Paper method: Nogueira, Figueiredo and da Silva, MSSP 261 (2026) 115013.
The source autoencoder is trained only on S01 healthy training observations.
The orthogonal rotation is estimated only from S01/S08 healthy observations in
the manifest training partition. Validation damage and locked test records are
never used for scaling, fitting, alignment, or threshold estimation.
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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_z24_strict_psd_scatterad as base
import z24_modal_transfer_common as modal


class SourceAE(nn.Module):
    def __init__(self, features: int, latent: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, 16), nn.LeakyReLU(.01), nn.Dropout(dropout),
            nn.Linear(16, 8), nn.LeakyReLU(.01), nn.Dropout(dropout),
            nn.Linear(8, latent), nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 8), nn.LeakyReLU(.01),
            nn.Linear(8, 16), nn.LeakyReLU(.01), nn.Linear(16, features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def loader(x: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch,
                      shuffle=shuffle, num_workers=0)


@torch.no_grad()
def reconstruction_errors(model, x, device, rotation=None):
    model.eval(); xt = torch.from_numpy(x).to(device); z = model.encoder(xt)
    if rotation is not None:
        z = z @ torch.from_numpy(rotation).to(device)
    return torch.mean((model.decoder(z) - xt) ** 2, dim=1).cpu().numpy()


@torch.no_grad()
def encode(model, x, device):
    model.eval()
    return model.encoder(torch.from_numpy(x).to(device)).cpu().numpy()


def fit_rotation(source_z: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    # Equation (4): min_R ||Z_t R - Z_s||_F^2, subject to R^T R = I.
    n = min(len(source_z), len(target_z))
    u, _, vt = np.linalg.svd(target_z[:n].T @ source_z[:n], full_matrices=False)
    return (u @ vt).astype(np.float32)


def train_source_ae(train, validation, args, device):
    model = SourceAE(train.shape[1], args.latent, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for (x,) in loader(train, args.batch_size, True):
            x = x.to(device); optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - x) ** 2); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        val = float(reconstruction_errors(model, validation, device).mean())
        tr = float(np.mean(losses)); history.append({"epoch": epoch, "train_loss": tr,
                                                      "source_val_loss": val})
        if val < best - 1e-8:
            best, state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch}/{args.epochs} train={tr:.6f} source_val={val:.6f}", flush=True)
        if stale >= args.patience:
            break
    if state is None:
        raise RuntimeError("No source AE checkpoint")
    model.load_state_dict(state)
    return model, pd.DataFrame(history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_procrustes_ae"))
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=75)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=.2)
    p.add_argument("--latent", type=int, default=2)
    p.add_argument("--fs", type=float, default=50.)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--stride", type=int, default=4096)
    p.add_argument("--fmin", type=float, default=3.)
    p.add_argument("--fmax", type=float, default=20.)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--mode-separation", type=float, default=.5)
    p.add_argument("--tracking-radius", type=float, default=.45)
    p.add_argument("--threshold-quantile", type=float, default=.99)
    p.add_argument("--record-quantile", type=float, default=.95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = base.resolve_manifest(args.manifest, args.data_root)
    source_train = frame[(frame.split == "train") & (frame.label == 0) & (frame.state == "01")].copy()
    target_train = frame[(frame.split == "train") & (frame.label == 0) & (frame.state == "08")].copy()
    source_val = frame[(frame.split == "val") & (frame.label == 0) & (frame.state == "01")].copy()
    target_val = frame[(frame.split == "val") & (frame.label == 0) & (frame.state == "08")].copy()
    val_rows = frame[frame.split == "val"].copy()
    test_rows = frame[frame.split == "test"].copy()
    expected = (12, 12, 3, 3, 132, 132)
    actual = tuple(map(len, (source_train, target_train, source_val, target_val, val_rows, test_rows)))
    if actual != expected:
        raise ValueError(f"Unexpected manifest split {actual}, expected {expected}")

    spectra, frequency = {}, None
    needed = pd.concat([source_train, target_train, source_val, target_val,
                        val_rows, test_rows]).drop_duplicates("record_id")
    for i, row in enumerate(needed.itertuples(), 1):
        f, s = modal.record_spectra(row.path, args.fs, args.window, args.stride,
                                    args.fmin, args.fmax)
        if frequency is None: frequency = f
        spectra[row.record_id] = s
        if i % 24 == 0 or i == len(needed):
            print(f"spectra={i}/{len(needed)}", flush=True)
    anchors = modal.learn_anchors([spectra[x] for x in source_train.record_id], frequency,
                                  args.modes, args.mode_separation)
    features = {key: modal.track_modes(frequency, value, anchors, args.tracking_radius)
                for key, value in spectra.items()}
    source_raw = np.concatenate([features[x] for x in source_train.record_id])
    center, spread = modal.robust_scaler(source_raw)
    scaled = {key: modal.scale(value, center, spread) for key, value in features.items()}
    source_x = np.concatenate([scaled[x] for x in source_train.record_id])
    target_x = np.concatenate([scaled[x] for x in target_train.record_id])
    source_val_x = np.concatenate([scaled[x] for x in source_val.record_id])

    started = time.time()
    model, history = train_source_ae(source_x, source_val_x, args, device)
    history.to_csv(args.output / "training_history.csv", index=False)
    rotation = fit_rotation(encode(model, source_x, device), encode(model, target_x, device))
    source_val_errors = reconstruction_errors(model, source_val_x, device)
    threshold = float(np.quantile(source_val_errors, args.threshold_quantile))

    def score_rows(rows, split):
        output = []
        for row in rows.itertuples():
            domain = modal.domain_for_state(row.state); x = scaled[row.record_id]
            error = reconstruction_errors(model, x, device,
                                           rotation if domain == "target" else None)
            score = modal.record_quantile(error, args.record_quantile)
            output.append({"record_id": row.record_id, "state": row.state,
                           "scenario": row.scenario, "excitation": row.excitation,
                           "setup": int(row.setup), "label": int(row.label),
                           "split": split, "domain": domain, "score": score,
                           "alarm": score > threshold})
        return pd.DataFrame(output)

    validation = score_rows(val_rows, "validation")
    test = score_rows(test_rows, "locked_test")
    validation.to_csv(args.output / "validation_record_scores.csv", index=False)
    test.to_csv(args.output / "locked_test_record_scores.csv", index=False)
    val_metrics = base.metrics(validation.label, validation.score, threshold)
    test_metrics = base.metrics(test.label, test.score, threshold)
    states = test.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("score", "median"), alarm_rate=("alarm", "mean"))
    states["alarm"] = states.alarm_rate >= .5
    states.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    summary = {"method": "MSSP2026_natural_frequency_AE_OrthogonalProcrustes",
               "source_healthy_train": len(source_train), "target_healthy_alignment": len(target_train),
               "damage_used_for_fit": False, "damage_used_for_threshold": False,
               "epochs_run": int(history.epoch.max()), "seconds": time.time() - started,
               "threshold": threshold,
               **{f"validation_{k}": v for k, v in val_metrics.items()},
               **{f"record_{k}": v for k, v in test_metrics.items()}}
    pd.DataFrame([summary]).to_csv(args.output / "metrics.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "center": center, "spread": spread,
                "anchors": anchors, "rotation": rotation, "threshold": threshold,
                "args": vars(args)}, args.output / "model.pt")
    metadata = {
        "paper": "Nogueira et al., MSSP 261 (2026) 115013",
        "doi": "10.1016/j.ymssp.2026.115013",
        "paper_equations": ["Zt* = Zt R", "R = argmin ||Zt R - Zs||_F",
                            "Xhat_t = Decoder_s(Zt*)", "MSE"],
        "z24_adapter": "fixed modal anchors learned from S01 healthy train; pre/post lifting defines source/target",
        "alignment": "12 S01 and 12 S08 healthy train records only; no damage",
        "causal_deconfounding_not_run": "record-level temperature/environment metadata absent from manifest",
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(history.epoch, history.train_loss, label="source train")
    axes[0].plot(history.epoch, history.source_val_loss, label="source healthy val")
    axes[0].legend(); axes[0].set_title("Source AE loss")
    fpr, tpr, _ = roc_curve(test.label, test.score)
    axes[1].plot(fpr, tpr, label=f"AUROC={test_metrics['auroc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray"); axes[1].legend(); axes[1].set_title("Locked-test ROC")
    ConfusionMatrixDisplay(confusion_matrix(test.label, test.alarm, labels=[0, 1]),
                           display_labels=["Healthy", "Damage"]).plot(ax=axes[2], cmap="Greens", colorbar=False)
    axes[2].set_title("Locked-test confusion"); fig.tight_layout()
    fig.savefig(args.output / "procrustes_results.png", dpi=190); plt.close(fig)
    print("\nRESULT\n" + pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()
