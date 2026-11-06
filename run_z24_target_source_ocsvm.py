"""Target-to-source healthy mapping + one-class SVM on the Z24 split.

Paper method: Heravi et al., MSSP 242 (2026) 113673.  The mapper is trained
only with S08 target-healthy observations and S01 source-healthy observations;
the one-class SVM is fitted only to S01 healthy observations.  The article does
not publish author code or the observation-pairing routine.  For this record
split, target healthy observations are paired to their nearest source healthy
modal observation using training data only; this necessary adapter is audited.
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
from scipy.spatial.distance import cdist
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_z24_strict_psd_scatterad as base
import z24_modal_transfer_common as modal


class TargetSourceMapper(nn.Module):
    def __init__(self, features: int, hidden: int, latent: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.ELU(),
            nn.Linear(hidden, latent), nn.ELU(),
            nn.Linear(latent, hidden), nn.ELU(),
            nn.Linear(hidden, features),
        )

    def forward(self, x):
        return self.network(x)


def nearest_source(target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argmin(cdist(target, source, metric="sqeuclidean"), axis=1)
    return source[indices], indices.astype(np.int32)


def pair_loader(x, y, batch, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def mapping_mse(model, x, y, device):
    model.eval(); xt = torch.from_numpy(x).to(device); yt = torch.from_numpy(y).to(device)
    return float(torch.mean((model(xt) - yt) ** 2).cpu())


@torch.no_grad()
def map_features(model, x, device):
    model.eval()
    return model(torch.from_numpy(x).to(device)).cpu().numpy()


def train_mapper(train_x, train_y, val_x, val_y, args, device):
    model = TargetSourceMapper(train_x.shape[1], args.hidden, args.latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in pair_loader(train_x, train_y, args.batch_size, True):
            x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        tr = float(np.mean(losses)); val = mapping_mse(model, val_x, val_y, device)
        history.append({"epoch": epoch, "mapping_train_loss": tr,
                        "healthy_mapping_val_loss": val})
        if val < best - 1e-8:
            best, state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch}/{args.epochs} mapping_train={tr:.6f} mapping_val={val:.6f}", flush=True)
        if stale >= args.patience:
            break
    if state is None:
        raise RuntimeError("No mapper checkpoint")
    model.load_state_dict(state)
    return model, pd.DataFrame(history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_target_source_ocsvm"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ocsvm-nu", type=float, default=.05)
    p.add_argument("--ocsvm-gamma", default="scale")
    p.add_argument("--fs", type=float, default=50.)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--stride", type=int, default=4096)
    p.add_argument("--fmin", type=float, default=3.)
    p.add_argument("--fmax", type=float, default=20.)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--mode-separation", type=float, default=.5)
    p.add_argument("--tracking-radius", type=float, default=.45)
    p.add_argument("--record-quantile", type=float, default=.95)
    p.add_argument("--preprocessing", choices=["modal", "regime_cointegration"], default="modal")
    p.add_argument("--cointegration-rank", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = base.resolve_manifest(args.manifest, args.data_root)
    source_train = frame[(frame.split == "train") & (frame.label == 0) & (frame.state == "01")].copy()
    target_train = frame[(frame.split == "train") & (frame.label == 0) & (frame.state == "08")].copy()
    target_val = frame[(frame.split == "val") & (frame.label == 0) & (frame.state == "08")].copy()
    val_rows = frame[frame.split == "val"].copy(); test_rows = frame[frame.split == "test"].copy()
    actual = tuple(map(len, (source_train, target_train, target_val, val_rows, test_rows)))
    if actual != (12, 12, 3, 132, 132):
        raise ValueError(f"Unexpected manifest split: {actual}")

    spectra, frequency = {}, None
    needed = pd.concat([source_train, target_train, target_val, val_rows, test_rows]).drop_duplicates("record_id")
    for i, row in enumerate(needed.itertuples(), 1):
        f, s = modal.record_spectra(row.path, args.fs, args.window, args.stride,
                                    args.fmin, args.fmax)
        if frequency is None: frequency = f
        spectra[row.record_id] = s
        if i % 24 == 0 or i == len(needed):
            print(f"spectra={i}/{len(needed)}", flush=True)
    anchors = modal.learn_anchors([spectra[x] for x in source_train.record_id], frequency,
                                  args.modes, args.mode_separation)
    tracker = modal.track_modes_centroid if args.preprocessing == "regime_cointegration" else modal.track_modes
    features = {key: tracker(frequency, value, anchors, args.tracking_radius)
                for key, value in spectra.items()}
    source_raw = np.concatenate([features[x] for x in source_train.record_id])
    center, spread = modal.robust_scaler(source_raw)
    scaled = {key: modal.scale(value, center, spread) for key, value in features.items()}
    cointegration = None
    if args.preprocessing == "regime_cointegration":
        scaled, cointegration = modal.cointegration_features(
            scaled, needed, source_train, args.cointegration_rank)
    source_x = np.concatenate([scaled[x] for x in source_train.record_id])
    target_x = np.concatenate([scaled[x] for x in target_train.record_id])
    target_val_x = np.concatenate([scaled[x] for x in target_val.record_id])
    paired_source, pair_index = nearest_source(target_x, source_x)
    paired_val, val_pair_index = nearest_source(target_val_x, source_x)

    started = time.time()
    mapper, history = train_mapper(target_x, paired_source, target_val_x, paired_val,
                                   args, device)
    history.to_csv(args.output / "training_history.csv", index=False)
    ocsvm = OneClassSVM(kernel="rbf", gamma=args.ocsvm_gamma, nu=args.ocsvm_nu)
    ocsvm.fit(source_x)
    # Native OCSVM boundary: decision_function < 0 is anomalous.
    threshold = 0.

    def anomaly_windows(x, domain):
        representation = map_features(mapper, x, device) if domain == "target" else x
        return -ocsvm.decision_function(representation)

    def score_rows(rows, split):
        output = []
        for row in rows.itertuples():
            domain = modal.domain_for_state(row.state)
            score = modal.record_quantile(anomaly_windows(scaled[row.record_id], domain),
                                          args.record_quantile)
            output.append({"record_id": row.record_id, "state": row.state,
                           "scenario": row.scenario, "excitation": row.excitation,
                           "setup": int(row.setup), "label": int(row.label),
                           "split": split, "domain": domain, "score": score,
                           "alarm": score > threshold})
        return pd.DataFrame(output)

    validation = score_rows(val_rows, "validation"); test = score_rows(test_rows, "locked_test")
    validation.to_csv(args.output / "validation_record_scores.csv", index=False)
    test.to_csv(args.output / "locked_test_record_scores.csv", index=False)
    val_metrics = base.metrics(validation.label, validation.score, threshold)
    test_metrics = base.metrics(test.label, test.score, threshold)
    states = test.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("score", "median"), alarm_rate=("alarm", "mean"))
    states["alarm"] = states.alarm_rate >= .5
    states.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    summary = {"method": f"MSSP2026_target_to_source_mapping_OCSVM_{args.preprocessing}",
               "source_healthy_train": len(source_train), "target_healthy_mapper_train": len(target_train),
               "damage_used_for_fit": False, "damage_used_for_threshold": False,
               "threshold_rule": "native OCSVM decision boundary",
               "epochs_run": int(history.epoch.max()), "seconds": time.time() - started,
               **{f"validation_{k}": v for k, v in val_metrics.items()},
               **{f"record_{k}": v for k, v in test_metrics.items()}}
    pd.DataFrame([summary]).to_csv(args.output / "metrics.csv", index=False)
    torch.save({"mapper_state_dict": mapper.state_dict(), "center": center, "spread": spread,
                "anchors": anchors, "pair_index": pair_index, "val_pair_index": val_pair_index,
                "cointegration_models": cointegration.models if cointegration else None,
                "ocsvm_support": ocsvm.support_vectors_, "args": vars(args)}, args.output / "model.pt")
    metadata = {
        "paper": "Heravi et al., MSSP 242 (2026) 113673",
        "doi": "10.1016/j.ymssp.2025.113673",
        "reproduced": ["target-healthy to source-healthy neural mapping", "source-only one-class SVM", "zero damage fitting"],
        "unspecified_by_paper_adapter": "nearest-source pairing of modal observations, fit on healthy train only",
        "domain_rule": "S01--S07 source/pre-lifting; S08+ target/post-lifting",
        "preprocessing": args.preprocessing,
        "cointegration_audit": cointegration.audit() if cointegration else None,
        "cointegration_note": "AVT/FVT-conditioned Johansen equation adaptation of Shi et al., MSSP 2018; continuous peak-centroid modes; S01/train fit only",
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(history.epoch, history.mapping_train_loss, label="mapping train")
    axes[0].plot(history.epoch, history.healthy_mapping_val_loss, label="target healthy val")
    axes[0].legend(); axes[0].set_title("Target-to-source mapping")
    fpr, tpr, _ = roc_curve(test.label, test.score)
    axes[1].plot(fpr, tpr, label=f"AUROC={test_metrics['auroc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray"); axes[1].legend(); axes[1].set_title("Locked-test ROC")
    ConfusionMatrixDisplay(confusion_matrix(test.label, test.alarm, labels=[0, 1]),
                           display_labels=["Healthy", "Damage"]).plot(ax=axes[2], cmap="Oranges", colorbar=False)
    axes[2].set_title("Locked-test confusion"); fig.tight_layout()
    fig.savefig(args.output / "target_source_ocsvm_results.png", dpi=190); plt.close(fig)
    print("\nRESULT\n" + pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

