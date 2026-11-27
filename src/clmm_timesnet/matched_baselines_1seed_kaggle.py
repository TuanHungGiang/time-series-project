"""Repeat the matched baseline protocol for one seed and export loss curves."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .matched_baseline_loss_curves import LABELS
from .matched_baselines_kaggle import BATCH_SIZE, EPOCHS, GROUP, MODELS, Z24_WINDOWS, run_dataset
from .whitening_multiseed_kaggle import CONDITIONS, DATASETS, METRICS, SEEDS
from .whitening_two_bridges_kaggle import INPUT, LENGTH, MASKS, Z24_INPUT


def export_curves(losses, out):
    keys = ["dataset", "model", "epoch", "objective"]
    summary = losses.groupby(keys, as_index=False).agg(
        train_loss=("train_loss", "mean"), val_loss=("val_loss", "mean"),
        groups=("group", "nunique"),
    )
    summary.to_csv(out / "epoch_loss_mean_by_bridge.csv", index=False)
    for dataset, bridge in summary.groupby("dataset"):
        figure, axes = plt.subplots(2, 3, figsize=(10.7, 5.6), sharex=True, layout="constrained")
        figure.suptitle(f"{dataset}  |  seed {SEEDS[0]}")
        for (model, title), ax in zip(LABELS.items(), axes.flat, strict=True):
            curve = bridge.loc[bridge.model.eq(model)].sort_values("epoch")
            ax.plot(curve.epoch, curve.train_loss, "o-", markersize=3.5, color="#2c5784",
                    label="Training loss")
            ax.plot(curve.epoch, curve.val_loss, "s--", markersize=3.5, color="#167268",
                    label="Held-out healthy loss")
            ax.set(title=title, xlabel="Epoch", ylabel=curve.objective.iloc[0],
                   xticks=range(1, EPOCHS + 1))
            ax.legend(fontsize=8, frameon=False)
            ax.grid(axis="y", alpha=0.20)
        label = out / f"train_val_loss_{dataset}"
        figure.savefig(label.with_suffix(".png"), dpi=170)
        figure.savefig(label.with_suffix(".pdf"))
        plt.close(figure)


def validate_metrics(metrics):
    keys = [*GROUP, "replicate", "seed"]
    expected = {
        (dataset, model, condition, policy, replicate, SEEDS[0])
        for dataset in DATASETS for model in MODELS for condition in CONDITIONS
        for policy in ("clean_frozen", "mask_matched")
        for replicate in ((0,) if condition == "clean" else MASKS)
    }
    actual = set(metrics[keys].itertuples(index=False, name=None))
    if actual != expected or len(actual) != len(metrics) or not np.isfinite(metrics[list(METRICS)]).all().all():
        raise ValueError("Missing, duplicated or invalid metrics.")


def validate_losses(losses):
    expected_groups = {"Z24": 18, "HellBridge": 1}
    keys = ["dataset", "group", "model", "epoch"]
    if losses.duplicated(keys).any() or not np.isfinite(losses[["train_loss", "val_loss"]]).all().all():
        raise ValueError("Duplicated or invalid epoch losses.")
    for dataset in DATASETS:
        bridge = losses.loc[losses.dataset.eq(dataset)]
        if not bridge.seed.eq(SEEDS[0]).all():
            raise ValueError(f"Unexpected seed in {dataset} losses")
        groups = set(bridge.group)
        if len(groups) != expected_groups[dataset]:
            raise ValueError(f"{dataset}: wrong number of fitted groups")
        expected = {(group, model, epoch) for group in groups for model in MODELS
                    for epoch in ((0,) if model == "pca_spe" else range(1, EPOCHS + 1))}
        if set(bridge[keys[1:]].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"{dataset}: epoch loss coverage is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, default=Z24_INPUT)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.z24_root / "metadata/records_manifest.csv", args.hell_h5,
                 args.baseline_dir / "run_hellbridge_source_baselines.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU on Kaggle.")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    source = importlib.import_module("run_hellbridge_source_baselines")
    args.epochs, args.batch_size, args.z24_windows = EPOCHS, BATCH_SIZE, Z24_WINDOWS
    args.layer_probe, args.replicates = False, MASKS

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=SEEDS[0], models=MODELS, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    window_points=LENGTH, z24_window_cap=Z24_WINDOWS, masks=MASKS,
                    z24_input=str(args.z24_root), hellbridge_input=str(args.hell_h5),
                    baseline_source=str(args.baseline_dir),
                    validation_role="same clean held-out healthy reference windows after each epoch",
                    score_calibration="same healthy validation threshold policies as the 3-seed benchmark",
                    pca_epoch_zero="single post-fit train and held-out reconstruction MSE")
    protocol_path = out / "protocol.json"
    text = json.dumps(protocol, indent=2)
    if protocol_path.is_file() and protocol_path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing output uses a different protocol: {out}")
    protocol_path.write_text(text, encoding="utf-8")
    score_parts, loss_parts = [], []
    for dataset in DATASETS:
        group = out / dataset / str(SEEDS[0])
        metrics_path, losses_path = group / "complete_metrics.csv", group / "epoch_losses.csv"
        if metrics_path.is_file() and losses_path.is_file():
            print(f"RESUME {dataset}", flush=True)
            metrics = pd.read_csv(metrics_path, float_precision="round_trip")
            loss_rows = pd.read_csv(losses_path, float_precision="round_trip")
        else:
            args.epoch_losses = []
            metrics = run_dataset(dataset, SEEDS[0], args, device, source, group)
            loss_rows = pd.DataFrame(args.epoch_losses)
            loss_rows.to_csv(losses_path, index=False)
            tmp = metrics_path.with_suffix(".tmp")
            metrics.to_csv(tmp, index=False)
            tmp.replace(metrics_path)
        score_parts.append(metrics)
        loss_parts.append(loss_rows)
    metrics = pd.concat(score_parts, ignore_index=True)
    validate_metrics(metrics)
    metrics.to_csv(out / "per_mask_metrics.csv", index=False)
    metrics.groupby(GROUP, as_index=False)[list(METRICS)].mean().to_csv(out / "summary_1seed.csv", index=False)
    losses = pd.concat(loss_parts, ignore_index=True)
    validate_losses(losses)
    losses.to_csv(out / "epoch_losses_by_group.csv", index=False)
    export_curves(losses, out)
    print(f"MATCHED_BASELINES_1SEED_LOSS_COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
