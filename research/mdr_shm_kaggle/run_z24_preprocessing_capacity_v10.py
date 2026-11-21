from __future__ import annotations

"""Z24 V10: preprocessing/decoder/capacity screening under sensor loss.

This runner isolates the suspected preprocessing/model mismatch in V9.  All
models fit healthy training windows only.  Scaler, score direction and record
threshold are selected from labeled clean/corrupted validation records; test
records are never used for fitting or selection.
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax, Normalizer, publication_metrics, seed_everything,
)
from run_z24_mir_usad_v0 import scenarios, sensor_corrupt  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)


VARIANTS = {
    "current_clip_sigmoid_large": {
        "scaler": "clip", "decoder": "sigmoid",
        "hidden1": 512, "hidden2": 128, "latent": 32,
    },
    "zscore_linear_small": {
        "scaler": "zscore", "decoder": "linear",
        "hidden1": 128, "hidden2": 32, "latent": 8,
    },
    "robust_linear_small": {
        "scaler": "robust", "decoder": "linear",
        "hidden1": 128, "hidden2": 32, "latent": 8,
    },
}
TAIL_COLUMNS = {
    "upper": "score_upper", "lower": "score_lower",
    "two_sided": "score_two_sided",
}


@dataclass
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "RobustScaler":
        center = np.median(values, axis=(0, 2), keepdims=True)
        q25 = np.quantile(values, 0.25, axis=(0, 2), keepdims=True)
        q75 = np.quantile(values, 0.75, axis=(0, 2), keepdims=True)
        scale = np.maximum((q75 - q25) / 1.349, 1e-6)
        return cls(center.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.center) / self.scale).astype(np.float32)


class IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, np.float32)


class Encoder(nn.Module):
    def __init__(self, width: int, hidden1: int, hidden2: int, latent: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, hidden1), nn.GELU(),
            nn.Linear(hidden1, hidden2), nn.GELU(),
            nn.Linear(hidden2, latent), nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class Decoder(nn.Module):
    def __init__(self, latent: int, hidden2: int, hidden1: int, width: int,
                 output: str):
        super().__init__()
        layers = [
            nn.Linear(latent, hidden2), nn.GELU(),
            nn.Linear(hidden2, hidden1), nn.GELU(),
            nn.Linear(hidden1, width),
        ]
        if output == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output != "linear":
            raise ValueError(output)
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CompactUSAD(nn.Module):
    def __init__(self, width: int, hidden1: int, hidden2: int, latent: int,
                 output: str):
        super().__init__()
        self.encoder = Encoder(width, hidden1, hidden2, latent)
        self.decoder1 = Decoder(latent, hidden2, hidden1, width, output)
        self.decoder2 = Decoder(latent, hidden2, hidden1, width, output)

    def parts(self, values: torch.Tensor):
        shape = values.shape
        flat = values.flatten(1)
        latent = self.encoder(flat)
        first = self.decoder1(latent)
        second = self.decoder2(latent)
        cycle = self.decoder2(self.encoder(first))
        return first.reshape(shape), second.reshape(shape), cycle.reshape(shape)


def usad_losses(truth, first, second, cycle, epoch):
    fraction = 1.0 / epoch
    loss1 = fraction * (truth - first).square().mean()
    loss1 += (1.0 - fraction) * (truth - cycle).square().mean()
    loss2 = fraction * (truth - second).square().mean()
    loss2 -= (1.0 - fraction) * (truth - cycle).square().mean()
    return loss1, loss2


def fit_model(fit, config, epochs, batch_size, device):
    width = int(np.prod(fit.shape[1:]))
    model = CompactUSAD(
        width, config["hidden1"], config["hidden2"], config["latent"],
        config["decoder"],
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)), batch_size=batch_size, shuffle=True,
    )
    parameters1 = list(model.encoder.parameters()) + list(model.decoder1.parameters())
    parameters2 = list(model.encoder.parameters()) + list(model.decoder2.parameters())
    optimizer1 = torch.optim.Adam(parameters1, lr=1e-3)
    optimizer2 = torch.optim.Adam(parameters2, lr=1e-3)
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer1.zero_grad(set_to_none=True)
            first, second, cycle = model.parts(batch)
            loss1, _ = usad_losses(batch, first, second, cycle, epoch)
            loss1.backward()
            nn.utils.clip_grad_norm_(parameters1, 5.0)
            optimizer1.step()
            optimizer2.zero_grad(set_to_none=True)
            first, second, cycle = model.parts(batch)
            _, loss2 = usad_losses(batch, first, second, cycle, epoch)
            loss2.backward()
            nn.utils.clip_grad_norm_(parameters2, 5.0)
            optimizer2.step()
            losses.append(float((loss1 + loss2).detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"    epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    return model.eval()


@torch.no_grad()
def score_model(model, values, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        first, _, cycle = model.parts(batch)
        error = 0.5 * (batch - first).square() + 0.5 * (batch - cycle).square()
        scores.extend(error.mean((1, 2)).cpu().numpy())
    return np.asarray(scores)


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_scaler(name, fit_raw):
    if name == "clip":
        primary = Normalizer.fit(fit_raw)
        fit_primary = primary.transform(fit_raw)
        return primary, FitMinMax(fit_primary)
    if name == "zscore":
        return Normalizer.fit(fit_raw), IdentityScaler()
    if name == "robust":
        return RobustScaler.fit(fit_raw), IdentityScaler()
    raise ValueError(name)


def transform(primary, secondary, values):
    return secondary.transform(primary.transform(values)).astype(np.float32)


def clipping_fraction(primary, secondary, values):
    if not isinstance(secondary, FitMinMax):
        return np.nan, np.nan, np.nan
    primary_values = primary.transform(values)
    below = primary_values < secondary.low
    above = primary_values > secondary.high
    return float(below.mean()), float(above.mean()), float((below | above).mean())


def group_file(output, setup, excitation):
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def audit_file(output, setup, excitation):
    return output / "groups" / f"setup{setup:02d}_{excitation}_saturation.csv"


def run_group(group, labels, seed, group_index, args, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    targets = group[group.split.isin(["val", "test"])]
    train_windows = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        ) for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_windows)
    raw_cache, counts = {}, {}
    for row in targets.itertuples():
        current = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        raw_cache[str(row.record_id)] = current.astype(np.float32)
        counts[str(row.record_id)] = len(current)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    frames, audits = [], []

    for variant_index, (variant, config) in enumerate(VARIANTS.items()):
        print(f"  variant={variant}", flush=True)
        primary, secondary = build_scaler(config["scaler"], fit_raw)
        fit = transform(primary, secondary, fit_raw)
        internal = transform(primary, secondary, internal_raw)
        seed_everything(seed + group_index * 1009 + variant_index * 101)
        model = fit_model(fit, config, args.epochs, args.batch_size, device)
        parameters = count_parameters(model)
        internal_scores = score_model(model, internal, args.batch_size, device)
        reference = robust_score_reference(internal_scores)
        print(
            f"    input={tuple(fit.shape)} parameters={parameters} "
            f"fit_range=({fit.min():.3f},{fit.max():.3f})", flush=True,
        )
        transformed_cache = {
            record_id: transform(primary, secondary, values)
            for record_id, values in raw_cache.items()
        }
        for split, split_records in targets.groupby("split", sort=False):
            for condition, kind, rate, replicate in scenarios():
                corrupted_arrays, record_counts, metadata = [], [], []
                for row in split_records.itertuples():
                    record_id = str(row.record_id)
                    values = transformed_cache[record_id]
                    corrupted, available = sensor_corrupt(
                        values, kind, rate, record_id, replicate, importance,
                    )
                    corrupted_arrays.append(corrupted)
                    record_counts.append(counts[record_id])
                    metadata.append({
                        "record_id": record_id, "split": split,
                        "label": int(row.label), "state": row.state,
                        "setup": row.setup, "excitation": row.excitation,
                        "condition": condition, "replicate": replicate,
                        "coverage": float(available.mean()),
                    })
                    low, high, total = clipping_fraction(
                        primary, secondary, raw_cache[record_id],
                    )
                    audits.append({
                        "variant": variant, "record_id": record_id,
                        "split": split, "label": int(row.label),
                        "condition": condition, "replicate": replicate,
                        "clip_low": low, "clip_high": high,
                        "clip_total": total,
                    })
                # One batched inference call per split/scenario avoids thousands
                # of tiny GPU launches while preserving record boundaries.
                window_scores = score_model(
                    model, np.concatenate(corrupted_arrays), args.batch_size, device,
                )
                upper, two_sided = aggregate_record_scores(
                    window_scores, record_counts, reference,
                )
                current = pd.DataFrame(metadata)
                current["variant"] = variant
                current["score_upper"] = upper
                current["score_lower"] = -np.asarray(upper)
                current["score_two_sided"] = two_sided
                current["parameters"] = parameters
                current["seed"] = seed
                frames.append(current)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    pd.DataFrame(audits).to_csv(
        audit_file(args.output_dir, int(group.setup.iloc[0]), str(group.excitation.iloc[0])),
        index=False,
    )


def select_variant_policy(frame):
    rows = []
    validation = frame[frame.split == "val"]
    for variant in VARIANTS:
        current = validation[validation.variant == variant]
        for tail, column in TAIL_COLUMNS.items():
            threshold, _ = threshold_from_labeled_validation(
                current.label.to_numpy(), current[column].to_numpy(),
            )
            condition_metrics = []
            for (condition, replicate), cohort in current.groupby(
                ["condition", "replicate"], sort=False,
            ):
                metric = publication_metrics(
                    cohort.label.to_numpy(), cohort[column].to_numpy(), threshold,
                )
                condition_metrics.append({
                    "condition": condition, "replicate": replicate, **metric,
                })
            metric = pd.DataFrame(condition_metrics)
            corrupted = metric[metric.condition != "clean"]
            targeted = metric[metric.condition == "targeted_0.50"]
            rows.append({
                "variant": variant, "tail": tail, "threshold": threshold,
                "validation_auc": float(metric.auc.mean()),
                "validation_ba": float(metric.ba.mean()),
                "corrupted_auc": float(corrupted.auc.mean()),
                "corrupted_ba": float(corrupted.ba.mean()),
                "corrupted_recall": float(corrupted.recall.mean()),
                "corrupted_fpr": float(corrupted.fpr.mean()),
                "targeted_recall": float(targeted.recall.mean()),
                "objective": float(
                    corrupted.auc.mean() + corrupted.ba.mean()
                    - 0.10 * corrupted.fpr.mean()
                ),
            })
    search = pd.DataFrame(rows)
    baseline = search[
        (search.variant == "current_clip_sigmoid_large")
    ].sort_values(["objective", "corrupted_recall"], ascending=False).iloc[0]
    search["recall_feasible"] = (
        (search.corrupted_recall >= baseline.corrupted_recall - 0.03)
        & (search.targeted_recall >= baseline.targeted_recall - 0.03)
    )
    feasible = search[search.recall_feasible]
    selected = (feasible if len(feasible) else search).sort_values(
        ["objective", "corrupted_ba", "corrupted_auc"], ascending=False,
    ).iloc[0]
    return selected, search


def finish(args, frame, saturation):
    selected, search = select_variant_policy(frame)
    search.to_csv(args.output_dir / "validation_search.csv", index=False)
    selection = {
        key: (bool(selected[key]) if key == "recall_feasible" else
              float(selected[key]) if isinstance(selected[key], (float, np.floating)) else
              selected[key])
        for key in selected.index
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8",
    )

    metric_rows, score_rows = [], []
    for _, policy in search.iterrows():
        variant, tail = policy.variant, policy.tail
        column, threshold = TAIL_COLUMNS[tail], float(policy.threshold)
        test = frame[(frame.split == "test") & (frame.variant == variant)].copy()
        test["tail"] = tail
        test["score"] = test[column]
        test["threshold"] = threshold
        score_rows.append(test)
        for (condition, replicate), cohort in test.groupby(
            ["condition", "replicate"], sort=False,
        ):
            metric_rows.append({
                "variant": variant, "tail": tail, "condition": condition,
                "replicate": int(replicate),
                **publication_metrics(cohort.label, cohort.score, threshold),
            })
    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(score_rows, ignore_index=True)
    summary = metrics.groupby(["variant", "tail", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
    )
    metrics.to_csv(args.output_dir / "metrics_all_tails.csv", index=False)
    scores.to_csv(args.output_dir / "record_scores_all_tails.csv", index=False)
    summary.to_csv(args.output_dir / "summary_all_tails.csv", index=False)
    saturation.to_csv(args.output_dir / "saturation_audit.csv", index=False)

    locked = summary.merge(
        search[["variant", "tail", "recall_feasible", "objective"]],
        on=["variant", "tail"], how="inner",
    )
    locked = locked.sort_values(["variant", "condition", "objective"], ascending=[True, True, False])
    locked = locked.groupby(["variant", "condition"], as_index=False).first()
    locked.to_csv(args.output_dir / "locked_test_summary.csv", index=False)

    pivot = locked.pivot(index="condition", columns="variant", values="auc_mean")
    axis = pivot.plot.bar(figsize=(13, 5.5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 V10: locked preprocessing/capacity ablation")
    axis.figure.tight_layout()
    axis.figure.savefig(args.output_dir / "v10_locked_auc.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== VALIDATION SELECTION ===", flush=True)
    print(json.dumps(selection, indent=2), flush=True)
    print("\n=== LOCKED TEST SUMMARY ===", flush=True)
    print(locked.to_string(index=False), flush=True)
    return locked, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} variants={list(VARIANTS)} "
        f"seed={args.seed}", flush=True,
    )
    started = time.time()
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, *key)
        audit = audit_file(args.output_dir, *key)
        if path.exists() and audit.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(group, labels, args.seed, group_index, args, device, path)

    frames = [pd.read_csv(group_file(args.output_dir, *key)) for key in sorted(sensor_map)]
    audits = [pd.read_csv(audit_file(args.output_dir, *key)) for key in sorted(sensor_map)]
    frame = pd.concat(frames, ignore_index=True)
    saturation = pd.concat(audits, ignore_index=True)
    keys = ["variant", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate variant/record/split/condition keys")
    locked, selection = finish(args, frame, saturation)
    decision = {
        "selected_on_validation": selection,
        "promote_to_three_seeds": bool(
            selection["variant"] != "current_clip_sigmoid_large"
            and selection["recall_feasible"]
        ),
        "interpretation": (
            "Preprocessing/capacity correction passed validation screening."
            if selection["variant"] != "current_clip_sigmoid_large" and selection["recall_feasible"]
            else "No corrected variant beat the current pipeline under the recall constraint."
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 preprocessing-decoder-capacity ablation V10",
        "fit": "healthy train windows only",
        "selection": "clean/corrupted labeled validation only",
        "test_used_for_fit_or_selection": False,
        "missing_sensor_value": "zero in normalized domain (healthy center)",
        "variants": VARIANTS, "conditions": scenarios(),
        "seed": args.seed, "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    print(f"Z24_V10_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

