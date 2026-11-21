from __future__ import annotations

"""One-seed Z24 observed-sensor evidence pooling experiment.

The proposed detector never imputes a missing sensor.  A shared temporal
autoencoder produces one anomaly score per observed sensor.  Healthy-only,
sensor-specific robust references standardize those scores before either mean
or top-k evidence pooling.  Missing sensors are excluded from pooling, so they
cannot contaminate representations of sensors that remain available.
"""

import argparse
import copy
import gc
import json
import math
import sys
import time
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
from run_z24_recovery_kd_v3 import (  # noqa: E402
    count_parameters, fit_compact_usad, score_compact,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)

METHODS = ("compact_raw", "sensorwise_mean", "sensorwise_topk")


class SensorwiseAutoencoder(nn.Module):
    """Shared temporal AE with a learned identity embedding for each sensor."""

    def __init__(self, channels: int, latent_channels: int = 32):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(8, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(16, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(24, latent_channels, 5, stride=2, padding=2), nn.GELU(),
        )
        self.sensor_embedding = nn.Embedding(channels, latent_channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 24, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(24, 16, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(8, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, length = values.shape
        if channels != self.channels:
            raise RuntimeError(f"Expected {self.channels} sensors, got {channels}")
        flat = values.reshape(batch * channels, 1, length)
        latent = self.encoder(flat)
        sensor_ids = torch.arange(channels, device=values.device).repeat(batch)
        identity = self.sensor_embedding(sensor_ids)[:, :, None]
        reconstructed = self.decoder(latent + identity)
        if reconstructed.shape[-1] != length:
            reconstructed = reconstructed[..., :length]
        return reconstructed.reshape(batch, channels, length)


@torch.no_grad()
def validation_loss(model, values, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    losses = []
    model.eval()
    for (batch,) in loader:
        batch = batch.to(device)
        losses.append(float((batch - model(batch)).pow(2).mean().cpu()))
    return float(np.mean(losses))


def fit_sensorwise(
    fit, internal, epochs, batch_size, latent_channels, patience, device,
):
    model = SensorwiseAutoencoder(fit.shape[1], latent_channels).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)), batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = (batch - reconstructed).pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        heldout = validation_loss(model, internal, batch_size, device)
        if heldout < best_loss - 1e-7:
            best_loss = heldout
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"  sensorwise epoch={epoch}/{epochs} "
                f"fit={np.mean(losses):.6f} internal={heldout:.6f} "
                f"best_epoch={best_epoch}", flush=True,
            )
        if stale >= patience:
            print(f"  sensorwise early-stop epoch={epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("Sensorwise model produced no finite checkpoint")
    model.load_state_dict(best_state)
    print(f"  sensorwise selected_epoch={best_epoch} internal={best_loss:.6f}", flush=True)
    return model.eval(), best_epoch, best_loss


@torch.no_grad()
def sensor_errors(model, values, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    rows = []
    model.eval()
    for (batch,) in loader:
        batch = batch.to(device)
        reconstructed = model(batch)
        rows.append((batch - reconstructed).pow(2).mean(2).cpu().numpy())
    return np.concatenate(rows)


def fit_sensor_reference(errors):
    values = np.log1p(np.maximum(np.asarray(errors, float), 0.0))
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center[None, :]), axis=0)
    standard = np.std(values, axis=0)
    global_scale = max(float(np.median(mad)), 0.25 * float(np.std(values)), 1e-6)
    raw_scale = np.maximum.reduce([mad, 0.25 * standard, np.full_like(mad, 1e-6)])
    # Small internal cohorts need partial pooling toward one global scale.
    scale = 0.5 * raw_scale + 0.5 * global_scale
    return center, scale


def standardized_sensor_scores(errors, reference):
    values = np.log1p(np.maximum(np.asarray(errors, float), 0.0))
    return (values - reference[0][None, :]) / reference[1][None, :]


def pool_window_scores(sensor_z, available, mode, top_fraction):
    upper, two_sided = [], []
    for row, mask in zip(sensor_z, available):
        observed = row[np.asarray(mask) > 0.5]
        if len(observed) == 0:
            raise RuntimeError("A window has no observed sensors")
        if mode == "mean":
            upper.append(float(np.mean(observed)))
            two_sided.append(float(np.mean(np.abs(observed))))
        elif mode == "topk":
            count = max(1, int(math.ceil(top_fraction * len(observed))))
            upper.append(float(np.mean(np.partition(observed, -count)[-count:])))
            absolute = np.abs(observed)
            two_sided.append(float(np.mean(np.partition(absolute, -count)[-count:])))
        else:
            raise ValueError(mode)
    return np.asarray(upper), np.asarray(two_sided)


def aggregate_windows(upper_windows, two_windows, counts):
    upper, two_sided, start = [], [], 0
    for count in counts:
        upper.append(float(np.median(upper_windows[start:start + count])))
        two_sided.append(float(np.median(two_windows[start:start + count])))
        start += count
    if start != len(upper_windows):
        raise RuntimeError("Window aggregation length mismatch")
    return upper, two_sided


def build_values(records, cache, condition, kind, rate, replicate, importance):
    arrays, masks, counts, metadata = [], [], [], []
    for row in records.itertuples():
        values, available = sensor_corrupt(
            cache[str(row.record_id)], kind, rate, str(row.record_id),
            replicate, importance,
        )
        arrays.append(values)
        masks.append(available)
        counts.append(len(values))
        metadata.append({
            "record_id": row.record_id,
            "split": row.split,
            "label": int(row.label),
            "state": row.state,
            "setup": row.setup,
            "excitation": row.excitation,
            "condition": condition,
            "replicate": replicate,
            "rate": rate,
        })
    return np.concatenate(arrays), np.concatenate(masks), counts, metadata


def make_frame(method, upper, two_sided, counts, metadata, available, parameters):
    frame = pd.DataFrame(metadata)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two_sided
    frame["windows"] = counts
    frame["parameters"] = parameters
    window_coverage = available.mean(1)
    start, coverage = 0, []
    for count in counts:
        coverage.append(float(window_coverage[start:start + count].mean()))
        start += count
    frame["coverage"] = coverage
    return frame


def group_file(output, setup, excitation):
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def run_group(group, labels, seed, group_index, args, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_by_record = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_by_record)
    normalizer = Normalizer.fit(fit_raw)
    fit_normalized = normalizer.transform(fit_raw)
    internal_normalized = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scaler = FitMinMax(fit_normalized)
    fit = scaler.transform(fit_normalized).astype(np.float32)
    internal = scaler.transform(internal_normalized).astype(np.float32)
    print(f"  fit={tuple(fit.shape)} internal={tuple(internal.shape)}", flush=True)

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    base_seed = seed + group_index * 1009
    seed_everything(base_seed)
    compact = fit_compact_usad(
        fit, args.compact_epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    compact_internal = score_compact(compact, internal, args.batch_size, device)
    compact_reference = robust_score_reference(compact_internal)

    seed_everything(base_seed + 97)
    sensorwise, selected_epoch, internal_loss = fit_sensorwise(
        fit, internal, args.sensor_epochs, args.batch_size,
        args.sensor_latent, args.patience, device,
    )
    internal_errors = sensor_errors(sensorwise, internal, args.batch_size, device)
    sensor_reference = fit_sensor_reference(internal_errors)
    diagnostic = pd.DataFrame({
        "setup": int(group.setup.iloc[0]),
        "excitation": str(group.excitation.iloc[0]),
        "sensor_index": np.arange(len(labels)),
        "sensor_label": labels,
        "healthy_center": sensor_reference[0],
        "healthy_scale": sensor_reference[1],
        "selected_epoch": selected_epoch,
        "internal_loss": internal_loss,
    })
    diagnostic.to_csv(output.with_name(output.stem + "_sensor_reference.csv"), index=False)
    parameters = {
        "compact_raw": count_parameters(compact),
        "sensorwise_mean": count_parameters(sensorwise),
        "sensorwise_topk": count_parameters(sensorwise),
    }
    print(f"  parameters={parameters}", flush=True)

    cohorts = [(validation, "clean", "clean", 0.0, 0)]
    cohorts.extend([
        (test, condition, kind, rate, replicate)
        for condition, kind, rate, replicate in scenarios()
    ])
    frames = []
    for records, condition, kind, rate, replicate in cohorts:
        values, available, counts, metadata = build_values(
            records, cache, condition, kind, rate, replicate, importance,
        )
        scaled = scaler.transform(values).astype(np.float32)
        compact_score = score_compact(compact, scaled, args.batch_size, device)
        compact_upper, compact_two = aggregate_record_scores(
            compact_score, counts, compact_reference,
        )
        frames.append(make_frame(
            "compact_raw", compact_upper, compact_two, counts, metadata,
            available, parameters["compact_raw"],
        ))

        errors = sensor_errors(sensorwise, scaled, args.batch_size, device)
        sensor_z = standardized_sensor_scores(errors, sensor_reference)
        for method, mode in (("sensorwise_mean", "mean"), ("sensorwise_topk", "topk")):
            upper_windows, two_windows = pool_window_scores(
                sensor_z, available, mode, args.top_fraction,
            )
            upper, two_sided = aggregate_windows(upper_windows, two_windows, counts)
            frames.append(make_frame(
                method, upper, two_sided, counts, metadata,
                available, parameters[method],
            ))
    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    result["top_fraction"] = args.top_fraction
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del compact, sensorwise
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finish(output, frame):
    metric_rows, scored = [], []
    for method in METHODS:
        validation = frame[(frame.method == method) & (frame.split == "val")].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                validation.label, validation[f"score_{tail}"],
            )
            candidates.append(((metric["ba"], metric["auc"], metric["f1"]), tail, threshold))
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        test["score"] = test[f"score_{tail}"]
        test["threshold"] = threshold
        test["score_tail"] = tail
        scored.append(test)
        rows, clean = [], None
        for (condition, replicate), current in test.groupby(["condition", "replicate"]):
            row = {
                "method": method, "condition": condition,
                "replicate": int(replicate), "score_tail": tail,
                "threshold": threshold,
                **publication_metrics(current.label, current.score, threshold),
                "coverage_mean": float(current.coverage.mean()),
                "parameters": int(current.parameters.iloc[0]),
            }
            rows.append(row)
            if condition == "clean":
                clean = row
        for row in rows:
            row["delta_auc"] = row["auc"] - clean["auc"]
            row["delta_ba"] = row["ba"] - clean["ba"]
            row["delta_fpr"] = row["fpr"] - clean["fpr"]
        metric_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(scored, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"), parameters=("parameters", "mean"),
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(12, 5.5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 observed-sensor evidence pooling")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "sensor_pool_auc.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== ONE-SEED SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary


def decision(summary):
    table = summary.set_index(["method", "condition"])
    corruptions = ["sensor_0.30", "sensor_0.50", "targeted_0.50"]
    comparisons = {}
    for method in ("sensorwise_mean", "sensorwise_topk"):
        comparisons[method] = {}
        for condition in ["clean", *corruptions]:
            proposed = table.loc[(method, condition)]
            baseline = table.loc[("compact_raw", condition)]
            comparisons[method][condition] = {
                "delta_auc": float(proposed.auc_mean - baseline.auc_mean),
                "delta_ba": float(proposed.ba_mean - baseline.ba_mean),
                "delta_fpr": float(proposed.fpr_mean - baseline.fpr_mean),
                "delta_recall": float(proposed.recall_mean - baseline.recall_mean),
            }
    candidates = []
    for method, effects in comparisons.items():
        average_auc = float(np.mean([effects[c]["delta_auc"] for c in corruptions]))
        average_ba = float(np.mean([effects[c]["delta_ba"] for c in corruptions]))
        promote = bool(
            average_auc > 0.0 and average_ba > 0.0
            and effects["targeted_0.50"]["delta_auc"] >= 0.0
            and effects["clean"]["delta_auc"] >= -0.02
        )
        candidates.append((promote, average_auc + average_ba, method))
    promote, _, best = max(candidates)
    return {
        "effects_vs_compact_raw": comparisons,
        "best_sensor_pooling": best,
        "promote_to_three_seeds": bool(promote),
        "decision": (
            f"PROMOTE {best} to a three-seed pilot."
            if promote else
            "STOP after one seed; observed-sensor pooling did not jointly improve corrupted AUC and BA."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--compact-epochs", type=int, default=5)
    parser.add_argument("--sensor-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--sensor-latent", type=int, default=32)
    parser.add_argument("--top-fraction", type=float, default=0.30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={METHODS} seed={args.seed} "
        f"compact_epochs={args.compact_epochs} sensor_epochs<={args.sensor_epochs}",
        flush=True,
    )
    started = time.time()
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, *key)
        if path.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(group, labels, args.seed, group_index, args, device, path)

    paths = [group_file(args.output_dir, *key) for key in sorted(sensor_map)]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(METHODS) * (132 + 132 * len(scenarios()))
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/record/condition keys")
    summary = finish(args.output_dir, frame)
    screening = decision(summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 observed-sensor evidence pooling v7",
        "seed": args.seed, "fit": "healthy train only",
        "selection": "clean labeled validation only",
        "test_used_for_fit_or_selection": False,
        "missing_sensor_handling": "exclude from score; no imputation",
        "methods": list(METHODS), "top_fraction": args.top_fraction,
        "sensor_model_selection": "healthy internal reconstruction loss only",
        "conditions": scenarios(), "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(screening, indent=2), flush=True)
    print(f"SENSOR_POOL_V7_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

