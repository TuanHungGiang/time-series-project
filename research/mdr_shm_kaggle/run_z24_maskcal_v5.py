from __future__ import annotations

"""One-seed Z24 screening of healthy-only mask-conditioned calibration.

The detector is identical in both arms. ``compact_raw`` uses the usual clean
healthy score reference, while ``compact_maskcal`` standardizes log anomaly
scores with a ridge model fitted only to synthetically masked *healthy
internal-calibration* windows. Damage test labels/scores never fit the model,
the calibrator, the score tail, or the threshold.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax,
    Normalizer,
    publication_metrics,
    seed_everything,
)
from run_z24_mir_usad_v0 import scenarios, sensor_corrupt  # noqa: E402
from run_z24_recovery_kd_v3 import (  # noqa: E402
    count_parameters,
    fit_compact_usad,
    score_compact,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores,
    load_manifest,
    read_stable_vertical,
    record_windows,
    robust_score_reference,
    split_training_windows,
    stable_sensor_map,
    threshold_from_labeled_validation,
)


METHODS = ("compact_raw", "compact_maskcal")


def mask_features(mask: np.ndarray, importance: np.ndarray) -> np.ndarray:
    """Encode observed sensors plus two physically interpretable summaries."""
    mask = np.asarray(mask, dtype=float).reshape(-1)
    weight = np.asarray(importance, dtype=float)
    weight = weight / max(float(weight.sum()), 1e-12)
    missing = 1.0 - mask
    coverage = float(mask.mean())
    criticality = float(np.dot(missing, weight))
    return np.r_[
        mask,
        coverage,
        criticality,
        coverage**2,
        criticality**2,
        coverage * criticality,
    ]


def calibration_patterns(
    channels: int,
    importance: np.ndarray,
    seed: int,
    random_patterns: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    patterns = [np.ones(channels, dtype=np.float32)]
    for rate in (0.30, 0.50):
        count = min(channels - 1, max(1, int(round(rate * channels))))
        for _ in range(random_patterns):
            mask = np.ones(channels, dtype=np.float32)
            mask[rng.permutation(channels)[:count]] = 0.0
            patterns.append(mask)
        # Include the exact stress-test family without using any test values.
        targeted = np.ones(channels, dtype=np.float32)
        targeted[np.argsort(importance)[-count:]] = 0.0
        patterns.append(targeted)
    # Remove accidental duplicate masks while preserving order.
    unique, seen = [], set()
    for pattern in patterns:
        key = tuple(pattern.astype(int).tolist())
        if key not in seen:
            unique.append(pattern)
            seen.add(key)
    return unique


class MaskCalibrator:
    """Predict healthy score location/scale from the observed-sensor mask."""

    def __init__(self, center: Ridge, log_scale: Ridge, importance: np.ndarray):
        self.center = center
        self.log_scale = log_scale
        self.importance = np.asarray(importance, dtype=float)

    def transform(self, score: np.ndarray, masks: np.ndarray) -> np.ndarray:
        features = np.stack([
            mask_features(mask, self.importance) for mask in np.asarray(masks)
        ])
        location = self.center.predict(features)
        scale = np.exp(self.log_scale.predict(features))
        scale = np.clip(scale, 1e-4, 10.0)
        values = np.log1p(np.maximum(np.asarray(score, dtype=float), 0.0))
        return (values - location) / scale


def fit_mask_calibrator(
    model,
    internal_normalized: np.ndarray,
    scaler: FitMinMax,
    importance: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
    random_patterns: int,
    ridge_alpha: float,
) -> tuple[MaskCalibrator, pd.DataFrame]:
    patterns = calibration_patterns(
        internal_normalized.shape[1], importance, seed, random_patterns,
    )
    rows = []
    for index, pattern in enumerate(patterns):
        masked = internal_normalized.copy()
        masked[:, pattern == 0, :] = 0.0
        scaled = scaler.transform(masked).astype(np.float32)
        score = score_compact(model, scaled, batch_size, device)
        log_score = np.log1p(np.maximum(score, 0.0))
        center = float(np.median(log_score))
        mad = 1.4826 * float(np.median(np.abs(log_score - center)))
        scale = max(mad, 0.25 * float(np.std(log_score)), 1e-4)
        rows.append({
            "pattern": index,
            "coverage": float(pattern.mean()),
            "criticality": float(np.dot(1.0 - pattern, importance / importance.sum())),
            "center": center,
            "scale": scale,
            "mask": "".join(str(int(value)) for value in pattern),
            "features": mask_features(pattern, importance),
        })
    x = np.stack([row.pop("features") for row in rows])
    center_target = np.asarray([row["center"] for row in rows])
    scale_target = np.log(np.asarray([row["scale"] for row in rows]))
    center_model = Ridge(alpha=ridge_alpha).fit(x, center_target)
    scale_model = Ridge(alpha=ridge_alpha).fit(x, scale_target)
    diagnostics = pd.DataFrame(rows)
    diagnostics["predicted_center"] = center_model.predict(x)
    diagnostics["predicted_scale"] = np.exp(scale_model.predict(x))
    calibrator = MaskCalibrator(center_model, scale_model, importance)
    return calibrator, diagnostics


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


def aggregate_calibrated(values: np.ndarray, counts: list[int]):
    upper, two_sided, start = [], [], 0
    for count in counts:
        current = values[start:start + count]
        upper.append(float(np.median(current)))
        two_sided.append(float(np.median(np.abs(current))))
        start += count
    if start != len(values):
        raise RuntimeError("Calibrated window-score aggregation mismatch")
    return upper, two_sided


def make_frame(
    method: str,
    upper,
    two_sided,
    counts,
    metadata,
    available,
    parameters: int,
):
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


def group_file(output: Path, setup: int, excitation: str) -> Path:
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
    print(
        f"  fit={tuple(fit.shape)} internal={tuple(internal_normalized.shape)}",
        flush=True,
    )

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    seed_everything(seed + group_index * 1009)
    model = fit_compact_usad(
        fit, args.epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    parameters = count_parameters(model)
    clean_internal = scaler.transform(internal_normalized).astype(np.float32)
    internal_score = score_compact(model, clean_internal, args.batch_size, device)
    raw_reference = robust_score_reference(internal_score)
    calibrator, diagnostics = fit_mask_calibrator(
        model, internal_normalized, scaler, importance,
        args.batch_size, device, seed + group_index * 1009 + 503,
        args.random_patterns, args.ridge_alpha,
    )
    diagnostics.insert(0, "setup", int(group.setup.iloc[0]))
    diagnostics.insert(1, "excitation", str(group.excitation.iloc[0]))
    diagnostics.to_csv(output.with_name(output.stem + "_calibration.csv"), index=False)
    print(
        f"  parameters={parameters:,} calibration_patterns={len(diagnostics)}",
        flush=True,
    )

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
        score = score_compact(
            model, scaler.transform(values).astype(np.float32),
            args.batch_size, device,
        )
        raw_upper, raw_two = aggregate_record_scores(score, counts, raw_reference)
        frames.append(make_frame(
            "compact_raw", raw_upper, raw_two, counts, metadata,
            available, parameters,
        ))
        calibrated = calibrator.transform(score, available)
        cal_upper, cal_two = aggregate_calibrated(calibrated, counts)
        frames.append(make_frame(
            "compact_maskcal", cal_upper, cal_two, counts, metadata,
            available, parameters,
        ))
    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finish(output: Path, frame: pd.DataFrame) -> pd.DataFrame:
    metric_rows, scored = [], []
    for method in METHODS:
        validation = frame[(frame.method == method) & (frame.split == "val")].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                validation.label, validation[f"score_{tail}"],
            )
            candidates.append(
                ((metric["ba"], metric["auc"], metric["f1"]), tail, threshold)
            )
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        test["score"] = test[f"score_{tail}"]
        test["threshold"] = threshold
        test["score_tail"] = tail
        scored.append(test)
        rows, clean = [], None
        for (condition, replicate), current in test.groupby(["condition", "replicate"]):
            row = {
                "method": method,
                "condition": condition,
                "replicate": int(replicate),
                "score_tail": tail,
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
        auc_mean=("auc", "mean"),
        ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"),
        f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"),
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="fpr_mean")
    axis = pivot.plot.bar(figsize=(11, 5), ylim=(0, 1.0), rot=15, grid=True)
    axis.set_ylabel("False-positive rate")
    axis.set_title("Z24: fixed threshold under sensor loss")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "maskcal_fpr.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== ONE-SEED SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary


def decision(summary: pd.DataFrame) -> dict:
    table = summary.set_index(["method", "condition"])
    corruptions = ["sensor_0.30", "sensor_0.50", "targeted_0.50"]
    effects = {}
    for condition in ["clean", *corruptions]:
        proposed = table.loc[("compact_maskcal", condition)]
        baseline = table.loc[("compact_raw", condition)]
        effects[condition] = {
            "delta_auc": float(proposed.auc_mean - baseline.auc_mean),
            "delta_ba": float(proposed.ba_mean - baseline.ba_mean),
            "delta_fpr": float(proposed.fpr_mean - baseline.fpr_mean),
            "delta_recall": float(proposed.recall_mean - baseline.recall_mean),
        }
    promote = bool(
        np.mean([effects[c]["delta_ba"] for c in corruptions]) > 0.0
        and np.mean([effects[c]["delta_fpr"] for c in corruptions]) < 0.0
        and effects["clean"]["delta_ba"] >= -0.02
    )
    return {
        "effects_maskcal_minus_raw": effects,
        "promote_to_three_seeds": promote,
        "decision": (
            "PROMOTE to 3 seeds; calibration improved corrupted BA/FPR without material clean loss."
            if promote else
            "STOP after 1 seed; mask calibration did not jointly improve corrupted BA and FPR."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--random-patterns", type=int, default=24)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={METHODS} "
        f"seed={args.seed} epochs={args.epochs}",
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
        "experiment": "Z24 compact USAD mask-conditioned healthy calibration v5",
        "seed": args.seed,
        "epochs": args.epochs,
        "fit": "healthy train only",
        "calibration": "synthetic masks on held-out healthy train windows only",
        "selection": "clean labeled validation only",
        "test_used_for_fit_calibration_or_selection": False,
        "methods": list(METHODS),
        "random_patterns_per_rate": args.random_patterns,
        "ridge_alpha": args.ridge_alpha,
        "conditions": scenarios(),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(screening, indent=2), flush=True)
    print(f"MASKCAL_V5_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

