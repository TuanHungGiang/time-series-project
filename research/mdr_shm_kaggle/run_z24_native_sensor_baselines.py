from __future__ import annotations

"""Native-signal healthy-only baselines for the project Z24 protocol.

All methods use the same complete-record split.  Z24 moves sensors between
setups and has no globally common physical channel label, so the default
``auto`` schema falls back to a fixed number of setup-local sensor slots. Input
values and labels from damage records never influence fitting, normalization,
score referencing, or threshold calibration.  Channel availability metadata
is used only to construct a fixed input shape.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, resample_poly
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    MODELS,
    Normalizer,
    publication_metrics,
    run_model,
    seed_everything,
    threshold_from_healthy,
)
from run_z24_global_pooled_baselines import fixed_internal_split  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores,
    load_manifest,
    parse_labels,
    read_stable_vertical,
    robust_score_reference,
)


def sensor_number(label: str) -> int:
    match = re.match(r"\d+", str(label))
    return int(match.group()) if match else 10**9


def sorted_labels(value: str) -> list[str]:
    return sorted(
        parse_labels(value), key=lambda label: (sensor_number(label), label)
    )


def resolve_sensor_schema(frame: pd.DataFrame, requested: str) -> dict:
    """Resolve global physical labels or deterministic setup-local slots."""
    label_lists = [sorted_labels(value) for value in frame.vertical_labels]
    common = sorted(
        set.intersection(*(set(labels) for labels in label_lists)),
        key=lambda label: (sensor_number(label), label),
    )
    minimum_count = min(map(len, label_lists))
    if requested == "intersection":
        if len(common) < 12:
            raise RuntimeError(
                f"Intersection schema requested but only {len(common)} labels "
                f"are common across layouts: {common}"
            )
        return {"mode": "global_physical_intersection", "labels": common,
                "sensor_count": len(common)}
    if requested == "auto" and len(common) >= 12:
        return {"mode": "global_physical_intersection", "labels": common,
                "sensor_count": len(common)}
    if minimum_count < 12:
        raise RuntimeError(f"Only {minimum_count} vertical sensor slots available")
    return {
        "mode": "setup_local_rank",
        "labels": [],
        "sensor_count": minimum_count,
        "global_intersection_count": len(common),
        "limitation": (
            "Slot k is the kth numerically sorted vertical channel within each "
            "setup; slots do not claim identical physical coordinates across setups."
        ),
    }


def native_windows(
    signals: np.ndarray,
    raw_window: int,
    stride: int,
    downsample: int,
) -> np.ndarray:
    starts = range(0, signals.shape[1] - raw_window + 1, stride)
    windows = []
    for start in starts:
        values = np.asarray(signals[:, start:start + raw_window], dtype=np.float64)
        values = detrend(values, axis=1, type="linear")
        # Polyphase resampling includes an anti-aliasing FIR filter and does
        # not use any healthy/damage statistic.
        values = resample_poly(values, up=1, down=downsample, axis=1)
        windows.append(values.astype(np.float32))
    output = np.stack(windows)
    if len(output) < 16 or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid native window tensor {output.shape}")
    return output


def record_cache(
    rows: pd.DataFrame,
    schema: dict,
    raw_window: int,
    stride: int,
    downsample: int,
) -> dict[str, np.ndarray]:
    cache = {}
    for index, row in enumerate(rows.itertuples(), 1):
        if schema["mode"] == "global_physical_intersection":
            labels = schema["labels"]
        else:
            labels = sorted_labels(row.vertical_labels)[:schema["sensor_count"]]
        signals = read_stable_vertical(Path(row.path), labels)
        cache[str(row.record_id)] = native_windows(
            signals, raw_window, stride, downsample
        )
        if index % 20 == 0 or index == len(rows):
            print(f"native_features={index}/{len(rows)}", flush=True)
    return cache


def concatenate_records(
    rows: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    arrays = [cache[str(record_id)] for record_id in rows.record_id]
    return np.concatenate(arrays), [len(array) for array in arrays]


def select_reference_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select two score-reference records per excitation without using damage."""
    fit_parts, reference_parts = [], []
    for _, group in rows.groupby("excitation", sort=True):
        ordered = group.sort_values(["state", "setup", "record_id"]).reset_index(drop=True)
        if len(ordered) < 6:
            raise RuntimeError("Too few healthy rows for cross-fit reference split")
        positions = sorted({len(ordered) // 3, (2 * len(ordered)) // 3})
        reference_parts.append(ordered.iloc[positions])
        fit_parts.append(ordered.drop(index=positions))
    return (
        pd.concat(fit_parts, ignore_index=True),
        pd.concat(reference_parts, ignore_index=True),
    )


def crossfit_healthy_scores(
    model_name: str,
    seed: int,
    development_rows: pd.DataFrame,
    cache: dict[str, np.ndarray],
    folds: int,
    epochs: int,
    batch_size: int,
    device,
) -> pd.DataFrame:
    """Generate one out-of-fold standardized record score per healthy record."""
    strata = (
        development_rows.state.astype(str)
        + "_"
        + development_rows.excitation.astype(str)
    )
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=20261113
    )
    output = []
    for fold, (fit_index, heldout_index) in enumerate(
        splitter.split(development_rows, strata)
    ):
        available = development_rows.iloc[fit_index].copy().reset_index(drop=True)
        heldout = development_rows.iloc[heldout_index].copy().reset_index(drop=True)
        fold_fit, fold_reference = select_reference_rows(available)
        fit, _ = concatenate_records(fold_fit, cache)
        reference, _ = concatenate_records(fold_reference, cache)
        target, counts = concatenate_records(heldout, cache)
        normalizer = Normalizer.fit(fit)
        fit = normalizer.transform(fit)
        reference = normalizer.transform(reference)
        target = normalizer.transform(target)
        seed_everything(seed + 1009 * (fold + 1))
        reference_scores, heldout_scores = run_model(
            model_name, fit, reference, target, epochs, batch_size, device
        )
        score_reference = robust_score_reference(reference_scores)
        upper, _ = aggregate_record_scores(
            heldout_scores, counts, score_reference
        )
        current = heldout[
            ["record_id", "split", "state", "setup", "excitation"]
        ].copy()
        current["fold"] = fold
        current["score"] = upper
        current["fit_records"] = len(fold_fit)
        current["reference_records"] = len(fold_reference)
        output.append(current)
        print(
            f"  crossfit fold={fold + 1}/{folds} model={model_name} "
            f"fit={len(fold_fit)} reference={len(fold_reference)} "
            f"heldout={len(heldout)}",
            flush=True,
        )
    frame = pd.concat(output, ignore_index=True)
    if len(frame) != len(development_rows) or frame.record_id.duplicated().any():
        raise RuntimeError("Cross-fit calibration did not score each healthy record once")
    return frame


def evaluate_run(
    model_name: str,
    seed: int,
    fit_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    cache: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    specificity_target: float,
    calibration_mode: str,
    crossfit_folds: int,
    device,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    fit, _ = concatenate_records(fit_rows, cache)
    reference, _ = concatenate_records(reference_rows, cache)
    validation, validation_counts = concatenate_records(validation_rows, cache)
    test, test_counts = concatenate_records(test_rows, cache)
    target = np.concatenate([validation, test])
    counts = validation_counts + test_counts

    normalizer = Normalizer.fit(fit)
    fit = normalizer.transform(fit)
    reference = normalizer.transform(reference)
    target = normalizer.transform(target)
    seed_everything(seed)
    reference_scores, target_scores = run_model(
        model_name, fit, reference, target, epochs, batch_size, device
    )
    score_reference = robust_score_reference(reference_scores)
    upper_scores, two_sided_scores = aggregate_record_scores(
        target_scores, counts, score_reference
    )

    records = pd.concat([validation_rows, test_rows], ignore_index=True)[
        ["record_id", "split", "label", "state", "setup", "excitation"]
    ].copy()
    records["score"] = upper_scores
    records["score_native_upper"] = upper_scores
    records["score_two_sided_diagnostic"] = two_sided_scores
    records["windows"] = counts
    records["model"] = model_name
    records["seed"] = seed

    calibration = records[(records.split == "val") & (records.label == 0)]
    test_records = records[records.split == "test"].copy()
    six_record_threshold = threshold_from_healthy(
        calibration.score.to_numpy(), specificity_target
    )
    if calibration_mode == "crossfit":
        development = pd.concat(
            [fit_rows, reference_rows, validation_rows], ignore_index=True
        )
        calibration_frame = crossfit_healthy_scores(
            model_name, seed, development, cache, crossfit_folds,
            epochs, batch_size, device,
        )
        threshold = threshold_from_healthy(
            calibration_frame.score.to_numpy(), specificity_target
        )
    else:
        calibration_frame = calibration[
            ["record_id", "split", "state", "setup", "excitation", "score"]
        ].copy()
        calibration_frame["fold"] = -1
        threshold = six_record_threshold
    metric = {
        "model": model_name,
        "seed": seed,
        "score": "native upper-tail record-median",
        "calibration_mode": calibration_mode,
        **publication_metrics(test_records.label, test_records.score, threshold),
        "threshold": threshold,
        "six_record_threshold_diagnostic": six_record_threshold,
        "calibration_specificity": float(
            np.mean(calibration_frame.score < threshold)
        ),
        "calibration_score_records": len(calibration_frame),
        "fit_healthy_records": len(fit_rows),
        "reference_healthy_records": len(reference_rows),
        "calibration_healthy_records": len(validation_rows),
        "test_healthy_records": int((test_records.label == 0).sum()),
        "test_damage_records": int((test_records.label == 1).sum()),
    }
    test_records["threshold"] = threshold
    return metric, test_records, calibration_frame


def refresh(output: Path) -> None:
    metric_paths = sorted((output / "runs").glob("*/metrics.json"))
    if not metric_paths:
        return
    metrics = pd.DataFrame([
        json.loads(path.read_text(encoding="utf-8")) for path in metric_paths
    ])
    scores = pd.concat([
        pd.read_csv(path.parent / "scores.csv") for path in metric_paths
    ], ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    aggregate = metrics.groupby("model", as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"),
        ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        specificity_mean=("specificity", "mean"), runs=("seed", "size"),
    )
    aggregate.to_csv(output / "aggregate.csv", index=False)
    axis = aggregate.set_index("model")[[
        "auc_mean", "ba_mean", "recall_mean", "specificity_mean", "f1_mean"
    ]].plot.bar(figsize=(13, 6), ylim=(0, 1.03), rot=30, grid=True)
    axis.set_title("Z24 sensor-preserving native baseline pilot")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "native_baseline_pilot.png", dpi=220, bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261113, 20261114, 20261115])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--specificity-target", type=float, default=.90)
    parser.add_argument(
        "--calibration-mode", choices=("six_val", "crossfit"),
        default="six_val",
    )
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument(
        "--sensor-schema", choices=("auto", "local_rank", "intersection"),
        default="auto",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    train = manifest[(manifest.split == "train") & (manifest.label == 0)].copy()
    validation = manifest[(manifest.split == "val") & (manifest.label == 0)].copy()
    test = manifest[manifest.split == "test"].copy()
    fit_rows, reference_rows = fixed_internal_split(train)
    needed = pd.concat([train, validation, test], ignore_index=True)
    schema = resolve_sensor_schema(needed, args.sensor_schema)
    print(f"sensor_schema={json.dumps(schema)}", flush=True)
    cache = record_cache(
        needed, schema, args.raw_window, args.window_stride, args.downsample
    )
    example = next(iter(cache.values()))

    protocol = {
        "dataset": "Z24",
        "task": "native-signal healthy-only unseen-damage detection",
        "split_unit": "complete MAT record",
        "conditioning": "global pooled records with fixed-size native channel slots",
        "sensor_schema_source": "channel-availability metadata only; no damage values or labels",
        "sensor_schema": schema,
        "common_sensor_labels": schema["labels"],
        "common_sensor_count": schema["sensor_count"],
        "preprocessing": "per-window linear detrend then polyphase anti-aliased downsampling",
        "raw_window": args.raw_window,
        "window_stride": args.window_stride,
        "downsample": args.downsample,
        "model_input_shape": list(example.shape[1:]),
        "windows_per_record_min": int(min(map(len, cache.values()))),
        "windows_per_record_max": int(max(map(len, cache.values()))),
        "fit_healthy_records": [str(value) for value in fit_rows.record_id],
        "score_reference_healthy_records": [str(value) for value in reference_rows.record_id],
        "calibration_healthy_records": [str(value) for value in validation.record_id],
        "locked_test_healthy_records": int((test.label == 0).sum()),
        "locked_test_damage_records": int((test.label == 1).sum()),
        "damage_values_or_labels_used_for_fit_normalization_reference_or_threshold": False,
        "score": "native upper-tail sequence score; median to complete-record score",
        "specificity_target": args.specificity_target,
        "calibration_mode": args.calibration_mode,
        "crossfit_folds": args.crossfit_folds if args.calibration_mode == "crossfit" else None,
        "crossfit_development_healthy_records": 30 if args.calibration_mode == "crossfit" else None,
        "epochs": args.epochs,
        "neural_seeds": args.seeds,
        "pca_runs": 1,
        "important_limitation": "Only six independent healthy calibration and six healthy test records.",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    for model_name in args.models:
        model_seeds = args.seeds[:1] if model_name.startswith("pca_") else args.seeds
        for seed in model_seeds:
            run_dir = args.output_dir / "runs" / f"{model_name}__{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "metrics.json").exists():
                print(f"SKIP complete model={model_name} seed={seed}", flush=True)
                continue
            print(f"\nmodel={model_name} seed={seed}", flush=True)
            started = time.time()
            metric, scores, calibration_scores = evaluate_run(
                model_name, seed, fit_rows, reference_rows, validation, test,
                cache, args.epochs, args.batch_size, args.specificity_target,
                args.calibration_mode, args.crossfit_folds, device,
            )
            metric["seconds"] = time.time() - started
            scores.to_csv(run_dir / "scores.csv", index=False)
            calibration_scores.to_csv(
                run_dir / "calibration_scores.csv", index=False
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(metric, indent=2), encoding="utf-8"
            )
            print(json.dumps(metric, indent=2), flush=True)
            refresh(args.output_dir)
    refresh(args.output_dir)
    print(pd.read_csv(args.output_dir / "aggregate.csv").to_string(index=False))
    print(f"Z24_NATIVE_SENSOR_BASELINES_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

