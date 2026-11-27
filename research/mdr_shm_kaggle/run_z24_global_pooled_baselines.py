from __future__ import annotations

"""Global pooled, healthy-only Z24 baselines.

The previous setup-by-excitation protocol left only one healthy peer in most
cells.  This runner uses sensor-order-invariant window descriptors so all
healthy records can be pooled without pretending that sensor indices are
aligned across changing layouts.  No damage record is used for fitting,
normalization, score referencing, threshold calibration, or model selection.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

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
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores,
    load_manifest,
    parse_labels,
    read_stable_vertical,
    robust_score_reference,
)


def descriptor_segments(
    signals: np.ndarray,
    segment_length: int,
    spectral_bins: int,
    representation: str,
    fs: float,
) -> np.ndarray:
    starts = range(0, signals.shape[1] - segment_length + 1, segment_length)
    grid = np.linspace(0.5, 30.0, spectral_bins)
    eps = np.finfo(np.float64).tiny
    descriptors = []
    for start in starts:
        values = detrend(
            np.asarray(signals[:, start:start + segment_length], dtype=np.float64),
            axis=1,
            type="linear",
        )
        rms = np.sqrt(np.mean(values**2, axis=1))
        derivative_rms = np.sqrt(np.mean(np.diff(values, axis=1) ** 2, axis=1))
        amplitude = np.concatenate([
            np.quantile(np.log10(np.maximum(rms, eps)), [0, .25, .5, .75, 1]),
            np.quantile(np.log10(np.maximum(derivative_rms, eps)), [0, .25, .5, .75, 1]),
        ])

        frequency, psd = welch(
            values,
            fs=fs,
            window="hann",
            nperseg=segment_length,
            noverlap=segment_length // 2,
            axis=1,
            detrend=False,
            scaling="density",
        )
        keep = (frequency >= 0.5) & (frequency <= 30.0)
        frequency, psd = frequency[keep], psd[:, keep]
        area = np.trapezoid(psd, frequency, axis=1)
        psd = psd / np.maximum(area[:, None], eps)
        shape = np.stack([
            np.interp(grid, frequency, np.log10(np.maximum(sensor, eps)))
            for sensor in psd
        ])
        shape -= shape.mean(axis=1, keepdims=True)
        spectral = np.concatenate([
            np.median(shape, axis=0),
            np.quantile(shape, .75, axis=0) - np.quantile(shape, .25, axis=0),
        ])
        descriptors.append(
            spectral if representation == "spectral_shape"
            else np.concatenate([amplitude, spectral])
        )
    output = np.asarray(descriptors, dtype=np.float32)
    if len(output) < 16 or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid descriptor matrix {output.shape}")
    return output


def descriptor_sequences(
    descriptors: np.ndarray,
    sequence_length: int,
    sequence_stride: int,
) -> np.ndarray:
    starts = range(0, len(descriptors) - sequence_length + 1, sequence_stride)
    sequences = np.stack([
        descriptors[start:start + sequence_length].T
        for start in starts
    ]).astype(np.float32)
    if not len(sequences):
        raise RuntimeError("No complete descriptor sequences")
    return sequences


def make_record_sequences(
    row,
    segment_length: int,
    spectral_bins: int,
    representation: str,
    sequence_length: int,
    sequence_stride: int,
    fs: float,
) -> np.ndarray:
    labels = parse_labels(row.vertical_labels)
    signals = read_stable_vertical(Path(row.path), labels)
    descriptors = descriptor_segments(
        signals, segment_length, spectral_bins, representation, fs
    )
    return descriptor_sequences(descriptors, sequence_length, sequence_stride)


def fixed_internal_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve two train records per excitation for score referencing."""
    fit_parts, internal_parts = [], []
    for _, group in train.groupby("excitation", sort=True):
        ordered = group.sort_values(["state", "setup", "record_id"]).reset_index(drop=True)
        if len(ordered) < 6:
            raise RuntimeError("Too few train records for fixed record-level holdout")
        # Evenly spaced positions avoid reserving only the last setup/state.
        positions = sorted({len(ordered) // 3, (2 * len(ordered)) // 3})
        internal_parts.append(ordered.iloc[positions])
        fit_parts.append(ordered.drop(index=positions))
    fit = pd.concat(fit_parts, ignore_index=True)
    internal = pd.concat(internal_parts, ignore_index=True)
    if len(fit) != 20 or len(internal) != 4:
        raise RuntimeError(f"Expected 20 fit and 4 internal records, got {len(fit)}, {len(internal)}")
    return fit, internal


def concatenate_records(
    rows: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    arrays = [cache[str(record_id)] for record_id in rows.record_id]
    return np.concatenate(arrays), [len(array) for array in arrays]


def evaluate_run(
    model_name: str,
    seed: int,
    fit_rows: pd.DataFrame,
    internal_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    cache: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    specificity_target: float,
    score_tail: str,
    device,
) -> tuple[dict, pd.DataFrame]:
    fit, _ = concatenate_records(fit_rows, cache)
    internal, _ = concatenate_records(internal_rows, cache)
    validation, validation_counts = concatenate_records(validation_rows, cache)
    test, test_counts = concatenate_records(test_rows, cache)
    targets = np.concatenate([validation, test])
    counts = validation_counts + test_counts

    normalizer = Normalizer.fit(fit)
    fit = normalizer.transform(fit)
    internal = normalizer.transform(internal)
    targets = normalizer.transform(targets)
    seed_everything(seed)
    internal_scores, target_scores = run_model(
        model_name, fit, internal, targets, epochs, batch_size, device
    )
    reference = robust_score_reference(internal_scores)
    upper, two_sided = aggregate_record_scores(target_scores, counts, reference)
    records = pd.concat([validation_rows, test_rows], ignore_index=True)[
        ["record_id", "split", "label", "state", "setup", "excitation"]
    ].copy()
    records["score_upper"] = upper
    records["score_two_sided"] = two_sided
    records["score"] = records[f"score_{score_tail}"]
    records["sequences"] = counts
    records["model"] = model_name
    records["seed"] = seed

    calibration = records[(records.split == "val") & (records.label == 0)]
    test_records = records[records.split == "test"].copy()
    threshold = threshold_from_healthy(calibration.score.to_numpy(), specificity_target)
    metric = {
        "model": model_name,
        "seed": seed,
        "score_tail": score_tail,
        **publication_metrics(test_records.label, test_records.score, threshold),
        "threshold": threshold,
        "calibration_specificity": float(np.mean(calibration.score < threshold)),
        "fit_healthy_records": len(fit_rows),
        "reference_healthy_records": len(internal_rows),
        "calibration_healthy_records": len(validation_rows),
        "test_healthy_records": int((test_records.label == 0).sum()),
        "test_damage_records": int((test_records.label == 1).sum()),
    }
    test_records["threshold"] = threshold
    return metric, test_records


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
    axis.set_title("Z24 global pooled healthy-only pilot")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "global_pooled_pilot.png", dpi=220, bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261113, 20261114, 20261115])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--representation", choices=("combined", "spectral_shape"), default="combined")
    parser.add_argument("--segment-length", type=int, default=1024)
    parser.add_argument("--spectral-bins", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-tail", choices=("upper", "two_sided"), default="two_sided")
    parser.add_argument("--specificity-target", type=float, default=.90)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    train = manifest[(manifest.split == "train") & (manifest.label == 0)].copy()
    validation = manifest[(manifest.split == "val") & (manifest.label == 0)].copy()
    test = manifest[manifest.split == "test"].copy()
    fit_rows, internal_rows = fixed_internal_split(train)

    needed = pd.concat([train, validation, test], ignore_index=True)
    cache = {}
    for index, row in enumerate(needed.itertuples(), 1):
        cache[str(row.record_id)] = make_record_sequences(
            row, args.segment_length, args.spectral_bins, args.representation,
            args.sequence_length, args.sequence_stride, args.fs,
        )
        if index % 20 == 0 or index == len(needed):
            print(f"features={index}/{len(needed)}", flush=True)

    example = next(iter(cache.values()))
    protocol = {
        "dataset": "Z24",
        "task": "global pooled healthy-only unseen-damage detection",
        "conditioning": "global",
        "representation": args.representation,
        "sensor_handling": "sensor-order-invariant median/IQR spectral shape; amplitude quantiles only in combined",
        "damage_used_for_training_or_calibration": False,
        "fit_healthy_records": [str(value) for value in fit_rows.record_id],
        "score_reference_healthy_records": [str(value) for value in internal_rows.record_id],
        "calibration_healthy_records": [str(value) for value in validation.record_id],
        "locked_test_healthy_records": int(((test.label == 0)).sum()),
        "locked_test_damage_records": int(((test.label == 1)).sum()),
        "segment_length": args.segment_length,
        "spectral_bins": args.spectral_bins,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "model_input_shape": list(example.shape[1:]),
        "sequences_per_record": int(len(example)),
        "score_tail": args.score_tail,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "important_limitation": "Only six healthy calibration and six healthy test records.",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    for model_name in args.models:
        for seed in args.seeds:
            run_dir = args.output_dir / "runs" / f"{model_name}__{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "metrics.json").exists():
                print(f"SKIP complete model={model_name} seed={seed}", flush=True)
                continue
            print(f"\nmodel={model_name} seed={seed}", flush=True)
            started = time.time()
            metric, scores = evaluate_run(
                model_name, seed, fit_rows, internal_rows, validation, test,
                cache, args.epochs, args.batch_size, args.specificity_target,
                args.score_tail, device,
            )
            metric["seconds"] = time.time() - started
            scores.to_csv(run_dir / "scores.csv", index=False)
            (run_dir / "metrics.json").write_text(
                json.dumps(metric, indent=2), encoding="utf-8"
            )
            print(json.dumps(metric, indent=2), flush=True)
            refresh(args.output_dir)
    refresh(args.output_dir)
    print(pd.read_csv(args.output_dir / "aggregate.csv").to_string(index=False))
    print(f"Z24_GLOBAL_POOLED_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

