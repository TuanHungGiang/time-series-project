from __future__ import annotations

"""Setup-conditioned, healthy-only source baselines for the Z24 bridge.

Each method is fitted independently in each of the 18 setup/excitation cells.
Only physical sensor labels stable within a cell are retained. Window scores
are aggregated to one score per complete MAT record before evaluation.
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
from scipy.io import loadmat
from scipy.signal import detrend, resample_poly, welch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    MODELS, Normalizer, publication_metrics, run_model, seed_everything,
    threshold_from_healthy,
)


def load_manifest(root: Path) -> pd.DataFrame:
    path = root / "metadata" / "records_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["split"] = frame.split.astype(str).str.lower()
    frame["excitation"] = frame.excitation.astype(str).str.lower()
    frame["setup"] = frame.setup.astype(int)
    frame["label"] = frame.label.astype(int)
    frame["state"] = frame.state.astype(int)
    frame = frame[frame.split.isin(["train", "val", "test"])].copy()
    frame["path"] = [str(root / Path(value)) for value in frame.relative_path]
    counts = {
        "train_healthy": int(((frame.split == "train") & (frame.label == 0)).sum()),
        "calibration_healthy": int(((frame.split == "val") & (frame.label == 0)).sum()),
        "test_healthy": int(((frame.split == "test") & (frame.label == 0)).sum()),
        "test_damage": int(((frame.split == "test") & (frame.label == 1)).sum()),
    }
    if counts != {"train_healthy": 24, "calibration_healthy": 6, "test_healthy": 6, "test_damage": 126}:
        raise RuntimeError(f"Unexpected locked Z24 split: {counts}")
    if frame.loc[frame.split == "train", "label"].ne(0).any():
        raise RuntimeError("Damage leakage in training split")
    return frame.reset_index(drop=True)


def parse_labels(value: str) -> list[str]:
    return [item.strip() for item in str(value).split("|") if item.strip()]


def stable_sensor_map(frame: pd.DataFrame) -> dict[tuple[int, str], list[str]]:
    output = {}
    for key, group in frame.groupby(["setup", "excitation"], sort=True):
        sets = [set(parse_labels(value)) for value in group.vertical_labels]
        stable = sorted(set.intersection(*sets), key=lambda value: int(re.match(r"\d+", value).group()))
        if len(stable) < 15:
            raise RuntimeError(f"Too few stable sensors for {key}: {stable}")
        output[(int(key[0]), str(key[1]))] = stable
    expected = {(setup, excitation) for setup in range(1, 10) for excitation in ("avt", "fvt")}
    if set(output) != expected:
        raise RuntimeError(f"Missing setup/excitation cells: {expected.difference(output)}")
    return output


def read_stable_vertical(path: Path, labels: list[str]) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    data = np.asarray(mat["data"], dtype=np.float32)
    raw = np.asarray(mat["labelshulp"]).squeeze().tolist()
    if not isinstance(raw, list):
        raw = [raw]
    observed = [str(value).strip() for value in raw]
    lookup = {label: index for index, label in enumerate(observed)}
    missing = [label for label in labels if label not in lookup]
    if missing:
        raise RuntimeError(f"{path.name} misses stable labels {missing}")
    signals = data[:, [lookup[label] for label in labels]].T
    if signals.shape[0] != len(labels) or not np.isfinite(signals).all():
        raise RuntimeError(f"Invalid selected matrix {path}: {signals.shape}")
    return signals


def record_windows(signals: np.ndarray, raw_length: int, stride: int,
                   representation: str, downsample: int,
                   frequency_bins: int) -> np.ndarray:
    """Create raw-temporal primary inputs or spectral ablation inputs."""
    starts = list(range(0, signals.shape[1] - raw_length + 1, stride))
    if len(starts) < 16:
        raise RuntimeError(f"Too few complete windows: {signals.shape}")
    grid = np.linspace(0.5, 30.0, frequency_bins)
    windows = []
    for start in starts:
        values = signals[:, start:start + raw_length]
        values = detrend(values, axis=1, type="linear")
        if representation == "raw_temporal":
            # Anti-aliasing is part of scipy's polyphase resampler. No test or
            # damage statistic is involved in this transformation.
            feature = resample_poly(values, up=1, down=downsample, axis=1)
            windows.append(feature.astype(np.float32))
            continue
        frequency, psd = welch(
            values, fs=100.0, window="hann", nperseg=min(1024, raw_length),
            noverlap=min(512, raw_length // 2), axis=1, detrend=False,
            scaling="density",
        )
        keep = (frequency >= 0.5) & (frequency <= 30.0)
        frequency, psd = frequency[keep], psd[:, keep]
        total = np.trapezoid(psd, frequency, axis=1)
        psd = psd / np.maximum(total[:, None], np.finfo(float).tiny)
        feature = np.stack([
            np.interp(grid, frequency, np.log10(sensor + 1e-18))
            for sensor in psd
        ])
        # Retain spectral shape while preventing a single broadband amplitude
        # offset from dominating every reconstruction baseline.
        feature -= feature.mean(axis=1, keepdims=True)
        windows.append(feature.astype(np.float32))
    return np.stack(windows)


def split_training_windows(windows_by_record: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    fit, internal = [], []
    for windows in windows_by_record:
        boundary = int(np.floor(0.75 * len(windows)))
        # One complete raw window is a guard band between fit and validation.
        fit.append(windows[:max(1, boundary - 1)])
        internal.append(windows[boundary:])
    return np.concatenate(fit), np.concatenate(internal)


def robust_score_reference(scores: np.ndarray) -> tuple[float, float]:
    values = np.log1p(np.maximum(np.asarray(scores, float), 0.0))
    center = float(np.median(values))
    scale = float(max(1.4826 * np.median(np.abs(values - center)), 0.25 * np.std(values), 1e-8))
    return center, scale


def aggregate_record_scores(scores: np.ndarray, counts: list[int], reference: tuple[float, float]) -> list[float]:
    values = np.log1p(np.maximum(np.asarray(scores, float), 0.0))
    values = (values - reference[0]) / reference[1]
    output, start = [], 0
    for count in counts:
        current = values[start:start + count]
        output.append(float(np.median(current)))
        start += count
    if start != len(values):
        raise RuntimeError("Window-score aggregation length mismatch")
    return output


def group_file(run_dir: Path, setup: int, excitation: str) -> Path:
    return run_dir / "groups" / f"setup{setup:02d}_{excitation}.csv"


def run_group(model_name: str, seed: int, group_index: int, group: pd.DataFrame,
              labels: list[str], raw_length: int, stride: int,
              representation: str, downsample: int, frequency_bins: int, epochs: int,
              batch_size: int, device, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    train = group[(group.split == "train") & (group.label == 0)].copy()
    targets = group[((group.split == "val") & (group.label == 0)) | (group.split == "test")].copy()
    if len(train) not in (1, 2):
        raise RuntimeError(f"Expected one or two healthy train records, got {len(train)}")

    train_windows = [record_windows(read_stable_vertical(Path(row.path), labels), raw_length, stride,
                                    representation, downsample, frequency_bins)
                     for row in train.itertuples()]
    fit, internal = split_training_windows(train_windows)
    target_windows, counts = [], []
    for row in targets.itertuples():
        current = record_windows(read_stable_vertical(Path(row.path), labels), raw_length, stride,
                                 representation, downsample, frequency_bins)
        target_windows.append(current); counts.append(len(current))
    target_matrix = np.concatenate(target_windows)

    normalizer = Normalizer.fit(fit)
    fit = normalizer.transform(fit)
    internal = normalizer.transform(internal)
    target_matrix = normalizer.transform(target_matrix)
    seed_everything(seed + group_index * 1009)
    internal_scores, target_scores = run_model(
        model_name, fit, internal, target_matrix, epochs, batch_size, device,
    )
    reference = robust_score_reference(internal_scores)
    record_scores = aggregate_record_scores(target_scores, counts, reference)
    result = targets[["record_id", "split", "label", "state", "setup", "excitation"]].copy()
    result["score"] = record_scores
    result["stable_sensors"] = len(labels)
    result["windows"] = counts
    result["score_reference_center"] = reference[0]
    result["score_reference_scale"] = reference[1]
    result.to_csv(output, index=False)


def collect_run(run_dir: Path, sensor_map: dict[tuple[int, str], list[str]]) -> pd.DataFrame | None:
    paths = [group_file(run_dir, setup, excitation)
             for setup, excitation in sorted(sensor_map)]
    if not all(path.exists() for path in paths):
        return None
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame.record_id.duplicated().any() or len(frame) != 138:
        raise RuntimeError(f"Expected 138 unique calibration/test records, got {len(frame)}")
    return frame


def finish_run(model_name: str, seed: int, run_dir: Path, frame: pd.DataFrame,
               specificity_target: float) -> None:
    calibration = frame[(frame.split == "val") & (frame.label == 0)]
    test = frame[frame.split == "test"].copy()
    if len(calibration) != 6 or len(test) != 132:
        raise RuntimeError("Locked record counts changed")
    threshold = threshold_from_healthy(calibration.score.to_numpy(), specificity_target)
    metric = {
        "model": model_name, "seed": seed,
        **publication_metrics(test.label, test.score, threshold),
        "threshold": threshold,
        "calibration_specificity": float(np.mean(calibration.score < threshold)),
        "records": len(test), "healthy_records": int((test.label == 0).sum()),
        "damage_records": int((test.label == 1).sum()),
    }
    test["model"] = model_name; test["seed"] = seed; test["threshold"] = threshold
    test.to_csv(run_dir / "scores.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")


def refresh(output: Path) -> None:
    metric_paths = sorted((output / "runs").glob("*/metrics.json"))
    if not metric_paths:
        return
    metrics = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in metric_paths])
    scores = pd.concat([pd.read_csv(path.parent / "scores.csv") for path in metric_paths], ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    aggregate = metrics.groupby("model", sort=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"),
        ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        specificity_mean=("specificity", "mean"), runs=("seed", "size"),
    ).reset_index()
    aggregate.to_csv(output / "aggregate.csv", index=False)
    chart = aggregate.set_index("model")[["auc_mean", "ap_mean", "ba_mean", "recall_mean", "precision_mean"]]
    axis = chart.plot.bar(figsize=(14, 6), ylim=(0, 1.03), rot=35, grid=True)
    axis.set_title("Z24 setup-conditioned healthy-only baselines")
    axis.figure.tight_layout(); axis.figure.savefig(output / "baseline_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20261113, 20261123)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--representation", choices=("raw_temporal", "spectral"), default="raw_temporal")
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--frequency-bins", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--specificity-target", type=float, default=0.90)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "runs").mkdir(exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_manifest(args.input_root)
    sensors = stable_sensor_map(frame)
    used = frame[(frame.split == "train") | ((frame.split == "val") & (frame.label == 0)) | (frame.split == "test")]
    print(f"device={device} cells={len(sensors)} models={args.models} seeds={args.seeds}", flush=True)

    protocol = {
        "dataset": "Z24", "task": "healthy-only unseen-damage detection",
        "split_unit": "complete MAT record", "train_healthy_records": 24,
        "calibration_healthy_records": 6, "locked_test_healthy_records": 6,
        "locked_test_damage_records": 126,
        "validation_damage_used": False, "test_used_for_selection": False,
        "conditioning": "18 independent setup x excitation models per method and seed",
        "sensor_selection": "physical-label intersection within each setup/excitation cell; metadata only",
        "representation": args.representation,
        "raw_window": args.raw_window, "stride": args.window_stride,
        "downsample": args.downsample if args.representation == "raw_temporal" else None,
        "frequency_range_hz": [0.5, 30.0] if args.representation == "spectral" else None,
        "model_input_length": (
            args.raw_window // args.downsample
            if args.representation == "raw_temporal" else args.frequency_bins
        ),
        "window_aggregation": "median standardized log1p anomaly score",
        "threshold": f"six healthy validation record scores at specificity target {args.specificity_target}",
        "point_adjustment": False, "seeds": args.seeds, "epochs": args.epochs,
        "stable_sensor_counts": {f"setup{key[0]:02d}_{key[1]}": len(value) for key, value in sensors.items()},
        "important_limitation": "Only six independent healthy calibration and six healthy test records.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for model_name in args.models:
        for seed in args.seeds:
            run_dir = args.output_dir / "runs" / f"{model_name}__{seed}"
            if (run_dir / "metrics.json").exists():
                print(f"SKIP complete model={model_name} seed={seed}", flush=True); continue
            started = time.time()
            for group_index, (key, labels) in enumerate(sorted(sensors.items())):
                path = group_file(run_dir, key[0], key[1])
                if path.exists():
                    print(f"SKIP group model={model_name} seed={seed} cell={key}", flush=True); continue
                print(f"\nmodel={model_name} seed={seed} cell={key} sensors={len(labels)}", flush=True)
                group = used[(used.setup == key[0]) & (used.excitation == key[1])]
                run_group(model_name, seed, group_index, group, labels, args.raw_window,
                          args.window_stride, args.representation, args.downsample,
                          args.frequency_bins, args.epochs,
                          args.batch_size, device, path)
            completed = collect_run(run_dir, sensors)
            if completed is None:
                raise RuntimeError(f"Incomplete groups for {model_name}/{seed}")
            finish_run(model_name, seed, run_dir, completed, args.specificity_target)
            metric = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metric["seconds"] = time.time() - started
            (run_dir / "metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
            print(json.dumps(metric, indent=2), flush=True); refresh(args.output_dir)
    refresh(args.output_dir)
    print(pd.read_csv(args.output_dir / "aggregate.csv").to_string(index=False))
    print(f"Z24_SOURCE_BASELINES_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

