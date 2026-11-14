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
from sklearn.metrics import roc_auc_score

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


def aggregate_record_scores(
    scores: np.ndarray,
    counts: list[int],
    reference: tuple[float, float],
) -> tuple[list[float], list[float]]:
    """Return upper-tail and two-sided record scores.

    The healthy-only reference is fitted before this function is called.  The
    two-sided score therefore detects either unusually large or unusually
    small model errors without learning a direction from damage labels.
    """
    values = np.log1p(np.maximum(np.asarray(scores, float), 0.0))
    values = (values - reference[0]) / reference[1]
    upper, two_sided, start = [], [], 0
    for count in counts:
        current = values[start:start + count]
        upper.append(float(np.median(current)))
        two_sided.append(float(np.median(np.abs(current))))
        start += count
    if start != len(values):
        raise RuntimeError("Window-score aggregation length mismatch")
    return upper, two_sided


def threshold_from_labeled_validation(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Select a record threshold using validation only.

    Balanced accuracy is primary; F1 and specificity are deterministic
    tie-breakers. Test scores never enter this function.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("Validation must contain healthy and damage records")
    unique = np.unique(scores)
    epsilon = max(float(np.ptp(unique)), 1.0) * 1e-9
    candidates = np.concatenate([
        [unique[0] - epsilon],
        (unique[:-1] + unique[1:]) / 2.0,
        [unique[-1] + epsilon],
    ])
    best = None
    for threshold in candidates:
        metric = publication_metrics(labels, scores, float(threshold))
        key = (metric["ba"], metric["f1"], metric["specificity"])
        if best is None or key > best[0]:
            best = (key, float(threshold), metric)
    return best[1], best[2]


def group_file(run_dir: Path, setup: int, excitation: str) -> Path:
    return run_dir / "groups" / f"setup{setup:02d}_{excitation}.csv"


def run_group(model_name: str, seed: int, group_index: int, group: pd.DataFrame,
              labels: list[str], raw_length: int, stride: int,
              representation: str, downsample: int, frequency_bins: int, epochs: int,
              batch_size: int, score_tail: str, device, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    train = group[(group.split == "train") & (group.label == 0)].copy()
    targets = group[group.split.isin(["val", "test"])].copy()
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
    upper_scores, two_sided_scores = aggregate_record_scores(target_scores, counts, reference)
    result = targets[["record_id", "split", "label", "state", "setup", "excitation"]].copy()
    result["score_upper"] = upper_scores
    result["score_lower"] = -np.asarray(upper_scores, dtype=float)
    result["score_two_sided"] = two_sided_scores
    result["score"] = result[
        "score_upper" if score_tail == "auto_validation" else f"score_{score_tail}"
    ]
    result["score_tail"] = score_tail
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
    if frame.record_id.duplicated().any() or len(frame) != 264:
        raise RuntimeError(f"Expected 264 unique validation/test records, got {len(frame)}")
    return frame


def finish_run(model_name: str, seed: int, run_dir: Path, frame: pd.DataFrame,
               specificity_target: float) -> None:
    validation = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    if (
        len(validation) != 132
        or int((validation.label == 0).sum()) != 6
        or int((validation.label == 1).sum()) != 126
        or len(test) != 132
    ):
        raise RuntimeError("Locked record counts changed")
    requested_tail = str(frame.score_tail.iloc[0])
    candidate_tails = (
        ("upper", "lower", "two_sided")
        if requested_tail == "auto_validation"
        else (requested_tail,)
    )
    selected = None
    for tail in candidate_tails:
        values = validation[f"score_{tail}"].to_numpy()
        candidate_threshold, candidate_metric = threshold_from_labeled_validation(
            validation.label.to_numpy(), values
        )
        key = (
            candidate_metric["ba"], candidate_metric["auc"],
            candidate_metric["f1"], candidate_metric["specificity"],
        )
        if selected is None or key > selected[0]:
            selected = (key, tail, candidate_threshold, candidate_metric)
    _, selected_tail, threshold, validation_metric = selected
    validation["score"] = validation[f"score_{selected_tail}"]
    test["score"] = test[f"score_{selected_tail}"]
    metric = {
        "model": model_name, "seed": seed,
        "score_tail_requested": requested_tail,
        "score_tail_selected": selected_tail,
        **publication_metrics(test.label, test.score, threshold),
        "threshold": threshold,
        **{
            f"validation_{name}": value
            for name, value in validation_metric.items()
            if name in {
                "auc", "ap", "ba", "recall", "fpr", "precision", "f1",
                "specificity",
            }
        },
        "validation_records": len(validation),
        "validation_healthy_records": int((validation.label == 0).sum()),
        "validation_damage_records": int((validation.label == 1).sum()),
        "records": len(test), "healthy_records": int((test.label == 0).sum()),
        "damage_records": int((test.label == 1).sum()),
    }
    validation["model"] = model_name
    validation["seed"] = seed
    validation["threshold"] = threshold
    validation.to_csv(run_dir / "validation_scores.csv", index=False)
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
    diagnostic_rows = []
    for (model, seed), group in scores.groupby(["model", "seed"], sort=False):
        diagnostic_rows.append({
            "model": model,
            "seed": seed,
            "signed_upper_auc": float(roc_auc_score(group.label, group.score_upper)),
            "signed_lower_auc": float(roc_auc_score(group.label, group.score_lower)),
            "two_sided_auc": float(roc_auc_score(group.label, group.score_two_sided)),
            "healthy_upper_median": float(group.loc[group.label == 0, "score_upper"].median()),
            "damage_upper_median": float(group.loc[group.label == 1, "score_upper"].median()),
            "healthy_two_sided_median": float(group.loc[group.label == 0, "score_two_sided"].median()),
            "damage_two_sided_median": float(group.loc[group.label == 1, "score_two_sided"].median()),
            "note": "descriptive locked-test diagnostic; never used for model or tail selection",
        })
    pd.DataFrame(diagnostic_rows).to_csv(output / "score_orientation_diagnostics.csv", index=False)
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
    parser.add_argument(
        "--score-tail", choices=("upper", "lower", "two_sided", "auto_validation"),
        default="upper",
        help=("upper is conventional high-error detection; lower detects "
              "unusually low reconstruction error; two_sided uses absolute "
              "healthy-standardized deviation; auto_validation selects among "
              "the three using validation only"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "runs").mkdir(exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_manifest(args.input_root)
    sensors = stable_sensor_map(frame)
    used = frame[frame.split.isin(["train", "val", "test"])].copy()
    print(f"device={device} cells={len(sensors)} models={args.models} seeds={args.seeds}", flush=True)

    protocol = {
        "dataset": "Z24",
        "task": "one-class detector training with labeled validation selection",
        "split_unit": "complete MAT record", "train_healthy_records": 24,
        "validation_healthy_records": 6, "validation_damage_records": 126,
        "locked_test_healthy_records": 6,
        "locked_test_damage_records": 126,
        "validation_damage_used_for_threshold_selection": True,
        "validation_damage_used_for_model_fit_or_normalization": False,
        "test_used_for_selection": False,
        "conditioning": "18 independent setup x excitation models per method and seed",
        "sensor_selection": "physical-label intersection within each setup/excitation cell; metadata only",
        "raw_mat_shape": "[N_record, 27]; N_record varies slightly by acquisition",
        "selected_record_shape": "[C_cell, N_record], C_cell listed in stable_sensor_counts",
        "canonical_window_pipeline": (
            "[C_cell,1024] -> per-channel linear detrend -> polyphase "
            "anti-aliased downsample by 2 -> train-only per-channel z-score"
        ),
        "canonical_model_input_shape": "[batch, C_cell, 512]",
        "model_adapters": {
            "pca_spe": "flatten to [batch, C_cell*512]",
            "pca_pyod": "flatten to [batch, C_cell*512]",
            "lstm_ae": "transpose to [batch, 512, C_cell]",
            "usad": "flatten to [batch, C_cell*512]",
            "gdn": "keep [batch, C_cell, 512]; sensors are nodes",
            "tranad": "adapter uses temporal sequence with C_cell features",
            "timesnet": "transpose to [batch, 512, C_cell]",
            "moderntcn": "canonical channel-first tensor",
            "scatterad": "canonical channel-first tensor",
        },
        "representation": args.representation,
        "raw_window": args.raw_window, "stride": args.window_stride,
        "downsample": args.downsample if args.representation == "raw_temporal" else None,
        "frequency_range_hz": [0.5, 30.0] if args.representation == "spectral" else None,
        "model_input_length": (
            args.raw_window // args.downsample
            if args.representation == "raw_temporal" else args.frequency_bins
        ),
        "score_tail": args.score_tail,
        "score_tail_selection": (
            "validation BA, then validation AUC/F1/specificity tie-breakers"
            if args.score_tail == "auto_validation" else "fixed before training"
        ),
        "window_aggregation": (
            "validation-selected upper, lower, or two-sided healthy-standardized score"
            if args.score_tail == "auto_validation"
            else "median absolute healthy-standardized log1p anomaly score"
            if args.score_tail == "two_sided"
            else "median signed healthy-standardized log1p anomaly score"
        ),
        "threshold": (
            "global labeled-validation record threshold maximizing balanced "
            "accuracy; F1 then specificity tie-breakers"
        ),
        "point_adjustment": False, "seeds": args.seeds, "epochs": args.epochs,
        "deterministic_model_runs": {
            "pca_spe": 1 if "pca_spe" in args.models else 0,
            "pca_pyod": 1 if "pca_pyod" in args.models else 0,
        },
        "stochastic_model_runs": len(args.seeds),
        "stable_sensor_counts": {f"setup{key[0]:02d}_{key[1]}": len(value) for key, value in sensors.items()},
        "important_limitation": "Only six independent healthy validation and six healthy test records.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for model_name in args.models:
        model_seeds = (
            args.seeds[:1]
            if model_name in {"pca_spe", "pca_pyod"}
            else args.seeds
        )
        for seed in model_seeds:
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
                          args.batch_size, args.score_tail, device, path)
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

