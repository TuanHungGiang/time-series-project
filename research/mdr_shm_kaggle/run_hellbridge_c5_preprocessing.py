from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_toeplitz
from scipy.signal import detrend
from scipy.stats import kurtosis
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score


FS = 100.0
DEFAULT_WINDOW = 1024  # 10.24 s: the test duration used in the HBTA reference study.
BANDS = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0))
NAME_PATTERN = re.compile(r"MVS_(P[12])_(UDS|DS\d+)_(SM|NM)_([YZ])_(\d+)")
METHODS = (
    "ag_axis_ar",
    "ag_biaxial_ar",
    "all_accel_ar",
    "all_accel_spectral",
    "all_accel_fusion",
)
BASELINE = "ag_axis_ar"
CANDIDATE = "all_accel_fusion"


def parse_name(name: str) -> dict[str, object]:
    match = NAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected HellBridge recording name: {name}")
    position, state, excitation, shaker_axis, repeat = match.groups()
    return {
        "record_id": name,
        "position": position,
        "state": state,
        "excitation": excitation,
        "shaker_axis": shaker_axis,
        "repeat": int(repeat),
        "regime": f"{position}_{excitation}_{shaker_axis}",
        "label": int(state != "UDS"),
    }


def read_channels(group: h5py.Group, shaker_axis: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    acceleration = group["acceleration"]
    ag = [f"AG{i:02d}" for i in range(1, 19)]
    al = [f"AL{i:02d}" for i in range(1, 41)]

    def stack(names: list[str], axis: str) -> np.ndarray:
        available = [name for name in names if name in acceleration and axis in acceleration[name]]
        if not available:
            raise ValueError(f"No channels found for axis={axis}, requested={names[:3]}...")
        arrays = [np.asarray(acceleration[name][axis], dtype=np.float64) for name in available]
        length = min(map(len, arrays))
        return np.stack([array[:length] for array in arrays])

    ag_y = stack(ag, "y")
    ag_z = stack(ag, "z")
    al_z = stack(al, "z")
    axis = shaker_axis.lower()
    if "AS" not in acceleration or axis not in acceleration["AS"]:
        raise ValueError(f"Missing shaker reference AS/{axis}")
    shaker = np.asarray(acceleration["AS"][axis], dtype=np.float64)
    length = min(ag_y.shape[1], ag_z.shape[1], al_z.shape[1], len(shaker))
    values = {
        "ag_y": ag_y[:, :length],
        "ag_z": ag_z[:, :length],
        "al_z": al_z[:, :length],
        "shaker": shaker[:length][None, :],
    }
    audit = {
        "ag_y_channels": int(ag_y.shape[0]),
        "ag_z_channels": int(ag_z.shape[0]),
        "al_z_channels": int(al_z.shape[0]),
        "shaker_channels": 1,
        "samples": int(length),
    }
    return values, audit


def segment(values: np.ndarray, window: int, max_windows: int | None) -> np.ndarray:
    count = values.shape[1] // window
    if max_windows is not None:
        count = min(count, max_windows)
    if count < 12:
        raise ValueError(f"Need >=12 windows, found {count} for shape={values.shape}")
    return values[:, : count * window].reshape(values.shape[0], count, window)


def ar_coefficients(standardized: np.ndarray, order: int) -> np.ndarray:
    sensors, windows, samples = standardized.shape
    result = np.zeros((sensors, windows, order), dtype=np.float64)
    correlations = np.empty((sensors, windows, order + 1), dtype=np.float64)
    correlations[:, :, 0] = np.mean(standardized**2, axis=2) + 1e-5
    for lag in range(1, order + 1):
        correlations[:, :, lag] = np.mean(
            standardized[:, :, :-lag] * standardized[:, :, lag:], axis=2
        )
    for sensor in range(sensors):
        for window in range(windows):
            values = correlations[sensor, window]
            try:
                result[sensor, window] = solve_toeplitz(
                    (values[:-1], values[:-1]), values[1:], check_finite=False
                )
            except np.linalg.LinAlgError:
                result[sensor, window] = 0.0
    return result


def aggregate_spatial(per_sensor: np.ndarray, blocks: int) -> np.ndarray:
    # Input: sensors x windows x features. Output retains coarse location blocks.
    outputs = []
    for indices in np.array_split(np.arange(per_sensor.shape[0]), blocks):
        block = per_sensor[indices]
        outputs.extend([np.median(block, axis=0), np.quantile(block, 0.90, axis=0)])
    return np.concatenate(outputs, axis=1)


def group_features(
    signals: np.ndarray,
    shaker: np.ndarray,
    window: int,
    order: int,
    blocks: int,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    x = segment(signals, window, max_windows)
    u = segment(shaker, window, max_windows)
    count = min(x.shape[1], u.shape[1])
    x, u = x[:, :count], u[:, :count]
    x = detrend(x, axis=2, type="linear")
    u = detrend(u, axis=2, type="linear")

    x_center = x - np.mean(x, axis=2, keepdims=True)
    x_scale = np.std(x_center, axis=2, keepdims=True)
    standardized = x_center / np.maximum(x_scale, 1e-10)
    ar = ar_coefficients(standardized, order)

    taper = np.hanning(window)[None, None, :]
    frequency = np.fft.rfftfreq(window, d=1.0 / FS)
    x_power = np.abs(np.fft.rfft(x * taper, axis=2)) ** 2
    u_power = np.abs(np.fft.rfft(u * taper, axis=2)) ** 2
    x_band, u_band = [], []
    for low, high in BANDS:
        keep = (frequency >= low) & (frequency < high)
        x_band.append(np.sum(x_power[:, :, keep], axis=2))
        u_band.append(np.sum(u_power[:, :, keep], axis=2))
    x_band = np.stack(x_band, axis=2)
    u_band = np.stack(u_band, axis=2)
    eps = np.finfo(float).tiny
    log_ratio = np.log10(np.maximum(x_band, eps)) - np.log10(np.maximum(u_band, eps))
    fractions = x_band / np.maximum(np.sum(x_band, axis=2, keepdims=True), eps)
    log_fraction = np.log10(np.maximum(fractions, eps))
    rms_ratio = (
        np.log10(np.maximum(np.sqrt(np.mean(x**2, axis=2)), eps))
        - np.log10(np.maximum(np.sqrt(np.mean(u**2, axis=2)), eps))
    )[:, :, None]
    shape = kurtosis(x, axis=2, fisher=False, bias=False)[:, :, None]
    spectral = np.concatenate([log_ratio, log_fraction, rms_ratio, shape], axis=2)
    return aggregate_spatial(ar, blocks), aggregate_spatial(spectral, blocks)


def extract_record_features(
    group: h5py.Group,
    metadata: dict[str, object],
    window: int,
    order: int,
    max_windows: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    channels, audit = read_channels(group, str(metadata["shaker_axis"]))
    shaker = channels["shaker"]
    ag_y_ar, ag_y_spectral = group_features(channels["ag_y"], shaker, window, order, 3, max_windows)
    ag_z_ar, ag_z_spectral = group_features(channels["ag_z"], shaker, window, order, 3, max_windows)
    al_z_ar, al_z_spectral = group_features(channels["al_z"], shaker, window, order, 4, max_windows)
    count = min(ag_y_ar.shape[0], ag_z_ar.shape[0], al_z_ar.shape[0])
    axis_ar = ag_y_ar if metadata["shaker_axis"] == "Y" else ag_z_ar
    methods = {
        "ag_axis_ar": axis_ar[:count],
        "ag_biaxial_ar": np.concatenate([ag_y_ar[:count], ag_z_ar[:count]], axis=1),
        "all_accel_ar": np.concatenate([ag_y_ar[:count], ag_z_ar[:count], al_z_ar[:count]], axis=1),
        "all_accel_spectral": np.concatenate(
            [ag_y_spectral[:count], ag_z_spectral[:count], al_z_spectral[:count]], axis=1
        ),
    }
    methods["all_accel_fusion"] = np.concatenate(
        [methods["all_accel_ar"], methods["all_accel_spectral"]], axis=1
    )
    audit.update({
        "record_id": metadata["record_id"],
        "regime": metadata["regime"],
        "state": metadata["state"],
        "windows": count,
        "window_samples": window,
        "window_seconds": window / FS,
        **{f"dimension_{name}": int(matrix.shape[1]) for name, matrix in methods.items()},
    })
    return methods, audit


@dataclass
class RobustDiagonal:
    center: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    top_fraction: float = 0.10

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "RobustDiagonal":
        center = np.median(matrix, axis=0)
        mad = 1.4826 * np.median(np.abs(matrix - center), axis=0)
        std = np.std(matrix, axis=0)
        scale = np.maximum(mad, 0.10 * std)
        finite = np.isfinite(center) & np.isfinite(scale)
        positive = scale > max(1e-8, float(np.nanmedian(scale[finite])) * 1e-4)
        keep = finite & positive
        if int(keep.sum()) < 3:
            raise RuntimeError(f"Only {int(keep.sum())} stable features remain")
        return cls(center=center, scale=np.maximum(scale, 1e-8), keep=keep)

    def score(self, matrix: np.ndarray) -> np.ndarray:
        z = np.abs((matrix[:, self.keep] - self.center[self.keep]) / self.scale[self.keep])
        z = np.clip(z, 0.0, 25.0)
        count = max(3, int(np.ceil(self.top_fraction * z.shape[1])))
        return np.mean(np.partition(z, -count, axis=1)[:, -count:], axis=1)


def split_fit_calibration(count: int, block: int = 5) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count)
    fit = ((indices // block) % 3) != 2
    calibration = ~fit
    if fit.sum() < 12 or calibration.sum() < 8:
        split = max(12, int(np.floor(0.65 * count)))
        fit = indices < split
        calibration = ~fit
    return fit, calibration


def bootstrap_statistic_threshold(
    calibration_scores: np.ndarray,
    record_windows: int,
    alpha: float,
    rng: np.random.Generator,
    draws: int = 2000,
    block_length: int = 4,
) -> float:
    values = np.asarray(calibration_scores, dtype=float)
    simulated = np.empty(draws, dtype=float)
    for draw in range(draws):
        parts = []
        while sum(len(part) for part in parts) < record_windows:
            start = int(rng.integers(0, len(values)))
            parts.append(values[(start + np.arange(block_length)) % len(values)])
        sample = np.concatenate(parts)[:record_windows]
        simulated[draw] = np.quantile(sample, 0.75)
    try:
        return float(np.quantile(simulated, 1.0 - alpha, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(simulated, 1.0 - alpha, interpolation="higher"))


def fit_fold(features: dict[str, np.ndarray], alpha: float, seed: int) -> dict[str, dict[str, object]]:
    fitted = {}
    for method_index, method in enumerate(METHODS):
        matrix = features[method]
        fit_mask, calibration_mask = split_fit_calibration(len(matrix))
        model = RobustDiagonal.fit(matrix[fit_mask])
        calibration_scores = model.score(matrix[calibration_mask])
        threshold = bootstrap_statistic_threshold(
            calibration_scores,
            len(matrix),
            alpha,
            np.random.default_rng(seed + 7919 * method_index),
        )
        fitted[method] = {
            "model": model,
            "threshold": max(threshold, 1e-8),
            "fit_windows": int(fit_mask.sum()),
            "calibration_windows": int(calibration_mask.sum()),
            "retained_features": int(model.keep.sum()),
        }
    return fitted


def score_record(fold: dict[str, dict[str, object]], features: dict[str, np.ndarray]) -> dict[str, float]:
    output = {}
    for method in METHODS:
        window_scores = fold[method]["model"].score(features[method])
        statistic = float(np.quantile(window_scores, 0.75))
        output[method] = statistic / float(fold[method]["threshold"])
    return output


def metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = scores > 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "alarm_rate": float(predictions.mean()),
    }


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in records.groupby("method", sort=False):
        rows.append({
            "method": method,
            "records": len(group),
            "healthy_records": int((group.label == 0).sum()),
            "damage_records": int((group.label == 1).sum()),
            **metric_dict(group.label.to_numpy(dtype=int), group.score.to_numpy(dtype=float)),
        })
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def subgroup_metrics(records: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for (method, value), group in records.groupby(["method", column], sort=False):
        if group.label.nunique() != 2:
            continue
        rows.append({
            "method": method,
            column: value,
            "records": len(group),
            **metric_dict(group.label.to_numpy(dtype=int), group.score.to_numpy(dtype=float)),
        })
    return pd.DataFrame(rows)


def state_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in records.groupby("method", sort=False):
        healthy = group[group.state == "UDS"]
        for state in sorted(set(group.state) - {"UDS"}, key=lambda x: int(x[2:])):
            damage = group[group.state == state]
            subset = pd.concat([healthy, damage], ignore_index=True)
            rows.append({
                "method": method,
                "state": state,
                "damage_records": len(damage),
                "auroc_vs_uds": float(roc_auc_score(subset.label, subset.score)),
                "damage_alarm_rate": float((damage.score > 1.0).mean()),
                "median_damage_score": float(damage.score.median()),
            })
    return pd.DataFrame(rows)


def paired_bootstrap(records: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    wide = records.pivot(index="record_id", columns="method", values="score")
    metadata = records.drop_duplicates("record_id").set_index("record_id").loc[wide.index]
    labels = metadata.label.to_numpy(dtype=int)
    healthy = np.flatnonzero(labels == 0)
    damage = np.flatnonzero(labels == 1)
    candidate = wide[CANDIDATE].to_numpy(dtype=float)
    baseline = wide[BASELINE].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    def difference(indices: np.ndarray) -> tuple[float, float]:
        y = labels[indices]
        auc = roc_auc_score(y, candidate[indices]) - roc_auc_score(y, baseline[indices])
        ba = balanced_accuracy_score(y, candidate[indices] > 1.0) - balanced_accuracy_score(
            y, baseline[indices] > 1.0
        )
        return float(auc), float(ba)

    estimate = difference(np.arange(len(labels)))
    draws = np.empty((samples, 2), dtype=float)
    for index in range(samples):
        selected = np.concatenate([
            rng.choice(healthy, len(healthy), replace=True),
            rng.choice(damage, len(damage), replace=True),
        ])
        draws[index] = difference(selected)
    rows = []
    for column, metric in enumerate(("auroc", "balanced_accuracy")):
        rows.append({
            "candidate": CANDIDATE,
            "baseline": BASELINE,
            "metric": metric,
            "estimate_candidate_minus_baseline": estimate[column],
            "ci_lower": float(np.quantile(draws[:, column], 0.025)),
            "ci_upper": float(np.quantile(draws[:, column], 0.975)),
            "probability_candidate_better": float(np.mean(draws[:, column] > 0)),
            "bootstrap_samples": samples,
        })
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(metrics))
    axes[0].bar(x, metrics.auroc, color="#2f6f9f")
    axes[1].bar(x, metrics.balanced_accuracy, color="#d47f35")
    for axis, title in zip(axes, ("AUROC", "Balanced accuracy at healthy-only threshold")):
        axis.set_xticks(x, metrics.method, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("HellBridge C5 preprocessing and channel ablation")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--window-samples", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--ar-order", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, object]] = {}
    cache: dict[str, dict[str, np.ndarray]] = {}
    audits = []
    with h5py.File(args.h5, "r") as source:
        names = [name for name in sorted(source.keys()) if NAME_PATTERN.fullmatch(name)]
        for index, name in enumerate(names):
            item = parse_name(name)
            if args.mode == "quick" and item["state"] not in ("UDS", "DS1", "DS8"):
                continue
            features, audit = extract_record_features(
                source[name], item, args.window_samples, args.ar_order, args.max_windows
            )
            metadata[name] = item
            cache[name] = features
            audits.append(audit)
            print(f"features {len(cache)}/{len(names)} {name}", flush=True)

    score_lists: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    folds = []
    for regime_index, regime in enumerate(sorted({str(item["regime"]) for item in metadata.values()})):
        healthy = sorted(
            name for name, item in metadata.items() if item["regime"] == regime and item["state"] == "UDS"
        )
        damage = sorted(
            name for name, item in metadata.items() if item["regime"] == regime and item["state"] != "UDS"
        )
        if len(healthy) != 2:
            raise ValueError(f"Expected two UDS records in {regime}, found {healthy}")
        for fold_index, heldout in enumerate(healthy):
            train_name = healthy[1 - fold_index]
            fold = fit_fold(cache[train_name], args.alpha, args.seed + 101 * regime_index + fold_index)
            for name in [heldout] + damage:
                values = score_record(fold, cache[name])
                for method, value in values.items():
                    score_lists[name][method].append(value)
            for method in METHODS:
                folds.append({
                    "regime": regime,
                    "fold": fold_index,
                    "method": method,
                    "train_uds": train_name,
                    "heldout_uds": heldout,
                    "fit_windows": fold[method]["fit_windows"],
                    "calibration_windows": fold[method]["calibration_windows"],
                    "retained_features": fold[method]["retained_features"],
                    "record_threshold": fold[method]["threshold"],
                })
        print(f"scored regime={regime}", flush=True)

    rows = []
    for name, item in sorted(metadata.items()):
        for method in METHODS:
            rows.append({
                **item,
                "method": method,
                "score": float(np.mean(score_lists[name][method])),
                "fold_scores": len(score_lists[name][method]),
            })
    records = pd.DataFrame(rows)
    metrics = summarize(records)
    regimes = subgroup_metrics(records, "regime")
    states = state_metrics(records)
    bootstrap = paired_bootstrap(records, args.bootstrap_samples, args.seed + 9001)

    records.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    regimes.to_csv(args.output_dir / "regime_metrics.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output_dir / "channel_audit.csv", index=False)
    pd.DataFrame(folds).to_csv(args.output_dir / "folds.csv", index=False)
    make_figure(metrics, args.output_dir / "fig_c5_preprocessing.png")

    protocol = {
        "experiment": "HellBridge C5 preprocessing and channel-aware diagnostic",
        "status": "exploratory development; HellBridge damage results were previously observed",
        "window_samples": args.window_samples,
        "window_seconds": args.window_samples / FS,
        "ar_order": args.ar_order,
        "channels": ["AG y", "AG z", "AL z", "AS shaker reference"],
        "features": ["spatial-block AR", "shaker-normalized band response", "spectral shape", "RMS ratio", "kurtosis"],
        "detector": "healthy-only robust diagonal top-tail distance",
        "record_statistic": "q75 window score",
        "calibration": "regime-matched moving-block pseudo-record bootstrap from training UDS only",
        "predeclared_baseline": BASELINE,
        "predeclared_candidate": CANDIDATE,
        "damage_labels_used_for_fit_or_calibration": False,
        "damage_labels_used_for_posthoc_evaluation": True,
        "alpha": args.alpha,
        "mode": args.mode,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(f"HELLBRIDGE_C5_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

