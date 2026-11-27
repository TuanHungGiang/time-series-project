from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from run_z24_b2 import Model, features, learn_anchors, normalized_sensor_spectra


NAME_PATTERN = re.compile(r"MVS_(P[12])_(UDS|DS\d+)_(SM|NM)_([YZ])_(\d+)")


def parse_name(name: str) -> dict[str, str]:
    match = NAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected HellBridge recording name: {name}")
    position, state, excitation, axis, repeat = match.groups()
    return {
        "record_id": name,
        "position": position,
        "state": state,
        "excitation": excitation,
        "axis": axis,
        "repeat": repeat,
        "regime": f"{position}_{excitation}_{axis}",
        "label": int(state != "UDS"),
    }


def extract_window_spectra(group, axis: str, window_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    sensor_names = [f"AG{index:02d}" for index in range(1, 19)]
    axis_name = axis.lower()
    missing = [name for name in sensor_names if name not in group["acceleration"]]
    if missing:
        raise ValueError(f"Missing required accelerometers: {missing}")
    sensor_names = [
        name for name in sensor_names
        if axis_name in group["acceleration"][name]
    ]
    if len(sensor_names) < 12:
        raise ValueError(f"Only {len(sensor_names)} accelerometers provide axis {axis}")
    values = np.stack(
        [np.asarray(group["acceleration"][name][axis_name], dtype=np.float32) for name in sensor_names]
    )
    window = int(100 * window_seconds)
    spectra = []
    frequencies = None
    for start in range(0, values.shape[1] - window + 1, window):
        frequencies, sensor_psd = normalized_sensor_spectra(values[:, start : start + window])
        spectra.append(np.median(sensor_psd, axis=0))
    if len(spectra) < 8:
        raise ValueError(f"Need at least eight non-overlapping windows, found {len(spectra)}")
    return frequencies, np.stack(spectra)


class Calibrator:
    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=float)
        self.threshold = float(np.max(values))
        center = float(np.median(values))
        self.scale = float(max(1.4826 * np.median(np.abs(values - center)), np.std(values), 1e-6))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return 1.0 + (np.asarray(values) - self.threshold) / self.scale


def fit_fold(train_spectra: np.ndarray, frequencies: np.ndarray) -> dict:
    split = max(4, int(np.floor(0.65 * len(train_spectra))))
    fit_spectra = train_spectra[:split]
    calibration_spectra = train_spectra[split:]
    anchors = learn_anchors(list(fit_spectra), frequencies)
    fit_features = np.stack([features(item, frequencies, anchors) for item in fit_spectra])
    calibration_features = np.stack([features(item, frequencies, anchors) for item in calibration_spectra])

    b2 = Model.fit(anchors, fit_features)
    b2_calibration = Calibrator(np.asarray([b2.raw_score(item) for item in calibration_features]))

    scaler = RobustScaler().fit(fit_features)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10).fit(scaler.transform(fit_features))
    ocsvm_calibration = Calibrator(-ocsvm.decision_function(scaler.transform(calibration_features)))
    return {
        "anchors": anchors,
        "b2": b2,
        "b2_calibration": b2_calibration,
        "scaler": scaler,
        "ocsvm": ocsvm,
        "ocsvm_calibration": ocsvm_calibration,
        "fit_windows": len(fit_spectra),
        "calibration_windows": len(calibration_spectra),
    }


def score_record(fold: dict, spectra: np.ndarray, frequencies: np.ndarray) -> dict[str, float]:
    matrix = np.stack([features(item, frequencies, fold["anchors"]) for item in spectra])
    b2_raw = np.asarray([fold["b2"].raw_score(item) for item in matrix])
    ocsvm_raw = -fold["ocsvm"].decision_function(fold["scaler"].transform(matrix))
    return {
        "b2": float(np.median(fold["b2_calibration"].transform(b2_raw))),
        "ocsvm": float(np.median(fold["ocsvm_calibration"].transform(ocsvm_raw))),
    }


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prediction = scores > 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def paired_bootstrap(scores: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    labels = scores.label.to_numpy(dtype=int)
    healthy = np.flatnonzero(labels == 0)
    damage = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    rows = []
    for metric_name in ("auroc", "balanced_accuracy"):
        def evaluate(indices):
            y = labels[indices]
            if metric_name == "auroc":
                return roc_auc_score(y, scores.b2.to_numpy()[indices]) - roc_auc_score(y, scores.ocsvm.to_numpy()[indices])
            return balanced_accuracy_score(y, scores.b2.to_numpy()[indices] > 1.0) - balanced_accuracy_score(y, scores.ocsvm.to_numpy()[indices] > 1.0)
        estimate = float(evaluate(np.arange(len(scores))))
        draws = []
        for _ in range(samples):
            indices = np.concatenate([rng.choice(healthy, len(healthy), replace=True), rng.choice(damage, len(damage), replace=True)])
            draws.append(evaluate(indices))
        draws = np.asarray(draws)
        rows.append({
            "metric": metric_name,
            "estimate_b2_minus_ocsvm": estimate,
            "ci_lower": float(np.quantile(draws, 0.025)),
            "ci_upper": float(np.quantile(draws, 0.975)),
            "probability_b2_better": float(np.mean(draws > 0)),
            "bootstrap_samples": samples,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hellbridge_b2_external"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectra_cache = {}
    metadata = {}
    with h5py.File(args.h5, "r") as source:
        all_names = sorted(source.keys())
        for name in all_names:
            item = parse_name(name)
            if args.mode == "quick" and item["state"] not in ("UDS", "DS1"):
                continue
            metadata[name] = item
            spectra_cache[name] = extract_window_spectra(source[name], item["axis"], args.window_seconds)
            print(f"loaded {len(spectra_cache)}/{len(all_names)} {name}", flush=True)

    regimes = sorted({item["regime"] for item in metadata.values()})
    score_lists = defaultdict(lambda: defaultdict(list))
    fold_rows = []
    for regime_index, regime in enumerate(regimes):
        healthy = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] == "UDS")
        damage = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] != "UDS")
        if len(healthy) != 2:
            raise ValueError(f"Expected two UDS recordings in {regime}, found {len(healthy)}")
        for fold_index, heldout in enumerate(healthy):
            train_name = healthy[1 - fold_index]
            frequencies, train_spectra = spectra_cache[train_name]
            fold = fit_fold(train_spectra, frequencies)
            for name in [heldout] + damage:
                record_frequencies, record_spectra = spectra_cache[name]
                values = score_record(fold, record_spectra, record_frequencies)
                for model, value in values.items():
                    score_lists[name][model].append(value)
            fold_rows.append({
                "regime": regime,
                "fold": fold_index,
                "train_uds": train_name,
                "heldout_uds": heldout,
                "fit_windows": fold["fit_windows"],
                "calibration_windows": fold["calibration_windows"],
            })

    rows = []
    for name, item in sorted(metadata.items()):
        rows.append({
            **item,
            "b2": float(np.mean(score_lists[name]["b2"])),
            "ocsvm": float(np.mean(score_lists[name]["ocsvm"])),
            "fold_scores": len(score_lists[name]["b2"]),
        })
    scores = pd.DataFrame(rows)
    metric_rows = []
    for model in ("b2", "ocsvm"):
        metric_rows.append({"model": model, **metrics(scores.label.to_numpy(), scores[model].to_numpy())})
    metric_frame = pd.DataFrame(metric_rows)

    severity_rows = []
    for state in sorted(scores.state.unique(), key=lambda value: (value != "UDS", value)):
        if state == "UDS":
            continue
        subset = scores[scores.state.isin(["UDS", state])]
        for model in ("b2", "ocsvm"):
            severity_rows.append({
                "state": state,
                "model": model,
                "damage_records": int((subset.state == state).sum()),
                "auroc_vs_uds": float(roc_auc_score(subset.label, subset[model])),
                "alarm_rate": float((scores.loc[scores.state == state, model] > 1.0).mean()),
                "median_score": float(scores.loc[scores.state == state, model].median()),
            })
    severity = pd.DataFrame(severity_rows)
    bootstrap = paired_bootstrap(scores, args.bootstrap_samples, 20261111)

    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metric_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    severity.to_csv(args.output_dir / "severity_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    protocol = {
        "dataset": "Hell Bridge Test Arena",
        "mode": args.mode,
        "protocol": "regime-matched two-fold leave-one-UDS-record-out",
        "adaptation": "algorithm and hyperparameters frozen; healthy reference re-estimated per HellBridge regime",
        "damage_labels_used_for_fit_or_calibration": False,
        "channels": "AG01-AG18; recording-name Y/Z axis",
        "sample_rate_hz": 100,
        "window_seconds": args.window_seconds,
        "window_overlap": 0,
        "record_score": "median normalized window anomaly score",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for index, model in enumerate(("b2", "ocsvm")):
        order = ["UDS"] + [f"DS{i}" for i in range(1, 9) if f"DS{i}" in set(scores.state)]
        axes[index].boxplot([scores.loc[scores.state == state, model] for state in order], tick_labels=order, showfliers=False)
        axes[index].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[index].set_title(model.upper())
        axes[index].set_xlabel("Structural state")
        axes[index].set_ylabel("Normalized anomaly score")
        axes[index].tick_params(axis="x", rotation=35)
        axes[index].spines[["top", "right"]].set_visible(False)
    figure.suptitle("Healthy-adapted external validation on HellBridge")
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig1_hellbridge_scores.png", dpi=220)
    plt.close(figure)

    print(metric_frame.to_string(index=False))
    print(severity.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(f"HELLBRIDGE_EXTERNAL_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

