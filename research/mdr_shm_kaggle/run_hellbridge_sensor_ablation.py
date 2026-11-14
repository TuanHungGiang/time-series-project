from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from run_hellbridge_b2_external import Calibrator, fit_fold, parse_name
from run_hellbridge_record_ablation import add_cross_regime_conformal
from run_z24_b2 import Model, features, learn_anchors, normalized_sensor_spectra


METHODS = ("global_psd_median", "sensor_median", "sensor_q75", "sensor_top3_mean")


def extract_sensor_spectra(group, axis: str, window_seconds: int):
    requested = [f"AG{index:02d}" for index in range(1, 19)]
    axis_name = axis.lower()
    missing_groups = [name for name in requested if name not in group["acceleration"]]
    if missing_groups:
        raise ValueError(f"Missing required accelerometers: {missing_groups}")
    sensor_names = [name for name in requested if axis_name in group["acceleration"][name]]
    if len(sensor_names) < 12:
        raise ValueError(f"Only {len(sensor_names)} accelerometers provide axis {axis}")
    values = np.stack([
        np.asarray(group["acceleration"][name][axis_name], dtype=np.float32)
        for name in sensor_names
    ])
    window = int(100 * window_seconds)
    spectra = []
    frequencies = None
    for start in range(0, values.shape[1] - window + 1, window):
        frequencies, psd = normalized_sensor_spectra(values[:, start : start + window])
        spectra.append(psd.astype(np.float32))
    if len(spectra) < 8:
        raise ValueError(f"Need at least eight non-overlapping windows, found {len(spectra)}")
    return sensor_names, frequencies, np.stack(spectra)


def common_anchors(fit_spectra: np.ndarray, frequencies: np.ndarray) -> list[float]:
    # One modal reference is shared across locations, but sensor responses remain separate.
    reference_windows = [np.median(window, axis=0) for window in fit_spectra]
    return learn_anchors(reference_windows, frequencies)


def fit_sensor_fold(train_spectra: np.ndarray, frequencies: np.ndarray) -> dict:
    split = max(4, int(np.floor(0.65 * len(train_spectra))))
    fit_spectra = train_spectra[:split]
    calibration_spectra = train_spectra[split:]
    anchors = common_anchors(fit_spectra, frequencies)
    sensor_count = train_spectra.shape[1]
    models = []
    calibrators = []
    for sensor in range(sensor_count):
        fit_matrix = np.stack([
            features(window[sensor], frequencies, anchors) for window in fit_spectra
        ])
        calibration_matrix = np.stack([
            features(window[sensor], frequencies, anchors) for window in calibration_spectra
        ])
        model = Model.fit(anchors, fit_matrix)
        calibrator = Calibrator(np.asarray([model.raw_score(item) for item in calibration_matrix]))
        models.append(model)
        calibrators.append(calibrator)
    return {
        "anchors": anchors,
        "models": models,
        "calibrators": calibrators,
        "fit_windows": len(fit_spectra),
        "calibration_windows": len(calibration_spectra),
    }


def sensor_window_scores(fold: dict, spectra: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    if spectra.shape[1] != len(fold["models"]):
        raise ValueError("Sensor count differs between healthy training and evaluated record")
    output = np.empty(spectra.shape[:2], dtype=float)
    for sensor, (model, calibrator) in enumerate(zip(fold["models"], fold["calibrators"])):
        matrix = np.stack([
            features(window[sensor], frequencies, fold["anchors"]) for window in spectra
        ])
        raw = np.asarray([model.raw_score(item) for item in matrix])
        output[:, sensor] = calibrator.transform(raw)
    return output


def aggregate_sensor_scores(matrix: np.ndarray, method: str) -> float:
    if method == "sensor_median":
        per_window = np.median(matrix, axis=1)
    elif method == "sensor_q75":
        per_window = np.quantile(matrix, 0.75, axis=1)
    elif method == "sensor_top3_mean":
        count = min(3, matrix.shape[1])
        per_window = np.mean(np.sort(matrix, axis=1)[:, -count:], axis=1)
    else:
        raise ValueError(method)
    # Temporal median was the winning temporal aggregation and is now frozen.
    return float(np.median(per_window))


def global_score(fold: dict, spectra: np.ndarray, frequencies: np.ndarray) -> float:
    vectors = np.median(spectra, axis=1)
    matrix = np.stack([features(item, frequencies, fold["anchors"]) for item in vectors])
    raw = np.asarray([fold["b2"].raw_score(item) for item in matrix])
    return float(np.median(fold["b2_calibration"].transform(raw)))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        labels = group.label.to_numpy(dtype=int)
        predictions = group.alarm.to_numpy(dtype=bool)
        rows.append({
            "method": method,
            "records": len(group),
            "healthy_records": int((labels == 0).sum()),
            "damage_records": int((labels == 1).sum()),
            "auroc": float(roc_auc_score(labels, group.score)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "specificity": float(recall_score(1 - labels, ~predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "alarm_rate": float(predictions.mean()),
        })
    return pd.DataFrame(rows).set_index("method").loc[list(METHODS)].reset_index()


def severity_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    states = sorted((x for x in frame.state.unique() if x != "UDS"), key=lambda x: int(x[2:]))
    for method, group in frame.groupby("method", sort=False):
        healthy = group[group.state == "UDS"]
        for state in states:
            damage = group[group.state == state]
            subset = pd.concat([healthy, damage])
            rows.append({
                "method": method,
                "state": state,
                "damage_records": len(damage),
                "auroc_vs_uds": float(roc_auc_score(subset.label, subset.score)),
                "damage_alarm_rate": float(damage.alarm.mean()),
                "median_damage_score": float(damage.score.median()),
            })
    return pd.DataFrame(rows)


def paired_bootstrap(frame: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    wide_score = frame.pivot(index="record_id", columns="method", values="score")
    wide_alarm = frame.pivot(index="record_id", columns="method", values="alarm")
    metadata = frame.drop_duplicates("record_id").set_index("record_id").loc[wide_score.index]
    labels = metadata.label.to_numpy(dtype=int)
    healthy = np.flatnonzero(labels == 0)
    damage = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    rows = []
    baseline_score = wide_score.global_psd_median.to_numpy(dtype=float)
    baseline_alarm = wide_alarm.global_psd_median.to_numpy(dtype=bool)
    for method in METHODS[1:]:
        candidate_score = wide_score[method].to_numpy(dtype=float)
        candidate_alarm = wide_alarm[method].to_numpy(dtype=bool)

        def difference(indices: np.ndarray) -> tuple[float, float]:
            y = labels[indices]
            auc = roc_auc_score(y, candidate_score[indices]) - roc_auc_score(y, baseline_score[indices])
            ba = balanced_accuracy_score(y, candidate_alarm[indices]) - balanced_accuracy_score(y, baseline_alarm[indices])
            return float(auc), float(ba)

        estimate = difference(np.arange(len(labels)))
        draws = np.empty((samples, 2), dtype=float)
        for index in range(samples):
            sampled = np.concatenate([
                rng.choice(healthy, len(healthy), replace=True),
                rng.choice(damage, len(damage), replace=True),
            ])
            draws[index] = difference(sampled)
        for column, metric in enumerate(("auroc", "balanced_accuracy")):
            rows.append({
                "candidate": method,
                "baseline": "global_psd_median",
                "metric": metric,
                "estimate_candidate_minus_baseline": estimate[column],
                "ci_lower": float(np.quantile(draws[:, column], 0.025)),
                "ci_upper": float(np.quantile(draws[:, column], 0.975)),
                "probability_candidate_better": float(np.mean(draws[:, column] > 0)),
                "bootstrap_samples": samples,
            })
    return pd.DataFrame(rows)


def localization_table(sensor_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(sensor_rows)
    rows = []
    for record_id, group in frame.groupby("record_id"):
        ordered = group.groupby("sensor", as_index=False).score.median().sort_values("score", ascending=False)
        top = ordered.head(3)
        positive = np.maximum(ordered.score.to_numpy(dtype=float), 0.0)
        top_positive = np.maximum(top.score.to_numpy(dtype=float), 0.0)
        rows.append({
            "record_id": record_id,
            "top_sensor_1": top.iloc[0].sensor,
            "top_sensor_2": top.iloc[1].sensor,
            "top_sensor_3": top.iloc[2].sensor,
            "top_sensor_1_score": float(top.iloc[0].score),
            "spatial_concentration_top3": float(top_positive.sum() / max(positive.sum(), 1e-8)),
        })
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#5B6C8F", "#4C956C", "#F2B134", "#D95D39"]
    axes[0].bar(metrics.method, metrics.auroc, color=colors)
    axes[1].bar(metrics.method, metrics.balanced_accuracy, color=colors)
    for axis, ylabel in zip(axes, ("AUROC", "Balanced accuracy")):
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Damage ranking")
    axes[1].set_title("Healthy-only conformal alarms")
    figure.suptitle("HellBridge spatial sensor-fusion ablation")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hellbridge_sensor_ablation"))
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    metadata = {}
    with h5py.File(args.h5, "r") as source:
        all_names = sorted(source.keys())
        for name in all_names:
            item = parse_name(name)
            if args.mode == "quick" and item["state"] not in ("UDS", "DS1"):
                continue
            metadata[name] = item
            cache[name] = extract_sensor_spectra(source[name], item["axis"], args.window_seconds)
            print(f"loaded {len(cache)}/{len(all_names)} {name}", flush=True)

    scores = defaultdict(lambda: defaultdict(list))
    fold_rows = []
    sensor_rows = []
    regimes = sorted({item["regime"] for item in metadata.values()})
    for regime in regimes:
        healthy = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] == "UDS")
        damage = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] != "UDS")
        if len(healthy) != 2:
            raise ValueError(f"Expected two UDS recordings in {regime}, found {len(healthy)}")
        for fold_index, heldout in enumerate(healthy):
            train_name = healthy[1 - fold_index]
            train_sensors, frequencies, train_spectra = cache[train_name]
            sensor_fold = fit_sensor_fold(train_spectra, frequencies)
            global_fold = fit_fold(np.median(train_spectra, axis=1), frequencies)
            for name in [heldout] + damage:
                sensor_names, record_frequencies, record_spectra = cache[name]
                if sensor_names != train_sensors:
                    raise ValueError(f"Sensor layout changed inside regime {regime}: {name}")
                matrix = sensor_window_scores(sensor_fold, record_spectra, record_frequencies)
                scores[name]["global_psd_median"].append(global_score(global_fold, record_spectra, record_frequencies))
                for method in METHODS[1:]:
                    scores[name][method].append(aggregate_sensor_scores(matrix, method))
                for window_index in range(matrix.shape[0]):
                    for sensor_index, sensor in enumerate(sensor_names):
                        sensor_rows.append({
                            "record_id": name,
                            "regime": regime,
                            "fold": fold_index,
                            "window": window_index,
                            "sensor": sensor,
                            "score": float(matrix[window_index, sensor_index]),
                        })
            fold_rows.append({
                "regime": regime,
                "fold": fold_index,
                "train_uds": train_name,
                "heldout_uds": heldout,
                "sensors": len(train_sensors),
                "fit_windows": sensor_fold["fit_windows"],
                "calibration_windows": sensor_fold["calibration_windows"],
            })
        print(f"scored regime={regime}", flush=True)

    rows = []
    for name, item in sorted(metadata.items()):
        for method in METHODS:
            rows.append({
                **item,
                "model": "b2",
                "aggregation": method,
                "method": method,
                "score": float(np.mean(scores[name][method])),
                "fold_scores": len(scores[name][method]),
            })
    records = add_cross_regime_conformal(pd.DataFrame(rows), args.alpha).drop(columns=["model", "aggregation"])
    metrics = summarize(records)
    severity = severity_table(records)
    bootstrap = paired_bootstrap(records, args.bootstrap_samples, 20261112)
    localization = localization_table(sensor_rows)

    records.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "sensor_ablation_metrics.csv", index=False)
    severity.to_csv(args.output_dir / "severity_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    localization.to_csv(args.output_dir / "sensor_localization.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    pd.DataFrame(sensor_rows).to_csv(args.output_dir / "sensor_window_scores.csv", index=False)
    make_figure(metrics, args.output_dir / "fig_sensor_ablation.png")

    protocol = {
        "experiment": "Step 1 - spatial sensor-fusion ablation",
        "frozen_components": [
            "Z24-compatible detrend/Welch/0.5-30Hz preprocessing",
            "four healthy modal anchors",
            "temporal median",
            "leave-target-regime-out healthy-only conformal calibration",
            f"alpha={args.alpha}",
        ],
        "changed_component": "spatial evidence fusion only",
        "methods": list(METHODS),
        "predeclared_candidate": "sensor_q75",
        "damage_labels_used_for_fit_model_selection_or_calibration": False,
        "status": "exploratory HellBridge development; retain all variants and bootstrap comparisons",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(f"HELLBRIDGE_SENSOR_ABLATION_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

