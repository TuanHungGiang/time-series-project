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

from run_hellbridge_b2_external import Calibrator, extract_window_spectra, parse_name
from run_hellbridge_record_ablation import add_cross_regime_conformal
from run_z24_b2 import BANDS, Model, features as full_features, learn_anchors


METHODS = (
    "full_b2_13d",
    "peak_frequency_4d",
    "grouped_modal_8d",
    "grouped_modal_band_13d",
    "band_energy_5d",
)


def peak_frequency_features(spectrum: np.ndarray, frequencies: np.ndarray, anchors: list[float]) -> np.ndarray:
    output = []
    for anchor in anchors:
        local = np.flatnonzero(np.abs(frequencies - anchor) <= 0.6)
        peak = local[int(np.argmax(spectrum[local]))]
        output.append(float(frequencies[peak]))
    return np.asarray(output)


def grouped_modal_features(spectrum: np.ndarray, frequencies: np.ndarray, anchors: list[float]) -> np.ndarray:
    output = []
    tiny = np.finfo(float).tiny
    for anchor in anchors:
        keep = np.abs(frequencies - anchor) <= 0.6
        local_f = frequencies[keep]
        local_p = spectrum[keep]
        area = float(np.trapezoid(local_p, local_f))
        # Spectral centroid is more stable than selecting one FFT/Welch bin.
        centroid = float(np.sum(local_f * local_p) / max(np.sum(local_p), tiny))
        output.extend([centroid, float(np.log10(area + 1e-16))])
    return np.asarray(output)


def band_energy_features(spectrum: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    output = []
    for low, high in BANDS:
        keep = (frequencies >= low) & (frequencies < high)
        output.append(float(np.log10(np.trapezoid(spectrum[keep], frequencies[keep]) + 1e-16)))
    return np.asarray(output)


def feature_vector(method: str, spectrum: np.ndarray, frequencies: np.ndarray, anchors: list[float]) -> np.ndarray:
    if method == "full_b2_13d":
        return full_features(spectrum, frequencies, anchors)
    if method == "peak_frequency_4d":
        return peak_frequency_features(spectrum, frequencies, anchors)
    if method == "grouped_modal_8d":
        return grouped_modal_features(spectrum, frequencies, anchors)
    if method == "grouped_modal_band_13d":
        return np.concatenate([
            grouped_modal_features(spectrum, frequencies, anchors),
            band_energy_features(spectrum, frequencies),
        ])
    if method == "band_energy_5d":
        return band_energy_features(spectrum, frequencies)
    raise ValueError(method)


def fit_fold(train_spectra: np.ndarray, frequencies: np.ndarray) -> dict:
    split = max(4, int(np.floor(0.65 * len(train_spectra))))
    fit_spectra = train_spectra[:split]
    calibration_spectra = train_spectra[split:]
    anchors = learn_anchors(list(fit_spectra), frequencies)
    fitted = {}
    for method in METHODS:
        fit_matrix = np.stack([
            feature_vector(method, item, frequencies, anchors) for item in fit_spectra
        ])
        calibration_matrix = np.stack([
            feature_vector(method, item, frequencies, anchors) for item in calibration_spectra
        ])
        model = Model.fit(anchors, fit_matrix)
        calibrator = Calibrator(np.asarray([model.raw_score(item) for item in calibration_matrix]))
        fitted[method] = {
            "model": model,
            "calibrator": calibrator,
            "dimension": fit_matrix.shape[1],
        }
    return {
        "anchors": anchors,
        "methods": fitted,
        "fit_windows": len(fit_spectra),
        "calibration_windows": len(calibration_spectra),
    }


def score_record(fold: dict, spectra: np.ndarray, frequencies: np.ndarray) -> dict[str, float]:
    output = {}
    for method in METHODS:
        matrix = np.stack([
            feature_vector(method, item, frequencies, fold["anchors"]) for item in spectra
        ])
        model = fold["methods"][method]["model"]
        calibrator = fold["methods"][method]["calibrator"]
        raw = np.asarray([model.raw_score(item) for item in matrix])
        output[method] = float(np.median(calibrator.transform(raw)))
    return output


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        labels = group.label.to_numpy(dtype=int)
        predictions = group.alarm.to_numpy(dtype=bool)
        rows.append({
            "method": method,
            "feature_dimension": int(group.feature_dimension.iloc[0]),
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


def regime_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, regime), group in frame.groupby(["method", "regime"], sort=False):
        if group.label.nunique() != 2:
            continue
        labels = group.label.to_numpy(dtype=int)
        predictions = group.alarm.to_numpy(dtype=bool)
        rows.append({
            "method": method,
            "regime": regime,
            "records": len(group),
            "auroc": float(roc_auc_score(labels, group.score)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "specificity": float(recall_score(1 - labels, ~predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
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
    baseline_score = wide_score.full_b2_13d.to_numpy(dtype=float)
    baseline_alarm = wide_alarm.full_b2_13d.to_numpy(dtype=bool)
    rows = []
    for method in METHODS[1:]:
        candidate_score = wide_score[method].to_numpy(dtype=float)
        candidate_alarm = wide_alarm[method].to_numpy(dtype=bool)

        def difference(indices: np.ndarray) -> tuple[float, float]:
            y = labels[indices]
            return (
                float(roc_auc_score(y, candidate_score[indices]) - roc_auc_score(y, baseline_score[indices])),
                float(balanced_accuracy_score(y, candidate_alarm[indices]) - balanced_accuracy_score(y, baseline_alarm[indices])),
            )

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
                "baseline": "full_b2_13d",
                "metric": metric,
                "estimate_candidate_minus_baseline": estimate[column],
                "ci_lower": float(np.quantile(draws[:, column], 0.025)),
                "ci_upper": float(np.quantile(draws[:, column], 0.975)),
                "probability_candidate_better": float(np.mean(draws[:, column] > 0)),
                "bootstrap_samples": samples,
            })
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, output: Path) -> None:
    labels = ["Full 13D", "Peak-f 4D", "Grouped 8D", "Grouped+band 13D", "Bands 5D"]
    colors = ["#5B6C8F", "#4C956C", "#F2B134", "#D95D39", "#8E6C88"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].bar(labels, metrics.auroc, color=colors)
    axes[1].bar(labels, metrics.balanced_accuracy, color=colors)
    for axis, ylabel in zip(axes, ("AUROC", "Balanced accuracy")):
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=28)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Damage ranking")
    axes[1].set_title("Healthy-only conformal alarms")
    figure.suptitle("HellBridge modal-focused feature ablation")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hellbridge_modal_ablation"))
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
            cache[name] = extract_window_spectra(source[name], item["axis"], args.window_seconds)
            print(f"loaded {len(cache)}/{len(all_names)} {name}", flush=True)

    scores = defaultdict(lambda: defaultdict(list))
    fold_rows = []
    regimes = sorted({item["regime"] for item in metadata.values()})
    for regime in regimes:
        healthy = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] == "UDS")
        damage = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] != "UDS")
        if len(healthy) != 2:
            raise ValueError(f"Expected two UDS recordings in {regime}, found {len(healthy)}")
        for fold_index, heldout in enumerate(healthy):
            train_name = healthy[1 - fold_index]
            frequencies, train_spectra = cache[train_name]
            fold = fit_fold(train_spectra, frequencies)
            for name in [heldout] + damage:
                record_frequencies, record_spectra = cache[name]
                values = score_record(fold, record_spectra, record_frequencies)
                for method, score in values.items():
                    scores[name][method].append(score)
            fold_rows.append({
                "regime": regime,
                "fold": fold_index,
                "train_uds": train_name,
                "heldout_uds": heldout,
                "fit_windows": fold["fit_windows"],
                "calibration_windows": fold["calibration_windows"],
                "anchors_hz": json.dumps(fold["anchors"]),
            })
        print(f"scored regime={regime}", flush=True)

    dimensions = {
        "full_b2_13d": 13,
        "peak_frequency_4d": 4,
        "grouped_modal_8d": 8,
        "grouped_modal_band_13d": 13,
        "band_energy_5d": 5,
    }
    rows = []
    for name, item in sorted(metadata.items()):
        for method in METHODS:
            rows.append({
                **item,
                "model": "b2",
                "aggregation": method,
                "method": method,
                "feature_dimension": dimensions[method],
                "score": float(np.mean(scores[name][method])),
                "fold_scores": len(scores[name][method]),
            })
    records = add_cross_regime_conformal(pd.DataFrame(rows), args.alpha).drop(columns=["model", "aggregation"])
    metrics = summarize(records)
    severity = severity_table(records)
    regimes_frame = regime_table(records)
    bootstrap = paired_bootstrap(records, args.bootstrap_samples, 20261112)

    records.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "modal_ablation_metrics.csv", index=False)
    severity.to_csv(args.output_dir / "severity_metrics.csv", index=False)
    regimes_frame.to_csv(args.output_dir / "regime_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds_and_anchors.csv", index=False)
    make_figure(metrics, args.output_dir / "fig_modal_ablation.png")

    protocol = {
        "experiment": "Step 2 - modal-focused representation ablation",
        "scientific_basis": [
            "Mahalanobis distance of modal frequencies with temporal false-alarm control",
            "grouped mode-focused frequency-domain healthy-only anomaly detection",
        ],
        "frozen_components": [
            "global median sensor PSD",
            "Z24-compatible detrend/Welch/0.5-30Hz preprocessing",
            "four healthy modal anchors",
            "temporal median",
            "leave-target-regime-out healthy-only conformal calibration",
            f"alpha={args.alpha}",
        ],
        "changed_component": "damage-sensitive feature representation only",
        "methods": list(METHODS),
        "predeclared_candidate": "grouped_modal_8d",
        "damage_labels_used_for_fit_model_selection_or_calibration": False,
        "status": "exploratory HellBridge development; retain all variants and paired bootstrap comparisons",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(f"HELLBRIDGE_MODAL_ABLATION_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

