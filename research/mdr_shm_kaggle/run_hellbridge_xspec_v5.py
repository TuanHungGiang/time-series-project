from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from multiview_v4_core import extract_views, learn_anchors, sensor_spectra
from run_hellbridge_multiview_v4 import parse_name, read_signals
from run_z24_b2 import Model, features
from xspec_v5_core import SubspaceDetector, calibrate, cross_spectral_features, normalize, robust_reference, standardize


def windows(signals: np.ndarray, seconds: int) -> list[np.ndarray]:
    width = 100 * seconds
    output = [signals[:, start : start + width] for start in range(0, signals.shape[1] - width + 1, width)]
    if len(output) < 8:
        raise ValueError(f"Need >=8 windows, found {len(output)}")
    return output


def metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prediction = scores > 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def fit_fold(items: list[np.ndarray], alpha: float, seed: int) -> dict:
    split = max(5, int(np.floor(0.65 * len(items))))
    fit_windows, calibration_windows = items[:split], items[split:]
    if len(calibration_windows) < 3:
        fit_windows, calibration_windows = items[:-3], items[-3:]

    spectra = [sensor_spectra(item) for item in fit_windows]
    frequency = spectra[0][0]
    anchors = learn_anchors([np.median(psd, axis=0) for _, psd in spectra], frequency)
    b2_matrix = np.stack([
        features(np.median(psd, axis=0), freq, anchors) for freq, psd in spectra
    ])
    b2 = Model.fit(anchors, b2_matrix)
    b2_reference = robust_reference(np.asarray([b2.raw_score(vector) for vector in b2_matrix]))

    x_matrix = np.stack([cross_spectral_features(item) for item in fit_windows])
    xspec = SubspaceDetector.fit(x_matrix)

    calibration_window_values = {"b2": [], "xspec": [], "fusion_80_20": []}
    for item in calibration_windows:
        freq, psd = sensor_spectra(item)
        b2_raw = b2.raw_score(features(np.median(psd, axis=0), freq, anchors))
        b2_z = standardize(b2_raw, b2_reference)
        x = xspec.components_score(cross_spectral_features(item))
        calibration_window_values["b2"].append(b2_z)
        calibration_window_values["xspec"].append(x["xspec_z"])
        calibration_window_values["fusion_80_20"].append(0.80 * b2_z + 0.20 * x["xspec_z"])

    rng = np.random.default_rng(seed)
    record_calibration = {}
    for model, values in calibration_window_values.items():
        values = np.asarray(values)
        pseudo_medians = np.median(rng.choice(values, (1000, len(items)), replace=True), axis=1)
        record_calibration[model] = calibrate(pseudo_medians, alpha)
    return {
        "anchors": anchors, "b2": b2, "b2_reference": b2_reference, "xspec": xspec,
        "calibration": record_calibration, "fit_windows": len(fit_windows),
        "calibration_windows": len(calibration_windows),
    }


def score_record(fold: dict, items: list[np.ndarray]) -> dict[str, float]:
    window_values = {"b2": [], "xspec": [], "fusion_80_20": []}
    diagnostics = {"b2_z": [], "xspec_z": [], "t2": [], "spe": []}
    for item in items:
        frequency, psd = sensor_spectra(item)
        b2_raw = fold["b2"].raw_score(features(np.median(psd, axis=0), frequency, fold["anchors"]))
        b2_z = standardize(b2_raw, fold["b2_reference"])
        x = fold["xspec"].components_score(cross_spectral_features(item))
        window_values["b2"].append(b2_z)
        window_values["xspec"].append(x["xspec_z"])
        window_values["fusion_80_20"].append(0.80 * b2_z + 0.20 * x["xspec_z"])
        diagnostics["b2_z"].append(b2_z)
        diagnostics["xspec_z"].append(x["xspec_z"])
        diagnostics["t2"].append(x["t2"])
        diagnostics["spe"].append(x["spe"])
    output = {
        model: normalize(float(np.median(values)), fold["calibration"][model])
        for model, values in window_values.items()
    }
    output.update({name: float(np.median(values)) for name, values in diagnostics.items()})
    output["domain_disagreement"] = abs(output["b2_z"] - output["xspec_z"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hellbridge_xspec_v5"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    healthy_windows = {}
    with h5py.File(args.h5, "r") as source:
        for name in sorted(source.keys()):
            item = parse_name(name)
            if args.mode == "quick" and item["state"] not in ("UDS", "DS1", "DS4"):
                continue
            metadata[name] = item
            if item["state"] == "UDS":
                healthy_windows[name] = windows(read_signals(source[name], str(item["axis"])), args.window_seconds)

    folds = {}
    fold_rows = []
    for regime_index, regime in enumerate(sorted({str(item["regime"]) for item in metadata.values()})):
        healthy = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] == "UDS")
        if len(healthy) != 2:
            raise ValueError(f"Expected two UDS in {regime}, found {healthy}")
        folds[regime] = []
        for index, train_name in enumerate(healthy):
            fold = fit_fold(healthy_windows[train_name], args.alpha, 20261112 + 17 * regime_index + index)
            fold["train_name"] = train_name
            fold["heldout_name"] = healthy[1 - index]
            folds[regime].append(fold)
            fold_rows.append({
                "regime": regime, "fold": index, "train_uds": train_name,
                "heldout_uds": healthy[1 - index], "fit_windows": fold["fit_windows"],
                "calibration_windows": fold["calibration_windows"],
            })
        print(f"fit regime={regime}", flush=True)

    rows = []
    with h5py.File(args.h5, "r") as source:
        for record_index, (name, item) in enumerate(sorted(metadata.items())):
            items = windows(read_signals(source[name], str(item["axis"])), args.window_seconds)
            relevant = folds[str(item["regime"])]
            if item["state"] == "UDS":
                relevant = [fold for fold in relevant if fold["heldout_name"] == name]
            values = [score_record(fold, items) for fold in relevant]
            common = {
                **item, "fold_scores": len(values),
                "b2_z": float(np.mean([value["b2_z"] for value in values])),
                "xspec_z": float(np.mean([value["xspec_z"] for value in values])),
                "t2": float(np.mean([value["t2"] for value in values])),
                "spe": float(np.mean([value["spe"] for value in values])),
                "domain_disagreement": float(np.mean([value["domain_disagreement"] for value in values])),
            }
            for model in ("b2", "xspec", "fusion_80_20"):
                value = float(np.mean([item_value[model] for item_value in values]))
                rows.append({**common, "model": model, "score": value, "prediction": value > 1.0})
            print(f"scored {record_index + 1}/{len(metadata)} {name}", flush=True)

    scores = pd.DataFrame(rows)
    metric_rows = []
    for model, subset in scores.groupby("model"):
        metric_rows.append({"model": model, "records": len(subset), **metric_dict(subset.label, subset.score)})
    metrics = pd.DataFrame(metric_rows)
    subgroup_rows = []
    for (model, regime), subset in scores.groupby(["model", "regime"]):
        subgroup_rows.append({
            "model": model, "group_type": "regime", "group": regime,
            **metric_dict(subset.label, subset.score),
        })
    # State-vs-UDS metrics are constructed explicitly to avoid record duplication.
    for model in scores.model.unique():
        model_scores = scores[scores.model == model]
        for state in sorted(set(model_scores.state) - {"UDS"}):
            subset = model_scores[model_scores.state.isin(["UDS", state])]
            subgroup_rows.append({
                "model": model, "group_type": "state", "group": state,
                **metric_dict(subset.label, subset.score),
            })
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "version": "xspec-v5", "dataset": "HellBridge", "mode": args.mode,
        "fit_labels": "UDS only", "damage_labels_used_for_selection": False,
        "fusion_weight": {"b2": 0.80, "xspec": 0.20},
        "protocol": "regime-matched leave-one-UDS-record-out",
        "record_calibration": "nested median pseudo-record bootstrap",
        "warning": "pseudo-records do not increase independent healthy record count",
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(pd.DataFrame(subgroup_rows).to_string(index=False))
    print(f"HELLBRIDGE_XSPEC_V5_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

