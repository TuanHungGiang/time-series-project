from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from run_z24_b2 import (
    Model, build_file_index, excitation, features, find_manifest, learn_anchors,
    load_vertical, record_spectrum, resolve_source,
)
from xspec_v5_core import SubspaceDetector, calibrate, cross_spectral_features, normalize, robust_reference, standardize


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prediction = scores > 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def protocol(root: Path) -> tuple[pd.DataFrame, Path]:
    manifest = find_manifest(root)
    frame = pd.read_csv(manifest)
    frame["label"] = frame.label.astype(int)
    if "role" not in frame:
        frame["role"] = frame["split"]
    frame["role"] = frame.role.astype(str).str.lower().replace({"val": "calibration"})
    frame.loc[frame.label == 1, "role"] = "test"
    frame = frame[frame.role.isin(["train", "calibration", "test"])].copy()
    if frame.loc[frame.role.isin(["train", "calibration"]), "label"].ne(0).any():
        raise RuntimeError("Leakage: damage appears in fit/calibration")
    dataset_root = manifest.parent.parent
    index = build_file_index(dataset_root)
    frame["resolved_path"] = [str(resolve_source(row, index, dataset_root)) for _, row in frame.iterrows()]
    frame["excitation_resolved"] = [excitation(row) for _, row in frame.iterrows()]
    return frame, manifest


def fit_group(group: pd.DataFrame, signals: dict[str, np.ndarray], alpha: float) -> dict:
    train = group[group.role == "train"]
    calibration = group[group.role == "calibration"]
    if len(train) < 8 or calibration.empty:
        raise RuntimeError(f"Insufficient healthy records: train={len(train)}, calibration={len(calibration)}")
    spectra = [record_spectrum(signals[row.record_id]) for row in train.itertuples()]
    frequency = spectra[0][0]
    anchors = learn_anchors([spectrum for _, spectrum in spectra], frequency)
    b2_matrix = np.stack([features(spectrum, freq, anchors) for freq, spectrum in spectra])
    b2 = Model.fit(anchors, b2_matrix)
    b2_train_raw = np.asarray([b2.raw_score(vector) for vector in b2_matrix])
    b2_reference = robust_reference(b2_train_raw)

    x_matrix = np.stack([cross_spectral_features(signals[row.record_id]) for row in train.itertuples()])
    dimensions = {len(vector) for vector in x_matrix}
    if len(dimensions) != 1:
        raise RuntimeError(f"Inconsistent cross-spectral dimensions: {dimensions}")
    xspec = SubspaceDetector.fit(x_matrix)

    calibration_values = {"b2": [], "xspec": [], "fusion": []}
    for row in calibration.itertuples():
        freq, spectrum = record_spectrum(signals[row.record_id])
        b2_raw = b2.raw_score(features(spectrum, freq, anchors))
        b2_z = standardize(b2_raw, b2_reference)
        x = xspec.components_score(cross_spectral_features(signals[row.record_id]))
        calibration_values["b2"].append(b2_z)
        calibration_values["xspec"].append(x["xspec_z"])
        calibration_values["fusion"].append(0.80 * b2_z + 0.20 * x["xspec_z"])
    return {
        "anchors": anchors, "b2": b2, "b2_reference": b2_reference, "xspec": xspec,
        "calibration": {name: calibrate(np.asarray(values), alpha) for name, values in calibration_values.items()},
        "healthy_calibration_records": len(calibration),
    }


def score(group: dict, signals: np.ndarray) -> dict[str, float]:
    frequency, spectrum = record_spectrum(signals)
    b2_raw = group["b2"].raw_score(features(spectrum, frequency, group["anchors"]))
    b2_z = standardize(b2_raw, group["b2_reference"])
    x = group["xspec"].components_score(cross_spectral_features(signals))
    fusion_raw = 0.80 * b2_z + 0.20 * x["xspec_z"]
    return {
        "b2": normalize(b2_z, group["calibration"]["b2"]),
        "xspec": normalize(x["xspec_z"], group["calibration"]["xspec"]),
        "fusion_80_20": normalize(fusion_raw, group["calibration"]["fusion"]),
        "b2_z": b2_z, "xspec_z": x["xspec_z"], "t2": x["t2"], "spe": x["spe"],
        "domain_disagreement": abs(b2_z - x["xspec_z"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/z24_xspec_v5"))
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, manifest = protocol(args.input_root)
    signals = {}
    for index, row in enumerate(frame.itertuples()):
        signals[row.record_id] = load_vertical(Path(row.resolved_path))
        if (index + 1) % 40 == 0:
            print(f"loaded {index + 1}/{len(frame)}", flush=True)

    models = {}
    for excitation_name in ("avt", "fvt"):
        healthy_group = frame[(frame.excitation_resolved == excitation_name) & (frame.label == 0)]
        models[excitation_name] = fit_group(healthy_group, signals, args.alpha)
        print(f"fit {excitation_name}", flush=True)

    rows = []
    test = frame[frame.role == "test"]
    for index, row in enumerate(test.itertuples()):
        values = score(models[row.excitation_resolved], signals[row.record_id])
        common = {
            "record_id": row.record_id, "label": int(row.label), "excitation": row.excitation_resolved,
            "scenario": getattr(row, "scenario", ""), "setup": getattr(row, "setup", ""),
            "b2_z": values["b2_z"], "xspec_z": values["xspec_z"], "t2": values["t2"],
            "spe": values["spe"], "domain_disagreement": values["domain_disagreement"],
        }
        for model in ("b2", "xspec", "fusion_80_20"):
            rows.append({**common, "model": model, "score": values[model], "prediction": values[model] > 1.0})
        if (index + 1) % 40 == 0:
            print(f"scored {index + 1}/{len(test)}", flush=True)
    scores = pd.DataFrame(rows)

    metric_rows = []
    for model, subset in scores.groupby("model"):
        metric_rows.append({"model": model, "records": len(subset), **metrics(subset.label.to_numpy(), subset.score.to_numpy())})
    metric_frame = pd.DataFrame(metric_rows)
    subgroup_rows = []
    for (model, excitation_name), subset in scores.groupby(["model", "excitation"]):
        subgroup_rows.append({
            "model": model, "excitation": excitation_name, "records": len(subset),
            **metrics(subset.label.to_numpy(), subset.score.to_numpy()),
        })
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metric_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "version": "xspec-v5", "dataset": "Z24", "fit_labels": "healthy only",
        "damage_labels_used_for_selection": False, "fusion_weight": {"b2": 0.80, "xspec": 0.20},
        "calibration": "excitation-specific held-out healthy records", "manifest": str(manifest),
    }, indent=2), encoding="utf-8")
    print(metric_frame.to_string(index=False))
    print(pd.DataFrame(subgroup_rows).to_string(index=False))
    print(f"Z24_XSPEC_V5_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

