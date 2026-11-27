from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, welch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_z24_b2 import build_file_index, excitation, find_manifest, load_vertical, resolve_source


FS = 100.0
FMIN, FMAX = 0.5, 30.0
BANDS = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0))
SHAPE_GRID = np.linspace(FMIN, FMAX, 96)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
REPRESENTATIONS = (
    "shape", "amplitude", "relative_sensor", "shape_amplitude",
    "modal_extended", "all_features",
)
PROBES = ("logistic", "rbf_svm", "random_forest", "hist_gbdt")


def infer_setup(row: pd.Series) -> int:
    value = row.get("setup", "")
    match = re.search(r"(\d+)", str(value))
    if match:
        setup = int(match.group(1))
        if 1 <= setup <= 9:
            return setup
    token = " ".join(str(row.get(key, "")) for key in ("record_id", "source_path", "relative_path"))
    match = re.search(r"setup[_-]?0?([1-9])", token, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Cannot infer setup for {row.get('record_id')}")
    return int(match.group(1))


def load_frame(root: Path) -> tuple[pd.DataFrame, Path]:
    manifest = find_manifest(root)
    frame = pd.read_csv(manifest)
    frame["label"] = frame.label.astype(int)
    if "role" not in frame:
        frame["role"] = frame["split"]
    frame["role"] = frame.role.astype(str).str.lower().replace({"val": "calibration"})
    frame.loc[frame.label == 1, "role"] = "test"
    frame = frame[frame.role.isin(["train", "calibration", "test"])].copy()
    dataset_root = manifest.parent.parent
    index = build_file_index(dataset_root)
    frame["resolved_path"] = [str(resolve_source(row, index, dataset_root)) for _, row in frame.iterrows()]
    frame["excitation_resolved"] = [excitation(row) for _, row in frame.iterrows()]
    frame["setup_resolved"] = [infer_setup(row) for _, row in frame.iterrows()]
    frame["audit_split"] = np.select(
        [frame.setup_resolved <= 5, frame.setup_resolved <= 7],
        ["train", "validation"], default="test",
    )
    return frame.reset_index(drop=True), manifest


def spectral_record(signals: np.ndarray) -> dict[str, np.ndarray]:
    values = detrend(np.asarray(signals, dtype=np.float64), axis=1, type="linear")
    frequency, raw = welch(
        values, fs=FS, window="hann", nperseg=min(4096, values.shape[1]),
        noverlap=min(2048, values.shape[1] // 2), axis=1, detrend=False,
        scaling="density",
    )
    keep = (frequency >= FMIN) & (frequency <= FMAX)
    frequency, raw = frequency[keep], raw[:, keep]
    area = np.trapezoid(raw, frequency, axis=1)
    normalized = raw / np.maximum(area[:, None], 1e-18)
    rms = np.sqrt(np.mean(values**2, axis=1))
    return {"frequency": frequency, "raw": raw, "normalized": normalized, "rms": rms}


def learn_anchors(cache: dict[str, dict[str, np.ndarray]], rows: pd.DataFrame) -> dict[str, list[float]]:
    anchors = {}
    for group in ("avt", "fvt"):
        healthy = rows[(rows.excitation_resolved == group) & (rows.label == 0)]
        spectra = [np.median(cache[row.record_id]["normalized"], axis=0) for row in healthy.itertuples()]
        if len(spectra) < 4:
            raise RuntimeError(f"Need >=4 healthy records for anchors in {group}")
        frequency = cache[healthy.iloc[0].record_id]["frequency"]
        reference = np.median(np.stack(spectra), axis=0)
        spacing = max(1, int(round(0.75 / (frequency[1] - frequency[0]))))
        peaks, _ = find_peaks(reference, distance=spacing)
        if len(peaks) < 4:
            raise RuntimeError(f"Insufficient peaks in {group}")
        strongest = peaks[np.argsort(reference[peaks])[-4:]]
        anchors[group] = sorted(float(frequency[index]) for index in strongest)
    return anchors


def fixed_features(record: dict[str, np.ndarray], group: str, anchors: dict[str, list[float]]) -> dict[str, np.ndarray]:
    frequency = record["frequency"]
    raw = record["raw"]
    normalized = record["normalized"]
    global_shape = np.median(normalized, axis=0)
    shape = np.interp(SHAPE_GRID, frequency, np.log10(global_shape + 1e-18))
    shape = shape - np.mean(shape)

    amplitude_values = list(np.quantile(np.log10(record["rms"] + 1e-18), QUANTILES))
    relative_values = []
    for low, high in BANDS:
        keep = (frequency >= low) & (frequency < high)
        energy = np.trapezoid(raw[:, keep], frequency[keep], axis=1)
        log_energy = np.log10(energy + 1e-18)
        amplitude_values.extend(np.quantile(log_energy, QUANTILES).tolist())
        centered = log_energy - np.median(log_energy)
        relative_values.extend(np.quantile(centered, QUANTILES).tolist())

    modal = []
    for anchor in anchors[group]:
        local = np.flatnonzero(np.abs(frequency - anchor) <= 0.6)
        peak = local[int(np.argmax(global_shape[local]))]
        weights = global_shape[local] / max(float(global_shape[local].sum()), 1e-18)
        centroid = float(np.sum(frequency[local] * weights))
        width = float(np.sqrt(np.sum((frequency[local] - centroid) ** 2 * weights)))
        local_floor = float(np.median(global_shape[local]))
        prominence = float(np.log10((global_shape[peak] + 1e-18) / (local_floor + 1e-18)))
        modal.extend((float(frequency[peak]), float(np.log10(global_shape[peak] + 1e-18)), prominence, width))
    for low, high in BANDS:
        keep = (frequency >= low) & (frequency < high)
        modal.append(float(np.log10(np.trapezoid(global_shape[keep], frequency[keep]) + 1e-18)))

    regime = np.asarray([float(group == "fvt")])
    shape = np.concatenate([shape, regime])
    amplitude = np.concatenate([np.asarray(amplitude_values), regime])
    relative = np.concatenate([np.asarray(relative_values), regime])
    modal = np.concatenate([np.asarray(modal), regime])
    return {
        "shape": shape,
        "amplitude": amplitude,
        "relative_sensor": relative,
        "shape_amplitude": np.concatenate([shape[:-1], amplitude]),
        "modal_extended": modal,
        "all_features": np.concatenate([shape[:-1], amplitude[:-1], relative[:-1], modal]),
    }


def make_probe(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=seed),
        )
    if name == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    if name == "hist_gbdt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(name)


def fit_probe(model, name: str, x: np.ndarray, y: np.ndarray):
    if name == "hist_gbdt":
        counts = np.bincount(y, minlength=2)
        weights = np.asarray([len(y) / (2 * max(counts[label], 1)) for label in y])
        model.fit(x, y, sample_weight=weights)
    else:
        model.fit(x, y)
    return model


def evaluate(model, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    prediction = model.predict(x).astype(int)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    else:
        score = model.decision_function(x)
    result = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, score)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - y, 1 - prediction, zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }
    return result, prediction, score


def matrices(frame: pd.DataFrame, feature_cache: dict[str, dict[str, np.ndarray]], representation: str, split: str):
    subset = frame[frame.audit_split == split]
    x = np.stack([feature_cache[row.record_id][representation] for row in subset.itertuples()])
    return subset, x, subset.label.to_numpy(dtype=int)


def split_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, setup, excitation_name, label), group in frame.groupby(
        ["audit_split", "setup_resolved", "excitation_resolved", "label"]
    ):
        rows.append({
            "split": split, "setup": int(setup), "excitation": excitation_name,
            "label": int(label), "records": len(group),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/z24_preprocessing_audit"))
    parser.add_argument("--phase", choices=("develop", "locked"), default="develop")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, manifest = load_frame(args.input_root)
    audit = split_audit(frame)
    audit.to_csv(args.output_dir / "split_audit.csv", index=False)
    overlap = frame.groupby("record_id").audit_split.nunique().max()
    if overlap != 1:
        raise RuntimeError("Record leakage across audit splits")
    print(frame.groupby(["audit_split", "label"]).size().unstack(fill_value=0), flush=True)

    spectral_cache = {}
    for index, row in enumerate(frame.itertuples()):
        spectral_cache[row.record_id] = spectral_record(load_vertical(Path(row.resolved_path)))
        if (index + 1) % 40 == 0:
            print(f"features {index + 1}/{len(frame)}", flush=True)

    anchor_rows = frame[frame.audit_split == "train"] if args.phase == "develop" else frame[frame.audit_split.isin(["train", "validation"])]
    anchors = learn_anchors(spectral_cache, anchor_rows)
    feature_cache = {
        row.record_id: fixed_features(spectral_cache[row.record_id], row.excitation_resolved, anchors)
        for row in frame.itertuples()
    }

    if args.phase == "develop":
        results = []
        for representation in REPRESENTATIONS:
            train_rows, x_train, y_train = matrices(frame, feature_cache, representation, "train")
            val_rows, x_val, y_val = matrices(frame, feature_cache, representation, "validation")
            for probe in PROBES:
                model = fit_probe(make_probe(probe, args.seed), probe, x_train, y_train)
                train_metrics, _, _ = evaluate(model, x_train, y_train)
                val_metrics, prediction, score = evaluate(model, x_val, y_val)
                results.append({
                    "representation": representation, "probe": probe,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"validation_{key}": value for key, value in val_metrics.items()},
                    "generalization_gap_ba": train_metrics["balanced_accuracy"] - val_metrics["balanced_accuracy"],
                })
        result_frame = pd.DataFrame(results).sort_values(
            ["validation_balanced_accuracy", "validation_auroc", "validation_macro_f1"],
            ascending=False,
        ).reset_index(drop=True)
        result_frame.to_csv(args.output_dir / "development_results.csv", index=False)
        winner = result_frame.iloc[0]
        selection = {
            "representation": winner.representation,
            "probe": winner.probe,
            "selected_by": ["validation_balanced_accuracy", "validation_auroc", "validation_macro_f1"],
            "train_setups": [1, 2, 3, 4, 5],
            "validation_setups": [6, 7],
            "locked_test_setups": [8, 9],
            "locked_test_metrics_viewed": False,
            "seed": args.seed,
            "manifest": str(manifest),
        }
        (args.output_dir / "selected_config.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(result_frame.to_string(index=False), flush=True)
        print(json.dumps(selection, indent=2), flush=True)
        print(f"Z24_PREPROCESSING_DEVELOP_PASS output={args.output_dir}", flush=True)
        return

    selection_path = args.output_dir / "selected_config.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run --phase develop first; selected_config.json is missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    representation, probe = selection["representation"], selection["probe"]
    fit_rows = frame[frame.audit_split.isin(["train", "validation"])]
    test_rows = frame[frame.audit_split == "test"]
    x_fit = np.stack([feature_cache[row.record_id][representation] for row in fit_rows.itertuples()])
    y_fit = fit_rows.label.to_numpy(dtype=int)
    x_test = np.stack([feature_cache[row.record_id][representation] for row in test_rows.itertuples()])
    y_test = test_rows.label.to_numpy(dtype=int)
    model = fit_probe(make_probe(probe, args.seed), probe, x_fit, y_fit)
    fit_metrics, _, _ = evaluate(model, x_fit, y_fit)
    test_metrics, prediction, score = evaluate(model, x_test, y_test)
    metrics_frame = pd.DataFrame([{
        "representation": representation, "probe": probe,
        **{f"fit_{key}": value for key, value in fit_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "fit_records": len(fit_rows), "test_records": len(test_rows),
    }])
    metrics_frame.to_csv(args.output_dir / "locked_test_metrics.csv", index=False)
    predictions = test_rows[["record_id", "label", "setup_resolved", "excitation_resolved"]].copy()
    predictions["score"] = score
    predictions["prediction"] = prediction
    predictions.to_csv(args.output_dir / "locked_test_predictions.csv", index=False)
    selection["locked_test_metrics_viewed"] = True
    selection["locked_test_result"] = test_metrics
    (args.output_dir / "locked_protocol.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    development = pd.read_csv(args.output_dir / "development_results.csv").head(12)
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = development.representation + " / " + development.probe
    axis.barh(np.arange(len(development)), development.validation_balanced_accuracy, color="#1769aa")
    axis.set_yticks(np.arange(len(development)), labels=labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Validation balanced accuracy")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig1_preprocessing_probe_validation.png", dpi=220)
    plt.close(figure)
    print(metrics_frame.to_string(index=False), flush=True)
    print(f"Z24_PREPROCESSING_LOCKED_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

