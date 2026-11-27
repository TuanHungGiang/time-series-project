from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_z24_b2 import load_vertical
from run_z24_crossfit_fusion_b11 import fit_modal_raw
from run_z24_model_a_v31 import corrupt_v31
from run_z24_supervised_consistency_c1 import (
    TEST_CONDITIONS,
    TRAIN_CONDITIONS,
    feature_from_signals as c1_feature,
    prepare_supervised,
)
from run_z24_shape_fusion_b1 import record_metadata


FS = 100.0
FMIN, FMAX = 0.5, 30.0
GRID = np.linspace(FMIN, FMAX, 64)
EXCITATIONS = ("avt", "fvt")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def spectral_components(
    signals: np.ndarray,
    availability: np.ndarray | None = None,
    nfft: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(signals, dtype=np.float64)
    if availability is None:
        availability = np.ones(values.shape[0], dtype=bool)
    values = values[np.asarray(availability, dtype=bool)]
    if values.shape[0] < 2:
        raise RuntimeError("C2 needs at least two observed sensors")
    values = detrend(values, axis=1, type="linear")
    width = min(nfft, values.shape[1])
    if width < 512:
        raise RuntimeError(f"C2 needs >=512 samples, got {values.shape[1]}")
    hop = max(1, width // 2)
    starts = np.arange(0, values.shape[1] - width + 1, hop)
    if starts.size == 0:
        starts = np.asarray([0])
    window = np.hanning(width)
    scale = max(float(np.sqrt(np.sum(window**2))), 1e-12)
    spectra = np.stack([
        np.fft.rfft(values[:, start : start + width] * window, n=nfft, axis=1) / scale
        for start in starts
    ])
    frequency = np.fft.rfftfreq(nfft, 1.0 / FS)
    keep = (frequency >= FMIN) & (frequency <= FMAX)
    frequency = frequency[keep]
    spectra = spectra[:, :, keep]
    auto = np.mean(np.abs(spectra) ** 2, axis=0) + 1e-18
    sensor_total = np.trapezoid(auto, frequency, axis=1)
    normalized_auto = auto / np.maximum(sensor_total[:, None], 1e-18)
    median_spectrum = np.median(normalized_auto, axis=0)
    cross = np.einsum(
        "asf,atf->fst", spectra, np.conjugate(spectra), optimize=True
    ) / len(spectra)
    trace = np.maximum(np.real(np.trace(cross, axis1=1, axis2=2)), 1e-18)
    cross = cross / trace[:, None, None]
    return frequency, normalized_auto, median_spectrum, cross


def learn_anchors(clean_signals: list[np.ndarray], count: int = 5) -> np.ndarray:
    spectra = [spectral_components(signal)[2] for signal in clean_signals]
    reference = np.median(np.stack(spectra), axis=0)
    frequency = spectral_components(clean_signals[0])[0]
    distance = max(1, int(round(0.60 / (frequency[1] - frequency[0]))))
    peaks, properties = find_peaks(reference, distance=distance, prominence=0)
    if len(peaks):
        prominence = properties.get("prominences", reference[peaks])
        ordered = peaks[np.argsort(prominence)[::-1]]
    else:
        ordered = np.asarray([], dtype=int)
    selected: list[int] = []
    for index in ordered:
        if all(abs(frequency[index] - frequency[item]) >= 0.60 for item in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.argsort(reference)[::-1]:
            if all(abs(frequency[index] - frequency[item]) >= 0.60 for item in selected):
                selected.append(int(index))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"Could not learn {count} modal anchors")
    return np.sort(frequency[selected]).astype(float)


def c2_feature(
    signals: np.ndarray,
    anchors: np.ndarray,
    availability: np.ndarray | None = None,
) -> np.ndarray:
    frequency, sensor_psd, spectrum, cross = spectral_components(signals, availability)
    log_shape = np.interp(GRID, frequency, np.log10(spectrum + 1e-18))
    log_shape -= np.mean(log_shape)
    output = list(log_shape)
    for anchor in anchors:
        band = np.flatnonzero(np.abs(frequency - anchor) <= 0.75)
        if band.size < 3:
            raise RuntimeError(f"Empty modal band around {anchor}")
        local = spectrum[band]
        peak_index = int(band[np.argmax(local)])
        band_frequency = frequency[band]
        band_energy = float(np.trapezoid(local, band_frequency))
        weights = local / max(float(local.sum()), 1e-18)
        centroid = float(np.sum(band_frequency * weights))
        entropy = float(-np.sum(weights * np.log(weights + 1e-18)) / np.log(len(weights)))
        eigenvalues = np.linalg.eigvalsh(cross[peak_index]).real[::-1]
        eigenvalues = np.maximum(eigenvalues, 0.0)
        eigenvalues /= max(float(eigenvalues.sum()), 1e-18)
        sensor_energy = np.trapezoid(sensor_psd[:, band], band_frequency, axis=1)
        spatial = np.log10(sensor_energy + 1e-18)
        spatial -= np.median(spatial)
        quantiles = np.quantile(spatial, np.linspace(0.1, 0.9, 6))
        output.extend([
            float(frequency[peak_index] - anchor),
            float(np.log10(spectrum[peak_index] + 1e-18)),
            float(np.log10(band_energy + 1e-18)),
            float(centroid - anchor), entropy,
            float(eigenvalues[0]),
            float(eigenvalues[1] if len(eigenvalues) > 1 else 0.0),
            *quantiles.tolist(),
        ])
    result = np.asarray(output, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite C2 feature")
    return result


def build_views(
    frame: pd.DataFrame,
    modal_models: dict,
    anchors: dict[str, np.ndarray],
    mode: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, signal_cache = [], {}
    for index, row in enumerate(frame.itertuples(), start=1):
        signals = load_vertical(Path(row.resolved_path))
        signal_cache[str(row.record_id)] = signals
        excitation = row.excitation_resolved
        state, setup = record_metadata(str(row.record_id))
        base = {
            "record_id": row.record_id, "label": int(row.label),
            "excitation": excitation, "state": state, "setup": setup,
            "split": row.supervised_split,
        }
        availability = np.ones(signals.shape[0], dtype=bool)
        rows.append({
            **base, "condition": "clean", "replicate": 0,
            "c1_feature": c1_feature(signals, excitation, availability),
            "c2_feature": c2_feature(signals, anchors[excitation], availability),
            "corrupted_sensors": "",
        })
        conditions = TRAIN_CONDITIONS if row.supervised_split != "test" else TEST_CONDITIONS[1:]
        repeats = 1 if mode == "quick" else (2 if row.supervised_split == "train" else 3)
        for condition in conditions:
            for replicate in range(repeats):
                changed, available, _, signature = corrupt_v31(
                    signals, condition, str(row.record_id), replicate,
                    modal_models[excitation],
                )
                rows.append({
                    **base, "condition": condition, "replicate": replicate,
                    "c1_feature": c1_feature(changed, excitation, available),
                    "c2_feature": c2_feature(changed, anchors[excitation], available),
                    "corrupted_sensors": signature,
                })
        if index % 24 == 0:
            print(f"features={index}/{len(frame)}", flush=True)
    return pd.DataFrame(rows), signal_cache


def metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions, dtype=bool)
    positive = labels == 1
    negative = ~positive
    recall = float(predictions[positive].mean()) if positive.any() else np.nan
    specificity = float((~predictions[negative]).mean()) if negative.any() else np.nan
    return {
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "recall": recall, "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2),
    }


def robust_threshold(group: pd.DataFrame, scores: np.ndarray, target: float) -> float:
    table = group[["record_id", "label"]].copy()
    table["score"] = scores
    healthy = table[table.label == 0].groupby("record_id").score.max().to_numpy()
    if healthy.size < 3:
        raise RuntimeError(f"Need >=3 healthy validation records, got {healthy.size}")
    candidates = np.r_[np.unique(healthy), np.nextafter(healthy.max(), np.inf)]
    for threshold in candidates:
        if float(np.mean(healthy < threshold)) >= target:
            return float(threshold)
    return float(np.nextafter(healthy.max(), np.inf))


class SharedModel:
    def __init__(self, estimator):
        self.estimator = estimator
        self.thresholds: dict[str, float] = {}

    def score(self, group: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(np.stack(group.c1_feature))[:, 1]


class ExcitationModel:
    def __init__(self, estimators: dict):
        self.estimators = estimators
        self.thresholds: dict[str, float] = {}

    def score(self, group: pd.DataFrame) -> np.ndarray:
        output = np.empty(len(group), dtype=float)
        for excitation, positions in group.groupby("excitation").indices.items():
            output[positions] = self.estimators[excitation].predict_proba(
                np.stack(group.iloc[positions].c2_feature)
            )[:, 1]
        return output


def predictions(model, group: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    thresholds = group.excitation.map(model.thresholds).to_numpy(dtype=float)
    return scores >= thresholds


def fit_models(views: pd.DataFrame, seed: int, specificity_target: float) -> dict:
    train_clean = views[(views.split == "train") & (views.condition == "clean")]
    train_all = views[views.split == "train"]
    shared = make_pipeline(
        StandardScaler(), SelectPercentile(f_classif, percentile=50),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5),
    ).fit(np.stack(train_clean.c1_feature), train_clean.label)
    models: dict[str, object] = {"c1_shape_logistic": SharedModel(shared)}
    for training_name, training in (("clean", train_clean), ("augmented", train_all)):
        logistic, trees = {}, {}
        for excitation in EXCITATIONS:
            subset = training[training.excitation == excitation]
            x, y = np.stack(subset.c2_feature), subset.label.to_numpy(dtype=int)
            logistic[excitation] = make_pipeline(
                StandardScaler(), SelectPercentile(f_classif, percentile=40),
                LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5),
            ).fit(x, y)
            trees[excitation] = ExtraTreesClassifier(
                n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ).fit(x, y)
        models[f"c2_{training_name}_logistic"] = ExcitationModel(logistic)
        models[f"c2_{training_name}_trees"] = ExcitationModel(trees)
    validation = views[views.split == "validation"].reset_index(drop=True)
    for model in models.values():
        score = model.score(validation)
        for excitation in EXCITATIONS:
            mask = validation.excitation == excitation
            model.thresholds[excitation] = robust_threshold(
                validation[mask], score[mask], specificity_target
            )
    return models


def evaluate(models: dict, views: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, prediction_rows = [], []
    evaluation = views[views.split == split].reset_index(drop=True)
    for name, model in models.items():
        for (condition, replicate), group in evaluation.groupby(["condition", "replicate"]):
            group = group.reset_index(drop=True)
            score = model.score(group)
            predicted = predictions(model, group, score)
            metric_rows.append({
                "model": name, "condition": condition, "replicate": int(replicate),
                **metrics(group.label, score, predicted),
            })
            for row, value, decision in zip(group.itertuples(), score, predicted):
                prediction_rows.append({
                    "model": name, "record_id": row.record_id, "label": int(row.label),
                    "condition": condition, "replicate": int(replicate),
                    "excitation": row.excitation, "state": int(row.state), "setup": int(row.setup),
                    "score": float(value), "prediction": int(decision),
                    "threshold": float(model.thresholds[row.excitation]), "split": split,
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def score_masked_record(
    name: str,
    model,
    signals: np.ndarray,
    excitation: str,
    anchors: dict[str, np.ndarray],
    availability: np.ndarray,
) -> float:
    row = {
        "excitation": excitation,
        "c1_feature": c1_feature(signals, excitation, availability),
        "c2_feature": c2_feature(signals, anchors[excitation], availability),
    }
    return float(model.score(pd.DataFrame([row]))[0])


def occlusion_stress_test(
    frame: pd.DataFrame,
    signal_cache: dict[str, np.ndarray],
    anchors: dict[str, np.ndarray],
    models: dict,
    selected_c2: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tested_models = ("c1_shape_logistic", selected_c2)
    test = frame[frame.supervised_split == "test"]
    for name in tested_models:
        model = models[name]
        for index, row in enumerate(test.itertuples(), start=1):
            signals = signal_cache[str(row.record_id)]
            excitation = row.excitation_resolved
            state, setup = record_metadata(str(row.record_id))
            available = np.ones(signals.shape[0], dtype=bool)
            clean_score = score_masked_record(
                name, model, signals, excitation, anchors, available
            )
            threshold = float(model.thresholds[excitation])
            base = {
                "model": name, "record_id": row.record_id, "label": int(row.label),
                "excitation": excitation, "state": state, "setup": setup,
                "threshold": threshold,
            }
            prediction_rows.append({
                **base, "condition": "clean", "score": clean_score,
                "prediction": int(clean_score >= threshold), "corrupted_sensors": "",
            })
            single_scores = []
            for sensor in range(signals.shape[0]):
                mask = available.copy()
                mask[sensor] = False
                single_scores.append(score_masked_record(
                    name, model, signals, excitation, anchors, mask
                ))
            single_scores = np.asarray(single_scores)
            for direction, order in (
                ("fn", np.argsort(single_scores)),
                ("fp", np.argsort(single_scores)[::-1]),
            ):
                for rate in (0.30, 0.50):
                    count = min(signals.shape[0] - 2, max(1, int(round(rate * signals.shape[0]))))
                    removed = np.sort(order[:count])
                    mask = available.copy()
                    mask[removed] = False
                    score = score_masked_record(
                        name, model, signals, excitation, anchors, mask
                    )
                    prediction_rows.append({
                        **base, "condition": f"occlusion_{direction}_{rate:.2f}",
                        "score": score, "prediction": int(score >= threshold),
                        "corrupted_sensors": ";".join(map(str, removed.tolist())),
                    })
            if index % 16 == 0:
                print(f"occlusion model={name} records={index}/{len(test)}", flush=True)
    predictions_frame = pd.DataFrame(prediction_rows)
    metric_rows = []
    for (name, condition), group in predictions_frame.groupby(["model", "condition"]):
        metric_rows.append({
            "model": name, "condition": condition,
            **metrics(group.label, group.score, group.prediction),
        })
    return pd.DataFrame(metric_rows), predictions_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--specificity-target", type=float, default=0.875)
    parser.add_argument("--run-occlusion", action="store_true")
    args = parser.parse_args()
    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, manifest = prepare_supervised(args.input_root)
    modal_models, anchors = {}, {}
    for excitation in EXCITATIONS:
        healthy_train = frame[
            (frame.supervised_split == "train") & (frame.label == 0)
            & (frame.excitation_resolved == excitation)
        ]
        modal_models[excitation] = fit_modal_raw(healthy_train)
        signals = [load_vertical(Path(row.resolved_path)) for row in healthy_train.itertuples()]
        anchors[excitation] = learn_anchors(signals)
        print(f"anchors {excitation}={anchors[excitation].round(3).tolist()}", flush=True)
    views, signal_cache = build_views(frame, modal_models, anchors, args.mode)
    models = fit_models(views, args.seed, args.specificity_target)
    validation_metrics, validation_predictions = evaluate(models, views, "validation")
    validation_summary = validation_metrics.groupby("model", as_index=False).agg(
        auroc=("auroc", "mean"), recall=("recall", "mean"),
        specificity=("specificity", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
    )
    c2_names = [name for name in models if name.startswith("c2_")]
    selected = max(
        c2_names,
        key=lambda name: tuple(
            validation_summary.set_index("model").loc[name, ["balanced_accuracy", "auroc"]]
        ),
    )
    validation_summary["selected_c2"] = validation_summary.model == selected
    test_metrics, test_predictions = evaluate(models, views, "test")
    for table in (validation_metrics, validation_predictions, test_metrics, test_predictions):
        table.insert(0, "seed", args.seed)
    validation_summary.insert(0, "seed", args.seed)
    validation_summary.to_csv(args.output_dir / "validation_summary.csv", index=False)
    validation_metrics.to_csv(args.output_dir / "validation_metrics.csv", index=False)
    validation_predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    test_metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)
    test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    views.drop(columns=["c1_feature", "c2_feature"]).to_csv(
        args.output_dir / "view_manifest.csv", index=False
    )
    for name, model in models.items():
        payload = {
            "thresholds": model.thresholds,
            "estimator": getattr(model, "estimator", None),
            "estimators": getattr(model, "estimators", None),
        }
        joblib.dump(payload, args.output_dir / f"{name}.joblib")
    if args.run_occlusion:
        occlusion_metrics, occlusion_predictions = occlusion_stress_test(
            frame, signal_cache, anchors, models, selected
        )
        occlusion_metrics.insert(0, "seed", args.seed)
        occlusion_predictions.insert(0, "seed", args.seed)
        occlusion_metrics.to_csv(args.output_dir / "occlusion_metrics.csv", index=False)
        occlusion_predictions.to_csv(args.output_dir / "occlusion_predictions.csv", index=False)
    artifact = {
        "protocol": "z24-excitation-conditioned-modal-band-c2",
        "seed": args.seed, "mode": args.mode, "manifest": str(manifest),
        "specificity_target": args.specificity_target,
        "train_setups": [1, 2, 3, 4, 5], "validation_setups": [6, 7],
        "test_setups": [8, 9], "selected_c2": selected,
        "selection_uses_validation_damage_labels": True,
        "test_labels_used_for_selection": False,
        "occlusion_stress_test": bool(args.run_occlusion),
        "anchors": {key: value.tolist() for key, value in anchors.items()},
    }
    (args.output_dir / "run.json").write_text(json.dumps(artifact, indent=2))
    print("\n=== VALIDATION SUMMARY ===")
    print(validation_summary.sort_values("balanced_accuracy", ascending=False).to_string(index=False))
    print("\n=== TEST SUMMARY ===")
    print(test_metrics.groupby(["model", "condition"], as_index=False)[
        ["auroc", "recall", "specificity", "balanced_accuracy"]
    ].mean().sort_values(["condition", "balanced_accuracy"], ascending=[True, False]).to_string(index=False))
    if args.run_occlusion:
        print("\n=== WHITE-BOX SENSOR OCCLUSION ===")
        print(occlusion_metrics.sort_values(["condition", "model"]).to_string(index=False))
    print(f"Z24_C2_PASS selected={selected} output={args.output_dir}")


if __name__ == "__main__":
    main()

