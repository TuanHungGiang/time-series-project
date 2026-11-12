from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_z24_b2 import load_vertical
from run_z24_crossfit_fusion_b11 import fit_modal_raw
from run_z24_excitation_spectral_c2 import spectral_components
from run_z24_model_a_v31 import corrupt_v31
from run_z24_shape_fusion_b1 import record_metadata
from run_z24_supervised_consistency_c1 import (
    TEST_CONDITIONS,
    TRAIN_CONDITIONS,
    feature_from_signals as c1_feature,
    prepare_supervised,
)


EXCITATIONS = ("avt", "fvt")
TARGETS = (0.80, 0.875, 0.95)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def record_peaks(signals: np.ndarray) -> list[dict]:
    frequency, _, spectrum, _ = spectral_components(signals)
    distance = max(1, int(round(0.55 / (frequency[1] - frequency[0]))))
    positive = spectrum[spectrum > 0]
    floor = float(np.median(positive)) if positive.size else 0.0
    peaks, _ = find_peaks(spectrum, distance=distance, prominence=max(0.05 * floor, 1e-18))
    if not len(peaks):
        return []
    prominences = peak_prominences(spectrum, peaks)[0]
    scale = max(float(np.median(prominences)), 1e-18)
    return [
        {"frequency": float(frequency[index]), "prominence": float(value / scale)}
        for index, value in zip(peaks, prominences)
    ]


def cluster_stable_peaks(
    records: list[tuple[str, np.ndarray]],
    excitation: str,
    tolerance: float = 0.35,
) -> pd.DataFrame:
    observations = []
    for record_id, signals in records:
        for item in record_peaks(signals):
            observations.append({"record_id": record_id, **item})
    observations.sort(key=lambda item: item["frequency"])
    clusters: list[list[dict]] = []
    for item in observations:
        if not clusters or abs(item["frequency"] - np.median([x["frequency"] for x in clusters[-1]])) > tolerance:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    rows = []
    record_count = max(len(records), 1)
    for cluster_id, cluster in enumerate(clusters):
        # Keep one strongest observation per source record.
        by_record: dict[str, dict] = {}
        for item in cluster:
            current = by_record.get(item["record_id"])
            if current is None or item["prominence"] > current["prominence"]:
                by_record[item["record_id"]] = item
        frequencies = np.asarray([item["frequency"] for item in by_record.values()])
        prominences = np.asarray([item["prominence"] for item in by_record.values()])
        support = len(by_record) / record_count
        iqr = float(np.quantile(frequencies, 0.75) - np.quantile(frequencies, 0.25)) if len(frequencies) > 1 else 0.0
        score = support * float(np.median(prominences)) / (iqr + 0.05)
        rows.append({
            "excitation": excitation, "cluster_id": cluster_id,
            "frequency_median": float(np.median(frequencies)),
            "frequency_iqr": iqr, "support": float(support),
            "prominence_median": float(np.median(prominences)),
            "stability_score": float(score), "records": len(by_record),
        })
    return pd.DataFrame(rows)


def select_anchors(stability: pd.DataFrame, b2_anchors: list[float], count: int = 5) -> np.ndarray:
    candidates = [(float(value), 1e9) for value in b2_anchors]
    eligible = stability[(stability.support >= 0.40) & (stability.frequency_iqr <= 0.50)]
    candidates.extend([
        (float(row.frequency_median), float(row.stability_score))
        for row in eligible.sort_values("stability_score", ascending=False).itertuples()
    ])
    selected: list[float] = []
    for frequency, _ in candidates:
        if all(abs(frequency - prior) >= 0.55 for prior in selected):
            selected.append(frequency)
        if len(selected) == count:
            break
    if len(selected) < count:
        for row in stability.sort_values("stability_score", ascending=False).itertuples():
            frequency = float(row.frequency_median)
            if all(abs(frequency - prior) >= 0.55 for prior in selected):
                selected.append(frequency)
            if len(selected) == count:
                break
    if len(selected) < 3:
        raise RuntimeError(f"Only {len(selected)} stable/B2 anchors found")
    return np.asarray(sorted(selected[:count]), dtype=float)


def compact_features(
    signals: np.ndarray,
    anchors: np.ndarray,
    availability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    frequency, _, spectrum, cross = spectral_components(signals, availability)
    modal, fdd, modal_names, fdd_names, detected = [], [], [], [], []
    for anchor_index, anchor in enumerate(anchors):
        band = np.flatnonzero(np.abs(frequency - anchor) <= 0.70)
        local = spectrum[band]
        peak_position = int(np.argmax(local))
        peak_index = int(band[peak_position])
        peak_frequency = float(frequency[peak_index])
        detected.append(peak_frequency)
        prominence = float(max(local[peak_position] - np.median(local), 0.0))
        prominence /= max(float(np.median(local)), 1e-18)
        energy = float(np.trapezoid(local, frequency[band]))
        shift = peak_frequency - anchor
        modal.extend([shift, shift / max(anchor, 1e-12), np.log1p(prominence), np.log10(energy + 1e-18)])
        modal_names.extend([
            f"a{anchor_index}_signed_shift", f"a{anchor_index}_relative_shift",
            f"a{anchor_index}_log_prominence", f"a{anchor_index}_log_energy",
        ])
        eigenvalues = np.maximum(np.linalg.eigvalsh(cross[peak_index]).real[::-1], 0.0)
        eigenvalues /= max(float(eigenvalues.sum()), 1e-18)
        first = float(eigenvalues[0])
        second = float(eigenvalues[1] if len(eigenvalues) > 1 else 0.0)
        fdd.extend([first, np.log((first + 1e-8) / (second + 1e-8))])
        fdd_names.extend([f"a{anchor_index}_fdd_concentration", f"a{anchor_index}_fdd_log_ratio"])
    for index in range(len(detected) - 1):
        modal.append(float(detected[index] / max(detected[index + 1], 1e-12)))
        modal_names.append(f"adjacent_frequency_ratio_{index}_{index + 1}")
    return (
        np.asarray(modal, dtype=float), np.asarray(fdd, dtype=float),
        modal_names, fdd_names,
    )


def build_views(frame, modal_models, anchors, mode: str) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(frame.itertuples(), start=1):
        signals = load_vertical(Path(row.resolved_path))
        excitation = row.excitation_resolved
        state, setup = record_metadata(str(row.record_id))
        base = {
            "record_id": row.record_id, "label": int(row.label), "excitation": excitation,
            "state": state, "setup": setup, "split": row.supervised_split,
        }
        available = np.ones(signals.shape[0], dtype=bool)
        modal, fdd, modal_names, fdd_names = compact_features(signals, anchors[excitation], available)
        rows.append({
            **base, "condition": "clean", "replicate": 0,
            "c1": c1_feature(signals, excitation, available), "modal": modal, "fdd": fdd,
        })
        conditions = TRAIN_CONDITIONS if row.supervised_split != "test" else TEST_CONDITIONS[1:]
        repeats = 1 if mode == "quick" else (2 if row.supervised_split == "train" else 3)
        for condition in conditions:
            for replicate in range(repeats):
                changed, availability, _, _ = corrupt_v31(
                    signals, condition, str(row.record_id), replicate, modal_models[excitation]
                )
                modal, fdd, _, _ = compact_features(changed, anchors[excitation], availability)
                rows.append({
                    **base, "condition": condition, "replicate": replicate,
                    "c1": c1_feature(changed, excitation, availability),
                    "modal": modal, "fdd": fdd,
                })
        if index % 24 == 0:
            print(f"features={index}/{len(frame)}", flush=True)
    views = pd.DataFrame(rows)
    views.attrs["modal_names"], views.attrs["fdd_names"] = modal_names, fdd_names
    return views


@dataclass
class ExcitationRobustScaler:
    centers: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]

    @classmethod
    def fit(cls, frame: pd.DataFrame, column: str):
        centers, scales = {}, {}
        for excitation, group in frame.groupby("excitation"):
            matrix = np.stack(group[column])
            center = np.median(matrix, axis=0)
            mad = 1.4826 * np.median(np.abs(matrix - center), axis=0)
            fallback = np.std(matrix, axis=0)
            centers[excitation] = center
            scales[excitation] = np.maximum(mad, np.maximum(0.25 * fallback, 1e-6))
        return cls(centers, scales)

    def transform(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        output = np.empty((len(frame), len(next(iter(self.centers.values())))), dtype=float)
        reset = frame.reset_index(drop=True)
        for excitation, positions in reset.groupby("excitation").indices.items():
            matrix = np.stack(reset.iloc[positions][column])
            output[positions] = np.clip(
                (matrix - self.centers[excitation]) / self.scales[excitation], -12, 12
            )
        return output


class FeatureModel:
    def __init__(self, estimator, kind: str, modal_scaler=None, fdd_scaler=None):
        self.estimator, self.kind = estimator, kind
        self.modal_scaler, self.fdd_scaler = modal_scaler, fdd_scaler

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        c1 = np.stack(frame.c1)
        if self.kind == "c1":
            return c1
        modal = self.modal_scaler.transform(frame, "modal")
        if self.kind == "modal":
            return np.concatenate([c1, modal], axis=1)
        fdd = self.fdd_scaler.transform(frame, "fdd")
        base = np.concatenate([c1, modal, fdd], axis=1)
        if self.kind in ("conditioned", "branch"):
            flag = (frame.excitation.to_numpy() == "fvt").astype(float)[:, None]
            conditioned = np.concatenate([base, modal * flag, fdd * flag], axis=1)
            if self.kind == "branch":
                return conditioned[:, c1.shape[1]:]
            return conditioned
        return base

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.matrix(frame))[:, 1]


class LateFusion:
    def __init__(self, shape_model: FeatureModel, branch_model: FeatureModel, alpha: float):
        self.shape_model, self.branch_model, self.alpha = shape_model, branch_model, alpha

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        shape = np.clip(self.shape_model.score(frame), 1e-6, 1 - 1e-6)
        branch = np.clip(self.branch_model.score(frame), 1e-6, 1 - 1e-6)
        shape_logit = np.log(shape / (1 - shape))
        branch_logit = np.log(branch / (1 - branch))
        logit = (1 - self.alpha) * shape_logit + self.alpha * branch_logit
        return 1 / (1 + np.exp(-np.clip(logit, -30, 30)))


def fit_logistic(matrix: np.ndarray, labels: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0),
    ).fit(matrix, labels)


def classification_metrics(labels, scores, threshold) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = scores >= threshold
    positive, negative = labels == 1, labels == 0
    recall = float(prediction[positive].mean()) if positive.any() else np.nan
    specificity = float((~prediction[negative]).mean()) if negative.any() else np.nan
    return {
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "recall": recall, "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2),
    }


def threshold_at_specificity(scores: np.ndarray, target: float) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    for threshold in candidates:
        realized = float(np.mean(scores < threshold))
        if realized >= target:
            return float(threshold), realized
    return float(np.nextafter(scores.max(), np.inf)), 1.0


def calibrate_thresholds(model, validation: pd.DataFrame) -> pd.DataFrame:
    score = model.score(validation)
    table = validation[["record_id", "label", "condition"]].copy()
    table["score"] = score
    healthy_clean = table[(table.label == 0) & (table.condition == "clean")].drop_duplicates("record_id")
    healthy_robust = table[table.label == 0].groupby("record_id", as_index=False).score.max()
    rows = []
    for mode, healthy in (("clean", healthy_clean), ("robust", healthy_robust)):
        for target in TARGETS if mode == "clean" else (0.875,):
            threshold, realized = threshold_at_specificity(healthy.score.to_numpy(), target)
            rows.append({
                "threshold_mode": mode, "specificity_target": target,
                "threshold": threshold, "healthy_records": len(healthy),
                "realized_validation_specificity": realized,
            })
    return pd.DataFrame(rows)


def healthy_specificities(model, validation: pd.DataFrame, threshold: float) -> dict:
    """Report false-alarm control separately for clean and corrupted healthy views."""
    score = model.score(validation)
    healthy = validation.label.to_numpy(dtype=int) == 0
    clean = healthy & (validation.condition.to_numpy() == "clean")
    corrupted = healthy & (validation.condition.to_numpy() != "clean")

    def specificity(mask: np.ndarray) -> float:
        return float(np.mean(score[mask] < threshold)) if mask.any() else np.nan

    return {
        "clean_healthy_specificity": specificity(clean),
        "corrupted_healthy_specificity": specificity(corrupted),
        "overall_healthy_specificity": specificity(healthy),
    }


def evaluate(model, frame: pd.DataFrame, threshold: float, model_name: str, split: str):
    metrics_rows, prediction_rows = [], []
    for (condition, replicate), group in frame.groupby(["condition", "replicate"]):
        group = group.reset_index(drop=True)
        score = model.score(group)
        values = classification_metrics(group.label, score, threshold)
        metrics_rows.append({
            "model": model_name, "split": split, "condition": condition,
            "replicate": int(replicate), **values,
        })
        for row, value in zip(group.itertuples(), score):
            prediction_rows.append({
                "model": model_name, "split": split, "record_id": row.record_id,
                "label": int(row.label), "condition": condition, "replicate": int(replicate),
                "excitation": row.excitation, "state": int(row.state), "setup": int(row.setup),
                "score": float(value), "threshold": float(threshold),
                "prediction": int(value >= threshold),
            })
    return metrics_rows, prediction_rows


def feature_diagnostics(
    views: pd.DataFrame,
    modal_names: list[str],
    fdd_names: list[str],
    modal_scaler: ExcitationRobustScaler,
    fdd_scaler: ExcitationRobustScaler,
) -> pd.DataFrame:
    validation = views[(views.split == "validation") & (views.condition == "clean")].reset_index(drop=True)
    modal = modal_scaler.transform(validation, "modal")
    fdd = fdd_scaler.transform(validation, "fdd")
    matrix = np.concatenate([modal, fdd], axis=1)
    names = modal_names + fdd_names
    rows = []
    for excitation in EXCITATIONS:
        mask = validation.excitation.to_numpy() == excitation
        labels = validation.label.to_numpy(dtype=int)[mask]
        for index, name in enumerate(names):
            values = matrix[mask, index]
            auc = float(roc_auc_score(labels, values)) if np.unique(labels).size == 2 else np.nan
            oriented = max(auc, 1 - auc) if np.isfinite(auc) else np.nan
            healthy, damage = values[labels == 0], values[labels == 1]
            pooled = np.sqrt((np.var(healthy) + np.var(damage)) / 2 + 1e-12)
            effect = float((np.mean(damage) - np.mean(healthy)) / pooled)
            rows.append({
                "excitation": excitation, "feature": name, "auroc": auc,
                "oriented_auroc": oriented, "standardized_effect": effect,
                "healthy_records": len(healthy), "damage_records": len(damage),
            })
    return pd.DataFrame(rows)


def diagnostic_figure(diagnostics: pd.DataFrame, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, excitation in zip(axes, EXCITATIONS):
        selected = diagnostics[diagnostics.excitation == excitation].nlargest(12, "oriented_auroc")
        axis.barh(selected.feature[::-1], selected.oriented_auroc[::-1])
        axis.axvline(0.65, color="crimson", linestyle="--", linewidth=1)
        axis.set_xlim(0.5, 1.0)
        axis.set_title(f"{excitation.upper()} validation feature separability")
        axis.set_xlabel("Oriented univariate AUROC")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, manifest = prepare_supervised(args.input_root)
    modal_models, anchors, stability_tables = {}, {}, []
    for excitation in EXCITATIONS:
        healthy = frame[
            (frame.supervised_split == "train") & (frame.label == 0)
            & (frame.excitation_resolved == excitation)
        ]
        modal_models[excitation] = fit_modal_raw(healthy)
        records = [
            (str(row.record_id), load_vertical(Path(row.resolved_path)))
            for row in healthy.itertuples()
        ]
        stability = cluster_stable_peaks(records, excitation)
        stability_tables.append(stability)
        anchors[excitation] = select_anchors(
            stability, list(modal_models[excitation].anchors), count=5
        )
        print(f"anchors {excitation}={anchors[excitation].round(3).tolist()}", flush=True)
    stability = pd.concat(stability_tables, ignore_index=True)
    stability.to_csv(args.output_dir / "anchor_stability.csv", index=False)
    (args.output_dir / "selected_anchors.json").write_text(json.dumps(
        {key: value.tolist() for key, value in anchors.items()}, indent=2
    ))
    views = build_views(frame, modal_models, anchors, args.mode)
    modal_names, fdd_names = views.attrs["modal_names"], views.attrs["fdd_names"]
    train_clean = views[(views.split == "train") & (views.condition == "clean")].reset_index(drop=True)
    validation = views[views.split == "validation"].reset_index(drop=True)
    test = views[views.split == "test"].reset_index(drop=True)
    modal_scaler = ExcitationRobustScaler.fit(train_clean, "modal")
    fdd_scaler = ExcitationRobustScaler.fit(train_clean, "fdd")
    diagnostics = feature_diagnostics(views, modal_names, fdd_names, modal_scaler, fdd_scaler)
    diagnostics.to_csv(args.output_dir / "feature_effects.csv", index=False)
    diagnostics.groupby("excitation", as_index=False).agg(
        features=("feature", "size"),
        features_auc_ge_065=("oriented_auroc", lambda values: int(np.sum(values >= 0.65))),
        best_oriented_auroc=("oriented_auroc", "max"),
        median_oriented_auroc=("oriented_auroc", "median"),
    ).to_csv(args.output_dir / "excitation_feature_metrics.csv", index=False)
    diagnostic_figure(diagnostics, args.output_dir / "feature_diagnostic.png")
    setup_rows = []
    clean_views = views[views.condition == "clean"].reset_index(drop=True)
    scaled_modal = modal_scaler.transform(clean_views, "modal")
    for position, row in enumerate(clean_views.itertuples()):
        for feature_index, name in enumerate(modal_names):
            setup_rows.append({
                "record_id": row.record_id, "split": row.split, "label": row.label,
                "excitation": row.excitation, "setup": row.setup,
                "feature": name, "value": scaled_modal[position, feature_index],
            })
    pd.DataFrame(setup_rows).groupby(
        ["split", "label", "excitation", "setup", "feature"], as_index=False
    ).value.median().to_csv(args.output_dir / "setup_feature_shift.csv", index=False)

    labels = train_clean.label.to_numpy(dtype=int)
    models: dict[str, object] = {}
    c1 = FeatureModel(None, "c1")
    c1.estimator = fit_logistic(c1.matrix(train_clean), labels)
    models["m0_locked_c1"] = c1
    for name, kind in (
        ("m1_c1_modal", "modal"),
        ("m2_c1_modal_fdd", "modal_fdd"),
        ("m3_conditioned", "conditioned"),
    ):
        model = FeatureModel(None, kind, modal_scaler, fdd_scaler)
        model.estimator = fit_logistic(model.matrix(train_clean), labels)
        models[name] = model
    # Modal/FDD branch for late fusion. Alpha=0 exactly recovers locked C1.
    branch = FeatureModel(None, "branch", modal_scaler, fdd_scaler)
    branch.estimator = fit_logistic(branch.matrix(train_clean), labels)
    best_late = None
    late_candidates = []
    validation_clean = validation[validation.condition == "clean"].reset_index(drop=True)
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5):
        candidate = LateFusion(c1, branch, alpha)
        thresholds = calibrate_thresholds(candidate, validation)
        threshold = float(thresholds[(thresholds.threshold_mode == "clean") & (thresholds.specificity_target == 0.875)].threshold.iloc[0])
        score = candidate.score(validation)
        overall = classification_metrics(validation.label, score, threshold)
        false_alarm = healthy_specificities(candidate, validation, threshold)
        recalls = []
        for excitation in EXCITATIONS:
            mask = validation.excitation == excitation
            recalls.append(classification_metrics(validation.loc[mask, "label"], score[mask], threshold)["recall"])
        eligible = (
            false_alarm["clean_healthy_specificity"] >= 0.875
            and false_alarm["overall_healthy_specificity"] >= 0.80
        )
        key = (eligible, overall["recall"], min(recalls), overall["auroc"], -alpha)
        late_candidates.append({
            "alpha": alpha, "eligible": eligible, **overall, **false_alarm,
            "worst_excitation_recall": min(recalls),
        })
        if best_late is None or key > best_late[0]:
            best_late = (key, alpha, candidate)
    assert best_late is not None
    models["m4_late_fusion"] = best_late[2]

    threshold_rows, validation_rows, validation_predictions = [], [], []
    selection_rows = []
    for name, model in models.items():
        table = calibrate_thresholds(model, validation)
        table.insert(0, "model", name)
        threshold_rows.append(table)
        primary = table[(table.threshold_mode == "clean") & (table.specificity_target == 0.875)].iloc[0]
        rows, preds = evaluate(model, validation, float(primary.threshold), name, "validation")
        validation_rows.extend(rows)
        validation_predictions.extend(preds)
        score = model.score(validation)
        overall = classification_metrics(validation.label, score, float(primary.threshold))
        false_alarm = healthy_specificities(model, validation, float(primary.threshold))
        excitation_recalls = []
        for excitation in EXCITATIONS:
            mask = validation.excitation == excitation
            excitation_recalls.append(classification_metrics(
                validation.loc[mask, "label"], score[mask], float(primary.threshold)
            )["recall"])
        selection_rows.append({
            "model": name, **overall, "worst_excitation_recall": min(excitation_recalls),
            "threshold": float(primary.threshold),
            "realized_clean_healthy_specificity": float(primary.realized_validation_specificity),
            **false_alarm,
            "late_fusion_alpha": best_late[1] if name == "m4_late_fusion" else np.nan,
        })
    selection = pd.DataFrame(selection_rows)
    selection["specificity_gate"] = (
        (selection.clean_healthy_specificity >= 0.875)
        & (selection.overall_healthy_specificity >= 0.80)
    )
    eligible = selection[selection.specificity_gate]
    if eligible.empty:
        raise RuntimeError(
            "No model passed the validation specificity gate: "
            "clean >= 0.875 and overall healthy >= 0.80"
        )
    selected = eligible.sort_values(
        ["recall", "worst_excitation_recall", "auroc", "model"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selection["selected"] = selection.model == selected.model
    feature_gate = {
        excitation: int(np.sum(group.oriented_auroc >= 0.65))
        for excitation, group in diagnostics.groupby("excitation")
    }
    feature_gate_pass = all(value >= 2 for value in feature_gate.values())
    model_gate_pass = selected.model != "m0_locked_c1"

    # Development-holdout ablation: every model uses its own validation-locked
    # clean threshold. This table diagnoses the representation; it never selects
    # a model from test labels.
    ablation_metric_rows, ablation_prediction_rows = [], []
    all_thresholds = pd.concat(threshold_rows, ignore_index=True)
    for name, model in models.items():
        primary = all_thresholds[
            (all_thresholds.model == name)
            & (all_thresholds.threshold_mode == "clean")
            & (all_thresholds.specificity_target == 0.875)
        ].iloc[0]
        rows, preds = evaluate(model, test, float(primary.threshold), name, "development_holdout")
        ablation_metric_rows.extend(rows)
        ablation_prediction_rows.extend(preds)

    test_metric_rows, test_prediction_rows = [], []
    selected_model = models[str(selected.model)]
    selected_thresholds = all_thresholds
    selected_thresholds = selected_thresholds[selected_thresholds.model == selected.model]
    for threshold_row in selected_thresholds.itertuples():
        rows, preds = evaluate(
            selected_model, test, float(threshold_row.threshold),
            str(selected.model), "test"
        )
        for row in rows:
            row.update({
                "threshold_mode": threshold_row.threshold_mode,
                "specificity_target": threshold_row.specificity_target,
            })
        for row in preds:
            row.update({
                "threshold_mode": threshold_row.threshold_mode,
                "specificity_target": threshold_row.specificity_target,
            })
        test_metric_rows.extend(rows)
        test_prediction_rows.extend(preds)

    all_thresholds.to_csv(args.output_dir / "thresholds.csv", index=False)
    pd.DataFrame(late_candidates).to_csv(args.output_dir / "late_fusion_validation.csv", index=False)
    selection.to_csv(args.output_dir / "validation_model_selection.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(args.output_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(validation_predictions).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(test_metric_rows).to_csv(args.output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(test_prediction_rows).to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(ablation_metric_rows).to_csv(args.output_dir / "development_holdout_ablation_metrics.csv", index=False)
    pd.DataFrame(ablation_prediction_rows).to_csv(args.output_dir / "development_holdout_ablation_predictions.csv", index=False)
    views.drop(columns=["c1", "modal", "fdd"]).to_csv(args.output_dir / "view_manifest.csv", index=False)
    joblib.dump({
        "model": selected_model, "thresholds": selected_thresholds,
        "anchors": anchors, "modal_names": modal_names, "fdd_names": fdd_names,
    }, args.output_dir / "selected_model.joblib")
    artifact = {
        "protocol": "z24-c3.1-specificity-gated", "seed": args.seed, "mode": args.mode,
        "manifest": str(manifest), "selected_model": str(selected.model),
        "feature_gate_counts": feature_gate, "feature_gate_pass": feature_gate_pass,
        "model_gate_pass": model_gate_pass, "late_fusion_alpha": best_late[1],
        "test_labels_used_for_selection": False,
        "test_status": "development holdout; previously inspected",
        "selection_gate": {"clean_healthy_specificity": 0.875, "overall_healthy_specificity": 0.80},
    }
    (args.output_dir / "run.json").write_text(json.dumps(artifact, indent=2))
    print("\n=== ANCHOR STABILITY TOP ===")
    print(stability.sort_values(["excitation", "stability_score"], ascending=[True, False]).groupby("excitation").head(8).to_string(index=False))
    print("\n=== FEATURE EFFECTS TOP ===")
    print(diagnostics.sort_values(["excitation", "oriented_auroc"], ascending=[True, False]).groupby("excitation").head(10).to_string(index=False))
    print("\n=== VALIDATION MODEL SELECTION ===")
    print(selection.sort_values(["recall", "auroc"], ascending=False).to_string(index=False))
    print("\n=== DEVELOPMENT HOLDOUT: ALL ABLATIONS (NOT USED FOR SELECTION) ===")
    print(pd.DataFrame(ablation_metric_rows).groupby(
        ["model", "condition"], as_index=False
    )[["auroc", "recall", "specificity", "balanced_accuracy"]].mean().to_string(index=False))
    print("\n=== SELECTED TEST OPERATING POINTS ===")
    print(pd.DataFrame(test_metric_rows).groupby(
        ["threshold_mode", "specificity_target", "condition"], as_index=False
    )[["auroc", "recall", "specificity", "balanced_accuracy"]].mean().to_string(index=False))
    print(
        f"Z24_C31_PASS selected={selected.model} feature_gate={feature_gate_pass} "
        f"model_gate={model_gate_pass} output={args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()

