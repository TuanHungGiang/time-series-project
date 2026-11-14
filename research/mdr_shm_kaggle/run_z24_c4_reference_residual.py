from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_z24_b2 import load_vertical
from run_z24_c31_specificity_gated import (
    EXCITATIONS,
    build_views,
    classification_metrics,
    cluster_stable_peaks,
    healthy_specificities,
    select_anchors,
    threshold_at_specificity,
)
from run_z24_crossfit_fusion_b11 import fit_modal_raw
from run_z24_supervised_consistency_c1 import prepare_supervised


SPECIFICITY_TARGET = 0.875
OVERALL_SPECIFICITY_GATE = 0.80


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def raw_modal_vector(row) -> np.ndarray:
    return np.concatenate([
        np.asarray(row.c1, dtype=float),
        np.asarray(row.modal, dtype=float),
    ])


def reference_candidates(
    views: pd.DataFrame,
    row,
    mode: str,
) -> pd.DataFrame:
    healthy = views[
        (views.label == 0)
        & (views.setup == row.setup)
        & (views.excitation == row.excitation)
    ]
    if mode == "clean":
        candidates = healthy[healthy.condition == "clean"]
    elif mode == "matched":
        candidates = healthy[
            (healthy.condition == row.condition)
            & (healthy.replicate == row.replicate)
        ]
        if candidates.empty:
            candidates = healthy[healthy.condition == row.condition]
    else:
        raise ValueError(mode)

    # A healthy record is never allowed to define its own reference score.
    # Damage records may use every pre-declared healthy baseline record.
    if int(row.label) == 0:
        candidates = candidates[candidates.record_id != row.record_id]

    if candidates.empty:
        candidates = healthy[healthy.condition == "clean"]
        if int(row.label) == 0:
            candidates = candidates[candidates.record_id != row.record_id]
    if candidates.empty:
        raise RuntimeError(
            "No independent healthy reference for "
            f"record={row.record_id}, setup={row.setup}, "
            f"excitation={row.excitation}, condition={row.condition}"
        )
    return candidates


def add_reference_features(views: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe within-setup/excitation residual features."""
    output = views.copy()
    output["feature_m0"] = [np.asarray(value, float) for value in output.c1]
    output["feature_m1"] = [raw_modal_vector(row) for row in output.itertuples()]

    clean_features, matched_features, audit_rows = [], [], []
    for position, row in enumerate(output.itertuples(), start=1):
        current = raw_modal_vector(row)
        generated = {}
        for mode in ("clean", "matched"):
            candidates = reference_candidates(output, row, mode)
            reference = np.median(
                np.stack([raw_modal_vector(item) for item in candidates.itertuples()]),
                axis=0,
            )
            delta = current - reference
            generated[mode] = np.concatenate([delta, np.abs(delta)])
            audit_rows.append({
                "record_id": row.record_id,
                "label": int(row.label),
                "split": row.split,
                "setup": int(row.setup),
                "excitation": row.excitation,
                "condition": row.condition,
                "replicate": int(row.replicate),
                "reference_mode": mode,
                "reference_records": int(candidates.record_id.nunique()),
                "reference_ids": "|".join(sorted(candidates.record_id.astype(str).unique())),
                "self_excluded": bool(
                    int(row.label) != 0
                    or str(row.record_id) not in set(candidates.record_id.astype(str))
                ),
                "residual_l2": float(np.linalg.norm(delta)),
            })
        clean_features.append(generated["clean"])
        matched_features.append(generated["matched"])
        if position % 1000 == 0:
            print(f"reference-features={position}/{len(output)}", flush=True)

    output["feature_ref_clean"] = clean_features
    output["feature_ref_matched"] = matched_features
    output.attrs["reference_audit"] = pd.DataFrame(audit_rows)
    return output


def record_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("record_id").size()
    return frame.record_id.map(lambda value: 1.0 / counts.loc[value]).to_numpy(float)


@dataclass
class VectorModel:
    estimator: object
    feature_column: str

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.stack(frame[self.feature_column].to_numpy())
        return self.estimator.predict_proba(matrix)[:, 1]


def fit_vector_model(
    frame: pd.DataFrame,
    feature_column: str,
    c_value: float,
    weighted: bool,
) -> VectorModel:
    matrix = np.stack(frame[feature_column].to_numpy())
    labels = frame.label.to_numpy(int)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=c_value,
            solver="liblinear",
            random_state=0,
        ),
    )
    kwargs = {}
    if weighted:
        kwargs["logisticregression__sample_weight"] = record_balanced_weights(frame)
    estimator.fit(matrix, labels, **kwargs)
    return VectorModel(estimator, feature_column)


def fit_candidates(train: pd.DataFrame) -> dict[str, VectorModel]:
    clean = train[train.condition == "clean"].reset_index(drop=True)
    return {
        "m0_c1_absolute": fit_vector_model(clean, "feature_m0", 1.0, False),
        "m1_modal_absolute": fit_vector_model(clean, "feature_m1", 1.0, False),
        "m2_ref_clean_train": fit_vector_model(clean, "feature_ref_clean", 0.20, False),
        "m3_ref_clean_aug": fit_vector_model(train, "feature_ref_clean", 0.20, True),
        "m4_ref_matched_aug": fit_vector_model(train, "feature_ref_matched", 0.20, True),
    }


def calibration_table(model, validation: pd.DataFrame, name: str) -> pd.DataFrame:
    scores = model.score(validation)
    table = validation[["record_id", "label", "condition"]].copy()
    table["score"] = scores
    clean = table[(table.label == 0) & (table.condition == "clean")].drop_duplicates("record_id")
    robust = table[table.label == 0].groupby("record_id", as_index=False).score.max()
    rows = []
    for mode, reference in (("clean_reference", clean), ("robust_reference", robust)):
        threshold, realized = threshold_at_specificity(
            reference.score.to_numpy(), SPECIFICITY_TARGET
        )
        rows.append({
            "model": name,
            "threshold_mode": mode,
            "specificity_target": SPECIFICITY_TARGET,
            "threshold": threshold,
            "healthy_records": len(reference),
            "realized_reference_specificity": realized,
        })
    return pd.DataFrame(rows)


def evaluate(model, frame, threshold, model_name, threshold_mode, split):
    metrics, predictions = [], []
    for (condition, replicate), group in frame.groupby(["condition", "replicate"]):
        group = group.reset_index(drop=True)
        scores = model.score(group)
        values = classification_metrics(group.label, scores, threshold)
        metrics.append({
            "model": model_name,
            "threshold_mode": threshold_mode,
            "split": split,
            "condition": condition,
            "replicate": int(replicate),
            **values,
        })
        for row, score in zip(group.itertuples(), scores):
            predictions.append({
                "model": model_name,
                "threshold_mode": threshold_mode,
                "split": split,
                "record_id": row.record_id,
                "label": int(row.label),
                "state": int(row.state),
                "setup": int(row.setup),
                "excitation": row.excitation,
                "condition": condition,
                "replicate": int(replicate),
                "score": float(score),
                "threshold": float(threshold),
                "prediction": int(score >= threshold),
            })
    return metrics, predictions


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
            (frame.supervised_split == "train")
            & (frame.label == 0)
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
        {name: values.tolist() for name, values in anchors.items()}, indent=2
    ))

    views = build_views(frame, modal_models, anchors, args.mode)
    views = add_reference_features(views)
    reference_audit = views.attrs["reference_audit"]
    reference_audit.to_csv(args.output_dir / "reference_audit.csv", index=False)
    if not reference_audit.self_excluded.all():
        raise RuntimeError("Healthy reference leakage detected")

    train = views[views.split == "train"].reset_index(drop=True)
    validation = views[views.split == "validation"].reset_index(drop=True)
    test = views[views.split == "test"].reset_index(drop=True)
    models = fit_candidates(train)

    threshold_tables = [
        calibration_table(model, validation, name)
        for name, model in models.items()
    ]
    thresholds = pd.concat(threshold_tables, ignore_index=True)
    thresholds.to_csv(args.output_dir / "thresholds.csv", index=False)

    selection_rows = []
    for row in thresholds.itertuples():
        model = models[row.model]
        scores = model.score(validation)
        overall = classification_metrics(validation.label, scores, row.threshold)
        false_alarm = healthy_specificities(model, validation, row.threshold)
        recalls = []
        for excitation in EXCITATIONS:
            mask = validation.excitation == excitation
            recalls.append(classification_metrics(
                validation.loc[mask, "label"], scores[mask], row.threshold
            )["recall"])
        selection_rows.append({
            "model": row.model,
            "threshold_mode": row.threshold_mode,
            "threshold": float(row.threshold),
            **overall,
            **false_alarm,
            "worst_excitation_recall": min(recalls),
        })
    selection = pd.DataFrame(selection_rows)
    selection["specificity_gate"] = (
        (selection.clean_healthy_specificity >= SPECIFICITY_TARGET)
        & (selection.overall_healthy_specificity >= OVERALL_SPECIFICITY_GATE)
    )
    eligible = selection[selection.specificity_gate]
    gate_pass = not eligible.empty
    pool = eligible if gate_pass else selection
    selected = pool.sort_values(
        ["balanced_accuracy", "worst_excitation_recall", "auroc",
         "overall_healthy_specificity", "model", "threshold_mode"],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]
    selection["selected"] = (
        (selection.model == selected.model)
        & (selection.threshold_mode == selected.threshold_mode)
    )
    selection.to_csv(args.output_dir / "validation_model_selection.csv", index=False)

    metric_rows, prediction_rows = [], []
    for row in thresholds.itertuples():
        metrics, predictions = evaluate(
            models[row.model], test, float(row.threshold),
            row.model, row.threshold_mode, "development_holdout",
        )
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "development_holdout_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "development_holdout_predictions.csv", index=False)

    # Class/excitation/state diagnostics expose where a candidate wins or fails.
    diagnostic_rows = []
    for (model_name, threshold_mode), group in predictions.groupby(["model", "threshold_mode"]):
        for keys, cohort in group.groupby(["condition", "excitation", "state"]):
            condition, excitation, state = keys
            diagnostic_rows.append({
                "model": model_name,
                "threshold_mode": threshold_mode,
                "condition": condition,
                "excitation": excitation,
                "state": int(state),
                "label": int(cohort.label.iloc[0]),
                "records": len(cohort),
                "detection_rate": float(cohort.prediction.mean()),
                "score_mean": float(cohort.score.mean()),
            })
    pd.DataFrame(diagnostic_rows).to_csv(
        args.output_dir / "state_excitation_diagnostics.csv", index=False
    )

    joblib.dump({
        "model": models[str(selected.model)],
        "threshold": float(selected.threshold),
        "threshold_mode": str(selected.threshold_mode),
        "anchors": anchors,
    }, args.output_dir / "selected_model.joblib")

    artifact = {
        "protocol": "z24-c4-matched-healthy-reference-residual-pilot",
        "seed": args.seed,
        "mode": args.mode,
        "manifest": str(manifest),
        "selected_model": str(selected.model),
        "selected_threshold_mode": str(selected.threshold_mode),
        "specificity_gate_pass": bool(gate_pass),
        "target_healthy_reference_used": True,
        "target_damage_labels_used_for_reference": False,
        "test_labels_used_for_selection": False,
        "test_status": "development holdout; previously inspected",
        "healthy_reference_protocol": "same setup/excitation; healthy LOO; clean versus condition-matched ablation",
    }
    (args.output_dir / "run.json").write_text(json.dumps(artifact, indent=2))

    print("\n=== VALIDATION SELECTION ===")
    print(selection.sort_values(
        ["specificity_gate", "balanced_accuracy", "auroc"], ascending=False
    ).to_string(index=False))
    print("\n=== DEVELOPMENT HOLDOUT (NOT USED FOR SELECTION) ===")
    print(metrics.groupby(
        ["model", "threshold_mode", "condition"], as_index=False
    )[["auroc", "recall", "specificity", "balanced_accuracy"]].mean().to_string(index=False))
    print(
        f"Z24_C4_PASS selected={selected.model}/{selected.threshold_mode} "
        f"gate={gate_pass} output={args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()

