from __future__ import annotations

import argparse
import json
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
    ExcitationRobustScaler,
    FeatureModel,
    LateFusion,
    build_views,
    classification_metrics,
    cluster_stable_peaks,
    evaluate,
    feature_diagnostics,
    healthy_specificities,
    seed_all,
    select_anchors,
    threshold_at_specificity,
)
from run_z24_crossfit_fusion_b11 import fit_modal_raw
from run_z24_supervised_consistency_c1 import prepare_supervised


PRIMARY_SPECIFICITY = 0.875
OVERALL_SPECIFICITY_GATE = 0.80
ROBUST_C = 0.25
FUSION_ALPHA = 0.20


def record_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each source record total weight one, regardless of view count."""
    counts = frame.groupby("record_id").size()
    return frame.record_id.map(lambda value: 1.0 / counts.loc[value]).to_numpy(float)


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
    c_value: float = 1.0,
):
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
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["logisticregression__sample_weight"] = sample_weight
    return estimator.fit(matrix, labels, **fit_kwargs)


def fit_candidates(train: pd.DataFrame) -> dict[str, object]:
    """Fit the four pre-registered C3.2 candidates on one training fold."""
    train = train.reset_index(drop=True)
    clean = train[train.condition == "clean"].reset_index(drop=True)
    modal_scaler = ExcitationRobustScaler.fit(clean, "modal")
    fdd_scaler = ExcitationRobustScaler.fit(clean, "fdd")

    models: dict[str, object] = {}

    m0 = FeatureModel(None, "c1")
    m0.estimator = fit_logistic(m0.matrix(clean), clean.label.to_numpy(int))
    models["m0_c1_clean"] = m0

    m1 = FeatureModel(None, "modal", modal_scaler, fdd_scaler)
    m1.estimator = fit_logistic(m1.matrix(clean), clean.label.to_numpy(int))
    models["m1_modal_clean"] = m1

    weights = record_balanced_weights(train)
    m1r = FeatureModel(None, "modal", modal_scaler, fdd_scaler)
    m1r.estimator = fit_logistic(
        m1r.matrix(train), train.label.to_numpy(int), weights, ROBUST_C
    )
    models["m1r_modal_augmented"] = m1r

    c1r = FeatureModel(None, "c1")
    c1r.estimator = fit_logistic(
        c1r.matrix(train), train.label.to_numpy(int), weights, ROBUST_C
    )
    branch = FeatureModel(None, "branch", modal_scaler, fdd_scaler)
    branch.estimator = fit_logistic(
        branch.matrix(train), train.label.to_numpy(int), weights, ROBUST_C
    )
    models["m4r_fusion_a020"] = LateFusion(c1r, branch, FUSION_ALPHA)
    return models


def grouped_oof_scores(train: pd.DataFrame) -> pd.DataFrame:
    """Leave one physical setup out; no view of a held setup enters its model."""
    rows = []
    setups = sorted(int(value) for value in train.setup.unique())
    if len(setups) < 3:
        raise RuntimeError(f"Need at least three training setups, found {setups}")
    for fold, held_setup in enumerate(setups):
        fit_frame = train[train.setup != held_setup].reset_index(drop=True)
        held_frame = train[train.setup == held_setup].reset_index(drop=True)
        if fit_frame.label.nunique() < 2 or held_frame.empty:
            raise RuntimeError(f"Invalid setup fold held_setup={held_setup}")
        models = fit_candidates(fit_frame)
        for name, model in models.items():
            scores = model.score(held_frame)
            for record, score in zip(held_frame.itertuples(), scores):
                rows.append({
                    "fold": fold,
                    "held_setup": held_setup,
                    "model": name,
                    "record_id": record.record_id,
                    "label": int(record.label),
                    "condition": record.condition,
                    "replicate": int(record.replicate),
                    "excitation": record.excitation,
                    "score": float(score),
                })
        print(f"oof setup={held_setup} records={held_frame.record_id.nunique()}", flush=True)
    return pd.DataFrame(rows)


def thresholds_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in oof.groupby("model"):
        healthy = group[group.label == 0]
        clean = healthy[healthy.condition == "clean"].drop_duplicates("record_id")
        robust = healthy.groupby("record_id", as_index=False).score.max()
        for mode, reference in (("clean_oof", clean), ("robust_oof", robust)):
            threshold, realized = threshold_at_specificity(
                reference.score.to_numpy(), PRIMARY_SPECIFICITY
            )
            rows.append({
                "model": model,
                "threshold_mode": mode,
                "specificity_target": PRIMARY_SPECIFICITY,
                "threshold": threshold,
                "healthy_records": len(reference),
                "realized_oof_specificity": realized,
            })
    return pd.DataFrame(rows)


def validation_row(
    name: str,
    model,
    validation: pd.DataFrame,
    threshold_mode: str,
    threshold: float,
) -> dict:
    scores = model.score(validation)
    overall = classification_metrics(validation.label, scores, threshold)
    false_alarm = healthy_specificities(model, validation, threshold)
    recalls = []
    for excitation in EXCITATIONS:
        mask = validation.excitation == excitation
        recalls.append(classification_metrics(
            validation.loc[mask, "label"], scores[mask], threshold
        )["recall"])
    return {
        "model": name,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        **overall,
        **false_alarm,
        "worst_excitation_recall": min(recalls),
    }


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
        {key: value.tolist() for key, value in anchors.items()}, indent=2
    ))

    views = build_views(frame, modal_models, anchors, args.mode)
    train = views[views.split == "train"].reset_index(drop=True)
    validation = views[views.split == "validation"].reset_index(drop=True)
    test = views[views.split == "test"].reset_index(drop=True)

    # Gap 1 evidence: stable excitation-aware modal representation.
    train_clean = train[train.condition == "clean"].reset_index(drop=True)
    modal_scaler = ExcitationRobustScaler.fit(train_clean, "modal")
    fdd_scaler = ExcitationRobustScaler.fit(train_clean, "fdd")
    diagnostics = feature_diagnostics(
        views, views.attrs["modal_names"], views.attrs["fdd_names"],
        modal_scaler, fdd_scaler,
    )
    diagnostics.to_csv(args.output_dir / "feature_effects.csv", index=False)

    # Gap 3: setup-grouped out-of-fold healthy calibration.
    oof = grouped_oof_scores(train)
    thresholds = thresholds_from_oof(oof)
    oof.to_csv(args.output_dir / "oof_calibration_predictions.csv", index=False)
    thresholds.to_csv(args.output_dir / "oof_thresholds.csv", index=False)

    # Gap 2: clean versus record-balanced corruption-augmented candidates.
    models = fit_candidates(train)
    selection_rows = []
    for row in thresholds.itertuples():
        selection_rows.append(validation_row(
            row.model, models[row.model], validation,
            row.threshold_mode, float(row.threshold),
        ))
    selection = pd.DataFrame(selection_rows)
    selection["specificity_gate"] = (
        (selection.clean_healthy_specificity >= PRIMARY_SPECIFICITY)
        & (selection.overall_healthy_specificity >= OVERALL_SPECIFICITY_GATE)
    )
    eligible = selection[selection.specificity_gate]
    if eligible.empty:
        # Preserve the failed gate as a scientific result rather than silently
        # tuning it down. Select the safest row only for diagnostic evaluation.
        pool = selection
        gate_pass = False
    else:
        pool = eligible
        gate_pass = True
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

    # Every candidate/threshold pair is evaluated on the development holdout,
    # but these labels never affect selection.
    metric_rows, prediction_rows = [], []
    for row in thresholds.itertuples():
        rows, predictions = evaluate(
            models[row.model], test, float(row.threshold),
            row.model, "development_holdout",
        )
        for item in rows:
            item["threshold_mode"] = row.threshold_mode
        for item in predictions:
            item["threshold_mode"] = row.threshold_mode
        metric_rows.extend(rows)
        prediction_rows.extend(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "development_holdout_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "development_holdout_predictions.csv", index=False)

    selected_threshold = float(selected.threshold)
    joblib.dump({
        "model": models[str(selected.model)],
        "threshold": selected_threshold,
        "threshold_mode": str(selected.threshold_mode),
        "anchors": anchors,
    }, args.output_dir / "selected_model.joblib")

    artifact = {
        "protocol": "z24-c3.2-record-balanced-grouped-calibration",
        "seed": args.seed,
        "mode": args.mode,
        "manifest": str(manifest),
        "selected_model": str(selected.model),
        "selected_threshold_mode": str(selected.threshold_mode),
        "specificity_gate_pass": bool(gate_pass),
        "selection_gate": {
            "clean_healthy_specificity": PRIMARY_SPECIFICITY,
            "overall_healthy_specificity": OVERALL_SPECIFICITY_GATE,
        },
        "robust_regularization_c": ROBUST_C,
        "fusion_alpha": FUSION_ALPHA,
        "test_labels_used_for_selection": False,
        "test_status": "development holdout; previously inspected",
    }
    (args.output_dir / "run.json").write_text(json.dumps(artifact, indent=2))

    print("\n=== OOF THRESHOLDS ===")
    print(thresholds.to_string(index=False))
    print("\n=== VALIDATION SELECTION ===")
    print(selection.sort_values(
        ["specificity_gate", "balanced_accuracy", "auroc"], ascending=False
    ).to_string(index=False))
    print("\n=== DEVELOPMENT HOLDOUT SUMMARY (NOT USED FOR SELECTION) ===")
    print(metrics.groupby(
        ["model", "threshold_mode", "condition"], as_index=False
    )[["auroc", "recall", "specificity", "balanced_accuracy"]].mean().to_string(index=False))
    print(
        f"Z24_C32_PASS selected={selected.model}/{selected.threshold_mode} "
        f"specificity_gate={gate_pass} output={args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()

