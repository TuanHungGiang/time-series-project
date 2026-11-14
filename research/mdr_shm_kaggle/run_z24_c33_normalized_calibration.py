from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_z24_b2 import load_vertical
from run_z24_c31_specificity_gated import (
    EXCITATIONS,
    ExcitationRobustScaler,
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
from run_z24_c32_robust_calibration import fit_candidates
from run_z24_crossfit_fusion_b11 import fit_modal_raw
from run_z24_supervised_consistency_c1 import prepare_supervised


SPECIFICITY_TARGETS = (0.80, 0.85, 0.90)
CLEAN_SPECIFICITY_GATE = 0.875
OVERALL_SPECIFICITY_GATE = 0.80


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


@dataclass
class ExcitationScoreNormalizer:
    model: object
    centers: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, model, healthy_clean: pd.DataFrame):
        raw = logit(model.score(healthy_clean))
        centers, scales = {}, {}
        for excitation in EXCITATIONS:
            mask = healthy_clean.excitation.to_numpy() == excitation
            values = raw[mask]
            if len(values) < 3:
                raise RuntimeError(
                    f"Need >=3 healthy reference records for {excitation}; found {len(values)}"
                )
            center = float(np.median(values))
            mad = float(1.4826 * np.median(np.abs(values - center)))
            std = float(np.std(values))
            # A modest floor prevents a nearly constant in-sample reference from
            # exploding small probability differences after logit scaling.
            scale = max(mad, 0.25 * std, 0.15)
            centers[excitation], scales[excitation] = center, scale
        return cls(model, centers, scales)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        raw = logit(self.model.score(frame))
        excitation = frame.excitation.to_numpy()
        normalized = np.empty(len(frame), dtype=float)
        for name in EXCITATIONS:
            mask = excitation == name
            normalized[mask] = (raw[mask] - self.centers[name]) / self.scales[name]
        return np.clip(normalized, -30, 30)

    def reference_rows(self, model_name: str, fold: str) -> list[dict]:
        return [
            {
                "model": model_name,
                "fold": fold,
                "excitation": excitation,
                "logit_center": self.centers[excitation],
                "logit_scale": self.scales[excitation],
            }
            for excitation in EXCITATIONS
        ]


def normalized_candidates(train: pd.DataFrame):
    base_models = fit_candidates(train)
    healthy_clean = train[
        (train.label == 0) & (train.condition == "clean")
    ].reset_index(drop=True)
    models, reference_rows = {}, []
    for name, base in base_models.items():
        normalized = ExcitationScoreNormalizer.fit(base, healthy_clean)
        models[name] = normalized
        reference_rows.extend(normalized.reference_rows(name, "final"))
    return models, reference_rows


def grouped_oof_scores(train: pd.DataFrame):
    rows, reference_rows = [], []
    setups = sorted(int(value) for value in train.setup.unique())
    if len(setups) < 3:
        raise RuntimeError(f"Need at least three training setups, found {setups}")
    for fold, held_setup in enumerate(setups):
        fit_frame = train[train.setup != held_setup].reset_index(drop=True)
        held_frame = train[train.setup == held_setup].reset_index(drop=True)
        base_models = fit_candidates(fit_frame)
        healthy_clean = fit_frame[
            (fit_frame.label == 0) & (fit_frame.condition == "clean")
        ].reset_index(drop=True)
        for name, base in base_models.items():
            model = ExcitationScoreNormalizer.fit(base, healthy_clean)
            reference_rows.extend(model.reference_rows(name, str(held_setup)))
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
                    "normalized_score": float(score),
                })
        print(f"normalized-oof setup={held_setup} records={held_frame.record_id.nunique()}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(reference_rows)


def thresholds_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in oof.groupby("model"):
        healthy = group[group.label == 0]
        clean = healthy[healthy.condition == "clean"].drop_duplicates("record_id")
        robust = healthy.groupby("record_id", as_index=False).normalized_score.max()
        for mode, reference in (("clean_oof_z", clean), ("robust_oof_z", robust)):
            for target in SPECIFICITY_TARGETS:
                threshold, realized = threshold_at_specificity(
                    reference.normalized_score.to_numpy(), target
                )
                rows.append({
                    "model": model,
                    "threshold_mode": mode,
                    "specificity_target": target,
                    "threshold": threshold,
                    "healthy_records": len(reference),
                    "realized_oof_specificity": realized,
                })
    return pd.DataFrame(rows)


def validation_row(name, model, validation, threshold_row) -> dict:
    threshold = float(threshold_row.threshold)
    scores = model.score(validation)
    overall = classification_metrics(validation.label, scores, threshold)
    false_alarm = healthy_specificities(model, validation, threshold)
    excitation_recalls = []
    for excitation in EXCITATIONS:
        mask = validation.excitation == excitation
        excitation_recalls.append(classification_metrics(
            validation.loc[mask, "label"], scores[mask], threshold
        )["recall"])
    return {
        "model": name,
        "threshold_mode": threshold_row.threshold_mode,
        "specificity_target": float(threshold_row.specificity_target),
        "threshold": threshold,
        **overall,
        **false_alarm,
        "worst_excitation_recall": min(excitation_recalls),
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

    # Preserve direct evidence for gap 1.
    train_clean = train[train.condition == "clean"].reset_index(drop=True)
    modal_scaler = ExcitationRobustScaler.fit(train_clean, "modal")
    fdd_scaler = ExcitationRobustScaler.fit(train_clean, "fdd")
    diagnostics = feature_diagnostics(
        views, views.attrs["modal_names"], views.attrs["fdd_names"],
        modal_scaler, fdd_scaler,
    )
    diagnostics.to_csv(args.output_dir / "feature_effects.csv", index=False)

    oof, fold_references = grouped_oof_scores(train)
    thresholds = thresholds_from_oof(oof)
    oof.to_csv(args.output_dir / "oof_normalized_predictions.csv", index=False)
    thresholds.to_csv(args.output_dir / "oof_normalized_thresholds.csv", index=False)

    models, final_references = normalized_candidates(train)
    references = pd.concat(
        [fold_references, pd.DataFrame(final_references)], ignore_index=True
    )
    references.to_csv(args.output_dir / "score_normalization_reference.csv", index=False)

    selection_rows = []
    for row in thresholds.itertuples():
        selection_rows.append(validation_row(
            row.model, models[row.model], validation, row
        ))
    selection = pd.DataFrame(selection_rows)
    selection["specificity_gate"] = (
        (selection.clean_healthy_specificity >= CLEAN_SPECIFICITY_GATE)
        & (selection.overall_healthy_specificity >= OVERALL_SPECIFICITY_GATE)
    )
    eligible = selection[selection.specificity_gate]
    gate_pass = not eligible.empty
    pool = eligible if gate_pass else selection
    selected = pool.sort_values(
        ["balanced_accuracy", "worst_excitation_recall", "auroc",
         "overall_healthy_specificity", "specificity_target", "model"],
        ascending=[False, False, False, False, False, True],
    ).iloc[0]
    selection["selected"] = (
        (selection.model == selected.model)
        & (selection.threshold_mode == selected.threshold_mode)
        & (selection.specificity_target == selected.specificity_target)
    )
    selection.to_csv(args.output_dir / "validation_model_selection.csv", index=False)

    metric_rows, prediction_rows = [], []
    for row in thresholds.itertuples():
        rows, predictions = evaluate(
            models[row.model], test, float(row.threshold),
            row.model, "development_holdout",
        )
        for item in rows:
            item.update({
                "threshold_mode": row.threshold_mode,
                "specificity_target": float(row.specificity_target),
            })
        for item in predictions:
            item.update({
                "threshold_mode": row.threshold_mode,
                "specificity_target": float(row.specificity_target),
            })
        metric_rows.extend(rows)
        prediction_rows.extend(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "development_holdout_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "development_holdout_predictions.csv", index=False)

    joblib.dump({
        "model": models[str(selected.model)],
        "threshold": float(selected.threshold),
        "threshold_mode": str(selected.threshold_mode),
        "specificity_target": float(selected.specificity_target),
        "anchors": anchors,
    }, args.output_dir / "selected_model.joblib")

    artifact = {
        "protocol": "z24-c3.3-excitation-normalized-oof-calibration",
        "seed": args.seed,
        "mode": args.mode,
        "manifest": str(manifest),
        "selected_model": str(selected.model),
        "selected_threshold_mode": str(selected.threshold_mode),
        "selected_specificity_target": float(selected.specificity_target),
        "specificity_gate_pass": bool(gate_pass),
        "test_labels_used_for_selection": False,
        "test_status": "development holdout; previously inspected",
        "score_transform": "excitation-specific healthy logit median/MAD",
    }
    (args.output_dir / "run.json").write_text(json.dumps(artifact, indent=2))

    print("\n=== NORMALIZED OOF THRESHOLDS ===")
    print(thresholds.to_string(index=False))
    print("\n=== VALIDATION SELECTION ===")
    print(selection.sort_values(
        ["specificity_gate", "balanced_accuracy", "auroc"], ascending=False
    ).to_string(index=False))
    print("\n=== DEVELOPMENT HOLDOUT (NOT USED FOR SELECTION) ===")
    print(metrics.groupby(
        ["model", "threshold_mode", "specificity_target", "condition"],
        as_index=False,
    )[["auroc", "recall", "specificity", "balanced_accuracy"]].mean().to_string(index=False))
    print(
        f"Z24_C33_PASS selected={selected.model}/{selected.threshold_mode}/"
        f"{selected.specificity_target:.2f} gate={gate_pass} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

