from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_z24_supervised_c1_final import (
    METRICS,
    bootstrap_metric_arrays,
    observed_condition_metrics,
    raw_metric_tables,
    subgroup_diagnostic,
)


def load_runs(root: Path):
    tests, validations, selections, occlusions = [], [], [], []
    for run_file in sorted(root.glob("seed_*/run.json")):
        artifact = json.loads(run_file.read_text())
        seed = int(artifact["seed"])
        run_dir = run_file.parent
        test = pd.read_csv(run_dir / "test_predictions.csv")
        validation = pd.read_csv(run_dir / "validation_predictions.csv")
        test["seed"], validation["seed"] = seed, seed
        tests.append(test)
        validations.append(validation)
        selections.append({
            "seed": seed, "selected_c2": artifact["selected_c2"],
            "specificity_target": artifact["specificity_target"],
        })
        path = run_dir / "occlusion_predictions.csv"
        if path.exists():
            item = pd.read_csv(path)
            item["seed"] = seed
            occlusions.append(item)
    if not tests:
        raise FileNotFoundError(f"No seed_*/run.json below {root}")
    return (
        pd.concat(tests, ignore_index=True),
        pd.concat(validations, ignore_index=True),
        pd.DataFrame(selections),
        pd.concat(occlusions, ignore_index=True) if occlusions else pd.DataFrame(),
    )


def selected_policy(test: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for row in selections.itertuples():
        part = test[(test.seed == row.seed) & (test.model == row.selected_c2)].copy()
        part["model"] = "validation_selected_c2"
        part["selected_source_model"] = row.selected_c2
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def paired_selected_bootstrap(
    test: pd.DataFrame,
    selections: pd.DataFrame,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for selection in selections.itertuples():
        seed_frame = test[test.seed == selection.seed]
        candidate = seed_frame[seed_frame.model == selection.selected_c2]
        baseline = seed_frame[seed_frame.model == "c1_shape_logistic"]
        for condition in sorted(seed_frame.condition.unique()):
            a, b = candidate[candidate.condition == condition], baseline[baseline.condition == condition]
            labels = a[["record_id", "label"]].drop_duplicates()
            positive_ids = labels.loc[labels.label == 1, "record_id"].to_numpy()
            negative_ids = labels.loc[labels.label == 0, "record_id"].to_numpy()
            positive_draw = rng.integers(0, len(positive_ids), (samples, len(positive_ids)))
            negative_draw = rng.integers(0, len(negative_ids), (samples, len(negative_ids)))
            am = bootstrap_metric_arrays(a, positive_ids, negative_ids, positive_draw, negative_draw)
            bm = bootstrap_metric_arrays(b, positive_ids, negative_ids, positive_draw, negative_draw)
            observed_a, observed_b = observed_condition_metrics(a), observed_condition_metrics(b)
            for metric in METRICS:
                difference = am[metric] - bm[metric]
                rows.append({
                    "seed": selection.seed, "condition": condition,
                    "candidate": selection.selected_c2, "baseline": "c1_shape_logistic",
                    "metric": metric, "delta": observed_a[metric] - observed_b[metric],
                    "ci_lower": float(np.quantile(difference, 0.025)),
                    "ci_upper": float(np.quantile(difference, 0.975)),
                    "probability_candidate_better": float(np.mean(difference > 0)),
                    "bootstrap_samples": samples,
                })
    return pd.DataFrame(rows)


def occlusion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    from analyze_z24_supervised_c1_final import binary_metrics
    rows = []
    for (seed, model, condition), group in frame.groupby(["seed", "model", "condition"]):
        rows.append({
            "seed": seed, "model": model, "condition": condition,
            **binary_metrics(group, "score", "threshold"),
        })
    per_seed = pd.DataFrame(rows)
    return per_seed.groupby(["model", "condition"], as_index=False).agg(
        auroc_mean=("auroc", "mean"), auroc_seed_std=("auroc", "std"),
        recall_mean=("recall", "mean"), specificity_mean=("specificity", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), seeds=("seed", "nunique"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test, validation, selections, occlusions = load_runs(args.runs_root)
    policy = selected_policy(test, selections)
    combined = pd.concat([test, policy], ignore_index=True)
    replicate, summary = raw_metric_tables(combined)
    bootstrap = paired_selected_bootstrap(
        test, selections, args.bootstrap_samples, np.random.default_rng(args.seed)
    )
    subgroup = subgroup_diagnostic(combined)
    occ_summary = occlusion_summary(occlusions)
    selections.to_csv(args.output_dir / "model_selection_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "detector_summary.csv", index=False)
    replicate.to_csv(args.output_dir / "replicate_metrics.csv", index=False)
    policy.to_csv(args.output_dir / "selected_policy_predictions.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_selected_vs_c1.csv", index=False)
    subgroup.to_csv(args.output_dir / "state_excitation_diagnostic.csv", index=False)
    if not occlusions.empty:
        occlusions.to_csv(args.output_dir / "occlusion_predictions.csv", index=False)
        occ_summary.to_csv(args.output_dir / "occlusion_summary.csv", index=False)
    print("\n=== C2 SELECTION ===")
    print(selections.to_string(index=False))
    print("\n=== C2 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== SELECTED C2 VS C1: CLEAN ===")
    print(bootstrap[bootstrap.condition == "clean"].to_string(index=False))
    print("\n=== STATE 9 / FVT ===")
    print(subgroup[(subgroup.state == 9) & (subgroup.excitation == "fvt")].to_string(index=False))
    if not occ_summary.empty:
        print("\n=== WHITE-BOX OCCLUSION ===")
        print(occ_summary.to_string(index=False))
    print(f"Z24_C2_ANALYSIS_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

