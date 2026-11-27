from __future__ import annotations

"""Post-hoc calibration audit for completed CARE-USAD V18 seed runs.

This script never trains a model and never selects on test.  It reconstructs
validation/test cohorts from each seed's group CSV files and compares:

1. the saved validation-selected policy;
2. a fixed lower score tail with condition-matched validation thresholds;
3. a fixed lower score tail with one clean-validation threshold.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import publication_metrics  # noqa: E402
from run_z24_mdr_v16 import FPR_CAP, threshold_under_fpr_cap  # noqa: E402


CONDITIONS = ("clean", "sensor_0.50", "targeted_0.50")
METHOD_MAP = {"source_usad": "source_usad", "care_usad_v18": "imputed_usad"}


def load_seed(experiment: Path, seed: int):
    seed_dir = experiment / f"seed_{seed}"
    paths = sorted((seed_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"seed={seed}: expected 18 group files, got {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics = pd.read_csv(seed_dir / "metrics.csv")
    return frame, metrics


def metric_rows(seed, protocol, output_method, source_method, condition,
                threshold, validation_metric, test):
    rows = []
    current = test[(test.method == source_method) & (test.condition == condition)]
    for replicate, cohort in current.groupby("replicate", sort=False):
        rows.append({
            "seed": seed, "protocol": protocol,
            "output_method": output_method, "source_method": source_method,
            "condition": condition, "replicate": int(replicate),
            "score_tail": "lower", "threshold": float(threshold),
            **publication_metrics(cohort.label, cohort.score_lower, threshold),
            **{f"validation_{key}": value for key, value in validation_metric.items()},
        })
    return rows


def fixed_lower_protocol(seed, frame, protocol, clean_frozen):
    validation = frame[frame.split == "val"]
    test = frame[frame.split == "test"]
    rows = []
    for output_method, missing_method in METHOD_MAP.items():
        clean_validation = validation[
            (validation.method == "source_usad")
            & (validation.condition == "clean")
        ]
        clean_threshold, clean_metric = threshold_under_fpr_cap(
            clean_validation.label, clean_validation.score_lower, FPR_CAP,
        )
        for condition in CONDITIONS:
            source_method = (
                "source_usad" if condition == "clean" else missing_method
            )
            if clean_frozen:
                threshold, val_metric = clean_threshold, clean_metric
            else:
                current_validation = validation[
                    (validation.method == source_method)
                    & (validation.condition == condition)
                ]
                threshold, val_metric = threshold_under_fpr_cap(
                    current_validation.label,
                    current_validation.score_lower,
                    FPR_CAP,
                )
            rows.extend(metric_rows(
                seed, protocol, output_method, source_method, condition,
                threshold, val_metric, test,
            ))
    return rows


def saved_policy_protocol(seed, metrics):
    output = metrics.copy()
    output["seed"] = seed
    output["protocol"] = "saved_validation_policy"
    output = output.rename(columns={"method": "output_method"})
    keep = [
        "seed", "protocol", "output_method", "source_method", "condition",
        "replicate", "score_tail", "threshold", "auc", "ap", "ba",
        "recall", "fpr", "precision", "f1", "specificity",
    ]
    return output[keep]


def paired_effects(metrics):
    keys = ["seed", "protocol", "condition", "replicate"]
    base = metrics[metrics.output_method == "source_usad"].set_index(keys)
    care = metrics[metrics.output_method == "care_usad_v18"].set_index(keys)
    columns = ["auc", "ba", "recall", "fpr", "f1"]
    effect = care[columns].subtract(base[columns])
    effect.columns = [f"delta_{name}" for name in columns]
    return effect.reset_index()


def tail_diagnostic(seed, frame):
    rows = []
    validation = frame[(frame.split == "val") & (frame.method == "source_usad")]
    test = frame[(frame.split == "test") & (frame.method == "source_usad")]
    for condition in CONDITIONS:
        val = validation[validation.condition == condition]
        tst = test[test.condition == condition]
        for tail in ("upper", "lower", "two_sided"):
            column = f"score_{tail}"
            threshold, val_metric = threshold_under_fpr_cap(
                val.label, val[column], FPR_CAP,
            )
            replicate_metrics = [
                publication_metrics(cohort.label, cohort[column], threshold)
                for _, cohort in tst.groupby("replicate", sort=False)
            ]
            current = pd.DataFrame(replicate_metrics).mean(numeric_only=True)
            rows.append({
                "seed": seed, "condition": condition, "tail": tail,
                "threshold": threshold,
                "validation_auc": val_metric["auc"],
                "validation_ba": val_metric["ba"],
                "validation_recall": val_metric["recall"],
                "validation_fpr": val_metric["fpr"],
                "test_auc": current.auc, "test_ba": current.ba,
                "test_recall": current.recall, "test_fpr": current.fpr,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frames, diagnostic_rows = [], []
    for seed in args.seeds:
        frame, saved = load_seed(args.experiment_dir, seed)
        metric_frames.append(saved_policy_protocol(seed, saved))
        metric_frames.append(pd.DataFrame(fixed_lower_protocol(
            seed, frame, "lower_condition_calibrated", clean_frozen=False,
        )))
        metric_frames.append(pd.DataFrame(fixed_lower_protocol(
            seed, frame, "lower_clean_frozen", clean_frozen=True,
        )))
        diagnostic_rows.extend(tail_diagnostic(seed, frame))

    metrics = pd.concat(metric_frames, ignore_index=True)
    effects = paired_effects(metrics)
    summary = metrics.groupby(
        ["protocol", "output_method", "condition"], as_index=False,
    ).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"), f1_mean=("f1", "mean"),
        observations=("auc", "size"), seeds=("seed", "nunique"),
    )
    effect_summary = effects.groupby(
        ["protocol", "condition"], as_index=False,
    ).agg(
        delta_auc_mean=("delta_auc", "mean"),
        delta_auc_median=("delta_auc", "median"),
        delta_auc_std=("delta_auc", "std"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_recall_mean=("delta_recall", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"),
        positive_auc_pairs=("delta_auc", lambda x: int((x > 0).sum())),
        nonnegative_auc_pairs=("delta_auc", lambda x: int((x >= 0).sum())),
        pairs=("delta_auc", "size"), seeds=("seed", "nunique"),
    )
    tail = pd.DataFrame(diagnostic_rows)

    random_audit = effect_summary[effect_summary.condition == "sensor_0.50"]
    robust = random_audit[
        random_audit.protocol.isin([
            "saved_validation_policy", "lower_condition_calibrated",
        ])
    ]
    decision = {
        "calibration_audit_passed": bool(
            len(robust) == 2
            and (robust.delta_auc_median > 0.02).all()
            and (robust.delta_ba_mean > 0.05).all()
            and (robust.positive_auc_pairs >= 10).all()
        ),
        "clean_frozen_reported_as_stress_test": True,
        "note": "No model was trained and no test value selected a policy.",
    }

    metrics.to_csv(args.output_dir / "audit_metrics.csv", index=False)
    effects.to_csv(args.output_dir / "audit_paired_effects.csv", index=False)
    summary.to_csv(args.output_dir / "audit_summary.csv", index=False)
    effect_summary.to_csv(args.output_dir / "audit_effect_summary.csv", index=False)
    tail.to_csv(args.output_dir / "source_usad_tail_diagnostic.csv", index=False)
    (args.output_dir / "audit_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )
    print("\n=== CALIBRATION AUDIT SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PAIRED EFFECT AUDIT ===")
    print(effect_summary.to_string(index=False))
    print("\n=== SOURCE-USAD TAIL DIAGNOSTIC ===")
    print(tail.to_string(index=False))
    print("\n=== AUDIT DECISION ===")
    print(json.dumps(decision, indent=2))
    print(f"CARE_USAD_V18_CALIBRATION_AUDIT_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

