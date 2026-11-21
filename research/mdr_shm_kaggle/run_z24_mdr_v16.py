from __future__ import annotations

"""MDR-SHM V16: validation-locked branch policy under an FPR constraint.

V16 leaves the V15 representation models unchanged.  For each observable mask
regime, validation data select one of Compact USAD, anchored mask recovery, or
criticality fusion, together with a lower-tail threshold constrained to
validation FPR <= 1/6.  Clean data always use Compact USAD.  Regimes whose
healthy-train criticality ratio exceeds the fixed fallback boundary use Compact
USAD as well.  The resulting policy is locked before test evaluation.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_mdr_v12 as v12  # noqa: E402
import run_z24_mdr_v14 as v14  # noqa: E402
import run_z24_mdr_v15 as v15  # noqa: E402
from run_hellbridge_source_baselines import publication_metrics  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    load_manifest, stable_sensor_map,
)


CONDITIONS = (
    "clean", "sensor_0.10", "sensor_0.30", "sensor_0.50",
    "targeted_0.30", "targeted_0.50",
)
CANDIDATES = v14.ALL_METHODS
FPR_CAP = 1.0 / 6.0
CRITICALITY_FALLBACK = 1.25


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def threshold_under_fpr_cap(labels, scores, fpr_cap=FPR_CAP):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("Threshold validation cohort must contain both classes")
    unique = np.unique(scores)
    epsilon = max(float(np.ptp(unique)), 1.0) * 1e-9
    candidates = np.concatenate([
        [unique[0] - epsilon],
        (unique[:-1] + unique[1:]) / 2.0,
        [unique[-1] + epsilon],
    ])
    best = None
    for threshold in candidates:
        metric = publication_metrics(labels, scores, float(threshold))
        if metric["fpr"] > fpr_cap + 1e-12:
            continue
        key = (
            metric["ba"], metric["recall"], metric["f1"],
            -metric["fpr"], metric["auc"],
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metric)
    if best is None:
        raise RuntimeError("No threshold satisfies the validation FPR cap")
    return best[1], best[2]


def select_policy(frame: pd.DataFrame):
    decisions, audit_rows = {}, []
    for condition in CONDITIONS:
        condition_validation = frame[
            (frame.split == "val") & (frame.condition == condition)
        ]
        meta = condition_validation[
            condition_validation.method == "compact_raw"
        ]
        coverage = float(meta.coverage.mean())
        relative_criticality = float(meta.relative_criticality.mean())
        forced_reason = None
        if condition == "clean":
            methods = ("compact_raw",)
            forced_reason = "clean_preserves_best_usad"
        elif relative_criticality >= CRITICALITY_FALLBACK:
            methods = ("compact_raw",)
            forced_reason = "high_healthy_train_criticality_fallback"
        else:
            methods = CANDIDATES

        ranked = []
        for method in methods:
            validation = condition_validation[
                condition_validation.method == method
            ]
            threshold, metric = threshold_under_fpr_cap(
                validation.label, validation.score_lower, FPR_CAP,
            )
            key = (
                metric["ba"], metric["recall"], metric["auc"],
                -metric["fpr"],
            )
            ranked.append((key, method, threshold, metric))
            audit_rows.append({
                "condition": condition,
                "method": method,
                "coverage": coverage,
                "relative_criticality": relative_criticality,
                "forced_reason": forced_reason,
                "threshold": threshold,
                **{f"validation_{name}": value for name, value in metric.items()},
            })
        _, method, threshold, metric = max(ranked, key=lambda item: item[0])
        decisions[condition] = {
            "method": method,
            "threshold": float(threshold),
            "coverage": coverage,
            "relative_criticality": relative_criticality,
            "forced_reason": forced_reason,
            "validation_ba": float(metric["ba"]),
            "validation_recall": float(metric["recall"]),
            "validation_fpr": float(metric["fpr"]),
        }
    return decisions, pd.DataFrame(audit_rows)


def evaluate_method(cohort, threshold):
    matched, realized = v15.matched_recall(
        cohort.label, cohort.score_lower, FPR_CAP,
    )
    return {
        **publication_metrics(cohort.label, cohort.score_lower, threshold),
        "partial_auc_fpr_0.20": v15.low_fpr_auc(
            cohort.label, cohort.score_lower,
        ),
        "matched_recall_fpr_1_over_6": matched,
        "matched_realized_fpr": realized,
    }


def finish(output: Path, base_frame: pd.DataFrame):
    frame = v14.add_criticality_fusion(base_frame)
    policy, audit = select_policy(frame)
    rows, selected_scores = [], []

    for condition in CONDITIONS:
        decision = policy[condition]
        selected_method = decision["method"]

        # Fair baseline: same condition-specific validation and FPR constraint.
        baseline_validation = frame[
            (frame.split == "val")
            & (frame.condition == condition)
            & (frame.method == "compact_raw")
        ]
        baseline_threshold, _ = threshold_under_fpr_cap(
            baseline_validation.label,
            baseline_validation.score_lower,
            FPR_CAP,
        )

        for output_method, source_method, threshold in (
            ("compact_raw", "compact_raw", baseline_threshold),
            ("v16_policy", selected_method, decision["threshold"]),
        ):
            test = frame[
                (frame.split == "test")
                & (frame.condition == condition)
                & (frame.method == source_method)
            ].copy()
            test["method"] = output_method
            test["source_method"] = source_method
            test["threshold"] = threshold
            test["score"] = test.score_lower
            selected_scores.append(test)
            for replicate, cohort in test.groupby("replicate", sort=False):
                rows.append({
                    "method": output_method,
                    "source_method": source_method,
                    "condition": condition,
                    "replicate": int(replicate),
                    "threshold": float(threshold),
                    **evaluate_method(cohort, threshold),
                    "coverage_mean": float(cohort.coverage.mean()),
                    "relative_criticality_mean": float(
                        cohort.relative_criticality.mean()
                    ),
                    "parameters": int(cohort.parameters.iloc[0]),
                })

    metrics = pd.DataFrame(rows)
    scores = pd.concat(selected_scores, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ba_mean=("ba", "mean"),
        ba_std=("ba", "std"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        partial_auc_fpr_0_20=("partial_auc_fpr_0.20", "mean"),
        matched_recall_fpr_1_over_6=("matched_recall_fpr_1_over_6", "mean"),
        matched_realized_fpr=("matched_realized_fpr", "mean"),
        coverage_mean=("coverage_mean", "mean"),
        relative_criticality_mean=("relative_criticality_mean", "mean"),
        parameters=("parameters", "mean"), runs=("replicate", "size"),
    )

    baseline = summary[summary.method == "compact_raw"].set_index("condition")
    proposed = summary[summary.method == "v16_policy"].set_index("condition")
    columns = [
        "auc_mean", "ba_mean", "recall_mean", "fpr_mean",
        "partial_auc_fpr_0_20", "matched_recall_fpr_1_over_6",
    ]
    effects = proposed[columns].subtract(baseline[columns])
    effects.columns = [
        "delta_auc", "delta_ba", "delta_recall", "delta_fpr",
        "delta_partial_auc_fpr_0_20", "delta_matched_recall_fpr_1_over_6",
    ]
    effects = effects.reset_index()
    effects["selected_method"] = effects.condition.map(
        {condition: value["method"] for condition, value in policy.items()}
    )

    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)
    audit.to_csv(output / "validation_policy_audit.csv", index=False)
    (output / "policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8",
    )

    pivot = summary.pivot(index="condition", columns="method", values="ba_mean")
    axis = pivot.plot.bar(figsize=(12, 5.5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level balanced accuracy")
    axis.set_title("MDR-SHM V16: validation-locked policy")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v16_balanced_accuracy.png", dpi=220)
    plt.close(axis.figure)

    print("\n=== LOCKED POLICY ===", flush=True)
    print(json.dumps(policy, indent=2), flush=True)
    print("\n=== V16 SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== V16 POLICY - FAIR BASELINE ===", flush=True)
    print(effects.to_string(index=False), flush=True)
    return summary, effects, policy


def decision(effects: pd.DataFrame):
    table = effects.set_index("condition")
    random_conditions = ["sensor_0.10", "sensor_0.30", "sensor_0.50"]
    mean_random_auc = float(table.loc[random_conditions, "delta_auc"].mean())
    mean_random_ba = float(table.loc[random_conditions, "delta_ba"].mean())
    promote = bool(
        table.loc["clean", "delta_auc"] >= -0.005
        and mean_random_auc > 0.01
        and mean_random_ba >= 0.0
        and table.loc["sensor_0.50", "delta_auc"] >= 0.02
        and table.loc["sensor_0.50", "delta_ba"] >= 0.0
        and table.loc["sensor_0.50", "delta_recall"] >= -0.02
        and table.loc[
            "sensor_0.50", "delta_matched_recall_fpr_1_over_6"
        ] >= 0.0
        and table.loc["targeted_0.50", "delta_auc"] >= -0.005
        and table.loc["targeted_0.50", "delta_ba"] >= 0.0
    )
    return {
        "mean_random_loss_delta_auc": mean_random_auc,
        "mean_random_loss_delta_ba": mean_random_ba,
        "promote_to_three_seeds": promote,
        "decision": (
            "PROMOTE V16 to three seeds."
            if promote else
            "STOP after one seed; the locked policy did not pass all gates."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--recovery-hidden", type=int, default=48)
    parser.add_argument("--recovery-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} candidates={CANDIDATES} "
        f"seed={args.seed} epochs={args.epochs} fpr_cap={FPR_CAP:.6f}",
        flush=True,
    )
    started = time.time()
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, *key)
        if path.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        v15.run_group(group, labels, args.seed, group_index, args, device, path)

    paths = [group_file(args.output_dir, *key) for key in sorted(sensor_map)]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate base score rows")
    expected = len(v14.BASE_METHODS) * 2 * 132 * len(v12.scenarios_v12())
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} base rows, got {len(frame)}")

    summary, effects, policy = finish(args.output_dir, frame)
    screening = decision(effects)
    (args.output_dir / "decision.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "MDR-SHM V16 validation-locked branch policy",
        "seed": args.seed,
        "epochs": args.epochs,
        "model_fit": "healthy train only",
        "candidate_branches": list(CANDIDATES),
        "policy_selection": "labeled condition-matched validation only",
        "validation_fpr_cap": FPR_CAP,
        "criticality_fallback_boundary": CRITICALITY_FALLBACK,
        "clean_branch_forced": "compact_raw",
        "targeted_training_masks": False,
        "test_used_for_fit_calibration_or_policy_selection": False,
        "locked_policy": policy,
        "conditions": v12.scenarios_v12(),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(screening, indent=2), flush=True)
    print(f"MDR_V16_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

