from __future__ import annotations

"""Three-seed confirmation for the locked MDR-SHM V16 policy.

The pilot-selected branch map is frozen for every seed:

* clean -> Compact USAD;
* random 10% -> anchored mask-aware recovery;
* random 30/50% -> criticality fusion;
* targeted 30/50% -> Compact-USAD fallback.

Only the numerical threshold is recalibrated on validation for each seed under
the same FPR <= 1/6 constraint.  Corruption masks are deterministic functions
of record IDs and replicate IDs, so baseline/proposed and all seeds see the
same missing-sensor patterns.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_mdr_v16 as v16  # noqa: E402
from run_hellbridge_source_baselines import publication_metrics  # noqa: E402


LOCKED_BRANCHES = {
    "clean": "compact_raw",
    "sensor_0.10": "maskaware_consistency",
    "sensor_0.30": "criticality_fusion",
    "sensor_0.50": "criticality_fusion",
    "targeted_0.30": "compact_raw",
    "targeted_0.50": "compact_raw",
}


def select_locked_policy(frame: pd.DataFrame):
    decisions, rows = {}, []
    for condition in v16.CONDITIONS:
        method = LOCKED_BRANCHES[condition]
        validation = frame[
            (frame.split == "val")
            & (frame.condition == condition)
            & (frame.method == method)
        ].copy()
        threshold, metric = v16.threshold_under_fpr_cap(
            validation.label, validation.score_lower, v16.FPR_CAP,
        )
        meta = frame[
            (frame.split == "val")
            & (frame.condition == condition)
            & (frame.method == "compact_raw")
        ]
        coverage = float(meta.coverage.mean())
        relative_criticality = float(meta.relative_criticality.mean())
        forced_reason = {
            "clean": "pilot_locked_clean_baseline",
            "targeted_0.30": "pilot_locked_high_criticality_fallback",
            "targeted_0.50": "pilot_locked_high_criticality_fallback",
        }.get(condition, "pilot_locked_validation_selection")
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
        rows.append({
            "condition": condition,
            "method": method,
            "coverage": coverage,
            "relative_criticality": relative_criticality,
            "forced_reason": forced_reason,
            "threshold": float(threshold),
            **{f"validation_{key}": value for key, value in metric.items()},
        })
    return decisions, pd.DataFrame(rows)


def run_seed(args, seed: int, run_dir: Path):
    required = [
        run_dir / "summary.csv",
        run_dir / "paired_effects.csv",
        run_dir / "decision.json",
        run_dir / "policy.json",
    ]
    if all(path.exists() for path in required):
        print(f"SKIP complete seed={seed} output={run_dir}", flush=True)
        return

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            str(Path(v16.__file__).resolve()),
            "--input-root", str(args.input_root),
            "--output-dir", str(run_dir),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--raw-window", str(args.raw_window),
            "--window-stride", str(args.window_stride),
            "--downsample", str(args.downsample),
            "--hidden1", str(args.hidden1),
            "--hidden2", str(args.hidden2),
            "--latent", str(args.latent),
            "--recovery-hidden", str(args.recovery_hidden),
            "--recovery-weight", str(args.recovery_weight),
            "--consistency-weight", str(args.consistency_weight),
        ]
        print(f"\n=== RUN SEED {seed} ===", flush=True)
        v16.main()
    finally:
        sys.argv = original_argv


def aggregate(output: Path, seeds: list[int]):
    summaries, effects, metrics, policies = [], [], [], []
    for seed in seeds:
        run_dir = output / f"seed_{seed}"
        summary = pd.read_csv(run_dir / "summary.csv")
        effect = pd.read_csv(run_dir / "paired_effects.csv")
        metric = pd.read_csv(run_dir / "metrics.csv")
        policy = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
        summary["seed"] = seed
        effect["seed"] = seed
        metric["seed"] = seed
        summaries.append(summary)
        effects.append(effect)
        metrics.append(metric)
        for condition, values in policy.items():
            policies.append({"seed": seed, "condition": condition, **values})

    per_seed_summary = pd.concat(summaries, ignore_index=True)
    per_seed_effects = pd.concat(effects, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    policy_frame = pd.DataFrame(policies)

    aggregate_summary = per_seed_summary.groupby(
        ["method", "condition"], as_index=False,
    ).agg(
        auc_mean=("auc_mean", "mean"),
        auc_seed_std=("auc_mean", "std"),
        partial_auc_mean=("partial_auc_fpr_0_20", "mean"),
        partial_auc_seed_std=("partial_auc_fpr_0_20", "std"),
        matched_recall_mean=("matched_recall_fpr_1_over_6", "mean"),
        matched_recall_seed_std=("matched_recall_fpr_1_over_6", "std"),
        ba_mean=("ba_mean", "mean"),
        ba_seed_std=("ba_mean", "std"),
        recall_mean=("recall_mean", "mean"),
        recall_seed_std=("recall_mean", "std"),
        specificity_mean=("specificity_mean", "mean"),
        fpr_mean=("fpr_mean", "mean"),
        f1_mean=("f1_mean", "mean"),
        seeds=("seed", "nunique"),
    )

    aggregate_effects = per_seed_effects.groupby(
        ["condition", "selected_method"], as_index=False,
    ).agg(
        delta_auc_mean=("delta_auc", "mean"),
        delta_auc_seed_std=("delta_auc", "std"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_ba_seed_std=("delta_ba", "std"),
        delta_recall_mean=("delta_recall", "mean"),
        delta_recall_seed_std=("delta_recall", "std"),
        delta_fpr_mean=("delta_fpr", "mean"),
        delta_partial_auc_mean=("delta_partial_auc_fpr_0_20", "mean"),
        delta_matched_recall_mean=(
            "delta_matched_recall_fpr_1_over_6", "mean",
        ),
        positive_auc_seeds=("delta_auc", lambda values: int((values > 0).sum())),
        nonnegative_ba_seeds=("delta_ba", lambda values: int((values >= 0).sum())),
        seeds=("seed", "nunique"),
    )

    random_rows = aggregate_effects[
        aggregate_effects.condition.isin(
            ["sensor_0.10", "sensor_0.30", "sensor_0.50"]
        )
    ]
    sensor50 = aggregate_effects[
        aggregate_effects.condition == "sensor_0.50"
    ].iloc[0]
    targeted50 = aggregate_effects[
        aggregate_effects.condition == "targeted_0.50"
    ].iloc[0]
    stable = bool(
        float(random_rows.delta_auc_mean.mean()) > 0.01
        and float(random_rows.delta_ba_mean.mean()) >= 0.0
        and float(sensor50.delta_auc_mean) >= 0.02
        and float(sensor50.delta_ba_mean) >= 0.0
        and int(sensor50.positive_auc_seeds) >= 2
        and float(targeted50.delta_auc_mean) >= -0.005
        and float(targeted50.delta_ba_mean) >= 0.0
    )
    decision = {
        "seeds": seeds,
        "locked_branches": LOCKED_BRANCHES,
        "mean_random_loss_delta_auc": float(random_rows.delta_auc_mean.mean()),
        "mean_random_loss_delta_ba": float(random_rows.delta_ba_mean.mean()),
        "sensor_0.50_positive_auc_seeds": int(sensor50.positive_auc_seeds),
        "stable_three_seed_result": stable,
        "decision": (
            "Three-seed confirmation passed; proceed to final 10 seeds and paired bootstrap."
            if stable else
            "Three-seed confirmation failed; do not expand to 10 seeds."
        ),
    }

    per_seed_summary.to_csv(output / "per_seed_summary.csv", index=False)
    per_seed_effects.to_csv(output / "per_seed_effects.csv", index=False)
    all_metrics.to_csv(output / "all_seed_metrics.csv", index=False)
    policy_frame.to_csv(output / "per_seed_thresholds.csv", index=False)
    aggregate_summary.to_csv(output / "three_seed_summary.csv", index=False)
    aggregate_effects.to_csv(output / "three_seed_effects.csv", index=False)
    (output / "three_seed_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    pivot = aggregate_summary.pivot(
        index="condition", columns="method", values="ba_mean",
    )
    axis = pivot.plot.bar(figsize=(12, 5.5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Balanced accuracy, mean over three seeds")
    axis.set_title("Locked MDR-SHM V16 three-seed confirmation")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v16_three_seed_ba.png", dpi=220)
    plt.close(axis.figure)

    print("\n=== THREE-SEED SUMMARY ===", flush=True)
    print(aggregate_summary.to_string(index=False), flush=True)
    print("\n=== THREE-SEED EFFECTS ===", flush=True)
    print(aggregate_effects.to_string(index=False), flush=True)
    print("\n=== THREE-SEED DECISION ===", flush=True)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[20261113, 20261129, 20261207],
    )
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
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) != 3:
        raise ValueError(f"Exactly three unique seeds are required, got {seeds}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Replace data-driven branch selection with the pilot-locked map.  The V16
    # evaluation still recalibrates each threshold from validation per seed.
    v16.select_policy = select_locked_policy
    for seed in seeds:
        run_seed(args, seed, args.output_dir / f"seed_{seed}")
    aggregate(args.output_dir, seeds)
    print(f"MDR_V16_THREE_SEED_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

