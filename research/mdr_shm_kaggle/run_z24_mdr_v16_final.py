from __future__ import annotations

"""Final locked Z24 evidence package for MDR-SHM V16.

Runs exactly ten seeds, reusing the frozen V16 branch map.  Produces:

* seed-level and aggregate test metrics;
* paired seed-bootstrap confidence intervals;
* paired Wilcoxon tests across seed-level effects;
* an ablation of Compact USAD, mask-aware recovery, criticality fusion, and
  the locked V16 policy under the same validation FPR constraint.

No branch, hyperparameter, corruption mask, or policy is selected from test.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_mdr_v14 as v14  # noqa: E402
import run_z24_mdr_v15 as v15  # noqa: E402
import run_z24_mdr_v16 as v16  # noqa: E402
import run_z24_mdr_v16_three_seed as pilot  # noqa: E402
from run_hellbridge_source_baselines import publication_metrics  # noqa: E402


DEFAULT_SEEDS = [
    20261113, 20261129, 20261207, 20261219, 20270103,
    20270117, 20270131, 20270213, 20270227, 20270313,
]
DELTA_COLUMNS = [
    "delta_auc", "delta_ba", "delta_recall", "delta_fpr",
    "delta_partial_auc_fpr_0_20", "delta_matched_recall_fpr_1_over_6",
]


def load_seed_outputs(output: Path, seeds: list[int]):
    summaries, effects, metrics, thresholds = [], [], [], []
    for seed in seeds:
        run_dir = output / f"seed_{seed}"
        summary = pd.read_csv(run_dir / "summary.csv")
        effect = pd.read_csv(run_dir / "paired_effects.csv")
        metric = pd.read_csv(run_dir / "metrics.csv")
        policy = pd.read_csv(run_dir / "validation_policy_audit.csv")
        for frame in (summary, effect, metric, policy):
            frame["seed"] = seed
        summaries.append(summary)
        effects.append(effect)
        metrics.append(metric)
        thresholds.append(policy)
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(effects, ignore_index=True),
        pd.concat(metrics, ignore_index=True),
        pd.concat(thresholds, ignore_index=True),
    )


def aggregate_final(per_seed_summary, per_seed_effects):
    summary = per_seed_summary.groupby(
        ["method", "condition"], as_index=False,
    ).agg(
        auc_mean=("auc_mean", "mean"), auc_seed_std=("auc_mean", "std"),
        partial_auc_mean=("partial_auc_fpr_0_20", "mean"),
        partial_auc_seed_std=("partial_auc_fpr_0_20", "std"),
        matched_recall_mean=("matched_recall_fpr_1_over_6", "mean"),
        matched_recall_seed_std=("matched_recall_fpr_1_over_6", "std"),
        ba_mean=("ba_mean", "mean"), ba_seed_std=("ba_mean", "std"),
        recall_mean=("recall_mean", "mean"),
        recall_seed_std=("recall_mean", "std"),
        specificity_mean=("specificity_mean", "mean"),
        fpr_mean=("fpr_mean", "mean"), f1_mean=("f1_mean", "mean"),
        seeds=("seed", "nunique"),
    )
    effects = per_seed_effects.groupby(
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
        positive_auc_seeds=("delta_auc", lambda x: int((x > 0).sum())),
        nonnegative_ba_seeds=("delta_ba", lambda x: int((x >= 0).sum())),
        seeds=("seed", "nunique"),
    )
    return summary, effects


def paired_seed_bootstrap(
    per_seed_effects: pd.DataFrame,
    samples: int,
    seed: int = 20270401,
):
    rng = np.random.default_rng(seed)
    rows = []
    for condition, current in per_seed_effects.groupby("condition", sort=False):
        current = current.sort_values("seed")
        for column in DELTA_COLUMNS:
            values = current[column].to_numpy(dtype=float)
            draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(1)
            rows.append({
                "condition": condition,
                "metric": column,
                "estimate": float(values.mean()),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "bootstrap_samples": samples,
                "paired_units": len(values),
                "unit": "training_seed",
            })
    return pd.DataFrame(rows)


def paired_tests(per_seed_effects: pd.DataFrame):
    rows = []
    for condition, current in per_seed_effects.groupby("condition", sort=False):
        for column in DELTA_COLUMNS:
            values = current[column].to_numpy(dtype=float)
            if np.allclose(values, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    values, alternative="two-sided", zero_method="wilcox",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append({
                "condition": condition,
                "metric": column,
                "mean_difference": float(values.mean()),
                "wilcoxon_statistic": statistic,
                "p_value_two_sided": p_value,
                "paired_seeds": len(values),
            })
    table = pd.DataFrame(rows)
    # Family-wise error control across all condition/metric confirmations.
    order = np.argsort(table.p_value_two_sided.to_numpy(dtype=float))
    raw = table.p_value_two_sided.to_numpy(dtype=float)[order]
    adjusted = np.minimum(
        1.0,
        np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw),
    )
    table["p_value_holm"] = np.nan
    table.loc[table.index[order], "p_value_holm"] = adjusted
    return table


def evaluate_candidate(cohort, threshold):
    matched, realized = v15.matched_recall(
        cohort.label, cohort.score_lower, v16.FPR_CAP,
    )
    return {
        **publication_metrics(cohort.label, cohort.score_lower, threshold),
        "partial_auc_fpr_0_20": v15.low_fpr_auc(
            cohort.label, cohort.score_lower,
        ),
        "matched_recall_fpr_1_over_6": matched,
        "matched_realized_fpr": realized,
    }


def ablation_for_seed(run_dir: Path, seed: int):
    paths = sorted((run_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 group files for seed {seed}, got {len(paths)}")
    base = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = v14.add_criticality_fusion(base)
    rows = []
    for method in v14.ALL_METHODS:
        for condition in v16.CONDITIONS:
            validation = frame[
                (frame.split == "val")
                & (frame.condition == condition)
                & (frame.method == method)
            ]
            threshold, _ = v16.threshold_under_fpr_cap(
                validation.label, validation.score_lower, v16.FPR_CAP,
            )
            test = frame[
                (frame.split == "test")
                & (frame.condition == condition)
                & (frame.method == method)
            ]
            replicate_rows = []
            for _, cohort in test.groupby("replicate", sort=False):
                replicate_rows.append(evaluate_candidate(cohort, threshold))
            metric = pd.DataFrame(replicate_rows).mean(numeric_only=True)
            rows.append({
                "seed": seed, "method": method, "condition": condition,
                "threshold": threshold, **metric.to_dict(),
            })

    # Add the locked policy from the already evaluated seed output.
    policy_summary = pd.read_csv(run_dir / "summary.csv")
    for row in policy_summary[policy_summary.method == "v16_policy"].itertuples():
        rows.append({
            "seed": seed, "method": "v16_policy", "condition": row.condition,
            "threshold": float("nan"), "auc": row.auc_mean, "ap": row.ap_mean,
            "ba": row.ba_mean, "recall": row.recall_mean,
            "specificity": row.specificity_mean, "fpr": row.fpr_mean,
            "precision": row.precision_mean, "f1": row.f1_mean,
            "partial_auc_fpr_0_20": row.partial_auc_fpr_0_20,
            "matched_recall_fpr_1_over_6": row.matched_recall_fpr_1_over_6,
            "matched_realized_fpr": row.matched_realized_fpr,
        })
    return pd.DataFrame(rows)


def make_ablation(output: Path, seeds: list[int]):
    per_seed = pd.concat([
        ablation_for_seed(output / f"seed_{seed}", seed) for seed in seeds
    ], ignore_index=True)
    summary = per_seed.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_seed_std=("auc", "std"),
        ba_mean=("ba", "mean"), ba_seed_std=("ba", "std"),
        recall_mean=("recall", "mean"), recall_seed_std=("recall", "std"),
        fpr_mean=("fpr", "mean"), f1_mean=("f1", "mean"),
        partial_auc_mean=("partial_auc_fpr_0_20", "mean"),
        matched_recall_mean=("matched_recall_fpr_1_over_6", "mean"),
        seeds=("seed", "nunique"),
    )
    return per_seed, summary


def final_decision(effects, bootstrap):
    table = effects.set_index("condition")
    ci = bootstrap[
        (bootstrap.metric == "delta_auc")
        & bootstrap.condition.isin(["sensor_0.30", "sensor_0.50"])
    ].set_index("condition")
    random_conditions = ["sensor_0.10", "sensor_0.30", "sensor_0.50"]
    passed = bool(
        float(table.loc[random_conditions, "delta_auc_mean"].mean()) > 0.01
        and float(table.loc[random_conditions, "delta_ba_mean"].mean()) >= 0.0
        and float(ci.loc["sensor_0.30", "ci_lower"]) > 0.0
        and float(ci.loc["sensor_0.50", "ci_lower"]) > 0.0
        and float(table.loc["targeted_0.50", "delta_auc_mean"]) >= -0.005
    )
    return {
        "locked_branches": pilot.LOCKED_BRANCHES,
        "ten_seed_confirmation_passed": passed,
        "decision": (
            "Z24 final confirmation passed. Freeze results and proceed to external validation."
            if passed else
            "Z24 final confirmation failed at least one pre-specified criterion."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
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
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) != 10:
        raise ValueError(f"Exactly ten unique seeds are required, got {seeds}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v16.select_policy = pilot.select_locked_policy
    for seed in seeds:
        pilot.run_seed(args, seed, args.output_dir / f"seed_{seed}")

    per_seed_summary, per_seed_effects, all_metrics, thresholds = load_seed_outputs(
        args.output_dir, seeds,
    )
    summary, effects = aggregate_final(per_seed_summary, per_seed_effects)
    bootstrap = paired_seed_bootstrap(
        per_seed_effects, args.bootstrap_samples,
    )
    tests = paired_tests(per_seed_effects)
    ablation_per_seed, ablation_summary = make_ablation(args.output_dir, seeds)
    decision = final_decision(effects, bootstrap)

    per_seed_summary.to_csv(args.output_dir / "per_seed_summary.csv", index=False)
    per_seed_effects.to_csv(args.output_dir / "per_seed_effects.csv", index=False)
    all_metrics.to_csv(args.output_dir / "all_seed_metrics.csv", index=False)
    thresholds.to_csv(args.output_dir / "per_seed_thresholds.csv", index=False)
    summary.to_csv(args.output_dir / "final_summary.csv", index=False)
    effects.to_csv(args.output_dir / "final_effects.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_ci.csv", index=False)
    tests.to_csv(args.output_dir / "paired_wilcoxon.csv", index=False)
    ablation_per_seed.to_csv(args.output_dir / "ablation_per_seed.csv", index=False)
    ablation_summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    (args.output_dir / "final_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    pivot = summary.pivot(index="condition", columns="method", values="ba_mean")
    axis = pivot.plot.bar(figsize=(12, 5.5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Balanced accuracy, mean over ten seeds")
    axis.set_title("MDR-SHM V16 final Z24 evaluation")
    axis.figure.tight_layout()
    axis.figure.savefig(args.output_dir / "v16_final_ba.png", dpi=220)
    plt.close(axis.figure)

    print("\n=== FINAL TEN-SEED SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== FINAL EFFECTS ===", flush=True)
    print(effects.to_string(index=False), flush=True)
    print("\n=== PAIRED BOOTSTRAP CI ===", flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    print("\n=== ABLATION SUMMARY ===", flush=True)
    print(ablation_summary.to_string(index=False), flush=True)
    print("\n=== FINAL DECISION ===", flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    print(f"MDR_V16_FINAL_Z24_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

