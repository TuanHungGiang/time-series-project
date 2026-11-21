from __future__ import annotations

"""Ten-seed confirmation of the validation-locked CLMM-RS detector.

This runner invokes the already frozen V21 implementation for ten independent
training seeds.  Architecture, branch, tail, score weight, FPR cap and five
random-50 masks are unchanged.  Completed seeds and completed group files are
resumable.  The final analysis separates seed-level means from paired mask
effects and reports a hierarchical seed-then-mask bootstrap confidence
interval.  Targeted-50 is reported only as a stress test.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (
    20270901,
    20270913,
    20270929,
    20271011,
    20271027,
    20271109,
    20271123,
    20271207,
    20271219,
    20280103,
)
METRICS = ("auc", "ba", "recall", "fpr", "f1")


def complete_seed(path: Path):
    required = (
        "metrics.csv",
        "paired_effects.csv",
        "effect_summary.csv",
        "validation_policy.csv",
        "protocol.json",
    )
    return all((path / name).exists() for name in required)


def run_seed(args, seed: int):
    output = args.output_dir / f"seed_{seed}"
    if complete_seed(output):
        print(f"SKIP completed seed={seed} output={output}", flush=True)
        return
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "run_z24_clmm_v21.py"),
        "--input-root", str(args.input_root),
        "--output-dir", str(output),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--adapter-epochs", str(args.adapter_epochs),
        "--batch-size", str(args.batch_size),
        "--raw-window", str(args.raw_window),
        "--window-stride", str(args.window_stride),
        "--downsample", str(args.downsample),
        "--embedding", str(args.embedding),
        "--heads", str(args.heads),
        "--hard-candidates", str(args.hard_candidates),
        "--score-weight", str(args.score_weight),
        "--fpr-cap", str(args.fpr_cap),
    ]
    print(f"\n=== RUN CLMM-RS SEED {seed} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not complete_seed(output):
        raise RuntimeError(f"Seed {seed} ended without complete outputs")


def load_outputs(output_dir: Path, seeds):
    metrics, effects, policies = [], [], []
    for seed in seeds:
        root = output_dir / f"seed_{seed}"
        current = pd.read_csv(root / "metrics.csv")
        current.insert(0, "seed", seed)
        metrics.append(current)

        current = pd.read_csv(root / "paired_effects.csv")
        current.insert(0, "seed", seed)
        effects.append(current)

        current = pd.read_csv(root / "validation_policy.csv")
        current.insert(0, "seed", seed)
        policies.append(current)
    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(effects, ignore_index=True),
        pd.concat(policies, ignore_index=True),
    )


def performance_summary(metrics):
    return metrics.groupby(
        ["method", "condition"], as_index=False,
    ).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        ba_mean=("ba", "mean"),
        ba_std=("ba", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"),
        f1_mean=("f1", "mean"),
        observations=("replicate", "count"),
        seeds=("seed", "nunique"),
    )


def per_seed_effects(effects):
    columns = {f"delta_{metric}": ["mean", "median"] for metric in METRICS}
    summary = effects.groupby(
        ["seed", "condition"], as_index=False,
    ).agg(columns)
    summary.columns = [
        "seed" if column[0] == "seed" else
        "condition" if column[0] == "condition" else
        f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return summary


def effect_summary(effects, seed_effects):
    rows = []
    for condition, current in effects.groupby("condition"):
        seeds = seed_effects[seed_effects.condition == condition]
        row = {
            "condition": condition,
            "paired_masks": len(current),
            "seeds": current.seed.nunique(),
            "positive_auc_masks": int((current.delta_auc > 0).sum()),
            "nonnegative_auc_masks": int((current.delta_auc >= 0).sum()),
            "positive_ba_masks": int((current.delta_ba > 0).sum()),
            "positive_recall_masks": int((current.delta_recall > 0).sum()),
            "positive_auc_seeds": int((seeds.delta_auc_mean > 0).sum()),
            "positive_ba_seeds": int((seeds.delta_ba_mean > 0).sum()),
        }
        for metric in METRICS:
            values = current[f"delta_{metric}"].to_numpy()
            seed_values = seeds[f"delta_{metric}_mean"].to_numpy()
            row[f"delta_{metric}_mask_mean"] = float(np.mean(values))
            row[f"delta_{metric}_mask_median"] = float(np.median(values))
            row[f"delta_{metric}_seed_mean"] = float(np.mean(seed_values))
            row[f"delta_{metric}_seed_std"] = float(np.std(seed_values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchical_bootstrap(effects, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for condition, current in effects.groupby("condition"):
        seed_ids = np.asarray(sorted(current.seed.unique()))
        estimates = {metric: np.empty(samples) for metric in METRICS}
        by_seed = {
            value: current[current.seed == value]
            for value in seed_ids
        }
        for index in range(samples):
            sampled_seeds = rng.choice(seed_ids, len(seed_ids), replace=True)
            for metric in METRICS:
                values = []
                for sampled_seed in sampled_seeds:
                    cohort = by_seed[sampled_seed][f"delta_{metric}"].to_numpy()
                    sampled_masks = rng.choice(cohort, len(cohort), replace=True)
                    values.extend(sampled_masks.tolist())
                estimates[metric][index] = np.mean(values)
        for metric, values in estimates.items():
            rows.append({
                "condition": condition,
                "metric": f"delta_{metric}",
                "estimate": float(current[f"delta_{metric}"].mean()),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_samples": samples,
                "bootstrap_unit": "seed_then_mask",
            })
    return pd.DataFrame(rows)


def finish(args):
    metrics, effects, policies = load_outputs(args.output_dir, args.seeds)
    summary = performance_summary(metrics)
    seed_effects = per_seed_effects(effects)
    effects_summary = effect_summary(effects, seed_effects)
    bootstrap = hierarchical_bootstrap(
        effects, args.bootstrap_samples, args.analysis_seed,
    )

    random_row = effects_summary[
        effects_summary.condition == "sensor_0.50"
    ].iloc[0]
    random_ci = bootstrap[
        bootstrap.condition.eq("sensor_0.50")
    ].set_index("metric")
    clean = effects_summary[
        effects_summary.condition == "clean"
    ].iloc[0]
    criteria = {
        "clean_auc_preserved": bool(abs(clean.delta_auc_seed_mean) < 1e-12),
        "clean_ba_preserved": bool(abs(clean.delta_ba_seed_mean) < 1e-12),
        "random50_delta_auc_mean_above_0_02": bool(
            random_row.delta_auc_seed_mean > 0.02
        ),
        "random50_delta_ba_mean_above_0_10": bool(
            random_row.delta_ba_seed_mean > 0.10
        ),
        "random50_delta_recall_mean_above_0_20": bool(
            random_row.delta_recall_seed_mean > 0.20
        ),
        "random50_delta_fpr_nonpositive": bool(
            random_row.delta_fpr_seed_mean <= 0.0
        ),
        "random50_positive_auc_masks_at_least_30_of_50": bool(
            random_row.positive_auc_masks >= 30
        ),
        "random50_positive_ba_masks_at_least_40_of_50": bool(
            random_row.positive_ba_masks >= 40
        ),
        "random50_positive_auc_seeds_at_least_7_of_10": bool(
            random_row.positive_auc_seeds >= 7
        ),
        "random50_auc_ci_lower_above_zero": bool(
            random_ci.loc["delta_auc", "ci_lower"] > 0.0
        ),
        "random50_ba_ci_lower_above_zero": bool(
            random_ci.loc["delta_ba", "ci_lower"] > 0.0
        ),
    }
    decision = {
        "ten_seed_confirmation_passed": bool(all(criteria.values())),
        "criteria": criteria,
        "model": "CLMM-RS frozen from V22 validation selection",
        "seeds": list(args.seeds),
        "random_masks_per_seed": 5,
        "architecture_reselected": False,
        "test_used_for_selection": False,
        "targeted50_role": "stress_test_only",
    }

    metrics.to_csv(args.output_dir / "all_metrics.csv", index=False)
    effects.to_csv(args.output_dir / "all_paired_effects.csv", index=False)
    policies.to_csv(args.output_dir / "all_validation_policies.csv", index=False)
    summary.to_csv(args.output_dir / "ten_seed_performance.csv", index=False)
    seed_effects.to_csv(args.output_dir / "per_seed_effects.csv", index=False)
    effects_summary.to_csv(args.output_dir / "ten_seed_effects.csv", index=False)
    bootstrap.to_csv(
        args.output_dir / "hierarchical_bootstrap_ci.csv", index=False,
    )
    (args.output_dir / "ten_seed_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    random_seed = seed_effects[
        seed_effects.condition == "sensor_0.50"
    ].sort_values("seed")
    axis = random_seed.plot(
        x="seed",
        y=["delta_auc_mean", "delta_ba_mean", "delta_recall_mean"],
        marker="o",
        figsize=(11, 5),
        grid=True,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("CLMM-RS minus Source-USAD")
    axis.set_title("CLMM-RS random-50 effects across ten seeds")
    axis.figure.tight_layout()
    axis.figure.savefig(
        args.output_dir / "clmm_v23_ten_seed.png", dpi=220,
    )
    plt.close(axis.figure)

    print("\n=== TEN-SEED PERFORMANCE ===")
    print(summary.to_string(index=False))
    print("\n=== PER-SEED EFFECTS ===")
    print(seed_effects.to_string(index=False))
    print("\n=== TEN-SEED EFFECT SUMMARY ===")
    print(effects_summary.to_string(index=False))
    print("\n=== HIERARCHICAL BOOTSTRAP CI ===")
    print(bootstrap.to_string(index=False))
    print("\n=== TEN-SEED DECISION ===")
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hard-candidates", type=int, default=4)
    parser.add_argument("--score-weight", type=float, default=0.10)
    parser.add_argument("--fpr-cap", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--analysis-seed", type=int, default=20261117)
    args = parser.parse_args()

    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("Exactly ten unique seeds are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"CLMM-RS ten-seed confirmation seeds={args.seeds}",
        flush=True,
    )
    for seed in args.seeds:
        run_seed(args, seed)
    finish(args)
    protocol = {
        "dataset": "Z24",
        "method": "CLMM-RS V23 ten-seed confirmation",
        "frozen_from": "V22 validation-only selection",
        "seeds": list(args.seeds),
        "random_masks_per_seed": 5,
        "training_labels": "healthy-only",
        "fixed_branch": "imputed_usad",
        "fixed_tail": "lower",
        "score_weight": args.score_weight,
        "threshold_calibration": "labelled validation under FPR cap",
        "fpr_cap": args.fpr_cap,
        "bootstrap": "hierarchical seed then mask",
        "test_used_for_selection": False,
        "targeted50_role": "stress_test_only",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"CLMM_V23_TEN_SEED_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

