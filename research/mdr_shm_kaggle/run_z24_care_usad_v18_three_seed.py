from __future__ import annotations

"""Three-seed confirmation of the frozen CARE-USAD V18 screening design."""

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

import run_z24_care_usad_v18 as v18  # noqa: E402


DEFAULT_SEEDS = (20270523, 20270607, 20270619)
PILOT_SEEDS = {20270511}


def run_seed(args, seed: int, output: Path):
    required = [
        output / "summary.csv", output / "metrics.csv",
        output / "effects_vs_source_usad.csv", output / "decision.json",
    ]
    if all(path.exists() for path in required):
        print(f"SKIP complete seed={seed} output={output}", flush=True)
        return
    original = sys.argv[:]
    try:
        sys.argv = [
            str(Path(v18.__file__).resolve()),
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
            "--fusion-weight", str(args.fusion_weight),
        ]
        print(f"\n=== RUN CONFIRMATION SEED {seed} ===", flush=True)
        v18.main()
    finally:
        sys.argv = original


def paired_mask_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "condition", "replicate"]
    baseline = metrics[metrics.method == "source_usad"].set_index(keys)
    proposed = metrics[metrics.method == "care_usad_v18"].set_index(keys)
    columns = ["auc", "ba", "recall", "fpr", "f1"]
    output = proposed[columns].subtract(baseline[columns])
    output.columns = [f"delta_{name}" for name in columns]
    return output.reset_index()


def aggregate(output: Path, seeds: list[int], bootstrap_samples: int):
    summaries, effects, metrics, policies = [], [], [], []
    for seed in seeds:
        current = output / f"seed_{seed}"
        for filename, collection in (
            ("summary.csv", summaries),
            ("effects_vs_source_usad.csv", effects),
            ("metrics.csv", metrics),
            ("validation_policy.csv", policies),
        ):
            frame = pd.read_csv(current / filename)
            frame["seed"] = seed
            collection.append(frame)
    per_seed_summary = pd.concat(summaries, ignore_index=True)
    per_seed_effects = pd.concat(effects, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_policies = pd.concat(policies, ignore_index=True)
    mask_effects = paired_mask_effects(all_metrics)

    final_summary = per_seed_summary.groupby(
        ["method", "condition"], as_index=False,
    ).agg(
        auc_mean=("auc_mean", "mean"), auc_seed_std=("auc_mean", "std"),
        ap_mean=("ap_mean", "mean"),
        ba_mean=("ba_mean", "mean"), ba_seed_std=("ba_mean", "std"),
        recall_mean=("recall_mean", "mean"),
        recall_seed_std=("recall_mean", "std"),
        specificity_mean=("specificity_mean", "mean"),
        fpr_mean=("fpr_mean", "mean"), f1_mean=("f1_mean", "mean"),
        parameters=("parameters", "mean"), seeds=("seed", "nunique"),
    )
    final_effects = per_seed_effects.groupby("condition", as_index=False).agg(
        delta_auc_mean=("delta_auc", "mean"),
        delta_auc_seed_std=("delta_auc", "std"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_ba_seed_std=("delta_ba", "std"),
        delta_recall_mean=("delta_recall", "mean"),
        delta_recall_seed_std=("delta_recall", "std"),
        delta_fpr_mean=("delta_fpr", "mean"),
        positive_auc_seeds=("delta_auc", lambda x: int((x > 0).sum())),
        positive_ba_seeds=("delta_ba", lambda x: int((x > 0).sum())),
        seeds=("seed", "nunique"),
    )

    rng = np.random.default_rng(20270630)
    bootstrap_rows = []
    for condition, current in per_seed_effects.groupby("condition", sort=False):
        for column in ("delta_auc", "delta_ba", "delta_recall", "delta_fpr"):
            values = current.sort_values("seed")[column].to_numpy(float)
            draws = rng.choice(
                values, size=(bootstrap_samples, len(values)), replace=True,
            ).mean(1)
            bootstrap_rows.append({
                "condition": condition, "metric": column,
                "estimate": float(values.mean()),
                "ci_lower_exploratory": float(np.quantile(draws, 0.025)),
                "ci_upper_exploratory": float(np.quantile(draws, 0.975)),
                "paired_seeds": len(values),
                "warning": "exploratory only; n=3 seeds",
            })
    bootstrap = pd.DataFrame(bootstrap_rows)

    random_masks = mask_effects[mask_effects.condition == "sensor_0.50"]
    mask_summary = pd.DataFrame([{
        "paired_masks": len(random_masks),
        "positive_auc_masks": int((random_masks.delta_auc > 0).sum()),
        "nonnegative_auc_masks": int((random_masks.delta_auc >= 0).sum()),
        "positive_ba_masks": int((random_masks.delta_ba > 0).sum()),
        "positive_recall_masks": int((random_masks.delta_recall > 0).sum()),
        "delta_auc_mean": float(random_masks.delta_auc.mean()),
        "delta_auc_std": float(random_masks.delta_auc.std()),
        "delta_ba_mean": float(random_masks.delta_ba.mean()),
        "delta_recall_mean": float(random_masks.delta_recall.mean()),
        "delta_fpr_mean": float(random_masks.delta_fpr.mean()),
    }])

    indexed = final_effects.set_index("condition")
    random = indexed.loc["sensor_0.50"]
    targeted = indexed.loc["targeted_0.50"]
    clean = indexed.loc["clean"]
    decision = {
        "three_seed_confirmation_passed": bool(
            random.delta_auc_mean > 0.02
            and random.delta_ba_mean > 0.08
            and random.delta_recall_mean > 0.15
            and int(random.positive_auc_seeds) >= 2
            and int(mask_summary.positive_auc_masks.iloc[0]) >= 10
            and targeted.delta_auc_mean >= 0.0
            and targeted.delta_ba_mean >= 0.0
            and clean.delta_auc_mean == 0.0
            and clean.delta_ba_mean == 0.0
        ),
        "pre_specified_criteria": {
            "random50_mean_delta_auc": "> 0.02",
            "random50_mean_delta_ba": "> 0.08",
            "random50_mean_delta_recall": "> 0.15",
            "positive_auc_seeds": ">= 2/3",
            "positive_auc_masks": ">= 10/15",
            "targeted50_auc_ba": "nonnegative",
            "clean_auc_ba": "exactly preserved",
        },
    }

    per_seed_summary.to_csv(output / "per_seed_summary.csv", index=False)
    per_seed_effects.to_csv(output / "per_seed_effects.csv", index=False)
    all_metrics.to_csv(output / "all_metrics.csv", index=False)
    all_policies.to_csv(output / "all_validation_policies.csv", index=False)
    mask_effects.to_csv(output / "paired_mask_effects.csv", index=False)
    final_summary.to_csv(output / "three_seed_summary.csv", index=False)
    final_effects.to_csv(output / "three_seed_effects.csv", index=False)
    mask_summary.to_csv(output / "random50_mask_summary.csv", index=False)
    bootstrap.to_csv(output / "exploratory_seed_bootstrap.csv", index=False)
    (output / "three_seed_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    pivot = final_summary.pivot(
        index="condition", columns="method", values="ba_mean",
    )
    axis = pivot.plot.bar(figsize=(10, 5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Balanced accuracy, mean over three seeds")
    axis.set_title("CARE-USAD V18 three-seed confirmation")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "care_usad_v18_three_seed.png", dpi=220)
    plt.close(axis.figure)

    print("\n=== THREE-SEED SUMMARY ===")
    print(final_summary.to_string(index=False))
    print("\n=== THREE-SEED EFFECTS ===")
    print(final_effects.to_string(index=False))
    print("\n=== RANDOM-50 PAIRED MASK SUMMARY ===")
    print(mask_summary.to_string(index=False))
    print("\n=== DECISION ===")
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hard-candidates", type=int, default=3)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) != 3:
        raise ValueError(f"Exactly three unique seeds required, got {seeds}")
    overlap = sorted(set(seeds) & PILOT_SEEDS)
    if overlap:
        raise ValueError(f"Confirmation seeds overlap screening pilot: {overlap}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        run_seed(args, seed, args.output_dir / f"seed_{seed}")
    aggregate(args.output_dir, seeds, args.bootstrap_samples)
    print(f"CARE_USAD_V18_THREE_SEED_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

