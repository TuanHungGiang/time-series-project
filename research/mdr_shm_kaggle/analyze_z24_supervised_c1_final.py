from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


METRICS = ("auroc", "balanced_accuracy", "recall", "specificity")


def binary_metrics(frame: pd.DataFrame, score_col: str, threshold_col: str | float) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    score = frame[score_col].to_numpy(dtype=float)
    if isinstance(threshold_col, str):
        threshold = frame[threshold_col].to_numpy(dtype=float)
    else:
        threshold = float(threshold_col)
    pred = score >= threshold
    positive = y == 1
    negative = ~positive
    recall = float(pred[positive].mean()) if positive.any() else np.nan
    specificity = float((~pred[negative]).mean()) if negative.any() else np.nan
    return {
        "auroc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else np.nan,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2),
    }


def read_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tests, validations, selections = [], [], []
    for run_file in sorted(root.glob("seed_*/run.json")):
        run_dir = run_file.parent
        artifact = json.loads(run_file.read_text())
        seed = int(artifact["seed"])
        test = pd.read_csv(run_dir / "test_predictions.csv")
        validation_path = run_dir / "validation_predictions.csv"
        if not validation_path.exists():
            raise FileNotFoundError(
                f"{validation_path} is missing. Re-run with the updated C1 script."
            )
        validation = pd.read_csv(validation_path)
        for frame in (test, validation):
            frame["seed"] = seed
        tests.append(test)
        validations.append(validation)
        selections.append({
            "seed": seed,
            "selected_model": artifact["selected_model"],
            "mode": artifact["mode"],
        })
    if not tests:
        raise FileNotFoundError(f"No seed_*/run.json found below {root}")
    return pd.concat(tests, ignore_index=True), pd.concat(validations, ignore_index=True), pd.DataFrame(selections)


def raw_metric_tables(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in test.groupby(["seed", "model", "condition", "replicate"]):
        rows.append(dict(zip(("seed", "model", "condition", "replicate"), keys)) | binary_metrics(group, "score", "threshold"))
    replicate = pd.DataFrame(rows)
    per_seed = replicate.groupby(["seed", "model", "condition"], as_index=False)[list(METRICS)].mean()
    corruption = replicate.groupby(["seed", "model", "condition"], as_index=False)[list(METRICS)].std(ddof=1)
    corruption = corruption.rename(columns={metric: f"{metric}_corruption_std" for metric in METRICS})
    per_seed = per_seed.merge(corruption, on=["seed", "model", "condition"], how="left")
    summary = per_seed.groupby(["model", "condition"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_seed_std": (metric, "std") for metric in METRICS},
        **{f"{metric}_corruption_std_mean": (f"{metric}_corruption_std", "mean") for metric in METRICS},
        seeds=("seed", "nunique"),
    )
    return replicate, summary


def threshold_for_specificity(validation: pd.DataFrame, target: float) -> float:
    healthy = validation.loc[validation.label == 0, "score"].to_numpy(dtype=float)
    candidates = np.r_[np.nextafter(healthy.min(), -np.inf), np.unique(healthy), np.nextafter(healthy.max(), np.inf)]
    for threshold in candidates:
        specificity = float(np.mean(healthy < threshold))
        if specificity >= target:
            return float(threshold)
    return float(np.nextafter(healthy.max(), np.inf))


def matched_specificity(test: pd.DataFrame, validation: pd.DataFrame, target: float) -> pd.DataFrame:
    rows = []
    for (seed, model), val_group in validation.groupby(["seed", "model"]):
        threshold = threshold_for_specificity(val_group, target)
        selected = test[(test.seed == seed) & (test.model == model)]
        for (condition, replicate), group in selected.groupby(["condition", "replicate"]):
            rows.append({
                "seed": seed, "model": model, "condition": condition,
                "replicate": int(replicate), "matched_threshold": threshold,
                "validation_specificity_target": target,
                **binary_metrics(group, "score", threshold),
            })
    return pd.DataFrame(rows)


def bootstrap_metric_arrays(
    frame: pd.DataFrame,
    positive_ids: np.ndarray,
    negative_ids: np.ndarray,
    positive_draw: np.ndarray,
    negative_draw: np.ndarray,
) -> dict[str, np.ndarray]:
    by_replicate = []
    for _, group in frame.groupby("replicate"):
        indexed = group.set_index("record_id")
        positive_score = indexed.loc[positive_ids, "score"].to_numpy()[positive_draw]
        negative_score = indexed.loc[negative_ids, "score"].to_numpy()[negative_draw]
        positive_threshold = indexed.loc[positive_ids, "threshold"].to_numpy()[positive_draw]
        negative_threshold = indexed.loc[negative_ids, "threshold"].to_numpy()[negative_draw]
        comparison = positive_score[:, :, None] - negative_score[:, None, :]
        auroc = np.mean((comparison > 0) + 0.5 * (comparison == 0), axis=(1, 2))
        recall = np.mean(positive_score >= positive_threshold, axis=1)
        specificity = np.mean(negative_score < negative_threshold, axis=1)
        by_replicate.append({
            "auroc": auroc, "recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
        })
    return {
        metric: np.mean([item[metric] for item in by_replicate], axis=0)
        for metric in METRICS
    }


def observed_condition_metrics(frame: pd.DataFrame) -> dict[str, float]:
    values = [binary_metrics(group, "score", "threshold") for _, group in frame.groupby("replicate")]
    return {metric: float(np.mean([item[metric] for item in values])) for metric in METRICS}


def paired_bootstrap(test: pd.DataFrame, samples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    comparisons = (("consistency_linear", "clean_logistic"), ("augmented_logistic", "clean_logistic"), ("consistency_linear", "augmented_logistic"))
    for (seed, condition), cohort in test.groupby(["seed", "condition"]):
        for candidate, baseline in comparisons:
            a = cohort[cohort.model == candidate]
            b = cohort[cohort.model == baseline]
            if a.empty or b.empty:
                continue
            labels = a[["record_id", "label"]].drop_duplicates()
            positive_ids = labels.loc[labels.label == 1, "record_id"].to_numpy()
            negative_ids = labels.loc[labels.label == 0, "record_id"].to_numpy()
            positive_draw = rng.integers(0, len(positive_ids), size=(samples, len(positive_ids)))
            negative_draw = rng.integers(0, len(negative_ids), size=(samples, len(negative_ids)))
            am = bootstrap_metric_arrays(a, positive_ids, negative_ids, positive_draw, negative_draw)
            bm = bootstrap_metric_arrays(b, positive_ids, negative_ids, positive_draw, negative_draw)
            observed_a = observed_condition_metrics(a)
            observed_b = observed_condition_metrics(b)
            for metric in METRICS:
                values = am[metric] - bm[metric]
                rows.append({
                    "seed": seed, "condition": condition, "candidate": candidate,
                    "baseline": baseline, "metric": metric,
                    "delta": observed_a[metric] - observed_b[metric],
                    "ci_lower": float(np.nanquantile(values, 0.025)),
                    "ci_upper": float(np.nanquantile(values, 0.975)),
                    "probability_candidate_better": float(np.nanmean(values > 0)),
                    "bootstrap_samples": samples,
                })
    return pd.DataFrame(rows)


def score_shift(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = test[test.condition == "clean"][["seed", "model", "record_id", "score", "prediction"]].rename(
        columns={"score": "clean_score", "prediction": "clean_prediction"}
    )
    corrupted = test[test.condition != "clean"].merge(clean, on=["seed", "model", "record_id"], validate="many_to_one")
    corrupted["score_delta"] = corrupted.score - corrupted.clean_score
    corrupted["tp_to_fn"] = ((corrupted.label == 1) & (corrupted.clean_prediction == 1) & (corrupted.prediction == 0)).astype(int)
    corrupted["tn_to_fp"] = ((corrupted.label == 0) & (corrupted.clean_prediction == 0) & (corrupted.prediction == 1)).astype(int)
    summary = corrupted.groupby(["model", "condition", "label", "excitation"], as_index=False).agg(
        score_delta_mean=("score_delta", "mean"), score_delta_median=("score_delta", "median"),
        score_delta_std=("score_delta", "std"), fraction_score_decreased=("score_delta", lambda x: float(np.mean(x < 0))),
        tp_to_fn_rate=("tp_to_fn", "mean"), tn_to_fp_rate=("tn_to_fp", "mean"), observations=("record_id", "size"),
    )
    return corrupted, summary


def subgroup_diagnostic(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in test.groupby(["seed", "model", "condition", "replicate", "state", "excitation"]):
        metrics = binary_metrics(group, "score", "threshold")
        rows.append(dict(zip(("seed", "model", "condition", "replicate", "state", "excitation"), keys)) | {
            "records": len(group), "damage_records": int(group.label.sum()),
            "recall": metrics["recall"], "specificity": metrics["specificity"],
        })
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    conditions = ["clean", "dropout_0.30", "missing_0.30", "noise_5", "targeted_0.30", "targeted_0.50"]
    models = ["clean_logistic", "augmented_logistic", "consistency_linear"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    x = np.arange(len(conditions))
    for model in models:
        selected = summary[summary.model == model].set_index("condition").reindex(conditions)
        for axis, metric in zip(axes, ("auroc", "balanced_accuracy")):
            axis.errorbar(x, selected[f"{metric}_mean"], yerr=selected[f"{metric}_seed_std"].fillna(0), marker="o", capsize=3, label=model)
            axis.set_title(metric.replace("_", " ").upper())
            axis.set_ylim(0.45, 1.01)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, [value.replace("_", "\n") for value in conditions], rotation=20)
    axes[0].set_ylabel("Metric (mean ± seed SD)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--matched-specificity", type=float, default=0.875)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test, validation, selections = read_runs(args.runs_root)
    replicate, summary = raw_metric_tables(test)
    matched = matched_specificity(test, validation, args.matched_specificity)
    bootstrap = paired_bootstrap(test, args.bootstrap_samples, np.random.default_rng(args.seed))
    shifts, shift_summary = score_shift(test)
    subgroups = subgroup_diagnostic(test)
    selections.to_csv(args.output_dir / "model_selection_by_seed.csv", index=False)
    replicate.to_csv(args.output_dir / "replicate_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "detector_summary.csv", index=False)
    matched.to_csv(args.output_dir / "matched_specificity_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_differences.csv", index=False)
    shifts.to_csv(args.output_dir / "paired_score_shifts.csv", index=False)
    shift_summary.to_csv(args.output_dir / "score_shift_summary.csv", index=False)
    subgroups.to_csv(args.output_dir / "state_excitation_diagnostic.csv", index=False)
    make_figure(summary, args.output_dir / "figure_robustness.png")
    selected_counts = selections.selected_model.value_counts().to_dict()
    report = {
        "runs": int(selections.seed.nunique()), "selected_model_counts": selected_counts,
        "bootstrap_samples": args.bootstrap_samples,
        "matched_validation_specificity": args.matched_specificity,
        "warning": "Do not rename a test-winning model as selected; selection is validation-only.",
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    print("\n=== MODEL SELECTION ===")
    print(selections.to_string(index=False))
    print("\n=== DETECTOR SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PAIRED BOOTSTRAP: CLEAN ===")
    print(bootstrap[bootstrap.condition == "clean"].to_string(index=False))
    weak = subgroups[(subgroups.state == 9) & (subgroups.excitation == "fvt")]
    print("\n=== STATE 9 / FVT ===")
    print(weak.to_string(index=False))
    print(f"C1_FINAL_ANALYSIS_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

