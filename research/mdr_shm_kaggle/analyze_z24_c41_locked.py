from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


CANDIDATE = "m1_modal_absolute"
BASELINE = "m0_c1_absolute"
PRIMARY_MODE = "robust_reference"
METRICS = ("auroc", "recall", "specificity", "balanced_accuracy")


def binary_metrics(labels, scores, threshold) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = scores >= float(threshold)
    positive, negative = labels == 1, labels == 0
    recall = float(predictions[positive].mean()) if positive.any() else np.nan
    specificity = float((~predictions[negative]).mean()) if negative.any() else np.nan
    return {
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2),
    }


def merge_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "record_id", "label", "state", "setup", "excitation",
        "condition", "replicate",
    ]
    candidate = predictions[
        (predictions.model == CANDIDATE)
        & (predictions.threshold_mode == PRIMARY_MODE)
    ][keys + ["score", "threshold"]].rename(columns={
        "score": "candidate_score", "threshold": "candidate_threshold"
    })
    baseline = predictions[
        (predictions.model == BASELINE)
        & (predictions.threshold_mode == PRIMARY_MODE)
    ][keys + ["score", "threshold"]].rename(columns={
        "score": "baseline_score", "threshold": "baseline_threshold"
    })
    merged = candidate.merge(baseline, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise RuntimeError(
            f"Unpaired predictions: candidate={len(candidate)}, "
            f"baseline={len(baseline)}, merged={len(merged)}"
        )
    return merged


def score_pair(frame: pd.DataFrame) -> tuple[dict, dict]:
    candidate = binary_metrics(
        frame.label,
        frame.candidate_score,
        frame.candidate_threshold.iloc[0],
    )
    baseline = binary_metrics(
        frame.label,
        frame.baseline_score,
        frame.baseline_threshold.iloc[0],
    )
    return candidate, baseline


def macro_condition_pair(frame: pd.DataFrame) -> tuple[dict, dict]:
    candidate_rows, baseline_rows = [], []
    for _, group in frame.groupby("condition"):
        candidate, baseline = score_pair(group)
        candidate_rows.append(candidate)
        baseline_rows.append(baseline)
    return (
        {metric: float(np.mean([row[metric] for row in candidate_rows])) for metric in METRICS},
        {metric: float(np.mean([row[metric] for row in baseline_rows])) for metric in METRICS},
    )


def stratified_record_sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sampled_parts = []
    record_labels = frame[["record_id", "label"]].drop_duplicates()
    for label, group in record_labels.groupby("label"):
        record_ids = group.record_id.to_numpy()
        sampled = rng.choice(record_ids, size=len(record_ids), replace=True)
        for draw, record_id in enumerate(sampled):
            part = frame[frame.record_id == record_id].copy()
            # Keep duplicate bootstrap draws as distinct record clusters.
            part["bootstrap_record"] = f"{label}:{draw}:{record_id}"
            sampled_parts.append(part)
    return pd.concat(sampled_parts, ignore_index=True)


def paired_bootstrap(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cohorts = [("macro_conditions", frame)] + [
        (str(condition), group.reset_index(drop=True))
        for condition, group in frame.groupby("condition")
    ]
    rows = []
    for cohort_name, cohort in cohorts:
        candidate_point, baseline_point = (
            macro_condition_pair(cohort)
            if cohort_name == "macro_conditions"
            else score_pair(cohort)
        )
        draws = {metric: [] for metric in METRICS}
        for _ in range(samples):
            sampled = stratified_record_sample(cohort, rng)
            candidate, baseline = (
                macro_condition_pair(sampled)
                if cohort_name == "macro_conditions"
                else score_pair(sampled)
            )
            for metric in METRICS:
                draws[metric].append(candidate[metric] - baseline[metric])
        for metric in METRICS:
            values = np.asarray(draws[metric], dtype=float)
            rows.append({
                "cohort": cohort_name,
                "metric": metric,
                "candidate": candidate_point[metric],
                "baseline": baseline_point[metric],
                "delta": candidate_point[metric] - baseline_point[metric],
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "probability_delta_gt_zero": float(np.mean(values > 0)),
                "bootstrap_samples": samples,
            })
        print(f"bootstrap cohort={cohort_name}", flush=True)
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    primary = predictions[
        predictions.threshold_mode == PRIMARY_MODE
    ]
    for keys, group in primary.groupby(["model"] + columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = binary_metrics(group.label, group.score, group.threshold.iloc[0])
        rows.append({
            **dict(zip(["model"] + columns, keys)),
            "records": int(group.record_id.nunique()),
            "views": len(group),
            **values,
        })
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, bootstrap: pd.DataFrame, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    subset = summary[summary.model.isin([BASELINE, CANDIDATE])]
    pivot = subset.pivot(index="condition", columns="model", values="balanced_accuracy")
    pivot = pivot[[BASELINE, CANDIDATE]]
    axis = pivot.plot(kind="bar", figsize=(10, 5), color=["#7f8c8d", "#2471a3"])
    axis.axhline(0.80, color="crimson", linestyle="--", linewidth=1)
    axis.set_ylim(0.5, 1.0)
    axis.set_ylabel("Balanced accuracy")
    axis.set_xlabel("")
    axis.set_title("Locked robust operating point")
    axis.legend(["M0 C1", "M1 C1 + modal"])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output / "condition_balanced_accuracy.png", dpi=220)
    plt.close()

    forest = bootstrap[
        (bootstrap.cohort == "macro_conditions")
        & (bootstrap.metric.isin(METRICS))
    ].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(7, 4))
    y = np.arange(len(forest))
    axis.errorbar(
        forest.delta, y,
        xerr=[forest.delta - forest.ci_lower, forest.ci_upper - forest.delta],
        fmt="o", color="#2471a3", capsize=4,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y, forest.metric)
    axis.set_xlabel("M1 - M0")
    axis.set_title("Paired record-bootstrap differences")
    fig.tight_layout()
    fig.savefig(output / "paired_bootstrap_forest.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.c4_output / "development_holdout_predictions.csv")
    thresholds = pd.read_csv(args.c4_output / "thresholds.csv")
    required = {(BASELINE, PRIMARY_MODE), (CANDIDATE, PRIMARY_MODE)}
    available = set(zip(predictions.model, predictions.threshold_mode))
    if not required.issubset(available):
        raise RuntimeError(f"Missing locked predictions: {required - available}")

    merged = merge_predictions(predictions)
    condition = grouped_metrics(predictions, ["condition"])
    excitation = grouped_metrics(predictions, ["condition", "excitation"])
    setup = grouped_metrics(predictions, ["setup"])
    bootstrap = paired_bootstrap(
        merged, args.bootstrap_samples, args.seed
    )

    condition.to_csv(args.output_dir / "condition_metrics.csv", index=False)
    excitation.to_csv(args.output_dir / "condition_excitation_metrics.csv", index=False)
    setup.to_csv(args.output_dir / "setup_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    thresholds[
        thresholds.model.isin([BASELINE, CANDIDATE])
        & (thresholds.threshold_mode == PRIMARY_MODE)
    ].to_csv(args.output_dir / "locked_thresholds.csv", index=False)
    make_figures(condition, bootstrap, args.output_dir)

    candidate_condition = condition[condition.model == CANDIDATE]
    macro = bootstrap[bootstrap.cohort == "macro_conditions"].set_index("metric")
    gates = {
        "mean_balanced_accuracy_ge_082": bool(candidate_condition.balanced_accuracy.mean() >= 0.82),
        "mean_specificity_ge_090": bool(candidate_condition.specificity.mean() >= 0.90),
        "worst_condition_ba_ge_075": bool(candidate_condition.balanced_accuracy.min() >= 0.75),
        "delta_auroc_ci_lower_gt_zero": bool(macro.loc["auroc", "ci_lower"] > 0),
    }
    artifact = {
        "protocol": "z24-c4.1-locked-robust-paired-bootstrap",
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "threshold_mode": PRIMARY_MODE,
        "bootstrap_samples": args.bootstrap_samples,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "development_holdout_previously_inspected": True,
    }
    (args.output_dir / "run.json").write_text(json.dumps(artifact, indent=2))

    print("\n=== LOCKED CONDITION METRICS ===")
    print(condition[condition.model.isin([BASELINE, CANDIDATE])].to_string(index=False))
    print("\n=== MACRO PAIRED BOOTSTRAP ===")
    print(bootstrap[bootstrap.cohort == "macro_conditions"].to_string(index=False))
    print("\n=== GATES ===")
    print(json.dumps(artifact, indent=2))
    print(f"C41_PASS={all(gates.values())} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

