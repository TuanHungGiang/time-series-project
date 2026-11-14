from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def metric(labels: np.ndarray, scores: np.ndarray, name: str) -> float:
    if name == "auroc":
        return float(roc_auc_score(labels, scores))
    return float(balanced_accuracy_score(labels, scores > 1.0))


def aggregate_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(
        ["record_id", "label", "condition", "replicate", "model"], as_index=False
    ).score.mean()


def paired_hierarchical_bootstrap(
    scores: pd.DataFrame,
    reference: str,
    challenger: str,
    condition: str,
    metric_name: str,
    samples: int,
    seed: int,
) -> dict:
    subset = scores[
        (scores.condition == condition)
        & scores.model.isin([reference, challenger])
    ]
    pivot = subset.pivot_table(
        index=["record_id", "label", "replicate"],
        columns="model",
        values="score",
        aggfunc="mean",
    ).dropna(subset=[reference, challenger]).reset_index()
    record_labels = pivot[["record_id", "label"]].drop_duplicates().sort_values("record_id")
    record_ids = record_labels.record_id.to_numpy()
    labels = record_labels.label.to_numpy(dtype=int)
    healthy = np.flatnonzero(labels == 0)
    damage = np.flatnonzero(labels == 1)
    replicates = sorted(pivot.replicate.unique())
    reference_matrix = pivot.pivot(index="record_id", columns="replicate", values=reference).reindex(index=record_ids, columns=replicates).to_numpy()
    challenger_matrix = pivot.pivot(index="record_id", columns="replicate", values=challenger).reindex(index=record_ids, columns=replicates).to_numpy()
    rng = np.random.default_rng(seed)

    def evaluate(indices: np.ndarray) -> float:
        values = []
        sampled_labels = labels[indices]
        for replicate_index in range(len(replicates)):
            values.append(
                metric(sampled_labels, reference_matrix[indices, replicate_index], metric_name)
                - metric(sampled_labels, challenger_matrix[indices, replicate_index], metric_name)
            )
        return float(np.mean(values))

    estimate = evaluate(np.arange(len(record_ids)))
    draws = []
    for _ in range(samples):
        indices = np.concatenate(
            [rng.choice(healthy, len(healthy), replace=True), rng.choice(damage, len(damage), replace=True)]
        )
        draws.append(evaluate(indices))
    draws = np.asarray(draws)
    return {
        "reference": reference,
        "challenger": challenger,
        "condition": condition,
        "metric": metric_name,
        "estimate_reference_minus_challenger": estimate,
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "probability_reference_better": float(np.mean(draws > 0.0)),
        "bootstrap_samples": samples,
        "healthy_records": int(len(healthy)),
        "damage_records": int(len(damage)),
    }


def variance_decomposition(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition), group in metrics.groupby(["model", "condition"]):
        for metric_name in ("auroc", "balanced_accuracy"):
            cells = group.pivot_table(index="seed", columns="replicate", values=metric_name, aggfunc="mean")
            matrix = cells.to_numpy(dtype=float)
            grand = float(np.nanmean(matrix))
            seed_means = np.nanmean(matrix, axis=1)
            replicate_means = np.nanmean(matrix, axis=0)
            seed_std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan")
            replicate_std = float(np.std(replicate_means, ddof=1)) if len(replicate_means) > 1 else float("nan")
            residual = matrix - seed_means[:, None] - replicate_means[None, :] + grand
            ss_seed = matrix.shape[1] * float(np.nansum((seed_means - grand) ** 2))
            ss_replicate = matrix.shape[0] * float(np.nansum((replicate_means - grand) ** 2))
            ss_interaction = float(np.nansum(residual**2))
            total = ss_seed + ss_replicate + ss_interaction
            rows.append({
                "model": model,
                "condition": condition,
                "metric": metric_name,
                "training_seeds": matrix.shape[0],
                "corruption_replicates": matrix.shape[1],
                "mean": grand,
                "between_seed_std": seed_std,
                "between_corruption_replicate_std": replicate_std,
                "seed_variance_share": ss_seed / total if total > 0 else float("nan"),
                "corruption_variance_share": ss_replicate / total if total > 0 else float("nan"),
                "seed_by_corruption_interaction_share": ss_interaction / total if total > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def merge_framework_evidence(
    detector_summary: pd.DataFrame,
    reliability_dir: Path,
    coverage: float,
) -> pd.DataFrame:
    delta = pd.read_csv(reliability_dir / "table4_proposed_vs_random.csv")
    delta = delta[np.isclose(delta.requested_coverage, coverage)].copy()
    aurc = pd.read_csv(reliability_dir / "table5_matched_aurc.csv")
    proposed = aurc[aurc.method == "proposed"][["condition", "mean"]].rename(columns={"mean": "proposed_matched_aurc"})
    random = aurc[aurc.method == "random_rejection"][["condition", "mean"]].rename(columns={"mean": "random_matched_aurc"})
    b2 = detector_summary[detector_summary.model == "b2"][["condition", "auroc_mean", "balanced_accuracy_mean"]].rename(
        columns={"auroc_mean": "b2_auroc", "balanced_accuracy_mean": "b2_balanced_accuracy"}
    )
    ocsvm = detector_summary[detector_summary.model == "ocsvm"][["condition", "auroc_mean", "balanced_accuracy_mean"]].rename(
        columns={"auroc_mean": "ocsvm_auroc", "balanced_accuracy_mean": "ocsvm_balanced_accuracy"}
    )
    output = b2.merge(ocsvm, on="condition", how="outer")
    output = output.merge(delta, on="condition", how="outer")
    output = output.merge(proposed, on="condition", how="outer").merge(random, on="condition", how="outer")
    output["b2_minus_ocsvm_auroc"] = output.b2_auroc - output.ocsvm_auroc
    output["proposed_minus_random_aurc"] = output.proposed_matched_aurc - output.random_matched_aurc
    return output.sort_values("condition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--reliability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/z24_final_statistics"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--coverage", type=float, default=0.70)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = aggregate_scores(pd.read_csv(args.detector_dir / "record_scores.csv"))
    metrics = pd.read_csv(args.detector_dir / "metrics.csv")
    detector_summary = pd.read_csv(args.detector_dir / "summary.csv")
    models = sorted(set(raw_scores.model) - {"b2"})
    conditions = sorted(raw_scores.condition.unique())
    bootstrap_rows = []
    for model_index, challenger in enumerate(models):
        for condition_index, condition in enumerate(conditions):
            for metric_name in ("auroc", "balanced_accuracy"):
                bootstrap_rows.append(paired_hierarchical_bootstrap(
                    raw_scores, "b2", challenger, condition, metric_name,
                    args.bootstrap_samples,
                    20261111 + model_index * 1009 + condition_index * 37 + (metric_name == "balanced_accuracy"),
                ))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "table1_paired_detector_bootstrap.csv", index=False)

    variance = variance_decomposition(metrics)
    variance.to_csv(args.output_dir / "table2_seed_corruption_variance.csv", index=False)
    evidence = merge_framework_evidence(detector_summary, args.reliability_dir, args.coverage)
    evidence.to_csv(args.output_dir / "table3_final_framework_evidence.csv", index=False)

    forest = bootstrap[(bootstrap.challenger == "ocsvm") & (bootstrap.metric == "auroc")].copy()
    forest = forest.sort_values("estimate_reference_minus_challenger")
    figure, axis = plt.subplots(figsize=(9, 7))
    y = np.arange(len(forest))
    estimate = forest.estimate_reference_minus_challenger.to_numpy()
    axis.hlines(y, forest.ci_lower, forest.ci_upper, color="#94D2BD", linewidth=2)
    axis.scatter(estimate, y, color="#005F73", zorder=3)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, forest.condition.str.replace("_", " "))
    axis.set_xlabel("Paired AUROC difference: B2 - OCSVM")
    axis.set_title("Hierarchical bootstrap comparison (95% CI)")
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig1_b2_vs_ocsvm_paired_ci.png", dpi=220)
    plt.close(figure)

    deep = variance[(variance.model.isin(["cnn_ae", "masked_cnn_ae", "deep_svdd"])) & (variance.metric == "auroc")]
    deep = deep.groupby("model")[["seed_variance_share", "corruption_variance_share", "seed_by_corruption_interaction_share"]].mean()
    figure, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(deep))
    colors = ["#005F73", "#EE9B00", "#AE2012"]
    for color, column in zip(colors, deep.columns):
        values = deep[column].fillna(0).to_numpy()
        axis.bar(deep.index.str.replace("_", " "), values, bottom=bottom, label=column.replace("_", " "), color=color)
        bottom += values
    axis.set_ylabel("Mean descriptive variance share")
    axis.set_title("Training-seed and corruption variability")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig2_variance_decomposition.png", dpi=220)
    plt.close(figure)

    metadata = {
        "bootstrap": "stratified record-level resampling; all corruption replicates retained per sampled record",
        "bootstrap_samples": args.bootstrap_samples,
        "reference_detector": "b2",
        "reliability_coverage": args.coverage,
        "damage_labels_used_for_selection": False,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(metadata, indent=2))
    print(bootstrap[(bootstrap.challenger == "ocsvm")].to_string(index=False))
    print(evidence.to_string(index=False))
    print(f"FINAL_STATISTICS_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

