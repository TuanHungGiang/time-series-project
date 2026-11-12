from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score


MODELS = ("b2", "xspec", "fusion_80_20")


def metrics(labels: np.ndarray, scores: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, decisions, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~decisions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
    }


def paired(frame: pd.DataFrame, candidate: str, samples: int, seed: int) -> pd.DataFrame:
    labels = frame.label.to_numpy(dtype=int)
    healthy = np.flatnonzero(labels == 0)
    damage = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)

    def difference(metric: str, indices: np.ndarray) -> float:
        y = labels[indices]
        if metric == "auroc":
            return float(
                roc_auc_score(y, frame[f"score_{candidate}"].to_numpy()[indices])
                - roc_auc_score(y, frame.score_b2.to_numpy()[indices])
            )
        if metric == "balanced_accuracy":
            return float(
                balanced_accuracy_score(y, frame[f"decision_{candidate}"].to_numpy()[indices])
                - balanced_accuracy_score(y, frame.decision_b2.to_numpy()[indices])
            )
        return float(
            recall_score(y, frame[f"decision_{candidate}"].to_numpy()[indices], zero_division=0)
            - recall_score(y, frame.decision_b2.to_numpy()[indices], zero_division=0)
        )

    rows = []
    for metric in ("auroc", "balanced_accuracy", "recall"):
        estimate = difference(metric, np.arange(len(frame)))
        draws = []
        for _ in range(samples):
            indices = np.concatenate((
                rng.choice(healthy, len(healthy), replace=True),
                rng.choice(damage, len(damage), replace=True),
            ))
            draws.append(difference(metric, indices))
        draws = np.asarray(draws)
        rows.append({
            "candidate": candidate, "baseline": "b2", "metric": metric,
            "estimate_candidate_minus_baseline": estimate,
            "ci_lower": float(np.quantile(draws, 0.025)),
            "ci_upper": float(np.quantile(draws, 0.975)),
            "probability_candidate_better": float(np.mean(draws > 0)),
            "bootstrap_samples": samples,
        })
    return pd.DataFrame(rows)


def analyze_dataset(dataset: str, path: Path, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(path)
    required = set(MODELS)
    if set(records.model) != required:
        raise ValueError(f"{dataset}: expected models {required}, found {set(records.model)}")
    labels = records.drop_duplicates("record_id").set_index("record_id").label
    score = records.pivot(index="record_id", columns="model", values="score").add_prefix("score_")
    decision = records.pivot(index="record_id", columns="model", values="prediction").astype(bool).add_prefix("decision_")
    wide = pd.concat([labels, score, decision], axis=1).reset_index()
    summary_rows = []
    y = wide.label.to_numpy(dtype=int)
    for model in MODELS:
        summary_rows.append({
            "dataset": dataset, "model": model, "records": len(wide),
            **metrics(y, wide[f"score_{model}"].to_numpy(), wide[f"decision_{model}"].to_numpy(dtype=bool)),
        })
    comparisons = []
    for index, candidate in enumerate(("xspec", "fusion_80_20")):
        result = paired(wide, candidate, samples, 20261112 + 101 * index + len(dataset))
        result.insert(0, "dataset", dataset)
        comparisons.append(result)
    return pd.DataFrame(summary_rows), pd.concat(comparisons, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-dir", type=Path, required=True)
    parser.add_argument("--hell-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/xspec_v5_analysis"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries, comparisons = [], []
    for dataset, directory in (("Z24", args.z24_dir), ("HellBridge", args.hell_dir)):
        summary, comparison = analyze_dataset(dataset, directory / "record_scores.csv", args.bootstrap_samples)
        summaries.append(summary)
        comparisons.append(comparison)
    summary = pd.concat(summaries, ignore_index=True)
    comparison = pd.concat(comparisons, ignore_index=True)
    summary.to_csv(args.output_dir / "table1_v5_performance.csv", index=False)
    comparison.to_csv(args.output_dir / "table2_v5_paired_bootstrap.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True)
    colors = ["#8a8f98", "#d97904", "#1769aa"]
    for axis, metric in zip(axes, ("auroc", "balanced_accuracy", "recall")):
        pivot = summary.pivot(index="dataset", columns="model", values=metric)
        pivot[list(MODELS)].plot.bar(ax=axis, color=colors, width=0.78)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend().remove()
    axes[0].set_ylabel("Metric")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig1_v5_comparison.png", dpi=240)
    plt.close(figure)

    gates = {}
    for dataset in ("Z24", "HellBridge"):
        candidate = summary[(summary.dataset == dataset) & (summary.model == "fusion_80_20")].iloc[0]
        ci = comparison[(comparison.dataset == dataset) & (comparison.candidate == "fusion_80_20")].set_index("metric")
        gates[dataset] = {
            "fusion_auroc_at_least_b2": bool(ci.loc["auroc", "ci_lower"] >= 0),
            "fusion_balanced_accuracy_at_least_b2": bool(ci.loc["balanced_accuracy", "ci_lower"] >= 0),
            "fusion_auroc_at_least_0_80": bool(candidate.auroc >= 0.80),
            "fusion_recall_at_least_0_50": bool(candidate.recall >= 0.50),
        }
    (args.output_dir / "v5_gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))
    print(json.dumps(gates, indent=2))
    print(f"XSPEC_V5_ANALYSIS_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

