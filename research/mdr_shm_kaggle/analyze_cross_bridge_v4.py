from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score


def clean_scores(path: Path, method: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "condition" in frame:
        frame = frame[frame.condition == "clean"]
    if "replicate" in frame:
        frame = frame[frame.replicate == 0]
    if method is not None and "method" in frame:
        frame = frame[frame.method == method]
    if frame.record_id.duplicated().any():
        raise ValueError(f"Duplicate clean record IDs in {path}")
    if "prediction" in frame:
        decision = frame["prediction"].astype(bool)
    elif "alarm" in frame:
        decision = frame["alarm"].astype(bool)
    else:
        decision = frame["score"].to_numpy(dtype=float) > 1.0
    output = frame[["record_id", "label", "score"]].copy()
    output["decision"] = np.asarray(decision, dtype=bool)
    return output


def operating_metrics(labels: np.ndarray, scores: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def paired_bootstrap(frame: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    labels = frame.label.to_numpy(dtype=int)
    candidate = frame.candidate.to_numpy(dtype=float)
    baseline = frame.baseline.to_numpy(dtype=float)
    candidate_decision = frame.candidate_decision.to_numpy(dtype=bool)
    baseline_decision = frame.baseline_decision.to_numpy(dtype=bool)
    healthy = np.flatnonzero(labels == 0)
    damage = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)

    def evaluate(metric: str, indices: np.ndarray) -> float:
        y = labels[indices]
        if metric == "auroc":
            return float(roc_auc_score(y, candidate[indices]) - roc_auc_score(y, baseline[indices]))
        if metric == "balanced_accuracy":
            return float(
                balanced_accuracy_score(y, candidate_decision[indices])
                - balanced_accuracy_score(y, baseline_decision[indices])
            )
        return float(
            recall_score(y, candidate_decision[indices], zero_division=0)
            - recall_score(y, baseline_decision[indices], zero_division=0)
        )

    rows = []
    for metric in ("auroc", "balanced_accuracy", "recall"):
        estimate = evaluate(metric, np.arange(len(frame)))
        draws = []
        for _ in range(samples):
            indices = np.concatenate((
                rng.choice(healthy, len(healthy), replace=True),
                rng.choice(damage, len(damage), replace=True),
            ))
            draws.append(evaluate(metric, indices))
        draws = np.asarray(draws)
        rows.append({
            "metric": metric, "estimate_candidate_minus_baseline": estimate,
            "ci_lower": float(np.quantile(draws, 0.025)),
            "ci_upper": float(np.quantile(draws, 0.975)),
            "probability_candidate_better": float(np.mean(draws > 0)),
            "bootstrap_samples": samples,
        })
    return pd.DataFrame(rows)


def compare(dataset: str, candidate_path: Path, baseline_path: Path, baseline_method: str | None, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = clean_scores(candidate_path).rename(columns={
        "score": "candidate", "label": "candidate_label", "decision": "candidate_decision",
    })
    baseline = clean_scores(baseline_path, baseline_method).rename(columns={
        "score": "baseline", "decision": "baseline_decision",
    })
    matched = candidate.merge(baseline, on="record_id", how="inner", validate="one_to_one")
    if not np.array_equal(matched.candidate_label, matched.label):
        raise ValueError(f"Label mismatch in {dataset}")
    matched = matched.drop(columns=["candidate_label"])
    labels = matched.label.to_numpy(dtype=int)
    rows = []
    for model in ("baseline", "candidate"):
        rows.append({
            "dataset": dataset, "model": model, "records": len(matched),
            **operating_metrics(
                labels, matched[model].to_numpy(), matched[f"{model}_decision"].to_numpy(dtype=bool)
            ),
        })
    bootstrap = paired_bootstrap(matched, samples, 20261112 + len(dataset))
    bootstrap.insert(0, "dataset", dataset)
    return pd.DataFrame(rows), bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-candidate", type=Path, required=True)
    parser.add_argument("--z24-baseline", type=Path, required=True)
    parser.add_argument("--hell-candidate", type=Path, required=True)
    parser.add_argument("--hell-baseline", type=Path, required=True)
    parser.add_argument("--hell-baseline-method", default="median_score")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/cross_bridge_v4_analysis"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    bootstraps = []
    for dataset, candidate, baseline, method in (
        ("Z24", args.z24_candidate, args.z24_baseline, None),
        ("HellBridge", args.hell_candidate, args.hell_baseline, args.hell_baseline_method),
    ):
        summary, bootstrap = compare(dataset, candidate, baseline, method, args.bootstrap_samples)
        summaries.append(summary)
        bootstraps.append(bootstrap)
    summary = pd.concat(summaries, ignore_index=True)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    summary.to_csv(args.output_dir / "table1_cross_bridge_performance.csv", index=False)
    bootstrap.to_csv(args.output_dir / "table2_paired_bootstrap.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    for axis, metric in zip(axes, ("auroc", "balanced_accuracy", "recall")):
        pivot = summary.pivot(index="dataset", columns="model", values=metric)
        pivot[["baseline", "candidate"]].plot.bar(ax=axis, color=["#9aa0a6", "#1769aa"], width=0.72)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend().remove()
    axes[0].set_ylabel("Metric")
    axes[-1].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig1_baseline_vs_v4.png", dpi=240)
    plt.close(figure)

    gates = {}
    for dataset in ("Z24", "HellBridge"):
        subset = bootstrap[bootstrap.dataset == dataset].set_index("metric")
        candidate_metrics = summary[(summary.dataset == dataset) & (summary.model == "candidate")].iloc[0]
        gates[dataset] = {
            "auroc_at_least_0_80": bool(candidate_metrics.auroc >= 0.80),
            "balanced_accuracy_at_least_0_70": bool(candidate_metrics.balanced_accuracy >= 0.70),
            "recall_at_least_0_60": bool(candidate_metrics.recall >= 0.60),
            "auroc_improvement_ci_excludes_zero": bool(subset.loc["auroc", "ci_lower"] > 0),
            "balanced_accuracy_improvement_ci_excludes_zero": bool(subset.loc["balanced_accuracy", "ci_lower"] > 0),
        }
    (args.output_dir / "publication_gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(json.dumps(gates, indent=2))
    print(f"CROSS_BRIDGE_V4_ANALYSIS_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

