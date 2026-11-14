from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from run_hellbridge_c9_unseen_damage import block_bootstrap


PRIMARY = "paper_score_geometric_fusion"
BASELINES = ("paper_ar5_al", "paper_ar5_ag")
FUSIONS = (
    PRIMARY,
    "paper_score_arithmetic_fusion",
    "paper_score_max_fusion",
    "paper_score_min_fusion",
)


def metric_dict(labels: np.ndarray, scores: np.ndarray, threshold: float = 1.0) -> dict[str, float]:
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    prediction = scores >= threshold
    positive, negative = labels == 1, labels == 0
    recall = float(prediction[positive].mean())
    specificity = float((~prediction[negative]).mean())
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": float(f1_score(labels, prediction)),
        "false_positive_rate": 1 - specificity,
        "false_negative_rate": 1 - recall,
    }


def paired_block_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    samples: int,
    seed: int,
    block_size: int = 4,
) -> pd.DataFrame:
    candidate = frame[frame.method == PRIMARY].set_index("sample_id")
    control = frame[frame.method == baseline].set_index("sample_id")
    merged = candidate[["record_id", "start", "label", "score", "threshold"]].join(
        control[["score", "threshold"]],
        lsuffix="_candidate", rsuffix="_baseline", validate="one_to_one",
    ).reset_index(drop=True)
    groups = []
    for _, group in merged.sort_values(["record_id", "start"]).groupby("record_id"):
        positions = group.index.to_numpy()
        groups.append([
            positions[start : start + block_size]
            for start in range(0, len(positions), block_size)
        ])
    names = ("auroc", "recall", "specificity", "balanced_accuracy")
    observed_candidate = metric_dict(
        merged.label, merged.score_candidate, float(merged.threshold_candidate.iloc[0])
    )
    observed_baseline = metric_dict(
        merged.label, merged.score_baseline, float(merged.threshold_baseline.iloc[0])
    )
    rng = np.random.default_rng(seed)
    distributions = {name: [] for name in names}
    for _ in range(samples):
        positions = []
        for blocks in groups:
            selected = rng.integers(0, len(blocks), size=len(blocks))
            positions.extend(np.concatenate([blocks[index] for index in selected]))
        subset = merged.loc[np.asarray(positions, int)]
        first = metric_dict(
            subset.label, subset.score_candidate, float(subset.threshold_candidate.iloc[0])
        )
        second = metric_dict(
            subset.label, subset.score_baseline, float(subset.threshold_baseline.iloc[0])
        )
        for name in names:
            distributions[name].append(first[name] - second[name])
    return pd.DataFrame([
        {
            "candidate": PRIMARY,
            "baseline": baseline,
            "metric": name,
            "candidate_value": observed_candidate[name],
            "baseline_value": observed_baseline[name],
            "delta": observed_candidate[name] - observed_baseline[name],
            "ci_lower": float(np.quantile(distributions[name], 0.025)),
            "ci_upper": float(np.quantile(distributions[name], 0.975)),
            "probability_delta_gt_zero": float(np.mean(np.asarray(distributions[name]) > 0)),
            "bootstrap_samples": samples,
            "block_size": block_size,
        }
        for name in names
    ])


def make_figure(metrics: pd.DataFrame, states: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    measures = ("auroc", "recall", "specificity", "balanced_accuracy")
    x = np.arange(len(metrics))
    width = 0.19
    for index, measure in enumerate(measures):
        axes[0].bar(x + (index - 1.5) * width, metrics[measure], width, label=measure)
    axes[0].set_xticks(x, metrics.method, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Paper-normalized score fusion, threshold=1")
    axes[0].legend(ncol=2)
    selected = states[states.method == PRIMARY]
    axes[1].bar(selected.state, selected.damage_recall)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Geometric fusion recall by unseen damage state")
    axes[1].set_ylabel("Recall")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c10-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.c10_scores)
    paper = scores[scores.method.isin(BASELINES)].copy()
    if set(paper.method.unique()) != set(BASELINES):
        raise ValueError(f"Missing paper score methods; found {sorted(paper.method.unique())}")
    metadata = paper[paper.method == "paper_ar5_al"][
        ["sample_id", "record_id", "start", "state", "label"]
    ].set_index("sample_id")
    pivot = paper.pivot(index="sample_id", columns="method", values="score")
    frame = metadata.join(pivot, validate="one_to_one").reset_index()
    if len(frame) != 720 or frame.sample_id.duplicated().any():
        raise ValueError(f"Expected 720 unique test samples, got {len(frame)}")
    al = np.maximum(frame.paper_ar5_al.to_numpy(float), 0.0)
    ag = np.maximum(frame.paper_ar5_ag.to_numpy(float), 0.0)
    fused = {
        PRIMARY: np.sqrt(al * ag),
        "paper_score_arithmetic_fusion": (al + ag) / 2,
        "paper_score_max_fusion": np.maximum(al, ag),
        "paper_score_min_fusion": np.minimum(al, ag),
    }

    rows = []
    # Preserve paper baselines in the same output table.
    for method, values in (
        ("paper_ar5_al", al),
        ("paper_ar5_ag", ag),
        *fused.items(),
    ):
        current = frame[["sample_id", "record_id", "start", "state", "label"]].copy()
        current["method"] = method
        current["score"] = values
        current["threshold"] = 1.0
        current["prediction"] = (values >= 1.0).astype(int)
        rows.append(current)
    output_scores = pd.concat(rows, ignore_index=True)

    metric_rows, state_rows = [], []
    for method, group in output_scores.groupby("method", sort=False):
        result = metric_dict(group.label, group.score, 1.0)
        metric_rows.append({
            "method": method,
            "primary": method == PRIMARY,
            "threshold": 1.0,
            "test_samples": len(group),
            **result,
        })
        for state in [f"DS{i}" for i in range(1, 9)]:
            damage = group.state.eq(state).to_numpy()
            cohort = group.state.isin(["UDS", state]).to_numpy()
            state_rows.append({
                "method": method,
                "state": state,
                "damage_samples": int(damage.sum()),
                "damage_recall": float(group.prediction.to_numpy()[damage].mean()),
                "auroc_vs_uds": float(roc_auc_score(
                    group.label.to_numpy()[cohort], group.score.to_numpy()[cohort]
                )),
            })
    metrics = pd.DataFrame(metric_rows)
    states = pd.DataFrame(state_rows)
    primary_frame = output_scores[output_scores.method == PRIMARY].reset_index(drop=True)
    primary_frame.index = np.arange(len(primary_frame))
    ci = block_bootstrap(primary_frame, args.bootstrap_samples, args.seed + 70001)
    paired = pd.concat([
        paired_block_bootstrap(
            output_scores, baseline, args.bootstrap_samples,
            args.seed + 80001 + index * 1009,
        )
        for index, baseline in enumerate(BASELINES)
    ], ignore_index=True)

    output_scores.to_csv(args.output_dir / "sample_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)
    ci.to_csv(args.output_dir / "primary_block_bootstrap_ci.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_paper_baselines.csv", index=False)
    make_figure(metrics, states, args.output_dir / "fig_c102_paper_score_fusion.png")
    protocol = {
        "protocol": "hellbridge-c10.2-paper-normalized-score-fusion",
        "input_scores": str(args.c10_scores),
        "damage_labels_used_to_choose_rule_or_threshold": False,
        "primary_method": PRIMARY,
        "primary_rule": "sqrt(normalized_AR5_AL * normalized_AR5_AG)",
        "primary_threshold": 1.0,
        "threshold_rationale": "each paper-normalized component has its natural alarm boundary at 1",
        "sensitivity_rules": [name for name in FUSIONS if name != PRIMARY],
        "test_labels_used_only_for_metrics": True,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("\n=== C10.2 METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== PRIMARY PAIRED VS PAPER BASELINES ===")
    print(paired.to_string(index=False))
    print("\n=== PRIMARY STATE METRICS ===")
    print(states[states.method == PRIMARY].to_string(index=False))
    print(f"HELLBRIDGE_C102_PAPER_SCORE_FUSION_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

