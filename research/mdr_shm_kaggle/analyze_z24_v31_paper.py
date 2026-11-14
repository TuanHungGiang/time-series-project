from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
METRICS = ("auroc", "recall", "specificity", "balanced_accuracy", "f1")
COMPONENT_COLUMNS = (
    "jackknife_instability",
    "coverage_deficit",
    "quality_deficit",
    "high_frequency_outlier",
    "spectral_roughness_outlier",
)
RELIABILITY_WEIGHTS = np.asarray([1.0, 0.25, 0.50, 1.0, 1.0])


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    differences = positive[:, None] - negative[None, :]
    return float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))


def metric_dict(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame.label.to_numpy(dtype=int)
    scores = frame.score.to_numpy(dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {metric: float("nan") for metric in METRICS}
    prediction = scores > 1.0
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    fn = int(np.sum(~prediction & positive))
    tn = int(np.sum(~prediction & negative))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1_denominator = 2 * tp + fp + fn
    return {
        "auroc": binary_auroc(labels, scores),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(2 * tp / f1_denominator if f1_denominator else 0.0),
    }


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def prepare_scores(input_dir: Path) -> tuple[pd.DataFrame, dict]:
    scores = pd.read_csv(input_dir / "record_scores.csv")
    model = json.loads((input_dir / "model.json").read_text())
    required = {
        "record_id",
        "label",
        "excitation",
        "condition",
        "replicate",
        "score",
        "reliability",
        "sensor_coverage",
        "observed_quality",
        "jackknife_instability",
        "high_frequency_outlier",
        "spectral_roughness_outlier",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"record_scores.csv is missing columns: {sorted(missing)}")
    scores["coverage_deficit"] = 1.0 - scores.sensor_coverage
    scores["quality_deficit"] = 1.0 - scores.observed_quality
    return scores, model


def add_ablation_scores(scores: pd.DataFrame, model: dict) -> pd.DataFrame:
    output = scores.copy()
    calibrations = model["reliability_calibrations"]
    masks = {
        "jackknife_only": np.asarray([1, 0, 0, 0, 0], dtype=bool),
        "coverage_quality": np.asarray([0, 1, 1, 0, 0], dtype=bool),
        "spectral_only": np.asarray([0, 0, 0, 1, 1], dtype=bool),
        "without_jackknife": np.asarray([0, 1, 1, 1, 1], dtype=bool),
        "without_spectral": np.asarray([1, 1, 1, 0, 0], dtype=bool),
    }
    for name in masks:
        output[f"ablation_{name}"] = np.nan

    for excitation, indices in output.groupby("excitation").groups.items():
        if excitation not in calibrations:
            raise KeyError(f"No reliability calibration for excitation={excitation}")
        calibration = calibrations[excitation]
        center = np.asarray(calibration["center"], dtype=float)
        scale = np.asarray(calibration["scale"], dtype=float)
        matrix = output.loc[indices, COMPONENT_COLUMNS].to_numpy(dtype=float)
        z = np.maximum((matrix - center[None, :]) / scale[None, :], 0.0)
        for name, mask in masks.items():
            weights = RELIABILITY_WEIGHTS * mask
            output.loc[indices, f"ablation_{name}"] = (
                np.sum(z * weights[None, :], axis=1) / weights.sum()
            )
    return output


def deterministic_subset(
    group: pd.DataFrame, uncertainty_column: str, coverage: float
) -> pd.DataFrame:
    count = max(2, int(np.ceil(len(group) * coverage)))
    return group.sort_values(
        [uncertainty_column, "record_id"], kind="stable"
    ).iloc[:count]


def random_subset(
    group: pd.DataFrame, coverage: float, repeat: int
) -> pd.DataFrame:
    count = max(2, int(np.ceil(len(group) * coverage)))
    seed = stable_seed(
        "paper-random",
        group.condition.iloc[0],
        int(group.replicate.iloc[0]),
        coverage,
        repeat,
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(group), count, replace=False)
    return group.iloc[np.sort(indices)]


def evaluate_subset(
    subset: pd.DataFrame,
    total_records: int,
    method: str,
    coverage: float,
    repeat: int,
) -> dict:
    class_counts = subset.label.value_counts()
    return {
        "method": method,
        "requested_coverage": coverage,
        "realized_coverage": len(subset) / total_records,
        "repeat": repeat,
        "records": len(subset),
        "healthy_records": int(class_counts.get(0, 0)),
        "damage_records": int(class_counts.get(1, 0)),
        "valid_two_class_cohort": subset.label.nunique() == 2,
        **metric_dict(subset),
    }


def matched_coverage_experiment(
    scores: pd.DataFrame, random_repeats: int
) -> pd.DataFrame:
    methods = {
        "proposed": "reliability",
        "confidence_margin": "confidence_uncertainty",
        "coverage_only": "coverage_deficit",
        "quality_only": "quality_uncertainty",
        "jackknife_only": "jackknife_instability",
    }
    working = scores.copy()
    working["confidence_uncertainty"] = -np.abs(working.score - 1.0)
    working["quality_uncertainty"] = (
        working.coverage_deficit + working.quality_deficit
    )
    rows = []
    for (condition, replicate), group in working.groupby(
        ["condition", "replicate"], sort=True
    ):
        prefix = {"condition": condition, "replicate": int(replicate)}
        for coverage in COVERAGES:
            for method, column in methods.items():
                subset = deterministic_subset(group, column, coverage)
                rows.append(
                    {
                        **prefix,
                        **evaluate_subset(
                            subset, len(group), method, coverage, repeat=0
                        ),
                    }
                )
            for repeat in range(random_repeats):
                subset = random_subset(group, coverage, repeat)
                rows.append(
                    {
                        **prefix,
                        **evaluate_subset(
                            subset,
                            len(group),
                            "random_rejection",
                            coverage,
                            repeat,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def ablation_experiment(scores: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "proposed": "reliability",
        "jackknife_only": "ablation_jackknife_only",
        "coverage_quality": "ablation_coverage_quality",
        "spectral_only": "ablation_spectral_only",
        "without_jackknife": "ablation_without_jackknife",
        "without_spectral": "ablation_without_spectral",
    }
    rows = []
    for (condition, replicate), group in scores.groupby(
        ["condition", "replicate"], sort=True
    ):
        for coverage in COVERAGES:
            for method, column in methods.items():
                subset = deterministic_subset(group, column, coverage)
                rows.append(
                    {
                        "condition": condition,
                        "replicate": int(replicate),
                        **evaluate_subset(
                            subset, len(group), method, coverage, repeat=0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_table(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    value_columns = [
        "realized_coverage",
        "healthy_records",
        "damage_records",
        *METRICS,
    ]
    aggregated = frame.groupby(group_columns)[value_columns].agg(["mean", "std"])
    aggregated.columns = [f"{name}_{stat}" for name, stat in aggregated.columns]
    return aggregated.reset_index()


def main_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "auroc",
        "recall",
        "specificity",
        "balanced_accuracy",
        "f1",
        "acceptance_rate",
        "healthy_acceptance_rate",
        "damage_acceptance_rate",
        "accepted_auroc",
        "accepted_recall",
        "accepted_balanced_accuracy",
        "accepted_f1",
        "reliability_error_auroc",
        "balanced_aurc",
    ]
    columns = [column for column in columns if column in metrics.columns]
    table = metrics.groupby("condition")[columns].agg(["mean", "std"])
    table.columns = [f"{name}_{stat}" for name, stat in table.columns]
    return table.reset_index()


def derive_reliability_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, replicate), group in scores.groupby(
        ["condition", "replicate"], sort=True
    ):
        prediction_error = (
            (group.score.to_numpy(dtype=float) > 1.0)
            != group.label.to_numpy(dtype=bool)
        ).astype(int)
        error_auroc = binary_auroc(
            prediction_error, group.reliability.to_numpy(dtype=float)
        )
        ordered = group.sort_values(["reliability", "record_id"], kind="stable")
        coverage = []
        risk = []
        for count in range(2, len(ordered) + 1):
            kept = ordered.iloc[:count]
            values = metric_dict(kept)
            if np.isfinite(values["balanced_accuracy"]):
                coverage.append(count / len(ordered))
                risk.append(1.0 - values["balanced_accuracy"])
        if len(coverage) >= 2 and coverage[-1] > coverage[0]:
            balanced_aurc = float(
                np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0])
            )
        else:
            balanced_aurc = float("nan")
        rows.append(
            {
                "condition": condition,
                "replicate": int(replicate),
                "derived_reliability_error_auroc": error_auroc,
                "derived_balanced_aurc": balanced_aurc,
            }
        )
    return pd.DataFrame(rows)


def enrich_metrics(metrics: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    derived = derive_reliability_metrics(scores)
    output = metrics.merge(derived, on=["condition", "replicate"], how="left")
    if "reliability_error_auroc" not in output:
        output["reliability_error_auroc"] = output[
            "derived_reliability_error_auroc"
        ]
    if "balanced_aurc" not in output:
        output["balanced_aurc"] = output["derived_balanced_aurc"]
    return output.drop(
        columns=["derived_reliability_error_auroc", "derived_balanced_aurc"]
    )


def proposed_vs_random_table(matched: pd.DataFrame) -> pd.DataFrame:
    deterministic = matched[matched.method == "proposed"].copy()
    random_mean = (
        matched[matched.method == "random_rejection"]
        .groupby(["condition", "replicate", "requested_coverage"])[
            list(METRICS)
        ]
        .mean()
        .reset_index()
    )
    merged = deterministic.merge(
        random_mean,
        on=["condition", "replicate", "requested_coverage"],
        suffixes=("_proposed", "_random"),
    )
    for metric in METRICS:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_proposed"] - merged[f"{metric}_random"]
        )
    delta_columns = [f"delta_{metric}" for metric in METRICS]
    output = (
        merged.groupby(["condition", "requested_coverage"])[delta_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    output.columns = [
        "_".join(filter(None, map(str, column)))
        if isinstance(column, tuple)
        else str(column)
        for column in output.columns
    ]
    return output


def discrete_aurc(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["condition", "replicate", "method", "repeat"]
    for keys, group in frame.groupby(group_columns, sort=True):
        valid = group[np.isfinite(group.balanced_accuracy)].sort_values(
            "realized_coverage"
        )
        if len(valid) < 2:
            value = float("nan")
        else:
            coverage = valid.realized_coverage.to_numpy(dtype=float)
            risk = 1.0 - valid.balanced_accuracy.to_numpy(dtype=float)
            span = coverage[-1] - coverage[0]
            value = (
                float(np.trapezoid(risk, coverage) / span)
                if span > 0
                else float("nan")
            )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "discrete_balanced_aurc_50_100": value,
            }
        )
    return pd.DataFrame(rows)


def style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)


def plot_risk_coverage(matched: pd.DataFrame, output_dir: Path) -> None:
    selected_conditions = [
        "clean",
        "dropout_0.30",
        "missing_0.30",
        "targeted_0.30",
        "noise_5",
    ]
    methods = [
        "proposed",
        "random_rejection",
        "confidence_margin",
        "coverage_only",
        "quality_only",
        "jackknife_only",
    ]
    colors = {
        "proposed": "#005F73",
        "random_rejection": "#9E9E9E",
        "confidence_margin": "#CA6702",
        "coverage_only": "#0A9396",
        "quality_only": "#BB3E03",
        "jackknife_only": "#6A4C93",
    }
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for axis, condition in zip(axes.flat, selected_conditions):
        subset = matched[matched.condition == condition]
        summary = (
            subset.groupby(["method", "requested_coverage"])
            .balanced_accuracy.mean()
            .reset_index()
        )
        summary["balanced_risk"] = 1.0 - summary.balanced_accuracy
        for method in methods:
            line = summary[summary.method == method].sort_values(
                "requested_coverage"
            )
            axis.plot(
                line.requested_coverage,
                line.balanced_risk,
                marker="o",
                linewidth=2 if method == "proposed" else 1.2,
                color=colors[method],
                label=method.replace("_", " "),
            )
        axis.set_title(condition.replace("_", " "))
        style_axes(axis)
    axes.flat[-1].axis("off")
    for axis in axes[-1, :2]:
        axis.set_xlabel("Coverage")
    for axis in axes[:, 0]:
        axis.set_ylabel("Balanced risk (lower is better)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.08))
    figure.suptitle("Matched-coverage selective risk", fontsize=15)
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(output_dir / "fig1_matched_risk_coverage.png", dpi=220)
    plt.close(figure)


def plot_coverage_severity(metrics: pd.DataFrame, output_dir: Path) -> None:
    parsed = metrics.copy()
    parts = parsed.condition.str.extract(
        r"^(dropout|missing|targeted|noise)_(.+)$"
    )
    parsed["corruption"] = parts[0]
    parsed["severity"] = pd.to_numeric(parts[1], errors="coerce")
    parsed = parsed[parsed.corruption.notna()]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    colors = {
        "dropout": "#005F73",
        "missing": "#0A9396",
        "targeted": "#CA6702",
        "noise": "#AE2012",
    }
    structured = parsed[parsed.corruption.isin(["dropout", "missing", "targeted"])]
    for corruption, group in structured.groupby("corruption"):
        summary = group.groupby("severity").acceptance_rate.agg(["mean", "std"])
        axes[0].errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
            linewidth=2,
            color=colors[corruption],
            label=corruption,
        )
    noise = parsed[parsed.corruption == "noise"]
    if not noise.empty:
        summary = noise.groupby("severity").acceptance_rate.agg(["mean", "std"])
        axes[1].errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
            linewidth=2,
            color=colors["noise"],
            label="noise",
        )
    axes[0].set_xlabel("Affected sensor fraction")
    axes[0].set_ylabel("Acceptance rate")
    axes[0].set_title("Structured sensor corruption")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Noise SNR (dB; lower is more severe)")
    axes[1].set_title("Additive noise")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        style_axes(axis)
    figure.suptitle("Coverage response to corruption severity", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "fig2_coverage_severity.png", dpi=220)
    plt.close(figure)


def plot_failure_detection(metrics: pd.DataFrame, output_dir: Path) -> None:
    summary = metrics.groupby("condition").reliability_error_auroc.agg(
        ["mean", "std"]
    )
    summary = summary.sort_values("mean")
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = ["#AE2012" if value < 0.5 else "#0A9396" for value in summary["mean"]]
    axis.barh(
        summary.index.str.replace("_", " "),
        summary["mean"],
        xerr=summary["std"].fillna(0),
        color=colors,
        alpha=0.9,
        capsize=3,
    )
    axis.axvline(0.5, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlim(0.35, 0.8)
    axis.set_xlabel("Reliability-to-error AUROC")
    axis.set_title("Does reliability identify prediction failures?")
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output_dir / "fig3_failure_detection_auroc.png", dpi=220)
    plt.close(figure)


def plot_ablation(ablation_aurc: pd.DataFrame, output_dir: Path) -> None:
    structured = ablation_aurc[
        ablation_aurc.condition.str.startswith(("dropout", "missing", "targeted"))
    ].copy()
    summary = (
        structured.groupby("method").discrete_balanced_aurc_50_100.mean().sort_values()
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = ["#005F73" if name == "proposed" else "#94D2BD" for name in summary.index]
    axis.barh(summary.index.str.replace("_", " "), summary.values, color=colors)
    axis.set_xlabel("Discrete balanced AURC, 50-100% coverage (lower is better)")
    axis.set_title("Reliability component ablation")
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output_dir / "fig4_reliability_ablation.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/model_a_v31_paper"),
    )
    parser.add_argument("--random-repeats", type=int, default=200)
    args = parser.parse_args()

    if args.random_repeats < 1:
        raise ValueError("--random-repeats must be positive")
    scores, model = prepare_scores(args.input_dir)
    scores = add_ablation_scores(scores, model)
    metrics = enrich_metrics(
        pd.read_csv(args.input_dir / "metrics.csv"), scores
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched = matched_coverage_experiment(scores, args.random_repeats)
    ablation = ablation_experiment(scores)
    matched_aurc = discrete_aurc(matched)
    ablation_aurc = discrete_aurc(ablation)
    matched.to_csv(args.output_dir / "matched_coverage_runs.csv", index=False)
    ablation.to_csv(args.output_dir / "ablation_runs.csv", index=False)
    aggregate_table(
        matched, ["condition", "method", "requested_coverage"]
    ).to_csv(args.output_dir / "table2_matched_coverage.csv", index=False)
    aggregate_table(
        ablation, ["condition", "method", "requested_coverage"]
    ).to_csv(args.output_dir / "table3_ablation.csv", index=False)
    main_results_table(metrics).to_csv(
        args.output_dir / "table1_main_results.csv", index=False
    )
    proposed_vs_random_table(matched).to_csv(
        args.output_dir / "table4_proposed_vs_random.csv", index=False
    )
    matched_aurc.to_csv(args.output_dir / "matched_aurc_runs.csv", index=False)
    aggregate_table_aurc = (
        matched_aurc.groupby(["condition", "method"])
        .discrete_balanced_aurc_50_100.agg(["mean", "std"])
        .reset_index()
    )
    aggregate_table_aurc.to_csv(
        args.output_dir / "table5_matched_aurc.csv", index=False
    )

    plot_risk_coverage(matched, args.output_dir)
    plot_coverage_severity(metrics, args.output_dir)
    plot_failure_detection(metrics, args.output_dir)
    plot_ablation(ablation_aurc, args.output_dir)

    metadata = {
        "protocol": "model-a-v3.1-paper-analysis",
        "input_dir": str(args.input_dir),
        "random_repeats": args.random_repeats,
        "coverages": list(COVERAGES),
        "selection_uses_damage_labels": False,
        "metrics_use_damage_labels": True,
        "matched_coverage_methods": sorted(matched.method.unique().tolist()),
        "ablation_methods": sorted(ablation.method.unique().tolist()),
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(metadata, indent=2))
    print("PAPER_ANALYSIS_PASS")
    print(f"output={args.output_dir}")
    print("tables=5 figures=4")


if __name__ == "__main__":
    main()

