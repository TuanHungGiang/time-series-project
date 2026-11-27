from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "b2": "#005F73",
    "pca_spe": "#94D2BD",
    "ocsvm": "#0A9396",
    "isolation_forest": "#E9D8A6",
    "cnn_ae": "#EE9B00",
    "masked_cnn_ae": "#CA6702",
    "deep_svdd": "#AE2012",
}


def style(axis) -> None:
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/z24_detector_paper"))
    args = parser.parse_args()
    metrics = pd.read_csv(args.input_dir / "metrics.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped = metrics.groupby(["model", "condition"]).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), runs=("auroc", "size"),
    ).reset_index()
    grouped.to_csv(args.output_dir / "table1_all_detector_results.csv", index=False)
    clean = grouped[grouped.condition == "clean"].sort_values("auroc_mean", ascending=False)
    clean.to_csv(args.output_dir / "table2_clean_comparison.csv", index=False)

    corrupted = grouped[grouped.condition != "clean"].groupby("model").agg(
        mean_corrupt_auroc=("auroc_mean", "mean"),
        worst_corrupt_auroc=("auroc_mean", "min"),
        mean_corrupt_balanced_accuracy=("balanced_accuracy_mean", "mean"),
        worst_corrupt_balanced_accuracy=("balanced_accuracy_mean", "min"),
    ).reset_index()
    clean_lookup = clean.set_index("model").auroc_mean
    corrupted["clean_auroc"] = corrupted.model.map(clean_lookup)
    corrupted["mean_auroc_drop"] = corrupted.clean_auroc - corrupted.mean_corrupt_auroc
    corrupted["worst_auroc_drop"] = corrupted.clean_auroc - corrupted.worst_corrupt_auroc
    corrupted = corrupted.sort_values(["mean_corrupt_auroc", "worst_corrupt_auroc"], ascending=False)
    corrupted.to_csv(args.output_dir / "table3_corruption_robustness.csv", index=False)

    family_rows = []
    for _, row in grouped[grouped.condition != "clean"].iterrows():
        family, _, severity = row.condition.partition("_")
        family_rows.append({**row.to_dict(), "family": family, "severity": float(severity)})
    family = pd.DataFrame(family_rows)
    family_summary = family.groupby(["model", "family"]).agg(
        auroc_mean=("auroc_mean", "mean"),
        balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
    ).reset_index()
    family_summary.to_csv(args.output_dir / "table4_corruption_family.csv", index=False)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    ordered = clean.sort_values("auroc_mean")
    axis.barh(
        ordered.model.str.replace("_", " "), ordered.auroc_mean,
        xerr=ordered.auroc_std.fillna(0),
        color=[COLORS.get(item, "#777777") for item in ordered.model], capsize=3,
    )
    axis.set_xlim(max(0.0, ordered.auroc_mean.min() - 0.10), min(1.0, ordered.auroc_mean.max() + 0.05))
    axis.set_xlabel("Clean AUROC")
    axis.set_title("Healthy-only Z24 detector comparison")
    style(axis)
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig1_clean_detector_auroc.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    ordered = corrupted.sort_values("mean_corrupt_auroc")
    axis.barh(
        ordered.model.str.replace("_", " "), ordered.mean_corrupt_auroc,
        color=[COLORS.get(item, "#777777") for item in ordered.model],
    )
    axis.scatter(ordered.worst_corrupt_auroc, np.arange(len(ordered)), color="black", marker="|", s=180, label="worst condition")
    axis.set_xlabel("AUROC across corruptions")
    axis.set_title("Mean and worst-case corruption robustness")
    axis.legend(frameon=False)
    style(axis)
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig2_corruption_robustness.png", dpi=220)
    plt.close(figure)

    families = [name for name in ("dropout", "missing", "noise", "targeted") if name in set(family.family)]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, family_name in zip(axes.flat, families):
        subset = family[family.family == family_name]
        for model_name, model_rows in subset.groupby("model"):
            values = model_rows.sort_values("severity")
            axis.plot(values.severity, values.auroc_mean, marker="o", linewidth=1.8, color=COLORS.get(model_name), label=model_name)
        axis.set_title(family_name.capitalize())
        axis.set_xlabel("SNR dB" if family_name == "noise" else "Affected fraction")
        style(axis)
    for axis in axes[:, 0]:
        axis.set_ylabel("AUROC")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, [x.replace("_", " ") for x in labels], loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Detector robustness versus corruption severity", fontsize=14)
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    figure.savefig(args.output_dir / "fig3_auroc_vs_severity.png", dpi=220)
    plt.close(figure)

    pivot = family_summary.pivot(index="model", columns="family", values="auroc_mean")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    image = axis.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=max(0.5, np.nanmin(pivot.to_numpy()) - 0.05), vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), [x.capitalize() for x in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [x.replace("_", " ") for x in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axis.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center")
    axis.set_title("Mean AUROC by corruption family")
    figure.colorbar(image, ax=axis, label="AUROC")
    figure.tight_layout()
    figure.savefig(args.output_dir / "fig4_corruption_heatmap.png", dpi=220)
    plt.close(figure)

    print(clean.to_string(index=False))
    print(corrupted.to_string(index=False))
    print(f"DETECTOR_ANALYSIS_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

