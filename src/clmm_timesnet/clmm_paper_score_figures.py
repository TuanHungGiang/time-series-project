"""Make paper figures from saved CLMM score reports. No model training is required."""

import argparse
from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

METHODS = ("timesnet_native", "clmm_whitened_residual")
NAMES = ("TimesNet", "CLMM whitening")
DATASETS = ("Z24", "HellBridge")
MASKS = set(range(300, 305))
CONDITIONS = ("sensor_0.10", "sensor_0.30", "sensor_0.50")
KEY = ["condition", "replicate", "method"]
COLORS = ("#315d83", "#a2543b")
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=195)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def open_report(path, stack):
    if path.is_dir():
        return lambda name: (path / name).read_bytes()
    archive = stack.enter_context(ZipFile(path))
    return archive.read


def read_csv(read, member):
    return pd.read_csv(BytesIO(read(member)), compression="gzip" if member.endswith(".gz") else None,
                       float_precision="round_trip")


def match_scores(read, bridge):
    protocol = json.loads(read(f"{bridge}/protocol.json"))
    seed = int(protocol["seed"])
    scores = read_csv(read, f"{bridge}/scores.csv.gz")
    scores = scores.loc[scores.condition.isin(CONDITIONS) & scores.split.eq("test")]
    if bridge == "Z24":
        metrics = read_csv(read, "Z24/metrics_by_mask.csv").query("policy == 'masked_q90'")
    else:
        metrics = read_csv(read, "HellBridge/calibration_by_mask.csv").query(
            "policy == 'same_mask_calibration' and alpha == 0.05")
    metrics = metrics.loc[metrics.condition.isin(CONDITIONS) & metrics.method.isin(METHODS)]
    if metrics.duplicated(KEY).any() or len(metrics) != len(CONDITIONS) * len(MASKS) * len(METHODS):
        raise ValueError(f"Unexpected threshold rows for {bridge}, seed {seed}")
    expected_counts = {0: 6, 1: 126} if bridge == "Z24" else {0: 80, 1: 640}
    expected = {(condition, mask) for condition in CONDITIONS for mask in MASKS}
    if set(scores[KEY[:2]].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"{bridge}: expected exactly five masks per loss level")
    for _, part in scores.groupby(KEY[:2]):
        if not part.sample_id.is_unique or part.label.value_counts().to_dict() != expected_counts:
            raise ValueError(f"{bridge}: unexpected test class counts")
    long = scores.melt(id_vars=[*KEY[:2], "sample_id", "label", "state"],
                       value_vars=METHODS, var_name="method", value_name="score")
    joined = long.merge(metrics[[*KEY, "threshold", "recall", "fpr", "auc"]], on=KEY, validate="many_to_one")
    if len(joined) != len(long):
        raise ValueError(f"Missing threshold entries for {bridge}")
    joined["margin"] = joined.score - joined.threshold
    joined["alarm"] = joined.score.ge(joined.threshold)
    actual = joined.groupby([*KEY, "label"]).alarm.mean().unstack()
    truth = metrics.set_index(KEY).loc[actual.index]
    np.testing.assert_allclose(actual[0], truth.fpr)
    np.testing.assert_allclose(actual[1], truth.recall)
    return joined.assign(dataset=bridge, seed=seed, state=joined.state.astype(str))


def read_reports(paths):
    all_rows, seen = [], set()
    for path in paths:
        with ExitStack() as stack:
            read = open_report(path, stack)
            parts = [match_scores(read, d) for d in DATASETS]
            seed = int(parts[0].seed.iloc[0])
            if any(not part.seed.eq(seed).all() for part in parts):
                raise ValueError(f"Seed mismatch inside {path}")
            if seed not in seen:
                all_rows.extend(parts)
                seen.add(seed)
    if not all_rows:
        raise ValueError("Supply a full seed report directory or *_report.zip")
    return pd.concat(all_rows, ignore_index=True)


def state_heatmap(scores, out):
    damage = scores.loc[scores.label.eq(1)]
    index = ["dataset", "seed", "state", "condition", "replicate"]
    rates = damage.groupby([*index, "method"], as_index=False).alarm.mean()
    paired = rates.pivot(index=index, columns="method", values="alarm").reset_index()
    if paired[list(METHODS)].isna().any().any():
        raise ValueError("Incomplete damage-state pairing")
    paired["delta_pp"] = 100 * (paired[METHODS[1]] - paired[METHODS[0]])
    by_seed = paired.groupby(["dataset", "seed", "state", "condition"], as_index=False)[
        [*METHODS, "delta_pp"]].mean()
    mean = by_seed.groupby(["dataset", "state", "condition"], as_index=False)[
        [*METHODS, "delta_pp"]].mean()
    by_seed.to_csv(out / "recall_by_state_seed.csv", index=False)
    mean.to_csv(out / "recall_by_state_summary.csv", index=False)
    bound = max(10, 10 * np.ceil(mean.delta_pp.abs().max() / 10))
    palette = LinearSegmentedColormap.from_list("recall_change", ["#a54e3d", "#f9f8f5", "#176f72"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), layout="constrained")
    for ax, bridge in zip(axes, DATASETS, strict=True):
        bridge_mean = mean.loc[mean.dataset.eq(bridge)]
        table = bridge_mean.pivot(index="state", columns="condition", values="delta_pp").reindex(columns=CONDITIONS)
        if bridge == "Z24":
            table = table.loc[sorted(table.index, key=int)]
            labels = [f"State {int(x):02d}" for x in table.index]
        else:
            labels = table.index
        image = ax.imshow(table.to_numpy(), cmap=palette, vmin=-bound, vmax=bound, aspect="auto")
        ax.set(title=bridge, xlabel="Nominal sensor loss", yticks=range(len(labels)), yticklabels=labels,
               xticks=[0, 1, 2], xticklabels=["10%", "30%", "50%"])
        ax.tick_params(length=0)
        for (row, col), val in np.ndenumerate(table.to_numpy()):
            val = 0.0 if abs(val) < .05 else float(val)
            ax.text(col, row, f"{val:+.1f}" if val else "0.0", ha="center", va="center", fontsize=8.8,
                    color="white" if abs(val) > .63 * bound else "#1d3346")
    colorbar = fig.colorbar(image, ax=axes, shrink=.85)
    colorbar.set_label("CLMM minus TimesNet recall (pp)")
    fig.suptitle(f"Damage-state recall at healthy thresholds | {scores.seed.nunique()} seed(s), 5 masks per level")
    save_figure(fig, out, "damage_state_recall")


def score_ecdf(scores, out):
    selected = scores.loc[scores.condition.eq("sensor_0.50")]
    selected[["dataset", "seed", "replicate", "sample_id", "state", "label", "method", "score", "threshold",
              "margin", "alarm"]].to_csv(out / "score_margins_50.csv.gz", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.0), sharey=True, layout="constrained")
    for row, bridge in enumerate(DATASETS):
        for col, (method, name) in enumerate(zip(METHODS, NAMES, strict=True)):
            ax = axes[row, col]
            part = selected.loc[selected.dataset.eq(bridge) & selected.method.eq(method)]
            summary = part.drop_duplicates(["seed", "replicate"])[["auc", "recall", "fpr"]].mean()
            for label, label_name, color in zip((0, 1), ("Healthy", "Damage"), COLORS, strict=True):
                margins = np.sort(part.loc[part.label.eq(label), "margin"].to_numpy())
                ax.step(margins, np.arange(1, len(margins) + 1) / len(margins), where="post",
                        label=label_name, color=color, lw=1.55)
            ax.axvline(0, color="#4e6578", linestyle="--", linewidth=.9, label="Alarm threshold")
            ax.set(title=f"{bridge} · {name}\nAUC {summary.auc:.3f}  |  Recall {summary.recall:.1%}  |  FPR {summary.fpr:.1%}",
                   xlabel="Score minus healthy threshold", xscale="symlog")
            if col == 0:
                ax.set_ylabel("Cumulative proportion")
            ax.grid(axis="y", alpha=.23)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Score distributions · 50% sensor loss · {scores.seed.nunique()} seed(s)")
    save_figure(fig, out, "healthy_damage_scores_50")


def split_diagram(out):
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.set(xlim=(0, 12), ylim=(0, 3.6))
    ax.axis("off")
    data = [
        (2.36, "Z24", "Record split", ["24 healthy records", "6 healthy + 126 damage",
                                      "6 healthy + 126 damage"]),
        (.85, "HellBridge", "Window split", ["50 healthy fit + 10 reference", "20 healthy calibration",
                                            "80 healthy + 640 damage"]),
    ]
    for y, bridge, unit, texts in data:
        ax.text(.10, y + .45, bridge, ha="left", va="center", fontsize=10.5, weight="bold")
        ax.text(.10, y + .12, unit, ha="left", va="center", fontsize=8.5, color="#526b7d")
        for i, (title, description) in enumerate(zip(("Fit / reference", "Validation", "Test"), texts, strict=True)):
            x = 2.02 + 3.29 * i
            ax.add_patch(FancyBboxPatch((x, y), 2.94, .83, boxstyle="round,pad=.012,rounding_size=.08",
                                        facecolor=["#eff5f9", "#edf6f3", "#f7f2eb"][i],
                                        edgecolor="#a9bdc8", lw=.75))
            ax.text(x + 1.47, y + .57, title, ha="center", va="center", fontsize=9.8, weight="bold")
            ax.text(x + 1.47, y + .24, description, ha="center", va="center", fontsize=8.35)
            if i < 2:
                ax.add_patch(FancyArrowPatch((x + 2.99, y + .41), (x + 3.23, y + .41), arrowstyle="-|>",
                                             color="#718496", mutation_scale=7))
    ax.text(.10, .17, "HellBridge healthy windows in development and test come from the same two source recordings.",
            ha="left", va="center", fontsize=8.4, color="#526b7d")
    fig.tight_layout(pad=.6)
    save_figure(fig, out, "two_bridge_split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_diagram(args.output_dir)
    if args.reports:
        scores = read_reports(args.reports)
        state_heatmap(scores, args.output_dir)
        score_ecdf(scores, args.output_dir)
        print(f"Plots use {scores.seed.nunique()} unique seed reports.", flush=True)
    print(f"CLMM_PAPER_SCORE_FIGURES_READY {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
