"""Plot alarm decisions and case traces from a saved bridge benchmark report."""

import argparse
import json
import re
import shutil
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = {"timesnet_native": "TimesNet", "clmm_whitened_residual": "CLMM-Net"}
CLASS_COLORS = {0: "#246b96", 1: "#bd542e"}
OUTCOMES = {
    "Detected both": "#28664f",
    "Missed after loss": "#cc5a39",
    "Detected after loss": "#5187a7",
    "Missed both": "#abb9c4",
}
KEY = ["condition", "replicate", "method"]
PRIMARY = "clmm_whitened_residual"


@contextmanager
def open_report(path):
    if path.is_dir():
        members = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
        yield members, lambda name: (path / name).read_bytes()
    else:
        with ZipFile(path) as archive:
            yield set(archive.namelist()), archive.read


def read_csv(read, path):
    return pd.read_csv(
        BytesIO(read(path)), compression="gzip" if path.endswith(".gz") else None,
        dtype={"state": str}, float_precision="round_trip",
    )


def load_shard(members, read, folder, bridge):
    points_match = re.search(r"(?:^|/)points_(\d+)(?:/|$)", folder)
    seed_match = re.search(r"(?:^|/)seed_(\d+)(?:/|$)", folder)
    if points_match and seed_match:
        points, seed = int(points_match[1]), int(seed_match[1])
    else:
        protocol = json.loads(read(f"{folder}/protocol.json"))
        input_points = protocol.get("points_per_window", protocol.get("native_points"))
        if input_points is None:
            raise ValueError(f"{folder}: input resolution is missing from protocol.json")
        points, seed = int(input_points), int(protocol["seed"])
    scores = read_csv(read, f"{folder}/scores.csv.gz")
    test = scores.loc[scores.split.eq("test")].copy()
    if test.empty or test.duplicated(["condition", "replicate", "sample_id"]).any():
        raise ValueError(f"{folder}: missing or duplicate test scores")
    clean_ids = set(test.loc[test.condition.eq("clean"), "sample_id"])
    if not clean_ids or any(set(part.sample_id) != clean_ids for _, part in test.groupby(KEY[:2])):
        raise ValueError(f"{folder}: clean and sensor-loss trials have different test cohorts")
    legacy_hell = f"{folder}/calibration_by_mask.csv" in members
    threshold_file = "calibration_by_mask.csv" if legacy_hell else "metrics_by_mask.csv"
    metrics = read_csv(read, f"{folder}/{threshold_file}")
    if legacy_hell:
        metrics = metrics.loc[metrics.policy.eq("same_mask_calibration") & np.isclose(metrics.alpha, .05)]
    else:
        metrics = metrics.loc[metrics.policy.isin(["mask_matched", "masked_q90"])]
    metrics = metrics.loc[metrics.method.isin(MODELS)]
    index = ["sample_id", "record_id", "group", "state", "label", "condition", "replicate"]
    long = test.melt(id_vars=index, value_vars=list(MODELS), var_name="method", value_name="score")
    joined = long.merge(metrics[[*KEY, "threshold", "auc", "recall", "fpr"]],
                        on=KEY, how="left", validate="many_to_one")
    if joined.threshold.isna().any() or not np.isfinite(joined[["score", "threshold"]]).all().all():
        raise ValueError(f"{folder}: unpaired or invalid test scores or thresholds")
    joined["margin"] = joined.score - joined.threshold
    joined["alarm"] = joined.score.ge(joined.threshold)
    measured = joined.groupby([*KEY, "label"]).alarm.mean().unstack()
    expected = metrics.set_index(KEY).reindex(measured.index)
    np.testing.assert_allclose(measured[1], expected.recall)
    np.testing.assert_allclose(measured[0], expected.fpr)
    joined = joined.assign(dataset=bridge, points=points, seed=seed, source=folder)

    path = f"{folder}/z24_window_scores.csv.gz"
    if path not in members:
        return joined, None
    raw = read_csv(read, path)
    raw = raw.loc[raw.split.eq("test")]
    window_scores = raw.melt(id_vars=[*index, "window_index"], value_vars=list(MODELS),
                             var_name="method", value_name="window_score")
    return joined, window_scores.assign(dataset=bridge, points=points, seed=seed)


def load_report(path):
    decisions, window_scores = [], []
    with open_report(path) as (members, read):
        for name in sorted(members):
            parts = PurePosixPath(name).parts
            if len(parts) < 2 or parts[-1] != "scores.csv.gz" or parts[-2] not in ("Z24", "HellBridge"):
                continue
            fold = "/".join(parts[:-1])
            scores, windows = load_shard(members, read, fold, parts[-2])
            decisions.append(scores)
            if windows is not None:
                window_scores.append(windows)
    if not decisions:
        raise ValueError(f"No Z24 or HellBridge per-sample score file in {path}")
    combined = pd.concat(decisions, ignore_index=True)
    if combined.duplicated(["dataset", "points", "seed", "sample_id", *KEY]).any():
        raise ValueError("Duplicate score shards inside the chosen report")
    return combined, pd.concat(window_scores, ignore_index=True) if window_scores else None


def colors_and_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#202d35", "text.color": "#202d35",
        "pdf.fonttype": 42, "axes.titleweight": "bold",
    })


def save(fig, out, name):
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=185, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def decision_figure(data, bridge, points, seed, condition, replicate, out):
    selected = data.loc[data.seed.eq(seed) & (
        data.condition.eq("clean") | (data.condition.eq(condition) & data.replicate.eq(replicate))
    )]
    conditions = ("clean", condition)
    labels = ("Complete input", f"{float(condition.split('_')[-1]):.0%} sensor loss")
    unit = "records" if bridge == "Z24" else "windows"
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharey="row")
    fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.825, wspace=.18, hspace=.39)
    fig.text(.13, .973, f"{bridge}  |  Test {unit} relative to the alarm threshold", fontsize=14, weight="bold")
    fig.text(.13, .924, f"{points:,} input points  ·  Seed {seed}  ·  Loss mask {replicate}", fontsize=9.5)
    for row, (method, method_name) in enumerate(MODELS.items()):
        for col, (status, title) in enumerate(zip(conditions, labels, strict=True)):
            ax = axes[row, col]
            values = selected.loc[selected.method.eq(method) & selected.condition.eq(status)]
            if values.empty:
                raise ValueError(f"Missing {bridge}/{method}/{status}/{replicate}")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.axhline(0, lw=1.1, color="#293742", linestyle="--")
            ax.grid(axis="y", color="#d9e1e6", alpha=.60, lw=.55)
            ax.set_axisbelow(True)
            for label in (0, 1):
                part = values.loc[values.label.eq(label)].sort_values("sample_id")
                rng = np.random.default_rng(31 + label)
                x = label + rng.uniform(-.27, .27, len(part))
                ax.scatter(x, part.margin, color=CLASS_COLORS[label], alpha=.72, s=15, linewidths=0, zorder=2)
                wrong = part.alarm.ne(bool(label)).to_numpy()
                ax.scatter(x[wrong], part.margin.to_numpy()[wrong], facecolors="none",
                           edgecolors="#343434", alpha=.68, s=18, linewidths=.40, zorder=3)
            counts = values.label.value_counts()
            ax.set_xticks([0, 1], [f"Healthy ({counts[0]})", f"Damaged ({counts[1]})"])
            ax.set_xlim(-.43, 1.43)
            if col == 0:
                ax.set_ylabel(f"{method_name}\nScore − threshold")
            ax.set_title(f"{title}   ·   Recall {values.recall.iloc[0]:.1%}   ·   FPR {values.fpr.iloc[0]:.1%}",
                         loc="left", fontsize=9.1, pad=7)
    fig.text(.13, .048, f"Each point is one {unit[:-1]}. Zero marks the alarm threshold. Outlines mark incorrect decisions.\n"
             "The symmetric-log axis expands the region near the threshold.", fontsize=8.3)
    save(fig, out, f"{bridge.lower()}_{points}_decision_margin")
    return selected


def state_transitions(data, bridge, points, condition, out):
    keys = ["seed", "sample_id", "method"]
    damaged = data.loc[data.label.eq(1)]
    clean = damaged.loc[damaged.condition.eq("clean"), [*keys, "alarm"]].rename(columns={"alarm": "clean_alarm"})
    lost = damaged.loc[damaged.condition.eq(condition)]
    table = lost.merge(clean, on=keys, how="left", validate="many_to_one")
    if table.clean_alarm.isna().any():
        raise ValueError("A masked test sample is missing its clean score")
    old, new = table.clean_alarm, table.alarm
    table["transition"] = np.select(
        [old & new, old & ~new, ~old & new], list(OUTCOMES)[:3], default="Missed both",
    )
    table["state"] = table.state.astype(str).map(lambda s: s.zfill(2) if s.isdecimal() else s)
    group = ["seed", "replicate", "method", "state"]
    encoded = pd.concat([table[group], pd.get_dummies(table.transition).reindex(columns=OUTCOMES, fill_value=0)], axis=1)
    by_seed = encoded.groupby(group)[list(OUTCOMES)].mean().groupby(["seed", "method", "state"]).mean()
    mean = by_seed.groupby(["method", "state"]).mean()
    seeds = table.seed.nunique()
    reps = table.replicate.nunique()
    loss_text = f"{float(condition.split('_')[-1]):.0%}"
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.45), sharey=True)
    fig.subplots_adjust(left=.14, right=.98, bottom=.236, top=.75, wspace=.13)
    fig.text(.14, .955, f"{bridge}  |  Which damage states are missed after sensor loss?", fontsize=14, weight="bold")
    seed_label = "seed" if seeds == 1 else "seeds"
    fig.text(.14, .894, f"Complete input → {loss_text} loss   ·   {points:,} input points   ·   {seeds} {seed_label}, {reps} masks",
             fontsize=9.3)
    for ax, (method, label) in zip(axes, MODELS.items(), strict=True):
        rows = mean.loc[method].sort_index(ascending=False)
        left = np.zeros(len(rows))
        for outcome, color in OUTCOMES.items():
            values = rows[outcome].to_numpy() * 100
            ax.barh(rows.index, values, left=left, color=color, label=outcome, height=.66)
            left = left + values
        ax.set(xlim=(0, 100), title=label, xlabel="Share of test damage units (%)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Damage state")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .07), ncol=4, fontsize=8.6, frameon=False)
    save(fig, out, f"{bridge.lower()}_{points}_state_transitions")
    return table, mean.reset_index()


def exemplar_rows(selected, condition):
    rows = selected.loc[selected.method.eq(PRIMARY) & selected.condition.eq(condition)]
    rows = rows.assign(outcome=np.select(
        [rows.label.eq(1) & rows.alarm, rows.label.eq(1), rows.alarm],
        ["Detected damage", "Missed damage", "False alarm"], default="Correct healthy",
    ))
    chosen = []
    for outcome in ("Correct healthy", "Detected damage", "Missed damage", "False alarm"):
        pool = rows.loc[rows.outcome.eq(outcome)].copy()
        if not pool.empty:
            pool["distance_to_median"] = (pool.margin - pool.margin.median()).abs()
            chosen.append(pool.sort_values(["distance_to_median", "sample_id"]).iloc[[0]])
    return pd.concat(chosen, ignore_index=True) if chosen else rows


def window_traces(windows, examples, out, z24_manifest):
    if windows is None or examples.empty:
        return None
    manifest = pd.read_csv(z24_manifest).set_index("record_id") if z24_manifest else None
    block = windows.loc[windows.method.eq(PRIMARY)].merge(
        examples[["points", "seed", "condition", "replicate", "sample_id", "threshold", "margin", "outcome"]],
        on=["points", "seed", "condition", "replicate", "sample_id"], validate="many_to_one",
    )
    block["window_margin"] = block.window_score - block.threshold
    scores = []
    fig, axs = plt.subplots(len(examples), 1, figsize=(8.9, 2.0 * len(examples) + .8), squeeze=False, layout="constrained")
    fig.suptitle("Z24  |  Window scores leading to the CLMM-Net record decision", weight="bold", fontsize=13)
    for ax, row in zip(axs.ravel(), examples.itertuples(), strict=True):
        data = block.loc[block.sample_id.eq(row.sample_id)].sort_values("window_index").copy()
        if data.empty or not np.isclose(data.window_margin.median(), row.margin):
            raise ValueError(f"Window median does not reproduce the decision on {row.sample_id}")
        if manifest is not None:
            n_windows = int(manifest.loc[row.record_id, "samples"]) // 1024
            indices = np.linspace(0, n_windows - 1, len(data), dtype=int)
            data["window_center_seconds"] = (indices * 1024 + 512) / 100
            x, xlabel = data.window_center_seconds, "Time in this record (s), sampled window centers"
        else:
            x, xlabel = data.window_index + 1, "Sampled window number within this record"
        scores.append(data)
        name = row.sample_id.replace("z24_", "").replace("_", " ")
        ax.plot(x, data.window_margin, color=CLASS_COLORS[row.label], marker="o", linestyle="none", ms=5)
        ax.axhline(0, color="#374652", lw=.9, linestyle="--")
        ax.axhline(row.margin, color=CLASS_COLORS[row.label], alpha=.50, lw=1.1)
        ax.set(title=f"{row.outcome}   ·   {name}   ·   Record margin {row.margin:+.2f}",
               ylabel="Score − threshold")
        ax.grid(axis="y", color="#dbe3e7", lw=.6)
    axs[-1, 0].set_xlabel(f"{xlabel}\nSolid line: median window score relative to the record-level threshold")
    save(fig, out, f"z24_{examples.points.iloc[0]}_case_window_scores")
    return pd.concat(scores, ignore_index=True)


def make_figures(report, out, loss=.50, seed=None, mask=None, z24_manifest=None):
    colors_and_style()
    out.mkdir(parents=True, exist_ok=True)
    scores, windows = load_report(report)
    for (bridge, points), data in scores.groupby(["dataset", "points"]):
        chosen_seed = int(data.seed.min() if seed is None else seed)
        condition = f"sensor_{loss:.2f}"
        available = data.loc[data.seed.eq(chosen_seed) & data.condition.eq(condition), "replicate"]
        if available.empty:
            raise ValueError(f"{bridge}/{points}: seed {chosen_seed} or condition {condition} unavailable")
        chosen_mask = int(available.min() if mask is None else mask)
        selected = decision_figure(data, bridge, points, chosen_seed, condition, chosen_mask, out)
        cases, mean = state_transitions(data, bridge, points, condition, out)
        stem = f"{bridge.lower()}_{points}"
        selected.to_csv(out / f"{stem}_plotted_margins.csv", index=False)
        mean.to_csv(out / f"{stem}_state_transitions_data.csv", index=False)
        cases.to_csv(out / f"{stem}_transition_cases.csv.gz", index=False)
        examples = exemplar_rows(selected, condition)
        examples.to_csv(out / f"{stem}_selected_cases.csv", index=False)
        if bridge == "Z24" and windows is not None:
            detail = window_traces(windows.loc[windows.points.eq(points)], examples, out, z24_manifest)
            if detail is not None:
                detail.to_csv(out / f"{stem}_case_window_scores.csv", index=False)
        print(f"{bridge} / {points} points: margins use seed {chosen_seed}, mask {chosen_mask}; "
              f"transitions aggregate {data.seed.nunique()} seed(s)", flush=True)
    if windows is None and scores.dataset.eq("Z24").any():
        print("The chosen report has record scores. Per-window traces require z24_window_scores.csv.gz.")
    return shutil.make_archive(str(out), "zip", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--z24-manifest", type=Path)
    parser.add_argument("--loss", type=float, choices=(.10, .30, .50), default=.50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mask", type=int)
    args = parser.parse_args()
    print(make_figures(args.report, args.output, args.loss, args.seed, args.mask, args.z24_manifest))


if __name__ == "__main__":
    main()
