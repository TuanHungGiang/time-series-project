"""Explain changes in PCA-SPE ranking and alarm decisions using a saved run."""

import argparse
import json
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASETS = ("Z24", "HellBridge")
COLORS = {"Z24": "#246587", "HellBridge": "#a74727"}
CAUSE = {"lost": "#bb5235", "gained": "#277994"}
RATES = (0, 10, 30, 50, 70, 90)
ALARM_METRICS = ("recall", "fpr", "f1")


@contextmanager
def report_reader(path):
    if path.is_dir():
        root = path / "results" if (path / "results/protocol.json").is_file() else path
        yield lambda name: (root / name).read_bytes()
        return
    with ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.endswith("Z24_scores.csv.gz")]
        if len(matches) != 1:
            raise ValueError("Choose the ZIP from the 1,024-point PCA-SPE run.")
        prefix = matches[0].removesuffix("Z24_scores.csv.gz")
        yield lambda name: archive.read(prefix + name)


def read_csv(read, name):
    # Older pandas needs round-trip parsing to preserve threshold comparisons
    # when a healthy-validation cutoff is one floating-point step above a score.
    return pd.read_csv(BytesIO(read(name)), compression="gzip" if name.endswith(".gz") else None,
                       dtype={"state": str}, float_precision="round_trip")


def alarm_metrics(y, values, threshold):
    alarm = values >= threshold
    tp, fp = int(alarm[y].sum()), int(alarm[~y].sum())
    return {"recall": tp / y.sum(), "fpr": fp / (~y).sum(),
            "f1": 2 * tp / (tp + fp + y.sum()), "tp": tp, "fp": fp}


def audit_bridge(scores, metrics, bridge, orientation, focus):
    metrics = metrics.loc[metrics.orientation.eq(orientation)]
    by_key = metrics.set_index(["loss", "replicate", "policy"])
    sign = int(metrics.score_sign.unique().item())
    test = scores.loc[scores.split.eq("test")]
    if not by_key.index.is_unique or test.duplicated(["loss", "replicate", "sample_id"]).any():
        raise ValueError(f"{bridge}: duplicate thresholds or score rows")
    clean = test.loc[test.loss.eq(0)].set_index("sample_id").sort_index()
    y = clean.label.to_numpy(dtype=bool)
    expected = (6, 126) if bridge == "Z24" else (80, 640)
    if ((~y).sum(), y.sum()) != expected:
        raise ValueError(f"{bridge}: unexpected test counts")
    clean_score = sign * clean.spe_score.to_numpy()
    clean_gap = clean_score[y, None] - clean_score[~y][None, :]
    clean_credit = (clean_gap > 0) + .5 * (clean_gap == 0)
    clean_threshold = by_key.loc[(0, 0, "mask_matched"), "threshold"]
    before = alarm_metrics(y, clean_score, clean_threshold)
    ranking, alarms, states, examples = [], [], [], {}
    for (loss, replicate), trial in test.groupby(["loss", "replicate"], sort=True):
        trial = trial.set_index("sample_id")
        if set(trial.index) != set(clean.index):
            raise ValueError(f"{bridge}/{loss}/{replicate}: changed test cohort")
        trial = trial.reindex(clean.index)
        score = sign * trial.spe_score.to_numpy()
        masked_gap = score[y, None] - score[~y][None, :]
        masked_credit = (masked_gap > 0) + .5 * (masked_gap == 0)
        delta = masked_credit - clean_credit
        lost = np.maximum(-delta, 0).mean()
        gained = np.maximum(delta, 0).mean()
        current = by_key.loc[(loss, replicate, "mask_matched")]
        frozen_row = by_key.loc[(loss, replicate, "clean_frozen")]
        frozen = alarm_metrics(y, score, clean_threshold)
        matched = alarm_metrics(y, score, current.threshold)
        np.testing.assert_allclose(
            [masked_credit.mean(), lost, gained], current[["auc", "rank_credit_lost", "rank_credit_gained"]].astype(float),
        )
        for policy_values, source in ((frozen, frozen_row), (matched, current)):
            np.testing.assert_allclose(list(policy_values.values()), source[list(policy_values)].astype(float))
        meta = dict(dataset=bridge, loss=int(loss), replicate=int(replicate))
        ranking.append(dict(**meta, auc_clean=clean_credit.mean(), auc_masked=masked_credit.mean(),
                            lost=lost, gained=gained, net=gained-lost, pairs=delta.size,
                            median_healthy_score_shift=np.median((score-clean_score)[~y]),
                            median_damage_score_shift=np.median((score-clean_score)[y]),
                            strictly_flipped=int(((clean_gap > 0) & (masked_gap < 0)).sum()),
                            strictly_recovered=int(((clean_gap < 0) & (masked_gap > 0)).sum())))
        for metric in ALARM_METRICS:
            alarms.append(dict(**meta, metric=metric, clean=before[metric], frozen=frozen[metric],
                               matched=matched[metric], score_shift=frozen[metric]-before[metric],
                               threshold_shift=matched[metric]-frozen[metric],
                               clean_threshold=clean_threshold, matched_threshold=current.threshold))
        pos_state = clean.loc[y, "state"].to_numpy()
        for state in sorted(set(pos_state), key=lambda s: int(s.removeprefix("DS"))):
            selected = pos_state == state
            state_delta = delta[selected]
            states.append(dict(**meta, state=state, positives=int(selected.sum()),
                               auc_clean=clean_credit[selected].mean(), auc_masked=masked_credit[selected].mean(),
                               net=state_delta.mean(), lost=np.maximum(-state_delta, 0).mean(),
                               gained=np.maximum(state_delta, 0).mean(),
                               recall_clean=(clean_score[y][selected] >= clean_threshold).mean(),
                               recall_masked=(score[y][selected] >= current.threshold).mean()))
        if loss == focus:
            examples[int(replicate)] = masked_gap
    rank = pd.DataFrame(ranking)
    nonzero = rank.loc[rank.loss.ne(0)].groupby("loss").replicate.nunique().to_dict()
    if nonzero != dict.fromkeys(RATES[1:], 30) or len(rank) != 151:
        raise ValueError(f"{bridge}: expected clean and thirty masks at each nonzero level")
    range_df = rank.loc[rank.loss.eq(focus)]
    median = range_df.net.median()
    picked = int(range_df.assign(distance=abs(range_df.net-median)).sort_values(["distance", "replicate"]).replicate.iloc[0])
    masked_gap = examples[picked]
    case = pd.DataFrame({
        "damage_id": np.repeat(clean.index[y].to_numpy(), (~y).sum()),
        "healthy_id": np.tile(clean.index[~y].to_numpy(), y.sum()),
        "damage_state": np.repeat(pos_state, (~y).sum()),
        "clean_gap": clean_gap.flatten(), "masked_gap": masked_gap.flatten(),
    })
    case = case.assign(dataset=bridge, loss=focus, replicate=picked)
    return rank, pd.DataFrame(alarms), pd.DataFrame(states), case


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=175, facecolor="white")
    plt.close(fig)


def rank_flow(summary, out):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.05), sharey=True, layout="constrained")
    figure.suptitle("Pairs that lose or gain correct ranking after sensor loss", fontsize=14, weight="bold")
    for ax, dataset in zip(axes, DATASETS, strict=True):
        df = summary.loc[summary.dataset.eq(dataset) & summary.loss.gt(0)]
        x = np.arange(len(df))
        ax.axhline(0, color="#82919a", lw=.8)
        ax.bar(x, -100 * df.lost, color=CAUSE["lost"], width=.69, label="Lost ranking credit")
        ax.bar(x, 100 * df.gained, color=CAUSE["gained"], width=.69, label="Gained ranking credit")
        ax.plot(x, 100 * df.net, "D-", ms=4.5, color="#172b3b", label="Net change in AUC")
        ax.set(title=dataset, ylabel="Share of all healthy–damage pairs (pp)",
               xlabel="Nominal whole-sensor loss (%)", xticks=x, xticklabels=df.loss.tolist())
        ax.grid(axis="y", lw=.5, alpha=.25)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    save(figure, out, "01_rank_lost_and_gained")


def pair_scatter(cases, focus, out):
    figure, axes = plt.subplots(1, 2, figsize=(9.45, 4.3), layout="constrained")
    figure.suptitle(f"Same healthy–damage pairs before and after {focus}% loss", fontsize=14, weight="bold")
    for ax, dataset in zip(axes, DATASETS, strict=True):
        points = cases.loc[cases.dataset.eq(dataset)]
        x, y = points.clean_gap, points.masked_gap
        lost, gained = (x > 0) & (y < 0), (x < 0) & (y > 0)
        for select, color, alpha, size in (
            (~(lost | gained), "#9ba8b1", .30, 7),
            (lost, CAUSE["lost"], .48, 9), (gained, CAUSE["gained"], .48, 9),
        ):
            ax.scatter(x[select], y[select], s=size, color=color, alpha=alpha, linewidths=0, rasterized=True)
        for line in (ax.axhline(0), ax.axvline(0)):
            line.set(color="#344a57", ls="--", lw=.85)
        span = np.median(np.abs(np.r_[x, y]))
        ax.set_xscale("symlog", linthresh=max(.02, span / 10))
        ax.set_yscale("symlog", linthresh=max(.02, span / 10))
        for get_lim, get_ticks, set_ticks in (
            (ax.get_xlim, ax.get_xticks, ax.set_xticks), (ax.get_ylim, ax.get_yticks, ax.set_yticks),
        ):
            low, high = get_lim()
            ticks = [v for v in get_ticks() if low <= v <= high and (v == 0 or abs(v) >= max(span / 6, 0.5))]
            set_ticks(ticks, [f"{v:g}" for v in ticks])
        ax.set(title=f"{dataset}  |  Mask {points.replicate.iloc[0]}",
               xlabel="Damage score − healthy score, clean",
               ylabel=f"Damage score − healthy score, {focus}% loss")
        ax.text(.03, .96, f"{lost.sum():,} flipped  ·  {gained.sum():,} recovered", transform=ax.transAxes,
                va="top", fontsize=8.2, bbox=dict(fc="white", ec="none", alpha=.88))
    figure.legend(handles=[
        Line2D([0], [0], color=c, marker="o", ls="none", label=l)
        for c,l in ((CAUSE["lost"], "Correct → incorrect"), (CAUSE["gained"], "Incorrect → correct"))
    ], fontsize=8, loc="outside lower center", ncol=2, frameon=False)
    save(figure, out, "02_pairs_crossing_zero")


def alarm_decomposition(summary, out):
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.25), layout="constrained")
    figure.suptitle("Why recall and F1 change: scores and threshold adjustment", fontsize=14, weight="bold")
    for idx, dataset in enumerate(DATASETS):
        for ax, metric in zip(axes[idx], ALARM_METRICS, strict=True):
            df = summary.loc[summary.dataset.eq(dataset) & summary.metric.eq(metric)]
            ax.axhline(0, color="#82919a", lw=.8, ls="--")
            for col, label, color, marker in (
                ("score_shift", "Mask input, keep clean threshold", "#a74932", "o"),
                ("threshold_shift", "Recalibrate threshold after masking", "#297c89", "s"),
            ):
                ax.plot(df.loss, 100 * df[col], color=color, marker=marker, ms=4.2, label=label)
            ax.plot(df.loss, 100*(df.matched-df.clean), color="#27424f", marker="D", ms=4, label="Observed net change")
            title = {"recall": "Recall", "fpr": "FPR", "f1": "F1"}[metric]
            ax.set(title=f"{dataset}  |  {title}", ylabel="Change (pp)", xlabel="Nominal loss (%)", xticks=RATES)
            ax.grid(axis="y", lw=.5, alpha=.24)
    handles, labels = axes[0,0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8.2, frameon=False)
    save(figure, out, "03_recall_f1_threshold_decomposition")


def state_heatmap(summary, out):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.25), layout="constrained")
    figure.suptitle("Ranking change by damage state after sensor loss", fontsize=14, weight="bold")
    df = summary.loc[summary.loss.ne(0)]
    vmax = max(abs(df.net).max() * 100, 1)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        p = df.loc[df.dataset.eq(dataset)].pivot(index="state", columns="loss", values="net")
        p = p.reindex(sorted(p.index, key=lambda s: int(str(s).removeprefix("DS"))))
        z = 100 * p.to_numpy()
        img = ax.imshow(z, cmap="RdBu", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        ax.set(title=dataset, xlabel="Nominal loss (%)", ylabel="Damage state")
        ax.set_xticks(np.arange(len(p.columns)), p.columns)
        states = [str(s).zfill(2) if str(s).isdecimal() else str(s) for s in p.index]
        ax.set_yticks(np.arange(len(states)), states)
        for (i,j), value in np.ndenumerate(z):
            ax.text(j,i,"0.0" if abs(value)<.05 else f"{value:+.1f}",ha="center",va="center",fontsize=8.5,
                    color="white" if abs(value)>.66*vmax else "#192e39")
    figure.colorbar(img, ax=axes, shrink=.83, label="Change in AUC vs clean (pp)")
    save(figure, out, "04_rank_loss_by_damage_state")


def explain(report, out, focus, orientation):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    with report_reader(report) as read:
        protocol = json.loads(read("protocol.json"))
        if protocol.get("samples") != 1024 or protocol.get("replicates") != 30:
            raise ValueError("This analysis expects the 1,024-point PCA-SPE report with 30 mask orders.")
        for bridge in DATASETS:
            parts.append(audit_bridge(read_csv(read, f"{bridge}_scores.csv.gz"),
                                      read_csv(read, f"{bridge}_metrics_by_mask.csv"), bridge, orientation, focus))
    rank, alarm, states, pairs = (pd.concat(tables, ignore_index=True) for tables in zip(*parts, strict=True))
    rank_summary = rank.groupby(["dataset", "loss"], as_index=False).mean(numeric_only=True).drop(columns="replicate")
    alarm_summary = alarm.groupby(["dataset", "loss", "metric"], as_index=False).mean(numeric_only=True).drop(columns="replicate")
    state_summary = states.groupby(["dataset", "loss", "state"], as_index=False).mean(numeric_only=True).drop(columns="replicate")
    for name, frame in (
        ("rank_by_mask.csv", rank), ("rank_summary.csv", rank_summary), ("alarms_by_mask.csv", alarm),
        ("alarm_decomposition.csv", alarm_summary), ("damage_state_ranking.csv", state_summary),
        ("pair_examples.csv.gz", pairs),
    ):
        frame.to_csv(out / name, index=False)
    rank_flow(rank_summary, out)
    pair_scatter(pairs, focus, out)
    alarm_decomposition(alarm_summary, out)
    state_heatmap(state_summary, out)
    (out / "analysis.json").write_text(json.dumps(dict(
        orientation=orientation, points=1024, masks=30, focus_loss=focus,
        pair_illustration="mask closest to median net AUC change at the selected level",
        score_path="clean score / clean threshold -> masked score / clean threshold -> masked score / matched threshold",
        interpretation="paired audit on fixed test units; threshold attribution is for this order",
    ), indent=2), encoding="utf-8")
    print("Rank credit at the selected loss, fraction of all pairs", flush=True)
    print(rank_summary.loc[rank_summary.loss.eq(focus), ["dataset", "auc_clean", "auc_masked", "lost", "gained", "net"]].to_string(index=False))
    print("Alarm changes at the selected loss, fraction of test units or F1", flush=True)
    show = ["dataset","metric","clean","frozen","matched","score_shift","threshold_shift"]
    print(alarm_summary.loc[alarm_summary.loss.eq(focus), show].round(4).to_string(index=False))
    print(f"PCA_EXPLANATION_COMPLETE {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus", type=int, choices=RATES[1:], default=50)
    parser.add_argument("--orientation", choices=("prior_protocol", "upper"), default="prior_protocol")
    args = parser.parse_args()
    explain(args.report, args.output, args.focus, args.orientation)
