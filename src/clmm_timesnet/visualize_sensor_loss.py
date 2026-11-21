"""Visual diagnosis of the paired score-calibration experiment."""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, resample

from .data.z24 import load_manifest, read_stable_vertical, record_windows, stable_sensor_map


CONDITIONS = ["clean", "sensor_0.10", "sensor_0.30", "sensor_0.50"]
TICKS = ["Clean", "10%", "30%", "50%"]
BRIDGES = ["Z24", "HellBridge"]
METHODS = {"observed_mean": "TimesNet", "mask_aware": "Mask-aware TimesNet"}
COLORS = {"observed_mean": "#596e83", "mask_aware": "#006d77"}


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def performance(metrics, out):
    fields = [
        ("auc", "ROC-AUC"), ("recall_at_fpr_05", "Recall at FPR ≤ 5% (ROC)"),
        ("recall", "Recall at Q90"), ("fpr", "FPR at Q90"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
    for row, bridge in enumerate(BRIDGES):
        subset = metrics[metrics.dataset.eq(bridge) & metrics.policy.eq("masked_q90")]
        for col, (field, title) in enumerate(fields):
            ax = axes[row, col]
            for method, label in METHODS.items():
                groups = subset[subset.method.eq(method)].groupby("condition")[field]
                stats = groups.agg(["mean", "min", "max"]).reindex(CONDITIONS) * 100
                x = np.arange(len(CONDITIONS))
                ax.fill_between(x, stats["min"], stats["max"], color=COLORS[method], alpha=.15)
                ax.plot(x, stats["mean"], "o-", label=label, color=COLORS[method])
            ax.set(title=title, xticks=range(4), xticklabels=TICKS)
            ax.grid(axis="y", alpha=.25)
            if col == 0:
                ax.set_ylabel(f"{bridge} (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sensor loss and detection performance · bands show mask min–max")
    save(fig, out, "01_performance_by_sensor_loss")


def test_scores(scores, metrics, metadata):
    keys = ["condition", "replicate"]
    selected = scores[scores.split.eq("test") & scores.method.eq("mask_aware")].copy()
    selected = selected.merge(metadata, on="sample_id", validate="many_to_one")
    thresholds = metrics[metrics.method.eq("mask_aware") & metrics.policy.eq("masked_q90")]
    selected = selected.merge(thresholds[keys + ["threshold"]], on=keys, validate="many_to_one")
    if len(selected) != len(scores[scores.split.eq("test") & scores.method.eq("mask_aware")]):
        raise ValueError("Missing metadata or threshold in test scores")
    selected["alarm"] = selected.oriented_score.ge(selected.threshold)
    return selected


def score_overlap(test_by_bridge, metrics, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    examples = []
    for row, (bridge, test) in enumerate(test_by_bridge.items()):
        trials = metrics[
            metrics.dataset.eq(bridge) & metrics.method.eq("mask_aware")
            & metrics.policy.eq("masked_q90") & metrics.condition.eq("sensor_0.50")
        ].sort_values(["recall_at_fpr_05", "replicate"])
        cases = [("clean", 0, "Clean"),
                 ("sensor_0.50", trials.iloc[0].replicate, "50% loss · lowest low-FPR recall"),
                 ("sensor_0.50", trials.iloc[-1].replicate, "50% loss · highest low-FPR recall")]
        for ax, (condition, rep, title) in zip(axes[row], cases, strict=True):
            group = test[test.condition.eq(condition) & test.replicate.eq(rep)]
            threshold = group.threshold.iloc[0]
            for label, name, color in [(0, "Healthy", "#32618a"), (1, "Damage", "#bf4e30")]:
                ordered = np.sort(group.loc[group.label.eq(label), "oriented_score"])
                ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                        where="post", label=name, color=color)
            ax.axvline(threshold, ls="--", color="#333333", label="Q90")
            ax.set(title=f"{bridge} · {title} ({int(rep)})", xlabel="Oriented anomaly score", ylabel="ECDF")
            ax.set_xscale("symlog", linthresh=1.0)
            ax.grid(alpha=.2)
            examples.append(dict(dataset=bridge, condition=condition, replicate=int(rep), panel=title))
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, out, "02_score_overlap")
    pd.DataFrame(examples).to_csv(out / "score_overlap_panels.csv", index=False)


def damage_states(test_by_bridge, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    tables = []
    for (bridge, test), ax in zip(test_by_bridge.items(), axes, strict=True):
        damage = test[test.label.eq(1)]
        by_mask = damage.groupby(["state", "condition", "replicate"], as_index=False).agg(
            recall=("alarm", "mean"), records_or_windows=("alarm", "size"),
        )
        tables.append(by_mask.assign(dataset=bridge))
        values = by_mask.pivot_table(index="state", columns="condition", values="recall").reindex(columns=CONDITIONS)
        img = ax.imshow(values * 100, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set(title=bridge, xticks=range(4), xticklabels=TICKS,
               yticks=range(len(values)), yticklabels=values.index, xlabel="Sensor loss")
        for i, line in enumerate(values.to_numpy() * 100):
            for j, value in enumerate(line):
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                            color="white" if value >= 63 else "#18212a", fontsize=8)
    fig.colorbar(img, ax=axes, label="Recall at Q90 (%)")
    save(fig, out, "03_recall_by_damage_state")
    pd.concat(tables, ignore_index=True).to_csv(out / "recall_by_damage_state_and_mask.csv", index=False)


def alarm_tradeoff(metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    selected = metrics[metrics.method.eq("mask_aware") & metrics.condition.eq("sensor_0.50")]
    for bridge, ax in zip(BRIDGES, axes, strict=True):
        data = selected[selected.dataset.eq(bridge)].pivot(index="replicate", columns="policy", values=["fpr", "recall"]) * 100
        for rep, row in data.iterrows():
            start = (row[("fpr", "clean_q90")], row[("recall", "clean_q90")])
            end = (row[("fpr", "masked_q90")], row[("recall", "masked_q90")])
            ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"->", "color":"#77808a"})
            ax.annotate(str(rep), end, textcoords="offset points", xytext=(4, 4), fontsize=8)
        for policy, label, color in [("clean_q90", "Clean Q90", "#bf4e30"), ("masked_q90", "Mask Q90", "#006d77")]:
            ax.scatter(data[("fpr", policy)], data[("recall", policy)], label=label, color=color)
        ax.set(title=bridge, xlabel="FPR (%)", ylabel="Recall (%)")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Changing the threshold at 50% sensor loss · one arrow per mask")
    save(fig, out, "04_q90_tradeoff")


def waveform_examples(z24, hb, hell_path, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    examples = []
    zgroup = z24[z24.setup.eq(3) & z24.excitation.eq("fvt") & z24.split.eq("test")]
    channels = stable_sensor_map(z24)[(3, "fvt")]
    idx = len(channels) // 2
    for label in (0, 1):
        row = zgroup[zgroup.label.eq(label)].sort_values("record_id").iloc[0]
        raw = read_stable_vertical(Path(row.path), channels)
        prepared = record_windows(raw, 1024, 1024, "raw_temporal", 2, 128)[0]
        name = "Healthy" if label == 0 else f"Damage state {row.state}"
        axes[0, 0].plot(raw[idx, :1024], alpha=.8, lw=.8, label=name)
        axes[0, 1].plot(prepared[idx], alpha=.8, lw=.8, label=name)
        examples.append(dict(dataset="Z24", sample_id=row.record_id, sensor=channels[idx], raw_length=1024,
                             input_length=prepared.shape[-1], input_channels=len(channels)))
    with h5py.File(hell_path, "r") as h5:
        for state in ("UDS", "DS1", "DS4"):
            row = hb[(hb.state.eq(state)) & hb.role.isin(["test_healthy","test_damage"])].sort_values("sample_id").iloc[0]
            raw = np.asarray(h5[row.record_id]["acceleration"]["AL20"]["z"][int(row.start):int(row.stop)])
            prepared = resample(detrend(raw, type="linear"), 128)
            axes[1, 0].plot(raw, alpha=.8, lw=.8, label=state)
            axes[1, 1].plot(prepared, alpha=.8, lw=.8, label=state)
            examples.append(dict(dataset="HellBridge", sample_id=row.sample_id, sensor="AL20/z",
                                 raw_length=len(raw), input_length=128, input_channels=40))
    for row, bridge in enumerate(BRIDGES):
        for col, stage in enumerate(("Raw window", "After detrending and resampling")):
            ax = axes[row, col]
            ax.set(title=f"{bridge} · {stage}", xlabel="Sample index", ylabel="Recorded amplitude")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=.2)
    save(fig, out, "05_waveform_examples")
    pd.DataFrame(examples).to_csv(out / "waveform_examples.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--hell-h5", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir
    out = run / "figures"
    out.mkdir(exist_ok=True)
    metrics = pd.read_csv(run / "metrics_per_mask.csv")
    z24 = load_manifest(args.z24_root)
    hb = pd.read_csv(run / "hellbridge_split" / "sample_manifest.csv")
    metadata = {
        "Z24": z24[["record_id", "state"]].rename(columns={"record_id": "sample_id"}),
        "HellBridge": hb[["sample_id", "state"]],
    }
    test = {bridge: test_scores(pd.read_csv(run / f"{bridge}_scores.csv.gz"),
                               metrics[metrics.dataset.eq(bridge)], metadata[bridge])
            for bridge in BRIDGES}
    performance(metrics, out)
    score_overlap(test, metrics, out)
    damage_states(test, out)
    alarm_tradeoff(metrics, out)
    waveform_examples(z24, hb, args.hell_h5, out)
    print(f"BRIDGE_VISUALIZATION_PASS {out}", flush=True)


if __name__ == "__main__":
    main()
