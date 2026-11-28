"""Generate the two-bridge sensor-loss atlas from PCA-SPE scores and two Z24 waveform cases."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from clmm_timesnet.data.masking import stable_rng
from clmm_timesnet.data.z24 import load_manifest, split_training_windows
from clmm_timesnet.explain_pca_sensor_loss import DATASETS, explain, plt, read_csv, report_reader, save
from clmm_timesnet.pca_sensor_loss_1024 import PCASPE, POINTS, read_z24_windows, select_windows


HEALTHY = "#277994"
DAMAGE = "#bb5235"
PROTECTED = ("431", "441")
CASES = (
    ("08/AVT", "431", "z24_s01_avt_setup08", "z24_s15_avt_setup08"),
    ("09/AVT", "441", "z24_s01_avt_setup09", "z24_s12_avt_setup09"),
)


def pipeline(out):
    fig, ax = plt.subplots(figsize=(10.6, 3.1), layout="constrained")
    ax.set(xlim=(0, 10.6), ylim=(0, 3.1))
    ax.set_axis_off()
    ax.set_title("From missing sensor locations to ranking and alarms", weight="bold", pad=12)
    nodes = [
        (1.15, 1.65, "1,024-point windows\nDetrend and healthy scaling", "#e6f1f4"),
        (3.65, 1.65, "Remove entire locations\nZero-fill all missing axes", "#f6eae4"),
        (6.15, 1.65, "Reconstruct with PCA\nHealthy-trained basis", "#e6f1f4"),
        (8.55, 1.65, "Residual score\nApply score direction", "#e6f1f4"),
        (8.55, 2.65, "Compare scores\nHealthy–damage ranking", "#e9edf1"),
        (8.55, .65, "Apply healthy threshold\nRecall, FPR, F1", "#e9edf1"),
    ]
    for x, y, label, fill in nodes:
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=.65", ec="none", fc=fill))
    for x1, x2 in ((2.19, 2.73), (4.6, 5.24), (7.04, 7.72)):
        ax.annotate("", (x2, 1.65), (x1, 1.65), arrowprops=dict(arrowstyle="->", color="#626c75"))
    for y1, y2 in ((1.98, 2.28), (1.32, 1.05)):
        ax.annotate("", (8.55, y2), (8.55, y1), arrowprops=dict(arrowstyle="->", color="#626c75"))
    save(fig, out, "00_scoring_pipeline")


def score_distributions(scores, metrics, examples, focus, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True, layout="constrained")
    fig.suptitle(f"Healthy and damage scores before and after {focus}% sensor loss", weight="bold")
    rows = []
    for axis, bridge in zip(axes, DATASETS, strict=True):
        picked = int(examples.loc[examples.dataset.eq(bridge), "replicate"].iloc[0])
        metric = metrics[bridge].query("orientation == 'prior_protocol' and policy == 'clean_frozen'")
        sign = int(metric.score_sign.iloc[0])
        clean_threshold = metric.query("loss == 0").threshold.item()
        test = scores[bridge].query("split == 'test'")
        selection = ((0, 0, "-"), (focus, picked, "--"))
        for loss, replicate, style in selection:
            cohort = test.loc[test.loss.eq(loss) & test.replicate.eq(replicate)]
            for label, name, color in ((0, "Healthy", HEALTHY), (1, "Damage", DAMAGE)):
                values = np.sort(sign * cohort.loc[cohort.label.eq(label), "spe_score"].to_numpy())
                axis.step(values, np.arange(1, len(values)+1) / len(values), where="post",
                          linewidth=1.4, color=color, linestyle=style, label=f"{name} · {loss}%")
                rows.append(dict(dataset=bridge, loss=loss, replicate=replicate, label=name,
                                 units=len(values), score_q25=np.quantile(values, .25),
                                 score_median=np.median(values), score_q75=np.quantile(values, .75)))
        axis.axvline(clean_threshold, lw=1, color="#444444", ls=":", label="Clean alarm threshold")
        axis.set(title=f"{bridge} · mask order {picked}", xlabel="Oriented PCA-SPE score",
                 ylabel="Fraction at or below score")
        axis.grid(axis="y", alpha=.22)
    axes[0].legend(frameon=False, fontsize=7.8, loc="upper left")
    pd.DataFrame(rows).to_csv(out / "score_distributions.csv", index=False)
    save(fig, out, "05_score_distributions")


def load_group(group, meta, cohort):
    layout = SimpleNamespace(channels=group["channels"])
    sites = np.asarray(group["sensors"])
    axis_sites = np.array([group["sensors"].index(channel[:-1]) for channel in layout.channels])

    def load_record(key):
        path = Path(meta.at[key, "path"])
        labels = loadmat(path, variable_names=["labelshulp"])["labelshulp"].ravel()
        offered = set(layout.channels).intersection(str(label).strip() for label in labels)
        return read_z24_windows(path, layout, offered)

    fit, reference = split_training_windows([load_record(k)[0] for k in group["fit_records"]])
    detector = PCASPE(select_windows(fit), select_windows(reference))
    records = {}
    for record in cohort.index:
        raw, available = load_record(record)
        windows = select_windows(raw)
        records[record] = (detector.terms(windows), available, windows)
    return detector, layout.channels, sites, axis_sites, records


def loss_comparison(group, cohort, sites, axis_sites, records, sign, size, orders):
    """Use the same device pool and pair denominator for both simulated outage designs."""
    common = np.logical_and.reduce([available for _, available, _ in records.values()])
    measured_sites = np.unique(axis_sites[common])
    eligible = measured_sites[~np.isin(sites[measured_sites], PROTECTED)]
    count = min(max(1, round(size / 100 * len(measured_sites))), len(measured_sites) - 1)
    if count > len(eligible):
        return []
    rows = []
    keys = list(records)
    label = cohort.loc[keys, "label"].to_numpy()
    for rep in range(1, orders+1):
        shared = stable_rng(group, "audit-common-mask-protect-431-441", rep).choice(eligible, count, replace=False)
        for design in ("shared", "independent"):
            scores, masks = [], []
            for key in keys:
                scorer, available, _ = records[key]
                drop = shared if design == "shared" else stable_rng(
                    key, "audit-independent-mask-protect-431-441", rep
                ).choice(eligible, count, replace=False)
                scores.append(sign * np.median(scorer(available & ~np.isin(axis_sites, drop))))
                masks.append("|".join(sorted(sites[drop])))
            for i in np.flatnonzero(label == 0):
                for j in np.flatnonzero(label == 1):
                    scores_clean = cohort.spe_score
                    gap_clean = sign * (scores_clean.loc[keys[j]] - scores_clean.loc[keys[i]])
                    gap_masked = scores[j] - scores[i]
                    rows.append(dict(group=group, design=design, replicate=rep, healthy=keys[i], damage=keys[j],
                                     dropped_sensor_count=count, common_sensor_count=len(measured_sites),
                                     healthy_mask=masks[i], damage_mask=masks[j], clean_gap=gap_clean,
                                     masked_gap=gap_masked, initially_correct=gap_clean > 0,
                                     flipped=gap_clean > 0 and gap_masked < 0))
    return rows


def z24_diagnostics(root, inventory, clean, sign, focus, orders):
    meta = load_manifest(root).set_index("record_id")
    single, masks, traces, case_scores, attribution = [], [], [], [], []
    with threadpool_limits(limits=4):
        for group in inventory:
            name = group["group"]
            cohort = clean.loc[clean.group.eq(name)]
            if cohort.label.nunique() < 2:
                continue
            print(f"Audit isolated location and matched masks | {name}", flush=True)
            detector, channels, sites, axis_sites, records = load_group(group, meta, cohort)
            for key, (scorer, available, _) in records.items():
                np.testing.assert_allclose(np.median(scorer(available)), cohort.spe_score.loc[key], atol=5e-3, rtol=2e-4)
            for i, sensor in enumerate(sites):
                available_scores = {
                    key: sign * np.median(score(available & (axis_sites != i)))
                    for key, (score, available, _) in records.items() if np.any(available & (axis_sites == i))
                }
                for h in cohort.loc[cohort.label.eq(0)].index.intersection(available_scores):
                    for d in cohort.loc[cohort.label.eq(1)].index.intersection(available_scores):
                        gap0 = sign * (cohort.spe_score.loc[d] - cohort.spe_score.loc[h])
                        gap1 = available_scores[d] - available_scores[h]
                        single.append(dict(group=name, sensor=sensor, healthy=h, damage=d,
                                           clean_gap=gap0, masked_gap=gap1, initially_correct=gap0 > 0,
                                           flipped=gap0 > 0 and gap1 < 0))
            masks.extend(loss_comparison(name, cohort, sites, axis_sites, records, sign, focus, orders))
            for _, sensor, healthy, damage in (case for case in CASES if case[0] == name):
                index = int(np.flatnonzero(sites == sensor).item())
                vertical = channels.index(sensor+"V")
                for key, label in ((healthy, "Healthy"), (damage, "Damage")):
                    scorer, available, windows = records[key]
                    normalized = detector.normalizer.transform(windows)
                    values = normalized[0, vertical]
                    flat = normalized.reshape(len(normalized), -1)
                    residual = (flat-detector.model.inverse_transform(detector.model.transform(flat)))**2
                    axis_error = residual.reshape(normalized.shape).sum(axis=(0, 2))
                    for j, error in enumerate(axis_error):
                        attribution.append(dict(group=name, sensor=sensor, sample_id=key, label=label,
                                                channel=channels[j], is_target_axis=axis_sites[j] == index,
                                                share=float(error / axis_error.sum())))
                    for j, value in enumerate(values):
                        traces.append(dict(group=name, sensor=sensor, sample_id=key, label=label,
                                           seconds=j / 100, standardized_acceleration=float(value)))
                    case_scores.append(dict(group=name, sensor=sensor, sample_id=key, label=label,
                                            clean=sign * np.median(scorer(available)),
                                            dropped=sign * np.median(scorer(available & (axis_sites != index)))))
    return (pd.DataFrame(table) for table in (single, masks, traces, case_scores, attribution))


def plot_mask_design(single, masks, focus, out):
    site = single.groupby(["group", "sensor"], as_index=False).flipped.sum().sort_values(
        ["flipped", "group", "sensor"], ascending=[False, True, True]
    )
    affected = site.loc[site.flipped.gt(0)]
    eligible = masks.loc[masks.initially_correct]
    by_design = eligible.groupby(["group", "design"]).flipped.agg(["sum", "size"]).reset_index()
    totals = eligible.groupby("design").flipped.agg(["sum", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    fig.suptitle("Z24 · sensor location and mask design", weight="bold")
    labels = affected.group + " · site " + affected.sensor
    axes[0].barh(labels, affected.flipped, color=DAMAGE, height=.52)
    axes[0].invert_yaxis()
    axes[0].set(title=f"Isolated removal · {len(site)} site–group trials",
                xlabel="Initially correct pairs that flipped", xlim=(0, max(1, affected.flipped.max()) + .8))
    for i, flips in enumerate(affected.flipped):
        axes[0].text(flips+.1, i, str(flips), va="center")
    groups = sorted(by_design.group.unique())
    pos = np.arange(len(groups))
    for i, (design, color) in enumerate((("shared", HEALTHY), ("independent", DAMAGE))):
        values = by_design.loc[by_design.design.eq(design)].set_index("group").loc[groups]
        offset = .17 * (2*i-1)
        pct = 100 * values["sum"] / values["size"]
        label = f"{design.capitalize()} ({totals.at[design, 'sum']}/{totals.at[design, 'size']})"
        axes[1].bar(pos+offset, pct, width=.32, color=color, label=label)
        axes[1].scatter(pos+offset, pct, color=color, s=10)
    axes[1].set(title=f"{focus}% loss · sites 431 and 441 retained where present",
                ylabel="% of initially correct pairs flipped", xlabel="Setup / excitation",
                xticks=pos, xticklabels=groups)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(alpha=.2)
    by_design.to_csv(out / "common_vs_independent_masks.csv", index=False)
    save(fig, out, "06_sensor_location_and_mask_design")


def waveform_cases(traces, scores, attribution, out):
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.3), layout="constrained",
                             width_ratios=(1.55, 1, 1))
    fig.suptitle("Z24 · waveform, PCA residual concentration and score reversal", weight="bold")
    for row, (group, site, _, _) in enumerate(CASES):
        ax0, ax1, ax2 = axes[row]
        for i, (label, color) in enumerate((("Healthy", HEALTHY), ("Damage", DAMAGE))):
            waveform = traces.loc[traces.group.eq(group) & traces.label.eq(label)]
            ax0.plot(waveform.seconds, waveform.standardized_acceleration, lw=.9, color=color,
                     alpha=.85, label=label)
            error = attribution.loc[attribution.group.eq(group) & attribution.label.eq(label)]
            pct = 100 * error.loc[error.is_target_axis, "share"].sum()
            ax1.bar(i, pct, width=.58, color=color)
            ax1.text(i, pct+2, f"{pct:.1f}%", ha="center", fontsize=8)
            record = scores.loc[scores.group.eq(group) & scores.label.eq(label)].iloc[0]
            ax2.plot([0, 1], [record.clean, record.dropped], "o-", color=color, label=label)
        ax0.set(title=f"{group} · {site}V · first 1,024-point window", ylabel="Healthy-scaled acceleration",
                xlabel="Time (s)", xlim=(0, (POINTS-1)/100))
        ax0.legend(frameon=False, fontsize=7, loc="upper right")
        ax1.set(title="Clean residual at this location", ylabel="% of total squared error",
                xticks=[0,1], xticklabels=["Healthy","Damage"], ylim=(0,112))
        ax2.set(title="Drop the location in both records", ylabel="Oriented PCA-SPE score",
                xticks=[0,1], xticklabels=["All sensors", f"Without {site}"], xlim=(-.15, 1.15))
        for ax in axes[row]:
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    save(fig, out, "07_waveform_residual_and_score")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--z24", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--focus", type=int, choices=(10, 30, 50, 70, 90), default=50)
    args = parser.parse_args()

    explain(args.report, args.out, args.focus, "prior_protocol")
    with report_reader(args.report) as read:
        inventory = json.loads(read("Z24_sensor_fit_inventory.json"))
        scores = {bridge: read_csv(read, f"{bridge}_scores.csv.gz") for bridge in DATASETS}
        metrics = {bridge: read_csv(read, f"{bridge}_metrics_by_mask.csv") for bridge in DATASETS}
    sample = pd.read_csv(args.out / "pair_examples.csv.gz")
    clean = scores["Z24"].query("split == 'test' and loss == 0").set_index("sample_id")
    sign = int(metrics["Z24"].query("orientation == 'prior_protocol'").score_sign.unique().item())
    pipeline(args.out)
    score_distributions(scores, metrics, sample, args.focus, args.out)
    single, masks, traces, case_scores, attribution = z24_diagnostics(args.z24, inventory, clean, sign, args.focus, 30)
    for stem, frame in (("single_location_pairs", single), ("mask_design_pairs", masks),
                        ("case_waveforms", traces), ("case_scores", case_scores), ("case_residual_shares", attribution)):
        frame.to_csv(args.out / f"{stem}.csv", index=False)
    plot_mask_design(single, masks, args.focus, args.out)
    waveform_cases(traces, case_scores, attribution, args.out)
    print("Isolated flips", single.groupby(["group", "sensor"]).flipped.sum().loc[lambda x: x.gt(0)])
    print("Shared and independent masks", masks.loc[masks.initially_correct].groupby("design").flipped.agg(["sum","size"]))
    print(f"PCA_SENSOR_LOSS_VISUALS_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
