"""Compare native time samples and Welch band-power features for Z24 PCA-SPE."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from threadpoolctl import threadpool_limits

from clmm_timesnet.pca_observed_sensor_comparison import MaskComparison, evaluate, paired_deltas, summarize
from clmm_timesnet.pca_sensor_loss_1024 import METRICS, POINTS, RATES, REPLICATES, z24_scores


SAMPLE_RATE = 100.0
WELCH_SEGMENT = 512
BAND_EDGES = np.linspace(.5, 30., 33)
METHODS = {
    "spe_score": "time_zero_fill",
    "observed_spe": "time_observed",
    "spectral_spe": "spectral_observed",
}
METRIC_KEYS = ["dataset", "orientation", "policy", "loss"]
STATE_KEYS = ["dataset", "orientation", "state", "loss"]


def log_band_power(windows):
    freqs, density = welch(
        windows, fs=SAMPLE_RATE, window="hann", nperseg=WELCH_SEGMENT,
        noverlap=WELCH_SEGMENT // 2, detrend=False, scaling="density", axis=-1,
    )
    start, stop = np.searchsorted(freqs, BAND_EDGES[:-1]), np.searchsorted(freqs, BAND_EDGES[1:])
    power = np.stack([density[..., low:high].mean(axis=-1) for low, high in zip(start, stop)], axis=-1)
    return np.log10(np.maximum(power, np.finfo(np.float32).tiny))


def log_band_shape(windows):
    log_power = log_band_power(windows)
    return log_power - log_power.mean(axis=-1, keepdims=True)


class RepresentationComparison:
    def __init__(self, fit, reference):
        self.time = MaskComparison(fit, reference)
        self.spectrum = MaskComparison(log_band_shape(fit), log_band_shape(reference))
        self.model = self.time.model

    def score_columns(self, raw, present):
        time = self.time.score_columns(raw, present)
        spectrum = self.spectrum.score_columns(log_band_shape(raw), present)

        def score(keep):
            values = time(keep)
            return dict(
                spe_score=values["spe_score"],
                observed_spe=values["observed_spe"],
                spectral_spe=spectrum(keep)["observed_spe"],
            )

        return score


def compare_to_time(frame, keys, fields):
    index = [*keys, "replicate"]
    paired = paired_deltas(frame, index, fields, reference="time_observed")
    observed = paired.loc[paired.method.eq("spectral_observed")].copy()
    clean_keys = [key for key in keys if key != "loss"]
    clean = observed.loc[observed.loss.eq(0), [*clean_keys, "delta_auc"]]
    observed = observed.merge(
        clean.rename(columns={"delta_auc": "clean_delta_auc"}),
        on=clean_keys, validate="many_to_one",
    )
    observed["delta_auc_retention"] = observed["delta_auc"] - observed["clean_delta_auc"]
    return observed


def plot_report(summary, paired, states, out):
    primary = summary.loc[summary.orientation.eq("prior_protocol") & summary.policy.eq("mask_matched")]
    delta = paired.loc[paired.orientation.eq("prior_protocol") & paired.policy.eq("mask_matched")]
    state_primary = states.loc[states.orientation.eq("prior_protocol")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained", width_ratios=(1, 1, 1.45))
    styles = {
        "time_zero_fill": ("0.45", "Time, zero fill"),
        "time_observed": ("#0072B2", "Time, observed only"),
        "spectral_observed": ("#D55E00", "Spectrum, observed only"),
    }
    for method, (color, label) in styles.items():
        part = primary.loc[primary.method.eq(method)]
        axes[0].errorbar(part.loss, part.auc_mean, yerr=part.auc_std.fillna(0), marker="o",
                         color=color, capsize=3, label=label)
    axes[0].set(title="Z24 damage ranking", ylabel="ROC-AUC")
    axes[0].legend(frameon=False, fontsize=8)
    for field, color, label in (
        ("delta_auc", "#D55E00", "Spectrum minus time"),
        ("delta_auc_retention", "#009E73", "Change in gap from clean"),
    ):
        axes[1].errorbar(delta.loss, 100 * delta[f"{field}_mean"],
                         yerr=100 * delta[f"{field}_std"].fillna(0), color=color,
                         marker="o", capsize=3, label=label)
    axes[1].axhline(0, color="0.5", linestyle="--", linewidth=.8)
    axes[1].set(title="Paired AUC change", ylabel="Difference (percentage points)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes[:2]:
        axis.set(xlabel="Nominal sensor loss (%)", xticks=RATES)
        axis.grid(alpha=.2)
    heat = state_primary.loc[state_primary.loss.gt(0)].pivot(
        index="state", columns="loss", values="delta_auc_mean"
    ) * 100
    limit = max(1., np.abs(heat.to_numpy()).max())
    image = axes[2].imshow(heat, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axes[2].set(title="Spectrum minus time by damage state",
                xlabel="Nominal sensor loss (%)", ylabel="Damage state",
                xticks=np.arange(len(heat.columns)), xticklabels=heat.columns,
                yticks=np.arange(len(heat.index)), yticklabels=heat.index)
    for i, row in enumerate(heat.to_numpy()):
        for j, value in enumerate(row):
            axes[2].text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(value) > .65 * limit else "black")
    fig.colorbar(image, ax=axes[2], label="AUC difference (pp)")
    fig.savefig(out / "z24_time_vs_spectrum.png", dpi=180)
    fig.savefig(out / "z24_time_vs_spectrum.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        dataset="Z24", input_root=str(args.z24), window_points=POINTS, sample_rate_hz=SAMPLE_RATE,
        welch_segment=WELCH_SEGMENT, welch_overlap=WELCH_SEGMENT // 2,
        spectrum="Mean PSD per band, log10, centered over bands per window and channel.",
        band_edges_hz=BAND_EDGES.tolist(), masks=REPLICATES, loss_rates=RATES,
        split="Original Z24 record split; keep the eight windows sampled by the audited baseline.",
        scaler="Separate healthy-fit per-channel standardization for each representation.",
        orientation="Original zero-fill direction selected on clean validation, fixed across variants; upper tail also exported.",
        delta_auc_retention="Spectrum-minus-time AUC at each loss minus spectrum-minus-time AUC on clean.",
    )
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with threadpool_limits(limits=4):
        data, masks, inventory = z24_scores(args.z24, RepresentationComparison)
        if not np.isfinite(data[list(METHODS)].to_numpy()).all():
            raise ValueError("Z24 comparison produced nonfinite scores")
        metrics, states = evaluate(data, "Z24", METHODS)
    data.to_csv(args.out / "scores.csv.gz", index=False)
    pd.DataFrame(masks).to_csv(args.out / "masks.csv.gz", index=False)
    (args.out / "sensor_fit_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    metrics.to_csv(args.out / "metrics_by_mask.csv", index=False)
    states.to_csv(args.out / "state_auc_by_mask.csv", index=False)
    summary = summarize(metrics, [*METRIC_KEYS, "method"], METRICS)
    paired = compare_to_time(metrics, METRIC_KEYS, ("auc", "diagnostic_recall", "ba"))
    paired_summary = summarize(paired, METRIC_KEYS, ("delta_auc", "delta_auc_retention", "delta_diagnostic_recall", "delta_ba"))
    state_paired = compare_to_time(states, STATE_KEYS, ("auc",))
    state_summary = summarize(state_paired, STATE_KEYS, ("delta_auc", "delta_auc_retention"))
    summary.to_csv(args.out / "metric_summary.csv", index=False)
    paired.to_csv(args.out / "paired_by_mask.csv", index=False)
    paired_summary.to_csv(args.out / "paired_summary.csv", index=False)
    state_summary.to_csv(args.out / "state_paired_summary.csv", index=False)
    plot_report(summary, paired_summary, state_summary, args.out)
    print(f"Z24_SPECTRAL_COMPARISON_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
