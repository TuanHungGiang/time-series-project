"""Isolate the healthy-reference log Welch power level under Z24 sensor loss."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from clmm_timesnet.pca_observed_sensor_comparison import MaskComparison, evaluate, paired_deltas, summarize
from clmm_timesnet.pca_sensor_loss_1024 import METRICS, POINTS, RATES, REPLICATES, z24_scores
from clmm_timesnet.pca_z24_spectral_comparison import BAND_EDGES, SAMPLE_RATE, WELCH_SEGMENT, log_band_power


SPECTRAL_METHODS = {
    "relative_spe": "welch_shape",
    "absolute_spe": "welch_shape_and_level",
    "level_spe": "welch_level_only",
}
METHODS = {"spe_score": "time_zero_fill", "observed_spe": "time_observed", **SPECTRAL_METHODS}


def welch_representations(windows):
    absolute = log_band_power(windows)
    level = absolute.mean(axis=-1, keepdims=True)
    return {"relative_spe": absolute - level, "absolute_spe": absolute, "level_spe": level}


class PowerAblation:
    def __init__(self, fit, reference):
        self.time = MaskComparison(fit, reference)
        fit_spectra = welch_representations(fit)
        reference_spectra = welch_representations(reference)
        self.spectral = {
            column: MaskComparison(fit_spectra[column], reference_spectra[column])
            for column in SPECTRAL_METHODS
        }
        self.model = self.time.model

    def score_columns(self, raw, present):
        time = self.time.score_columns(raw, present)
        features = welch_representations(raw)
        spectral = {
            column: model.score_columns(features[column], present)
            for column, model in self.spectral.items()
        }

        def score(keep):
            values = time(keep)
            scores = {column: scorer(keep)["observed_spe"] for column, scorer in spectral.items()}
            return dict(spe_score=values["spe_score"], observed_spe=values["observed_spe"], **scores)

        return score


def validation_auc(data):
    validation = data.loc[data["split"].eq("val")]
    clean = validation.loc[validation.loss.eq(0)]
    sign = -1 if roc_auc_score(clean.label, clean.spe_score) < .5 else 1
    rows = []
    for (loss, replicate), group in validation.groupby(["loss", "replicate"]):
        if group.label.value_counts().to_dict() != {1: 126, 0: 6}:
            raise ValueError("Unexpected healthy or damage count in masked validation")
        for column, method in METHODS.items():
            for orientation, direction in (("prior_protocol", sign), ("upper", 1)):
                rows.append(dict(
                    split="val", loss=loss, replicate=replicate, orientation=orientation,
                    method=method, auc=roc_auc_score(group.label, direction * group[column]),
                ))
    return pd.DataFrame(rows)


def power_deltas(metrics, keys, fields):
    rows = []
    for reference in ("welch_shape", "time_observed", "time_zero_fill"):
        compared = paired_deltas(metrics, [*keys, "replicate"], fields, reference=reference)
        rows.append(compared.loc[compared.method.eq("welch_shape_and_level")].assign(reference=reference))
    return pd.concat(rows, ignore_index=True)


def plot_report(auc, state_deltas, out):
    primary = auc.loc[auc.orientation.eq("prior_protocol")]
    labels = {
        "time_zero_fill": ("#555555", "Time, zero fill"),
        "time_observed": ("#0072B2", "Time, observed only"),
        "welch_shape": ("#D55E00", "Welch shape"),
        "welch_shape_and_level": ("#009E73", "Welch shape + level"),
        "welch_level_only": ("#AA4499", "Welch level"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained", width_ratios=(1, 1, 1.45))
    for split, axis in zip(("val", "test"), axes[:2]):
        for method, (color, label) in labels.items():
            values = primary.loc[primary["split"].eq(split) & primary.method.eq(method)]
            axis.errorbar(values.loss, values.auc_mean, yerr=values.auc_std.fillna(0),
                          color=color, marker="o", capsize=2, label=label)
        axis.axhline(.5, color="0.7", linestyle="--", linewidth=.8)
        axis.set(title="Validation" if split == "val" else "Test", ylabel="ROC-AUC",
                 xlabel="Nominal sensor loss (%)", xticks=RATES)
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7)

    data = state_deltas.loc[
        state_deltas.orientation.eq("prior_protocol") & state_deltas.reference.eq("welch_shape")
    ]
    heat = data.pivot(index="state", columns="loss", values="delta_auc_mean") * 100
    limit = max(1., np.abs(heat.to_numpy()).max())
    image = axes[2].imshow(heat, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axes[2].set(title="Test AUC gain from keeping power level",
                ylabel="Damage state", xlabel="Nominal sensor loss (%)",
                xticks=np.arange(len(heat.columns)), xticklabels=heat.columns,
                yticks=np.arange(len(heat.index)), yticklabels=heat.index)
    for i, row in enumerate(heat.to_numpy()):
        for j, value in enumerate(row):
            axes[2].text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(value) > .65 * limit else "black")
    fig.colorbar(image, ax=axes[2], label="AUC difference (pp)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"z24_welch_power_ablation.{ext}", dpi=180)
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
        band_edges_hz=BAND_EDGES.tolist(), loss_rates=RATES, paired_mask_orders=REPLICATES,
        decomposition="Absolute log-band power = centered log-band shape + mean log-band power.",
        score_orientation="Previous Z24 sign from clean labeled validation, with upper-tail sensitivity.",
        ranking="AUC on matched validation and test masks; neither cohort enters PCA fitting.",
        threshold="Test thresholds calibrated on healthy validation, both frozen and mask-matched.",
        study_scope="Follow-up on an existing test partition; confirm the fixed method on independent data.",
    )
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with threadpool_limits(limits=4):
        data, masks, inventory = z24_scores(args.z24, PowerAblation, mask_damage_validation=True)
        if not np.isfinite(data[list(METHODS)].to_numpy()).all():
            raise ValueError("Nonfinite Z24 ablation scores")
        validation = validation_auc(data)
        metrics, states = evaluate(data, "Z24", METHODS)
    test_auc = metrics.loc[metrics.policy.eq("mask_matched"), validation.columns.drop("split")].assign(split="test")
    auc = pd.concat([validation, test_auc], ignore_index=True)
    auc_keys = ["split", "orientation", "loss"]
    auc_summary = summarize(auc, [*auc_keys, "method"], ("auc",))
    deltas = power_deltas(auc, auc_keys, ("auc",))
    delta_summary = summarize(deltas, [*auc_keys, "reference"], ("delta_auc",))
    state_keys = ["orientation", "loss", "state"]
    state_delta = power_deltas(states, state_keys, ("auc",))
    state_summary = summarize(state_delta, [*state_keys, "reference"], ("delta_auc",))

    data.to_csv(args.out / "scores.csv.gz", index=False)
    pd.DataFrame(masks).to_csv(args.out / "masks.csv.gz", index=False)
    (args.out / "sensor_fit_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    auc_summary.to_csv(args.out / "auc_summary.csv", index=False)
    delta_summary.to_csv(args.out / "paired_auc_summary.csv", index=False)
    deltas.to_csv(args.out / "paired_auc_by_mask.csv", index=False)
    summarize(metrics, ["orientation", "policy", "loss", "method"], METRICS).to_csv(
        args.out / "test_metric_summary.csv", index=False
    )
    states.to_csv(args.out / "test_state_auc_by_mask.csv", index=False)
    state_summary.to_csv(args.out / "test_state_paired_summary.csv", index=False)
    plot_report(auc_summary, state_summary, args.out)
    print(f"Z24_WELCH_POWER_ABLATION_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
