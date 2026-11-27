"""Compare zero-fill, observed-only projection and subset-refit PCA on two bridges."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from clmm_timesnet.pca_sensor_loss_1024 import (
    METRICS, POINTS, RATES, REPLICATES, PCASPE, hell_scores, measure, z24_scores,
)


METHODS = {
    "spe_score": "zero_fill",
    "observed_spe": "observed_projection",
    "refit_spe": "subset_refit",
}
RIDGE = 1e-6
VARIANCE = 0.95


class MaskComparison(PCASPE):
    def __init__(self, fit, reference):
        super().__init__(fit, reference)
        # Standardization uses the original healthy training set for all variants.
        norm = self.normalizer.transform(fit).astype(np.float64)
        self.refit_mean = norm.mean(axis=0)
        self.centered_train = np.ascontiguousarray(
            (norm - self.refit_mean).transpose(1, 0, 2)
        )
        self.basis = np.ascontiguousarray(
            self.model.components_.reshape(-1, *norm.shape[1:]).transpose(1, 0, 2),
            dtype=np.float64,
        )
        self.full_mean = self.model.mean_.reshape(norm.shape[1:])
        self.axis_basis_gram = self.basis @ self.basis.transpose(0, 2, 1)
        self.axis_train_gram = self.centered_train @ self.centered_train.transpose(0, 2, 1)
        self.mask_models = {}

    def for_mask(self, keep):
        key = keep.tobytes()
        if key not in self.mask_models:
            gram = self.axis_basis_gram[keep].sum(axis=0)
            # The unit-normalized full PCA basis has spectral norm one.
            factored = cho_factor(gram + RIDGE * np.eye(len(gram)))

            # Dual PCA fits on healthy windows restricted to the surviving axes.
            train_gram = self.axis_train_gram[keep].sum(axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(train_gram)
            eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
            eigenvectors = eigenvectors[:, ::-1]
            total = eigenvalues.sum()
            count = 0
            if total > 0:
                count = int(np.searchsorted(np.cumsum(eigenvalues), VARIANCE * total, side="right") + 1)
            dual_basis = eigenvectors[:, :count] / np.sqrt(eigenvalues[:count])
            self.mask_models[key] = gram, factored, dual_basis
        return self.mask_models[key]

    def score_columns(self, raw, present):
        baseline = self.terms(raw)
        norm = self.normalizer.transform(raw).astype(np.float64).transpose(1, 0, 2)
        x_full = norm - self.full_mean[:, None, :]
        x_refit = norm - self.refit_mean[:, None, :]
        full_cross = x_full @ self.basis.transpose(0, 2, 1)
        refit_cross = x_refit @ self.centered_train.transpose(0, 2, 1)
        full_energy = np.einsum("cnt,cnt->cn", x_full, x_full)
        refit_energy = np.einsum("cnt,cnt->cn", x_refit, x_refit)

        def columns(keep):
            if not keep.any() or np.any(keep & ~present):
                raise ValueError("Each mask must retain at least one naturally observed channel")
            standard = baseline(keep)
            if keep.all():
                return dict(spe_score=standard, observed_spe=standard, refit_spe=standard)

            gram, factored, dual_basis = self.for_mask(keep)
            cross = full_cross[keep].sum(axis=0)
            latents = cho_solve(factored, cross.T).T
            # Residual energy excludes the ridge penalty used to solve the projection.
            cross_term = np.einsum("nk,nk->n", latents, cross)
            fitted_energy = np.einsum("nk,nk->n", latents @ gram, latents)
            observed = full_energy[keep].sum(axis=0) - 2 * cross_term + fitted_energy

            projection = refit_cross[keep].sum(axis=0) @ dual_basis
            refit = refit_energy[keep].sum(axis=0) - np.einsum("nk,nk->n", projection, projection)
            width = int(keep.sum()) * raw.shape[-1]
            return {
                "spe_score": standard,
                "observed_spe": self.orient_neutral(observed / width),
                "refit_spe": self.orient_neutral(refit / width),
            }

        return columns


def evaluate(data, dataset, methods=METHODS):
    validation = data.loc[data["split"].eq("val") & data["loss"].eq(0)]
    sign = -1 if dataset == "Z24" and roc_auc_score(validation.label, validation.spe_score) < .5 else 1
    metrics, states = [], []
    metadata = data.drop(columns=list(methods))
    for column, method in methods.items():
        scores = metadata.assign(spe_score=data[column])
        metrics.append(measure(scores, dataset, prior_sign=sign).assign(method=method))
        test = scores.loc[scores["split"].eq("test")]
        for (loss, replicate), trial in test.groupby(["loss", "replicate"]):
            for state, damage in trial.loc[trial.label.eq(1)].groupby("state"):
                comparison = pd.concat([trial.loc[trial.label.eq(0)], damage])
                for orientation, direction in (("prior_protocol", sign), ("upper", 1)):
                    states.append(dict(
                        dataset=dataset, method=method, loss=loss, replicate=replicate,
                        state=state, orientation=orientation,
                        auc=roc_auc_score(comparison.label, direction * comparison.spe_score),
                        healthy_units=int(comparison.label.eq(0).sum()), damage_units=len(damage),
                    ))
    return pd.concat(metrics, ignore_index=True), pd.DataFrame(states)


def paired_deltas(metrics, keys, fields, reference="zero_fill"):
    baseline = metrics.loc[metrics.method.eq(reference), [*keys, *fields]]
    alternative = metrics.loc[metrics.method.ne(reference)]
    paired = alternative.merge(baseline, on=keys, suffixes=("", "_baseline"), validate="many_to_one")
    for field in fields:
        paired[f"delta_{field}"] = paired[field] - paired[f"{field}_baseline"]
    return paired


def summarize(metrics, keys, fields):
    summary = metrics.groupby(keys)[list(fields)].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary["masks"] = metrics.groupby(keys).size()
    return summary.reset_index()


def plot_auc(metrics, paired, out):
    metrics = metrics.loc[metrics.orientation.eq("prior_protocol") & metrics.policy.eq("mask_matched")]
    paired = paired.loc[paired.orientation.eq("prior_protocol") & paired.policy.eq("mask_matched")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    names = {"zero_fill": "Zero-fill PCA-SPE", "observed_projection": "Observed-only projection",
             "subset_refit": "Subset-refit PCA"}
    for i, name in enumerate(("Z24", "HellBridge")):
        for j, (frame, metric) in enumerate(((metrics, "auc"), (paired, "delta_auc"))):
            axis = axes[i, j]
            for method, group in frame.loc[frame.dataset.eq(name)].groupby("method", sort=False):
                values = group.groupby("loss")[metric].agg(["mean", "std"])
                scale = 100 if j else 1
                error = values["std"].fillna(0) * scale
                color = dict(zero_fill="0.35", observed_projection="#0072B2", subset_refit="#D55E00")[method]
                axis.errorbar(values.index, values["mean"] * scale, yerr=error, color=color,
                              marker="o", capsize=3, label=names[method])
            axis.axhline(0 if j else .5, color="0.65", linestyle="--", linewidth=.8)
            axis.set(title=name, ylabel="AUC gain vs zero-fill (pp)" if j else "ROC-AUC", xticks=RATES)
            axis.grid(alpha=.2)
            if i == 1:
                axis.set_xlabel("Nominal sensor loss (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "pca_mask_comparison.png", dpi=175)
    fig.savefig(out / "pca_mask_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24", required=True, type=Path)
    parser.add_argument("--hell", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        window_points=POINTS, simulated_loss_rates=RATES, mask_orders=REPLICATES,
        ridge=RIDGE, pca_variance=VARIANCE,
        preprocessing="Existing native 1024-point runner with healthy-fit standardization",
        masks="Original nested whole-device masks. Natural absence also excluded by the two observed-only methods.",
        common_scaling="Per-group center and scale from full-basis healthy reference residuals for all methods",
        refit="Exact observed-feature PCA computed from a healthy training Gram matrix for each retained subset.",
        orientation="Prior clean-Z24-validation sign learned once on zero-fill and reused. Upper tail exported separately.",
        test_units="Z24 source records; HellBridge windows, with healthy source recordings shared with development",
        statistics="SD across matched masks on the fixed cohorts. Clean has one evaluation.",
    )
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    all_metrics, all_states = [], []
    with threadpool_limits(limits=4):
        for dataset, runner in (("Z24", lambda: z24_scores(args.z24, MaskComparison)),
                                ("HellBridge", lambda: hell_scores(args.hell, args.out, MaskComparison))):
            data, masks, inventory = runner()
            if not np.isfinite(data[list(METHODS)].to_numpy()).all():
                raise ValueError(f"{dataset} has nonfinite scores")
            data.to_csv(args.out / f"{dataset}_scores.csv.gz", index=False)
            pd.DataFrame(masks).to_csv(args.out / f"{dataset}_masks.csv.gz", index=False)
            (args.out / f"{dataset}_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            metrics, states = evaluate(data, dataset)
            all_metrics.append(metrics)
            all_states.append(states)
            print(f"{dataset} complete | units={data.loc[data['loss'].eq(0) & data['split'].eq('test')].shape[0]}", flush=True)
    metrics, states = pd.concat(all_metrics, ignore_index=True), pd.concat(all_states, ignore_index=True)
    keys = ["dataset", "orientation", "policy", "loss"]
    paired = paired_deltas(metrics, [*keys, "replicate"], METRICS)
    summary = summarize(metrics, [*keys, "method"], METRICS)
    delta_fields = [f"delta_{metric}" for metric in METRICS]
    summarize(paired, [*keys, "method"], delta_fields).to_csv(args.out / "paired_summary.csv", index=False)
    summary.to_csv(args.out / "metric_summary.csv", index=False)
    metrics.to_csv(args.out / "metrics_by_mask.csv", index=False)
    paired.to_csv(args.out / "paired_by_mask.csv", index=False)
    states.to_csv(args.out / "state_auc_by_mask.csv", index=False)
    state_keys = ["dataset", "orientation", "loss", "state"]
    state_paired = paired_deltas(states, [*state_keys, "replicate"], ("auc",))
    summarize(state_paired, [*state_keys, "method"], ("delta_auc",)).to_csv(args.out / "state_paired_summary.csv", index=False)
    plot_auc(metrics, paired, args.out)
    print(f"PCA_OBSERVED_SENSOR_COMPARISON_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
