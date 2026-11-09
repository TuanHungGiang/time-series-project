"""Z24 evaluation of the distributional Bayesian core of Phys-SpanIDK.

This is a Z24 adaptation, not the full two-span Bautzen architecture.  The
predeclared proposed path is:

continuous modal-centroid windows -> IDK record embedding -> train-only PCA ->
hierarchical NIW conditioned on pre/post-lifting domain and AVT/FVT regime.

All fitted objects and the conformal threshold use healthy train records only.
Setup-grouped cross-fitting supplies out-of-fold calibration scores. Damage
validation/test labels are used only for metrics, never for method selection.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal


METHODS = (
    "gaussian_modal_mean",
    "idk_l2",
    "modal_niw_regime_domain",
    "idk_niw_pooled",
    "modal_idk_niw_regime_domain",
)
PROPOSED = "modal_idk_niw_regime_domain"


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)


class IsolationDistributionalKernel:
    def __init__(self, partitions: int, anchors_per_partition: int, seed: int):
        self.partitions = partitions
        self.anchors_per_partition = anchors_per_partition
        self.seed = seed
        self.anchors_: list[np.ndarray] = []
        self.radii_: list[np.ndarray] = []

    @property
    def output_dim(self):
        return self.partitions * self.anchors_per_partition

    @staticmethod
    def distances(a, b):
        return np.maximum((a * a).sum(1, keepdims=True) +
                          (b * b).sum(1)[None, :] - 2. * a @ b.T, 0.)

    def fit(self, x):
        x = np.asarray(x, np.float64)
        rng = np.random.default_rng(self.seed)
        psi = min(self.anchors_per_partition, len(x))
        if x.ndim != 2 or psi < 2:
            raise ValueError("IDK needs a 2-D array with at least two windows")
        self.anchors_per_partition = psi
        for _ in range(self.partitions):
            anchors = x[rng.choice(len(x), psi, replace=False)]
            distances = self.distances(anchors, anchors)
            np.fill_diagonal(distances, np.inf)
            self.anchors_.append(anchors)
            self.radii_.append(np.sqrt(np.min(distances, axis=1)))
        return self

    def transform_points(self, x):
        x = np.asarray(x, np.float64)
        result = np.zeros((len(x), self.output_dim), np.float32)
        for part, (anchors, radii) in enumerate(zip(self.anchors_, self.radii_)):
            distances = np.sqrt(self.distances(x, anchors))
            nearest = np.argmin(distances, axis=1)
            accepted = distances[np.arange(len(x)), nearest] <= radii[nearest]
            rows = np.flatnonzero(accepted)
            result[rows, part * self.anchors_per_partition + nearest[rows]] = 1.
        return result

    def transform_distribution(self, x):
        return self.transform_points(x).mean(axis=0)


@dataclass
class NIWPosterior:
    mu: np.ndarray
    kappa: float
    lam: np.ndarray
    nu: float
    precision: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        dimension = self.mu.size
        dof = self.nu - dimension + 1.
        if dof <= 0: raise ValueError("Invalid NIW degrees of freedom")
        scale = ((self.kappa + 1.) / (self.kappa * dof)) * self.lam
        self.precision = pinvh(scale)

    def score(self, value):
        delta = np.asarray(value, float) - self.mu
        return float(delta @ self.precision @ delta)


def empirical_prior(x, strength: float):
    x = np.asarray(x, float); dimension = x.shape[1]
    covariance = LedoitWolf().fit(x).covariance_
    covariance += np.eye(dimension) * max(np.trace(covariance) / dimension, 1e-6) * 1e-4
    nu = dimension + 2. + strength
    return NIWPosterior(x.mean(0), strength,
                        covariance * (nu - dimension - 1.), nu)


def weak_update(x):
    x = np.asarray(x, float); n, dimension = x.shape
    prior = NIWPosterior(np.zeros(dimension), 1e-2,
                         np.eye(dimension) * 1e-2, dimension + 2.)
    return update_niw(prior, x)


def update_niw(prior, x):
    x = np.asarray(x, float); n = len(x); mean = x.mean(0)
    centered = x - mean; kappa = prior.kappa + n; nu = prior.nu + n
    delta = mean - prior.mu
    lam = (prior.lam + centered.T @ centered +
           (prior.kappa * n / kappa) * np.outer(delta, delta))
    mu = (prior.kappa * prior.mu + n * mean) / kappa
    return NIWPosterior(mu, kappa, lam, nu)


def robust_location_scale(values):
    values = np.asarray(values, float); center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if scale < 1e-8: scale = float(np.std(values))
    return center, max(scale, 1e-8)


def condition_key(row):
    return (modal.domain_for_state(str(row.state)), str(row.excitation).upper())


class Z24DistributionalModels:
    def __init__(self, args, seed):
        self.args, self.seed = args, seed

    def fit(self, rows: pd.DataFrame, features: dict[str, np.ndarray]):
        pooled_windows = np.concatenate([features[key] for key in rows.record_id])
        self.idk = IsolationDistributionalKernel(
            self.args.idk_partitions, self.args.idk_anchors, self.seed
        ).fit(pooled_windows)
        raw_means = np.stack([features[key].mean(0) for key in rows.record_id])
        idk = np.stack([self.idk.transform_distribution(features[key])
                        for key in rows.record_id])
        self.modal_scaler = RobustScaler().fit(raw_means)
        modal_values = self.modal_scaler.transform(raw_means)
        components = max(1, min(self.args.idk_components, len(idk) - 2,
                                idk.shape[1] - 1))
        self.idk_pca = PCA(n_components=components, random_state=self.seed).fit(idk)
        projected = self.idk_pca.transform(idk)
        self.idk_scaler = RobustScaler().fit(projected)
        idk_values = self.idk_scaler.transform(projected)

        self.gaussian = LedoitWolf().fit(modal_values)
        self.idk_center = np.median(idk_values, axis=0)
        self.modal_global_prior = empirical_prior(modal_values, self.args.prior_strength)
        self.idk_global_prior = empirical_prior(idk_values, self.args.prior_strength)
        self.modal_group, self.idk_group = {}, {}
        keys = [condition_key(row) for row in rows.itertuples()]
        for key in sorted(set(keys)):
            selected = np.asarray([value == key for value in keys])
            self.modal_group[key] = update_niw(self.modal_global_prior,
                                               modal_values[selected])
            self.idk_group[key] = update_niw(self.idk_global_prior,
                                             idk_values[selected])
        self.idk_pooled = weak_update(idk_values)
        self.fit_rows = rows.copy()
        self.fit_features = features
        raw_scores = {method: [] for method in METHODS}
        for row in rows.itertuples():
            scores = self.raw_scores(row, features[row.record_id])
            for method in METHODS: raw_scores[method].append(scores[method])
        self.references = {method: robust_location_scale(values)
                           for method, values in raw_scores.items()}
        return self

    def representations(self, value):
        modal_value = self.modal_scaler.transform(value.mean(0, keepdims=True))[0]
        distribution = self.idk.transform_distribution(value)[None, :]
        idk_value = self.idk_scaler.transform(
            self.idk_pca.transform(distribution)
        )[0]
        return modal_value, idk_value

    def raw_scores(self, row, value):
        modal_value, idk_value = self.representations(value)
        key = condition_key(row)
        if key not in self.idk_group:
            raise ValueError(
                f"Condition {key} was absent from healthy fitting data; "
                "the strict protocol cannot borrow a damage record to create it"
            )
        modal_niw = self.modal_group[key].score(modal_value)
        idk_niw = self.idk_group[key].score(idk_value)
        return {
            "gaussian_modal_mean": float(self.gaussian.mahalanobis(
                modal_value[None, :]
            )[0]),
            "idk_l2": float(np.mean((idk_value - self.idk_center) ** 2)),
            "modal_niw_regime_domain": modal_niw,
            "idk_niw_pooled": self.idk_pooled.score(idk_value),
            # Both terms are squared Mahalanobis distances. Dividing by their
            # dimensions gives an equal per-dimension contribution before the
            # healthy-train robust score normalization below.
            PROPOSED: (modal_niw / modal_value.size + idk_niw / idk_value.size),
        }

    def scores(self, row, value):
        raw = self.raw_scores(row, value)
        return {method: (raw[method] - self.references[method][0]) /
                self.references[method][1] for method in METHODS}


def feature_cache(rows, fit_rows, spectra, frequency, args):
    anchors = modal.learn_anchors(
        [spectra[key] for key in fit_rows.record_id], frequency,
        args.modes, args.mode_separation,
    )
    return ({row.record_id: modal.track_modes_centroid(
        frequency, spectra[row.record_id], anchors, args.tracking_radius
    ) for row in rows.itertuples()}, anchors)


def conformal_threshold(values, alpha):
    values = np.sort(np.asarray(values, float)); n = len(values)
    rank = min(int(np.ceil((n + 1) * (1. - alpha))), n)
    return float(values[rank - 1]), rank


def evaluate(rows, model, cache, thresholds, split):
    output = []
    for row in rows.itertuples():
        scores = model.scores(row, cache[row.record_id])
        for method, score in scores.items():
            output.append({
                "method": method, "split": split,
                "record_id": row.record_id, "state": row.state,
                "scenario": row.scenario, "excitation": row.excitation,
                "setup": int(row.setup), "label": int(row.label),
                "score": score, "threshold": thresholds[method],
                "margin": score - thresholds[method],
                "alarm": score > thresholds[method],
            })
    return pd.DataFrame(output)


def summarize(details):
    rows = []
    for (split, method), group in details.groupby(["split", "method"]):
        rows.append({"evaluation": split, "method": method,
                     **common.metrics(group.label, group.margin, 0.)})
    return pd.DataFrame(rows)


def markdown_table(frame, columns):
    values = frame[columns].copy()
    for column in values:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda x: f"{x:.4f}")
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |"
          for row in values.itertuples(index=False, name=None)],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/z24_modal_idk_niw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--fs", type=float, default=50.)
    parser.add_argument("--modal-window", type=int, default=4096)
    parser.add_argument("--modal-stride", type=int, default=2048)
    parser.add_argument("--fmin", type=float, default=3.)
    parser.add_argument("--modal-fmax", type=float, default=20.)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--mode-separation", type=float, default=.50)
    parser.add_argument("--tracking-radius", type=float, default=.45)
    parser.add_argument("--idk-partitions", type=int, default=32)
    parser.add_argument("--idk-anchors", type=int, default=4)
    parser.add_argument("--idk-components", type=int, default=8)
    parser.add_argument("--prior-strength", type=float, default=4.)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed); started = time.time()

    frame = common.resolve_manifest(args.manifest, args.data_root)
    train = frame[(frame.split == "train") & (frame.label.astype(int) == 0)].copy()
    val = frame[frame.split == "val"].copy(); test = frame[frame.split == "test"].copy()
    if tuple(map(len, (train, val, test))) != (24, 132, 132):
        raise ValueError("Unexpected manifest split")
    print("[1/4] Extracting modal-centroid spectra...", flush=True)
    spectra, frequency = {}, None
    for index, row in enumerate(frame.itertuples(), 1):
        f, value = modal.record_spectra(
            row.path, args.fs, args.modal_window, args.modal_stride,
            args.fmin, args.modal_fmax,
        )
        if frequency is None: frequency = f
        elif not np.array_equal(frequency, f): raise ValueError("Frequency grid mismatch")
        spectra[row.record_id] = value
        if index % 48 == 0 or index == len(frame):
            print(f"      records={index}/{len(frame)}", flush=True)

    splitter = GroupKFold(n_splits=args.oof_folds)
    oof_rows, anchor_rows = [], []
    for fold, (fit_index, held_index) in enumerate(splitter.split(
            train, groups=train.setup.astype(int)), 1):
        fit = train.iloc[fit_index].copy(); held = train.iloc[held_index].copy()
        cache, anchors = feature_cache(pd.concat([fit, held]), fit,
                                       spectra, frequency, args)
        model = Z24DistributionalModels(args, args.seed + fold).fit(fit, cache)
        for row in held.itertuples():
            for method, score in model.scores(row, cache[row.record_id]).items():
                oof_rows.append({"record_id": row.record_id, "fold": fold,
                                 "setup": int(row.setup), "method": method,
                                 "score": score})
        anchor_rows.append({"fold": fold, "anchors_hz": json.dumps(anchors.tolist())})
        print(f"[2/4] OOF fold {fold}/{args.oof_folds} complete", flush=True)
    oof = pd.DataFrame(oof_rows)
    thresholds, threshold_rows = {}, []
    for method, group in oof.groupby("method"):
        threshold, rank = conformal_threshold(group.score, args.alpha)
        thresholds[method] = threshold
        threshold_rows.append({"method": method, "threshold": threshold,
                               "n_oof": len(group), "rank": rank,
                               "alpha": args.alpha})

    cache, anchors = feature_cache(frame, train, spectra, frequency, args)
    final = Z24DistributionalModels(args, args.seed).fit(train, cache)
    details = pd.concat([
        evaluate(val, final, cache, thresholds, "validation"),
        evaluate(test, final, cache, thresholds, "locked_test"),
    ], ignore_index=True)
    metrics = summarize(details)
    proposed = metrics[metrics.method == PROPOSED].copy()
    print("[3/4] Proposed and ablations evaluated", flush=True)

    oof.to_csv(args.output / "healthy_oof_scores.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.output / "thresholds.csv", index=False)
    details.to_csv(args.output / "record_scores.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(args.output / "oof_anchors.csv", index=False)
    pd.DataFrame({"modal_anchor_hz": anchors}).to_csv(
        args.output / "final_anchors.csv", index=False
    )

    report = f"""# Z24 Modal-IDK-NIW report

## Protocol

- Input: continuous modal centroid, 4 modes, window 4096, stride 2048
- Train: 24 whole healthy records only
- Calibration: setup-grouped 3-fold healthy OOF scores
- Damage used for fitting, projection, NIW or threshold: no
- Primary method declared before evaluation: `{PROPOSED}`
- This is the distributional Bayesian core adapted to Z24, not full two-span Phys-SpanIDK
- NIW hierarchy: empirical-Bayes global healthy shrinkage followed by condition-specific healthy update
- Fusion: equal per-dimension sum of modal-NIW and IDK-NIW distances; no damage-tuned weight

## Proposed result

{markdown_table(proposed, ['evaluation', 'tn', 'fp', 'fn', 'tp', 'balanced_accuracy', 'precision', 'recall', 'f1', 'specificity', 'auroc', 'auprc'])}

## Ablations

{markdown_table(metrics, ['evaluation', 'method', 'balanced_accuracy', 'recall', 'f1', 'specificity', 'auroc', 'auprc'])}

## Interpretation guard

The primary claim must use `{PROPOSED}` even if an ablation scores higher.
The locked test has been inspected in earlier project iterations, so this is a
comparative development result rather than a pristine external confirmation.
"""
    (args.output / "Z24_PROPOSED_REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "primary_method": PROPOSED, "methods": list(METHODS),
        "damage_used_for_fit_or_threshold": False,
        "test_used_for_method_selection": False,
        "niw_interpretation": "empirical-Bayes shrinkage, not an exact independent-prior posterior",
        "parameters": vars(args), "final_anchors_hz": anchors.tolist(),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    test_metrics = metrics[metrics.evaluation == "locked_test"]
    axis.bar(test_metrics.method, test_metrics.auroc)
    axis.set_ylim(0, 1); axis.set_ylabel("AUROC")
    axis.set_title("Z24 proposed method and ablations")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=.25); figure.tight_layout()
    figure.savefig(args.output / "proposed_ablation_auroc.png", dpi=190)
    plt.close(figure)
    print("[4/4] Complete", flush=True)
    print(f"REPORT={args.output / 'Z24_PROPOSED_REPORT.md'}", flush=True)


if __name__ == "__main__": main()

