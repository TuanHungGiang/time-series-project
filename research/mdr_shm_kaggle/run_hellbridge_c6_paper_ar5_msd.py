from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score


METHODS = ("paper_ar5_msd_al40", "paper_ar5_msd_ag35")


def read_record_channels(group: h5py.Group) -> dict[str, np.ndarray]:
    acceleration = group["acceleration"]
    al_names = [f"AL{i:02d}" for i in range(1, 41)]
    ag_names = [f"AG{i:02d}" for i in range(1, 19)]
    al = [np.asarray(acceleration[name]["z"], dtype=np.float64) for name in al_names]
    ag = []
    # The paper's global group contains 35 channels: 18 z and 17 y (AG09-y absent).
    for name in ag_names:
        ag.append(np.asarray(acceleration[name]["z"], dtype=np.float64))
    for name in ag_names:
        if "y" in acceleration[name]:
            ag.append(np.asarray(acceleration[name]["y"], dtype=np.float64))
    if len(al) != 40 or len(ag) != 35:
        raise RuntimeError(f"Expected AL40/AG35, obtained AL{len(al)}/AG{len(ag)}")
    length = min(*(len(item) for item in al), *(len(item) for item in ag))
    return {
        "al40": np.stack([item[:length] for item in al]),
        "ag35": np.stack([item[:length] for item in ag]),
    }


def ar_ols(signals: np.ndarray, order: int) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float64)
    values = values - values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    values = values / np.maximum(scales, 1e-12)
    windows = sliding_window_view(values, order + 1, axis=1)
    design = windows[:, :, :-1][:, :, ::-1]
    target = windows[:, :, -1]
    gram = np.einsum("sni,snj->sij", design, design, optimize=True)
    rhs = np.einsum("sni,sn->si", design, target, optimize=True)
    ridge = 1e-8 * np.maximum(np.trace(gram, axis1=1, axis2=2) / order, 1.0)
    gram += ridge[:, None, None] * np.eye(order)[None, :, :]
    try:
        coefficients = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        coefficients = np.stack([np.linalg.pinv(g) @ r for g, r in zip(gram, rhs)])
    return coefficients.reshape(-1)


def extract_features(h5_path: Path, manifest: pd.DataFrame, order: int) -> dict[str, np.ndarray]:
    al_features = np.empty((len(manifest), 40 * order), dtype=np.float64)
    ag_features = np.empty((len(manifest), 35 * order), dtype=np.float64)
    with h5py.File(h5_path, "r") as source:
        for record_index, (record_id, rows) in enumerate(manifest.groupby("record_id", sort=False)):
            channels = read_record_channels(source[record_id])
            for row in rows.itertuples():
                start, stop = int(row.start), int(row.stop)
                al_features[row.Index] = ar_ols(channels["al40"][:, start:stop], order)
                ag_features[row.Index] = ar_ols(channels["ag35"][:, start:stop], order)
            print(
                f"features record={record_index + 1}/{manifest.record_id.nunique()} "
                f"name={record_id} samples={len(rows)}",
                flush=True,
            )
    return {"paper_ar5_msd_al40": al_features, "paper_ar5_msd_ag35": ag_features}


@dataclass
class MSDModel:
    mean: np.ndarray
    inverse_covariance: np.ndarray
    threshold: float
    noise_sigma: np.ndarray
    augmented_rows: int

    def raw_score(self, matrix: np.ndarray) -> np.ndarray:
        delta = np.asarray(matrix, dtype=float) - self.mean
        return np.einsum("ni,ij,nj->n", delta, self.inverse_covariance, delta, optimize=True)

    def normalized_score(self, matrix: np.ndarray) -> np.ndarray:
        return self.raw_score(matrix) / self.threshold


def monte_carlo_max_msd_threshold(
    observations: int,
    dimensions: int,
    trials: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    # Faithful, tractable implementation of the paper's maximum-MSD Monte Carlo
    # threshold. Under N(0, I), individual MSDs follow chi-square(k).
    maxima = np.empty(trials, dtype=np.float64)
    batch = 250
    for start in range(0, trials, batch):
        count = min(batch, trials - start)
        maxima[start : start + count] = rng.chisquare(
            dimensions, size=(count, observations)
        ).max(axis=1)
    try:
        return float(np.quantile(maxima, 1.0 - alpha, method="higher"))
    except TypeError:
        return float(np.quantile(maxima, 1.0 - alpha, interpolation="higher"))


def fit_paper_model(
    training: np.ndarray,
    snr: float,
    copies: int,
    mc_trials: int,
    alpha: float,
    seed: int,
) -> MSDModel:
    rng = np.random.default_rng(seed)
    # The paper defines SNR = mean absolute feature value / noise RMS.
    sigma = np.mean(np.abs(training), axis=0) / snr
    sigma = np.maximum(sigma, 1e-10)
    augmented = np.tile(training, (copies, 1))
    augmented += rng.normal(0.0, sigma, size=augmented.shape)
    mean = augmented.mean(axis=0)
    covariance = np.cov(augmented, rowvar=False)
    covariance += np.eye(covariance.shape[0]) * max(
        1e-12, float(np.trace(covariance) / covariance.shape[0]) * 1e-10
    )
    inverse = np.linalg.pinv(covariance, hermitian=True, rcond=1e-12)
    threshold = monte_carlo_max_msd_threshold(
        len(augmented), training.shape[1], mc_trials, alpha, rng
    )
    return MSDModel(mean, inverse, threshold, sigma, len(augmented))


def metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prediction = scores > 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "false_positive_rate": float(np.mean(prediction[labels == 0])),
        "false_negative_rate": float(np.mean(~prediction[labels == 1])),
        "alarm_rate": float(prediction.mean()),
    }


def bootstrap_metrics(
    frame: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for method, group in frame.groupby("method", sort=False):
        labels = group.label.to_numpy(dtype=int)
        scores = group.score.to_numpy(dtype=float)
        healthy = np.flatnonzero(labels == 0)
        damage = np.flatnonzero(labels == 1)
        draws = {"auroc": [], "balanced_accuracy": []}
        for _ in range(samples):
            indices = np.concatenate([
                rng.choice(healthy, len(healthy), replace=True),
                rng.choice(damage, len(damage), replace=True),
            ])
            metrics = metric_dict(labels[indices], scores[indices])
            for key in draws:
                draws[key].append(metrics[key])
        point = metric_dict(labels, scores)
        for metric, values in draws.items():
            values = np.asarray(values)
            rows.append({
                "method": method,
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_samples": samples,
            })
    return pd.DataFrame(rows)


def make_figure(scores: pd.DataFrame, output: Path) -> None:
    states = ["UDS"] + [f"DS{i}" for i in range(1, 9)]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, method in zip(axes, METHODS):
        subset = scores[scores.method == method]
        values = [subset.loc[subset.state == state, "score"] for state in states]
        axis.boxplot(values, tick_labels=states, showfliers=False)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_ylabel("MSD / Monte Carlo threshold")
        axis.set_title(method)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Structural state")
    figure.suptitle("Paper-aligned HellBridge AR(5)-MSD reproduction")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--noise-copies", type=int, default=20)
    parser.add_argument("--mc-trials", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.split_dir / "sample_manifest.csv")
    split_protocol = json.loads((args.split_dir / "protocol.json").read_text(encoding="utf-8"))
    if split_protocol["profile"] != "svendsen2022":
        raise ValueError(f"Expected svendsen2022 split, got {split_protocol['profile']}")
    if not split_protocol["paper_counts_exact"]:
        raise ValueError("Paper counts are not exact; reproduction label would be invalid")
    if manifest.sample_id.duplicated().any():
        raise RuntimeError("Duplicate sample IDs")
    train_mask = manifest.role.eq("train_healthy").to_numpy()
    test_mask = manifest.role.isin(["test_healthy", "test_damage"]).to_numpy()
    if manifest.loc[train_mask, "label"].ne(0).any():
        raise RuntimeError("Damage leakage into training")

    feature_sets = extract_features(args.h5, manifest, order=5)
    result_rows, model_rows = [], []
    test_manifest = manifest.loc[test_mask].reset_index(drop=True)
    for method_index, method in enumerate(METHODS):
        matrix = feature_sets[method]
        model = fit_paper_model(
            matrix[train_mask], args.snr, args.noise_copies, args.mc_trials,
            args.alpha, args.seed + 10007 * method_index,
        )
        test_scores = model.normalized_score(matrix[test_mask])
        for row, score in zip(test_manifest.itertuples(index=False), test_scores):
            result_rows.append({
                "sample_id": row.sample_id,
                "record_id": row.record_id,
                "state": row.state,
                "label": int(row.label),
                "repeat": int(row.repeat),
                "window_index": int(row.window_index),
                "method": method,
                "score": float(score),
                "alarm": bool(score > 1.0),
            })
        model_rows.append({
            "method": method,
            "feature_dimension": matrix.shape[1],
            "measured_training_samples": int(train_mask.sum()),
            "augmented_training_samples": model.augmented_rows,
            "snr": args.snr,
            "threshold_raw_msd": model.threshold,
            "covariance_rank": int(np.linalg.matrix_rank(model.inverse_covariance)),
        })

    scores = pd.DataFrame(result_rows)
    metric_rows = []
    for method, group in scores.groupby("method", sort=False):
        metric_rows.append({
            "method": method,
            "test_samples": len(group),
            "healthy_test_samples": int((group.label == 0).sum()),
            "damage_test_samples": int((group.label == 1).sum()),
            **metric_dict(group.label.to_numpy(dtype=int), group.score.to_numpy(dtype=float)),
        })
    metrics = pd.DataFrame(metric_rows)
    state_rows = []
    for method, group in scores.groupby("method", sort=False):
        healthy = group[group.state == "UDS"]
        for state in [f"DS{i}" for i in range(1, 9)]:
            damage = group[group.state == state]
            subset = pd.concat([healthy, damage], ignore_index=True)
            state_rows.append({
                "method": method,
                "state": state,
                "auroc_vs_uds": float(roc_auc_score(subset.label, subset.score)),
                "damage_alarm_rate": float(damage.alarm.mean()),
                "median_damage_score": float(damage.score.median()),
            })
    states = pd.DataFrame(state_rows)
    bootstrap = bootstrap_metrics(scores, args.bootstrap_samples, args.seed + 70001)

    scores.to_csv(args.output_dir / "sample_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_ci.csv", index=False)
    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_audit.csv", index=False)
    np.savez_compressed(
        args.output_dir / "ar5_feature_cache.npz",
        sample_id=manifest.sample_id.to_numpy(dtype=str),
        role=manifest.role.to_numpy(dtype=str),
        state=manifest.state.to_numpy(dtype=str),
        label=manifest.label.to_numpy(dtype=int),
        al40=feature_sets["paper_ar5_msd_al40"],
        ag35=feature_sets["paper_ar5_msd_ag35"],
    )
    make_figure(scores, args.output_dir / "fig_ar5_msd_reproduction.png")
    protocol = {
        "experiment": "HellBridge C6 paper-aligned AR(5)-MSD reproduction",
        "reference": "Svendsen et al. 2022, JCSHM 12:101-115",
        "split_manifest": str(args.split_dir / "sample_manifest.csv"),
        "split_profile": split_protocol["profile"],
        "paper_counts_exact": split_protocol["paper_counts_exact"],
        "window_samples": split_protocol["window_samples"],
        "ar_estimator": "batched no-intercept OLS after per-sample centering/scaling",
        "sensor_groups": {"SG1": "AL40 x AR5 = 200", "SG2": "AG35 x AR5 = 175"},
        "training": f"80 measured UDS samples x {args.noise_copies} SNR-{args.snr:g} noisy copies",
        "threshold": (
            f"Monte Carlo maximum chi-square MSD approximation, alpha={args.alpha}, "
            f"trials={args.mc_trials}"
        ),
        "damage_labels_used_for_fit_or_threshold": False,
        "reproduction_note": (
            "Faithful independent implementation; original author code and exact random noise "
            "realizations are unavailable, so numerical equality is not claimed"
        ),
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(states.to_string(index=False))
    print(f"HELLBRIDGE_C6_AR5_MSD_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

