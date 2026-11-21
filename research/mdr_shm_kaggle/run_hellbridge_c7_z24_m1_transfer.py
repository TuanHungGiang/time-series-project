from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.metrics import f1_score, roc_auc_score

from run_hellbridge_c6_paper_ar5_msd import read_record_channels
from run_z24_supervised_consistency_c1 import feature_from_signals as c1_feature


METHODS = (
    "z24_m1_al40_ensemble",
    "z24_m1_ag35_ensemble",
    "z24_m1_al_ag_ensemble",
)


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return np.log(value / (1 - value))


def expit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, float), -30, 30)
    return 1.0 / (1.0 + np.exp(-value))


def modal_from_components(
    frequency: np.ndarray,
    spectrum: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Exact C4.1 modal formula, reusing one spectral decomposition."""
    modal, detected = [], []
    for anchor in np.asarray(anchors, float):
        band = np.flatnonzero(np.abs(frequency - anchor) <= 0.70)
        if not len(band):
            raise RuntimeError(f"No spectral bin around frozen Z24 anchor {anchor}")
        local = spectrum[band]
        peak_position = int(np.argmax(local))
        peak_index = int(band[peak_position])
        peak_frequency = float(frequency[peak_index])
        detected.append(peak_frequency)
        prominence = float(max(local[peak_position] - np.median(local), 0.0))
        prominence /= max(float(np.median(local)), 1e-18)
        energy = float(np.trapezoid(local, frequency[band]))
        shift = peak_frequency - anchor
        modal.extend([
            shift,
            shift / max(float(anchor), 1e-12),
            np.log1p(prominence),
            np.log10(energy + 1e-18),
        ])
        # The frozen m1 model does not consume the FDD terms. Computing the
        # eigenvalues here would only add cost and cannot affect its score.
    for index in range(len(detected) - 1):
        modal.append(float(detected[index] / max(detected[index + 1], 1e-12)))
    return np.asarray(modal, dtype=float)


def modal_spectrum(signals: np.ndarray, nfft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """C4.1/C2 median normalized spectrum without unused cross-spectral matrices."""
    values = detrend(np.asarray(signals, dtype=np.float64), axis=1, type="linear")
    width = min(nfft, values.shape[1])
    if width < 512:
        raise RuntimeError(f"C4.1 transfer needs >=512 samples, got {values.shape[1]}")
    hop = max(1, width // 2)
    starts = np.arange(0, values.shape[1] - width + 1, hop)
    if starts.size == 0:
        starts = np.asarray([0])
    window = np.hanning(width)
    scale = max(float(np.sqrt(np.sum(window**2))), 1e-12)
    spectra = np.stack([
        np.fft.rfft(values[:, start : start + width] * window, n=nfft, axis=1) / scale
        for start in starts
    ])
    frequency = np.fft.rfftfreq(nfft, 1.0 / 100.0)
    keep = (frequency >= 0.5) & (frequency <= 30.0)
    frequency = frequency[keep]
    auto = np.mean(np.abs(spectra[:, :, keep]) ** 2, axis=0) + 1e-18
    sensor_total = np.trapezoid(auto, frequency, axis=1)
    normalized_auto = auto / np.maximum(sensor_total[:, None], 1e-18)
    return frequency, np.median(normalized_auto, axis=0)


def frozen_feature_views(signals: np.ndarray, anchors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # C1 differs between AVT and FVT only through its final excitation flag.
    c1_avt = c1_feature(signals, "avt")
    c1_fvt = c1_avt.copy()
    c1_fvt[-1] = 1.0
    frequency, spectrum = modal_spectrum(signals)
    return {
        "avt": np.concatenate([
            c1_avt,
            modal_from_components(frequency, spectrum, anchors["avt"]),
        ]),
        "fvt": np.concatenate([
            c1_fvt,
            modal_from_components(frequency, spectrum, anchors["fvt"]),
        ]),
    }


def extract_matrices(
    h5_path: Path,
    manifest: pd.DataFrame,
    anchors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    storage: dict[str, list[np.ndarray | None]] = {
        "al40_avt": [None] * len(manifest),
        "al40_fvt": [None] * len(manifest),
        "ag35_avt": [None] * len(manifest),
        "ag35_fvt": [None] * len(manifest),
    }
    with h5py.File(h5_path, "r") as source:
        groups = manifest.groupby("record_id", sort=False)
        total = manifest.record_id.nunique()
        for record_index, (record_id, rows) in enumerate(groups, start=1):
            channels = read_record_channels(source[record_id])
            for row in rows.itertuples():
                start, stop = int(row.start), int(row.stop)
                for sensor_group in ("al40", "ag35"):
                    views = frozen_feature_views(
                        channels[sensor_group][:, start:stop], anchors
                    )
                    storage[f"{sensor_group}_avt"][row.Index] = views["avt"]
                    storage[f"{sensor_group}_fvt"][row.Index] = views["fvt"]
            print(
                f"transfer-features record={record_index}/{total} "
                f"name={record_id} samples={len(rows)}",
                flush=True,
            )
    return {
        name: np.stack([np.asarray(item, float) for item in items])
        for name, items in storage.items()
    }


def model_probability(model, matrix: np.ndarray) -> np.ndarray:
    if getattr(model, "feature_column", None) != "feature_m1":
        raise ValueError(
            "Expected the selected Z24 C4.1 m1_modal_absolute artifact "
            f"with feature_m1, got {getattr(model, 'feature_column', None)!r}"
        )
    frame = pd.DataFrame({"feature_m1": list(matrix)})
    return np.asarray(model.score(frame), float)


def transferred_scores(model, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    probabilities = {
        name: model_probability(model, matrix)
        for name, matrix in matrices.items()
    }
    al = expit((logit(probabilities["al40_avt"]) + logit(probabilities["al40_fvt"])) / 2)
    ag = expit((logit(probabilities["ag35_avt"]) + logit(probabilities["ag35_fvt"])) / 2)
    fusion = expit((logit(al) + logit(ag)) / 2)
    return {
        "z24_m1_al40_ensemble": al,
        "z24_m1_ag35_ensemble": ag,
        "z24_m1_al_ag_ensemble": fusion,
    }


def threshold_at_specificity(scores: np.ndarray, target: float) -> tuple[float, float]:
    scores = np.asarray(scores, float)
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    for threshold in candidates:
        realized = float(np.mean(scores < threshold))
        if realized >= target:
            return float(threshold), realized
    return float(np.nextafter(scores.max(), np.inf)), 1.0


def metric_dict(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    prediction = scores >= threshold
    positive, negative = labels == 1, labels == 0
    recall = float(prediction[positive].mean())
    specificity = float((~prediction[negative]).mean())
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": float(f1_score(labels, prediction)),
        "false_positive_rate": 1 - specificity,
        "false_negative_rate": 1 - recall,
    }


def paired_bootstrap(
    candidate: pd.DataFrame,
    baselines: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    merged = candidate.merge(
        baselines[["sample_id", "method", "score"]],
        on="sample_id",
        suffixes=("_candidate", "_baseline"),
        validate="many_to_many",
    )
    rows, rng = [], np.random.default_rng(seed)
    for (threshold_mode, baseline_method), group in merged.groupby(
        ["threshold_mode", "method"], sort=False
    ):
        group = group.reset_index(drop=True)
        labels = group.label.to_numpy(int)
        candidate_score = group.score_candidate.to_numpy(float)
        baseline_score = group.score_baseline.to_numpy(float)
        candidate_threshold = float(group.threshold.iloc[0])
        negative = np.flatnonzero(labels == 0)
        positive = np.flatnonzero(labels == 1)
        distributions = {name: [] for name in ("auroc", "recall", "specificity", "balanced_accuracy")}
        observed_candidate = metric_dict(labels, candidate_score, candidate_threshold)
        observed_baseline = metric_dict(labels, baseline_score, 1.0)
        for _ in range(samples):
            index = np.r_[
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
            first = metric_dict(labels[index], candidate_score[index], candidate_threshold)
            second = metric_dict(labels[index], baseline_score[index], 1.0)
            for metric in distributions:
                distributions[metric].append(first[metric] - second[metric])
        for metric, values in distributions.items():
            values = np.asarray(values, float)
            rows.append({
                "candidate": "z24_m1_al_ag_ensemble",
                "threshold_mode": threshold_mode,
                "baseline": baseline_method,
                "metric": metric,
                "candidate_value": observed_candidate[metric],
                "baseline_value": observed_baseline[metric],
                "delta": observed_candidate[metric] - observed_baseline[metric],
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "probability_delta_gt_zero": float(np.mean(values > 0)),
                "bootstrap_samples": samples,
            })
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, output: Path) -> None:
    selected = metrics[metrics.threshold_mode == "target_healthy_calibrated"]
    measures = ["auroc", "recall", "specificity", "balanced_accuracy"]
    x = np.arange(len(selected))
    width = 0.19
    fig, axis = plt.subplots(figsize=(12, 5))
    for index, measure in enumerate(measures):
        axis.bar(x + (index - 1.5) * width, selected[measure], width, label=measure)
    axis.set_xticks(x, [name.replace("z24_m1_", "") for name in selected.method], rotation=12)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Metric")
    axis.set_title("Frozen Z24 m1 transfer to HellBridge")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--z24-model", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--specificity-target", type=float, default=0.875)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.split_dir / "sample_manifest.csv")
    protocol = json.loads((args.split_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("profile") != "svendsen2022" or not protocol.get("paper_counts_exact"):
        raise ValueError("C7 requires the exact locked Svendsen-2022 manifest")
    artifact = joblib.load(args.z24_model)
    model = artifact["model"]
    if getattr(model, "feature_column", None) != "feature_m1":
        raise ValueError("The Z24 artifact is not m1_modal_absolute")
    anchors = {name: np.asarray(value, float) for name, value in artifact["anchors"].items()}
    if set(anchors) != {"avt", "fvt"}:
        raise ValueError(f"Unexpected frozen anchor groups: {sorted(anchors)}")
    strict_threshold = float(artifact["threshold"])

    matrices = extract_matrices(args.h5, manifest, anchors)
    expected = int(model.estimator.n_features_in_)
    actual = {name: matrix.shape[1] for name, matrix in matrices.items()}
    if set(actual.values()) != {expected}:
        raise ValueError(f"Feature mismatch: frozen model={expected}, HellBridge={actual}")
    scores_by_method = transferred_scores(model, matrices)

    train_mask = manifest.role.eq("train_healthy").to_numpy()
    test_mask = manifest.role.isin(["test_healthy", "test_damage"]).to_numpy()
    if manifest.loc[train_mask, "label"].ne(0).any():
        raise RuntimeError("Damage leakage into HellBridge calibration")

    threshold_rows, score_rows, metric_rows, state_rows = [], [], [], []
    test_manifest = manifest.loc[test_mask].reset_index(drop=True)
    for method in METHODS:
        scores = scores_by_method[method]
        target_threshold, realized = threshold_at_specificity(
            scores[train_mask], args.specificity_target
        )
        operating_points = (
            ("strict_z24_frozen", strict_threshold, np.nan),
            ("target_healthy_calibrated", target_threshold, realized),
        )
        for threshold_mode, threshold, calibration_specificity in operating_points:
            threshold_rows.append({
                "method": method,
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "specificity_target": args.specificity_target if threshold_mode != "strict_z24_frozen" else np.nan,
                "calibration_healthy_samples": int(train_mask.sum()) if threshold_mode != "strict_z24_frozen" else 0,
                "realized_calibration_specificity": calibration_specificity,
            })
            test_scores = scores[test_mask]
            metric_rows.append({
                "method": method,
                "threshold_mode": threshold_mode,
                "test_samples": len(test_manifest),
                **metric_dict(test_manifest.label.to_numpy(int), test_scores, threshold),
            })
            for row, score in zip(test_manifest.itertuples(index=False), test_scores):
                score_rows.append({
                    "sample_id": row.sample_id,
                    "record_id": row.record_id,
                    "state": row.state,
                    "label": int(row.label),
                    "method": method,
                    "threshold_mode": threshold_mode,
                    "score": float(score),
                    "threshold": threshold,
                    "prediction": int(score >= threshold),
                })
            for state in [f"DS{i}" for i in range(1, 9)]:
                subset = test_manifest.state.isin(["UDS", state]).to_numpy()
                labels = test_manifest.loc[subset, "label"].to_numpy(int)
                state_scores = test_scores[subset]
                damage = test_manifest.state.eq(state).to_numpy()
                state_rows.append({
                    "method": method,
                    "threshold_mode": threshold_mode,
                    "state": state,
                    "auroc_vs_uds": float(roc_auc_score(labels, state_scores)),
                    "damage_recall": float(np.mean(test_scores[damage] >= threshold)),
                })

    metrics = pd.DataFrame(metric_rows)
    scores = pd.DataFrame(score_rows)
    thresholds = pd.DataFrame(threshold_rows)
    states = pd.DataFrame(state_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    scores.to_csv(args.output_dir / "sample_scores.csv", index=False)
    thresholds.to_csv(args.output_dir / "thresholds.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)

    if args.baseline_scores and args.baseline_scores.exists():
        candidate = scores[scores.method == "z24_m1_al_ag_ensemble"].copy()
        baseline = pd.read_csv(args.baseline_scores)
        paired = paired_bootstrap(
            candidate, baseline, args.bootstrap_samples, args.seed + 90001
        )
        paired.to_csv(args.output_dir / "paired_vs_ar5_msd.csv", index=False)
        print("\n=== PAIRED AGAINST PAPER BASELINES ===")
        print(paired.to_string(index=False))

    make_figure(metrics, args.output_dir / "fig_z24_m1_transfer.png")
    run = {
        "protocol": "hellbridge-c7-frozen-z24-m1-external-transfer",
        "source_model": str(args.z24_model),
        "source_model_expected": "Z24 C4.1 m1_modal_absolute",
        "source_damage_labels_used_for_training": True,
        "hellbridge_damage_labels_used_for_fit_calibration_or_selection": False,
        "target_adaptation": "threshold only, using 80 target healthy training windows",
        "bridge_adapter": "fixed logit ensemble over AVT/FVT assumptions and AL40/AG35 groups",
        "feature_dimension": expected,
        "specificity_target": args.specificity_target,
        "test_labels_used_only_for_final_evaluation": True,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("\n=== EXTERNAL TRANSFER METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== THRESHOLDS ===")
    print(thresholds.to_string(index=False))
    print(f"HELLBRIDGE_C7_Z24_M1_TRANSFER_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

