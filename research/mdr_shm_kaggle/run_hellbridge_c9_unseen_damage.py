from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from run_hellbridge_c6_paper_ar5_msd import extract_features as extract_ar5
from run_hellbridge_c8_m1_from_scratch import extract_base, learn_anchors
from run_hellbridge_c7_z24_m1_transfer import modal_from_components


BASE_VIEWS = ("ar5_al", "ar5_ag", "modal_al", "modal_ag")
METHODS = BASE_VIEWS + ("fusion_equal", "fusion_softmax")
ALPHAS = (0.10, 0.05)


@dataclass
class HealthyPCA:
    center: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    latent_center: np.ndarray
    latent_scale: np.ndarray
    residual_center: float
    residual_scale: float

    @classmethod
    def fit(cls, matrix: np.ndarray, maximum_components: int = 6) -> "HealthyPCA":
        values = np.asarray(matrix, dtype=float)
        center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - center), axis=0)
        fallback = np.std(values, axis=0)
        scale = np.maximum(mad, np.maximum(0.25 * fallback, 1e-6))
        standardized = np.clip((values - center) / scale, -12, 12)
        _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
        variance = singular**2
        cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
        needed = int(np.searchsorted(cumulative, 0.95) + 1)
        count = max(2, min(maximum_components, values.shape[0] - 2, needed))
        components = vt[:count]
        latent = standardized @ components.T
        latent_center = np.median(latent, axis=0)
        latent_mad = 1.4826 * np.median(np.abs(latent - latent_center), axis=0)
        latent_scale = np.maximum(latent_mad, np.maximum(0.25 * np.std(latent, axis=0), 1e-4))
        reconstructed = latent @ components
        residual = np.sqrt(np.mean((standardized - reconstructed) ** 2, axis=1))
        residual_center = float(np.median(residual))
        residual_scale = float(max(
            1.4826 * np.median(np.abs(residual - residual_center)),
            0.25 * np.std(residual),
            1e-4,
        ))
        return cls(
            center, scale, components, latent_center, latent_scale,
            residual_center, residual_scale,
        )

    def score(self, matrix: np.ndarray) -> np.ndarray:
        standardized = np.clip((np.asarray(matrix, float) - self.center) / self.scale, -12, 12)
        latent = standardized @ self.components.T
        latent_distance = np.sqrt(np.mean(
            ((latent - self.latent_center) / self.latent_scale) ** 2,
            axis=1,
        ))
        residual = np.sqrt(np.mean(
            (standardized - latent @ self.components) ** 2,
            axis=1,
        ))
        residual_excess = np.maximum(
            0.0, (residual - self.residual_center) / self.residual_scale
        )
        return np.sqrt(latent_distance**2 + residual_excess**2)


def modal_matrix(
    indices: np.ndarray,
    group_name: str,
    anchors: np.ndarray,
    frequency: np.ndarray,
    shapes: dict[str, np.ndarray],
    spectra: dict[str, np.ndarray],
) -> np.ndarray:
    modal = np.stack([
        modal_from_components(frequency, spectra[group_name][index], anchors)
        for index in indices
    ])
    return np.concatenate([shapes[group_name][indices], modal], axis=1)


def feature_sets(
    indices: np.ndarray,
    anchors: dict[str, np.ndarray],
    frequency: np.ndarray,
    shapes: dict[str, np.ndarray],
    spectra: dict[str, np.ndarray],
    ar5: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "ar5_al": ar5["paper_ar5_msd_al40"][indices],
        "ar5_ag": ar5["paper_ar5_msd_ag35"][indices],
        "modal_al": modal_matrix(indices, "al40", anchors["al40"], frequency, shapes, spectra),
        "modal_ag": modal_matrix(indices, "ag35", anchors["ag35"], frequency, shapes, spectra),
    }


def learn_fold_anchors(
    fit_indices: np.ndarray,
    frequency: np.ndarray,
    spectra: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "al40": learn_anchors(spectra["al40"][fit_indices], frequency, count=5),
        "ag35": learn_anchors(spectra["ag35"][fit_indices], frequency, count=5),
    }


def loo_evidence(oof_raw: np.ndarray) -> np.ndarray:
    values = np.asarray(oof_raw, float)
    output = np.empty(len(values), dtype=float)
    for index, value in enumerate(values):
        reference = np.delete(values, index)
        p_value = (1.0 + np.sum(reference >= value)) / (len(reference) + 1.0)
        output[index] = -np.log(p_value)
    return output


def test_evidence(oof_raw: np.ndarray, test_raw: np.ndarray) -> np.ndarray:
    reference = np.asarray(oof_raw, float)
    values = np.asarray(test_raw, float)
    counts = np.sum(reference[None, :] >= values[:, None], axis=1)
    p_value = (1.0 + counts) / (len(reference) + 1.0)
    return -np.log(p_value)


def fuse(evidence: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    matrix = np.column_stack([evidence[name] for name in BASE_VIEWS])
    equal = matrix.mean(axis=1)
    beta = 2.0
    maximum = matrix.max(axis=1, keepdims=True)
    softmax = (
        maximum[:, 0]
        + np.log(np.mean(np.exp(beta * (matrix - maximum)), axis=1)) / beta
    )
    return {**evidence, "fusion_equal": equal, "fusion_softmax": softmax}


def threshold_at_specificity(scores: np.ndarray, target: float) -> tuple[float, float]:
    values = np.asarray(scores, float)
    candidates = np.r_[np.unique(values), np.nextafter(values.max(), np.inf)]
    for threshold in candidates:
        realized = float(np.mean(values < threshold))
        if realized >= target:
            return float(threshold), realized
    return float(np.nextafter(values.max(), np.inf)), 1.0


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


def block_bootstrap(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
    block_size: int = 4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups: list[list[np.ndarray]] = []
    for _, group in frame.sort_values(["record_id", "start"]).groupby("record_id", sort=False):
        positions = group.index.to_numpy()
        blocks = [positions[start : start + block_size] for start in range(0, len(positions), block_size)]
        groups.append(blocks)
    names = ("auroc", "recall", "specificity", "balanced_accuracy", "f1")
    distributions = {name: [] for name in names}
    observed = metric_dict(frame.label, frame.score, float(frame.threshold.iloc[0]))
    for _ in range(samples):
        sampled = []
        for blocks in groups:
            selected = rng.integers(0, len(blocks), size=len(blocks))
            sampled.extend(np.concatenate([blocks[index] for index in selected]))
        subset = frame.loc[np.asarray(sampled, int)]
        result = metric_dict(subset.label, subset.score, float(subset.threshold.iloc[0]))
        for name in names:
            distributions[name].append(result[name])
    return pd.DataFrame([
        {
            "metric": name,
            "estimate": observed[name],
            "ci_lower": float(np.quantile(distributions[name], 0.025)),
            "ci_upper": float(np.quantile(distributions[name], 0.975)),
            "bootstrap_samples": samples,
            "block_size": block_size,
        }
        for name in names
    ])


def make_figure(aggregate: pd.DataFrame, states: pd.DataFrame, output: Path) -> None:
    primary = aggregate[aggregate.alpha.eq(0.10)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    measures = ("auroc_mean", "recall_mean", "specificity_mean", "balanced_accuracy_mean")
    x = np.arange(len(primary))
    width = 0.19
    for index, measure in enumerate(measures):
        axes[0].bar(x + (index - 1.5) * width, primary[measure], width, label=measure.replace("_mean", ""))
    axes[0].set_xticks(x, primary.method, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Unseen-damage repeat-holdout (alpha=0.10)")
    axes[0].legend(ncol=2)
    selected = states[(states.method == "fusion_equal") & states.alpha.eq(0.10)]
    pivot = selected.pivot(index="state", columns="fold", values="damage_recall")
    pivot.plot(kind="bar", ax=axes[1])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Primary fusion recall by unseen damage state")
    axes[1].set_ylabel("Recall")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--paper-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.paper_split_dir / "sample_manifest.csv")
    manifest = manifest.sort_values(["record_id", "start"]).reset_index(drop=True)
    if len(manifest) != 800 or manifest.sample_id.duplicated().any():
        raise ValueError("Expected the locked 800-sample Svendsen manifest")
    frequency, shapes, spectra = extract_base(args.h5, manifest)
    ar5 = extract_ar5(args.h5, manifest, order=5)

    metric_rows, threshold_rows, score_rows, state_rows, anchor_audit = [], [], [], [], []
    bootstrap_tables = []
    for fold_index, source_repeat in enumerate((1, 2), start=1):
        target_repeat = 2 if source_repeat == 1 else 1
        source = manifest.index[
            manifest.state.eq("UDS") & manifest["repeat"].eq(source_repeat)
        ].to_numpy()
        healthy_test = manifest.index[
            manifest.state.eq("UDS") & manifest["repeat"].eq(target_repeat)
        ].to_numpy()
        damage_test = manifest.index[manifest.label.eq(1)].to_numpy()
        test_indices = np.r_[healthy_test, damage_test]
        if len(source) != 80 or len(healthy_test) != 80 or len(damage_test) != 640:
            raise RuntimeError(
                f"Invalid fold counts source={len(source)} healthy_test={len(healthy_test)} "
                f"damage_test={len(damage_test)}"
            )
        source = source[np.argsort(manifest.loc[source, "start"].to_numpy())]
        oof_raw = {name: np.empty(80, dtype=float) for name in BASE_VIEWS}
        for block in range(5):
            held_positions = np.arange(block * 16, (block + 1) * 16)
            fit_positions = np.setdiff1d(np.arange(80), held_positions)
            fit_indices, held_indices = source[fit_positions], source[held_positions]
            anchors = learn_fold_anchors(fit_indices, frequency, spectra)
            fit_features = feature_sets(fit_indices, anchors, frequency, shapes, spectra, ar5)
            held_features = feature_sets(held_indices, anchors, frequency, shapes, spectra, ar5)
            for name in BASE_VIEWS:
                detector = HealthyPCA.fit(fit_features[name])
                oof_raw[name][held_positions] = detector.score(held_features[name])
            print(
                f"crossfit fold={fold_index} source_repeat={source_repeat} block={block + 1}/5",
                flush=True,
            )

        final_anchors = learn_fold_anchors(source, frequency, spectra)
        for group_name, values in final_anchors.items():
            anchor_audit.append({
                "fold": fold_index,
                "source_repeat": source_repeat,
                "group": group_name,
                "anchors": "|".join(f"{value:.6f}" for value in values),
            })
        source_features = feature_sets(source, final_anchors, frequency, shapes, spectra, ar5)
        test_features = feature_sets(test_indices, final_anchors, frequency, shapes, spectra, ar5)
        test_raw = {}
        for name in BASE_VIEWS:
            detector = HealthyPCA.fit(source_features[name])
            test_raw[name] = detector.score(test_features[name])

        oof_evidence = fuse({name: loo_evidence(oof_raw[name]) for name in BASE_VIEWS})
        test_evidences = fuse({
            name: test_evidence(oof_raw[name], test_raw[name])
            for name in BASE_VIEWS
        })
        test_manifest = manifest.loc[test_indices].reset_index(drop=True)
        labels = test_manifest.label.to_numpy(int)
        for method in METHODS:
            for alpha in ALPHAS:
                threshold, realized = threshold_at_specificity(
                    oof_evidence[method], 1.0 - alpha
                )
                result = metric_dict(labels, test_evidences[method], threshold)
                metric_rows.append({
                    "fold": fold_index,
                    "source_repeat": source_repeat,
                    "test_healthy_repeat": target_repeat,
                    "method": method,
                    "alpha": alpha,
                    "test_samples": len(labels),
                    "healthy_test_samples": int((labels == 0).sum()),
                    "damage_test_samples": int((labels == 1).sum()),
                    **result,
                })
                threshold_rows.append({
                    "fold": fold_index,
                    "method": method,
                    "alpha": alpha,
                    "threshold": threshold,
                    "oof_healthy_scores": len(oof_evidence[method]),
                    "target_specificity": 1.0 - alpha,
                    "realized_oof_specificity": realized,
                })
                predictions = test_evidences[method] >= threshold
                for row, score, prediction in zip(
                    test_manifest.itertuples(index=False), test_evidences[method], predictions
                ):
                    score_rows.append({
                        "fold": fold_index,
                        "source_repeat": source_repeat,
                        "sample_id": row.sample_id,
                        "record_id": row.record_id,
                        "start": int(row.start),
                        "state": row.state,
                        "label": int(row.label),
                        "method": method,
                        "alpha": alpha,
                        "score": float(score),
                        "threshold": threshold,
                        "prediction": int(prediction),
                    })
                for state in [f"DS{i}" for i in range(1, 9)]:
                    damage = test_manifest.state.eq(state).to_numpy()
                    cohort = test_manifest.state.isin(["UDS", state]).to_numpy()
                    state_rows.append({
                        "fold": fold_index,
                        "method": method,
                        "alpha": alpha,
                        "state": state,
                        "damage_samples": int(damage.sum()),
                        "damage_recall": float(predictions[damage].mean()),
                        "auroc_vs_uds": float(roc_auc_score(
                            labels[cohort], test_evidences[method][cohort]
                        )),
                    })

        primary = pd.DataFrame(score_rows)
        primary = primary[
            (primary.fold == fold_index)
            & (primary.method == "fusion_equal")
            & primary.alpha.eq(0.10)
        ].reset_index(drop=True)
        primary.index = np.arange(len(primary))
        ci = block_bootstrap(
            primary, args.bootstrap_samples, args.seed + fold_index * 1009
        )
        ci.insert(0, "fold", fold_index)
        bootstrap_tables.append(ci)

    fold_metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    scores = pd.DataFrame(score_rows)
    states = pd.DataFrame(state_rows)
    aggregate = fold_metrics.groupby(["method", "alpha"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        specificity_mean=("specificity", "mean"),
        specificity_std=("specificity", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        f1_mean=("f1", "mean"),
    )
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    thresholds.to_csv(args.output_dir / "thresholds.csv", index=False)
    scores.to_csv(args.output_dir / "sample_scores.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)
    pd.DataFrame(anchor_audit).to_csv(args.output_dir / "anchor_audit.csv", index=False)
    pd.concat(bootstrap_tables, ignore_index=True).to_csv(
        args.output_dir / "primary_block_bootstrap_ci.csv", index=False
    )
    make_figure(aggregate, states, args.output_dir / "fig_c9_unseen_damage.png")

    protocol = {
        "protocol": "hellbridge-c9-unseen-damage-two-repeat-holdout",
        "training_labels": "healthy only",
        "damage_labels_used_for_fit_anchor_calibration_fusion_or_threshold": False,
        "folds": {
            "1": "source UDS repeat 1; healthy test UDS repeat 2; damage test DS1-DS8",
            "2": "source UDS repeat 2; healthy test UDS repeat 1; damage test DS1-DS8",
        },
        "healthy_calibration": "five contiguous blocks; out-of-fold conformal evidence",
        "views": list(BASE_VIEWS),
        "primary_method": "fusion_equal",
        "secondary_method": "fusion_softmax",
        "operating_points": {"primary_alpha": 0.10, "sensitivity_alpha": 0.05},
        "fusion_weights_selected_with_damage_labels": False,
        "test_labels_used_only_for_final_metrics": True,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print("\n=== FOLD METRICS ===")
    print(fold_metrics.to_string(index=False))
    print("\n=== TWO-FOLD AGGREGATE ===")
    print(aggregate.to_string(index=False))
    print("\n=== PRIMARY STATE RECALL ===")
    print(states[(states.method == "fusion_equal") & states.alpha.eq(0.10)].to_string(index=False))
    print(f"HELLBRIDGE_C9_UNSEEN_DAMAGE_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

