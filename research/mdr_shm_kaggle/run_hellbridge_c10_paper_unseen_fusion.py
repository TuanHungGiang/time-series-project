from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from run_hellbridge_c6_paper_ar5_msd import extract_features as extract_ar5, fit_paper_model
from run_hellbridge_c7_z24_m1_transfer import modal_from_components
from run_hellbridge_c8_m1_from_scratch import extract_base, learn_anchors
from run_hellbridge_c9_unseen_damage import HealthyPCA, block_bootstrap


PRIMARY = "ar5_al_ag_fusion"
METHODS = (
    "paper_ar5_al",
    "paper_ar5_ag",
    "ar5_al_calibrated",
    "ar5_ag_calibrated",
    PRIMARY,
    "ar5_modal_ag_fusion",
)


def robust_reference(values: np.ndarray) -> tuple[float, float]:
    # AR/MSD and reconstruction scores are positive and strongly right-skewed.
    # Log compression prevents a few healthy OOF outliers from forcing a
    # saturated threshold while preserving score ordering.
    values = np.log1p(np.maximum(np.asarray(values, float), 0.0))
    center = float(np.median(values))
    scale = float(max(
        1.4826 * np.median(np.abs(values - center)),
        0.25 * np.std(values),
        1e-8,
    ))
    return center, scale


def standardize(values: np.ndarray, reference: tuple[float, float]) -> np.ndarray:
    values = np.log1p(np.maximum(np.asarray(values, float), 0.0))
    # Do not upper-clip: the prior ceiling at 30 made the 90%-specificity
    # threshold slightly greater than every possible test score.
    return np.maximum((values - reference[0]) / reference[1], -20.0)


def modal_matrix(
    indices: np.ndarray,
    group: str,
    anchors: np.ndarray,
    frequency: np.ndarray,
    shapes: dict[str, np.ndarray],
    spectra: dict[str, np.ndarray],
) -> np.ndarray:
    modal = np.stack([
        modal_from_components(frequency, spectra[group][index], anchors)
        for index in indices
    ])
    return np.concatenate([shapes[group][indices], modal], axis=1)


def healthy_blocks(manifest: pd.DataFrame, train_indices: np.ndarray) -> list[np.ndarray]:
    by_record = []
    for _, group in manifest.loc[train_indices].sort_values("start").groupby("record_id"):
        indices = group.index.to_numpy()
        if len(indices) != 40:
            raise RuntimeError(f"Expected 40 healthy training windows per repeat, got {len(indices)}")
        by_record.append([indices[start : start + 8] for start in range(0, 40, 8)])
    if len(by_record) != 2:
        raise RuntimeError(f"Expected two healthy repeats, got {len(by_record)}")
    return [np.r_[by_record[0][fold], by_record[1][fold]] for fold in range(5)]


def crossfit_scores(
    manifest: pd.DataFrame,
    train_indices: np.ndarray,
    ar5: dict[str, np.ndarray],
    frequency: np.ndarray,
    shapes: dict[str, np.ndarray],
    spectra: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    position = {index: offset for offset, index in enumerate(train_indices)}
    output = {name: np.empty(len(train_indices), float) for name in ("ar5_al", "ar5_ag", "modal_ag")}
    for fold, held in enumerate(healthy_blocks(manifest, train_indices)):
        fit = np.setdiff1d(train_indices, held)
        anchors_ag = learn_anchors(spectra["ag35"][fit], frequency, count=5)
        fit_modal = modal_matrix(fit, "ag35", anchors_ag, frequency, shapes, spectra)
        held_modal = modal_matrix(held, "ag35", anchors_ag, frequency, shapes, spectra)
        modal_model = HealthyPCA.fit(fit_modal)
        modal_reference = robust_reference(modal_model.score(fit_modal))
        modal_held = standardize(modal_model.score(held_modal), modal_reference)

        for view, key in (
            ("ar5_al", "paper_ar5_msd_al40"),
            ("ar5_ag", "paper_ar5_msd_ag35"),
        ):
            model = fit_paper_model(
                ar5[key][fit], snr=20.0, copies=20,
                mc_trials=200, alpha=0.01,
                seed=seed + fold * 1009 + (0 if view == "ar5_al" else 503),
            )
            reference = robust_reference(model.raw_score(ar5[key][fit]))
            held_score = standardize(model.raw_score(ar5[key][held]), reference)
            for index, value in zip(held, held_score):
                output[view][position[index]] = value
        for index, value in zip(held, modal_held):
            output["modal_ag"][position[index]] = value
        print(f"healthy-crossfit block={fold + 1}/5", flush=True)
    return output


def threshold_at_specificity(scores: np.ndarray, target: float) -> tuple[float, float]:
    values = np.asarray(scores, float)
    candidates = np.r_[np.unique(values), np.nextafter(values.max(), np.inf)]
    for threshold in candidates:
        realized = float(np.mean(values < threshold))
        if realized >= target:
            return float(threshold), realized
    return float(np.nextafter(values.max(), np.inf)), 1.0


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
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


def paired_block_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    samples: int,
    seed: int,
    block_size: int = 4,
) -> pd.DataFrame:
    candidate = frame[frame.method == PRIMARY].set_index("sample_id")
    control = frame[frame.method == baseline].set_index("sample_id")
    merged = candidate[["record_id", "start", "label", "score", "threshold"]].join(
        control[["score", "threshold"]], lsuffix="_candidate", rsuffix="_baseline",
        validate="one_to_one",
    ).reset_index(drop=True)
    groups = []
    for _, group in merged.sort_values(["record_id", "start"]).groupby("record_id"):
        positions = group.index.to_numpy()
        groups.append([positions[start : start + block_size] for start in range(0, len(positions), block_size)])
    rng = np.random.default_rng(seed)
    names = ("auroc", "recall", "specificity", "balanced_accuracy")
    candidate_observed = metrics(
        merged.label, merged.score_candidate, float(merged.threshold_candidate.iloc[0])
    )
    baseline_observed = metrics(
        merged.label, merged.score_baseline, float(merged.threshold_baseline.iloc[0])
    )
    distributions = {name: [] for name in names}
    for _ in range(samples):
        selected = []
        for blocks in groups:
            chosen = rng.integers(0, len(blocks), len(blocks))
            selected.extend(np.concatenate([blocks[index] for index in chosen]))
        subset = merged.loc[np.asarray(selected, int)]
        first = metrics(subset.label, subset.score_candidate, float(subset.threshold_candidate.iloc[0]))
        second = metrics(subset.label, subset.score_baseline, float(subset.threshold_baseline.iloc[0]))
        for name in names:
            distributions[name].append(first[name] - second[name])
    return pd.DataFrame([
        {
            "candidate": PRIMARY,
            "baseline": baseline,
            "metric": name,
            "candidate_value": candidate_observed[name],
            "baseline_value": baseline_observed[name],
            "delta": candidate_observed[name] - baseline_observed[name],
            "ci_lower": float(np.quantile(distributions[name], 0.025)),
            "ci_upper": float(np.quantile(distributions[name], 0.975)),
            "probability_delta_gt_zero": float(np.mean(np.asarray(distributions[name]) > 0)),
            "bootstrap_samples": samples,
            "block_size": block_size,
        }
        for name in names
    ])


def make_figure(summary: pd.DataFrame, states: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    measures = ("auroc", "recall", "specificity", "balanced_accuracy")
    x = np.arange(len(summary))
    width = 0.19
    for index, measure in enumerate(measures):
        axes[0].bar(x + (index - 1.5) * width, summary[measure], width, label=measure)
    axes[0].set_xticks(x, summary.method, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Paper-aligned unseen-damage test")
    axes[0].legend(ncol=2)
    selected = states[states.method == PRIMARY]
    axes[1].bar(selected.state, selected.damage_recall)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Primary AR5 AL-AG fusion by damage state")
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
    parser.add_argument("--specificity-target", type=float, default=0.90)
    parser.add_argument("--mc-trials", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.paper_split_dir / "sample_manifest.csv")
    manifest = manifest.sort_values(["record_id", "start"]).reset_index(drop=True)
    train = manifest.index[manifest.role.eq("train_healthy")].to_numpy()
    test = manifest.index[manifest.role.isin(["test_healthy", "test_damage"])].to_numpy()
    if len(train) != 80 or len(test) != 720 or manifest.loc[train, "label"].ne(0).any():
        raise ValueError("Expected exact paper split: 80 healthy train and 720 test")

    frequency, shapes, spectra = extract_base(args.h5, manifest)
    ar5 = extract_ar5(args.h5, manifest, order=5)
    oof = crossfit_scores(manifest, train, ar5, frequency, shapes, spectra, args.seed)

    final_anchors_ag = learn_anchors(spectra["ag35"][train], frequency, count=5)
    train_modal = modal_matrix(train, "ag35", final_anchors_ag, frequency, shapes, spectra)
    test_modal = modal_matrix(test, "ag35", final_anchors_ag, frequency, shapes, spectra)
    modal_model = HealthyPCA.fit(train_modal)
    modal_reference = robust_reference(modal_model.score(train_modal))
    test_z = {"modal_ag": standardize(modal_model.score(test_modal), modal_reference)}

    paper_scores, test_z_ar = {}, {}
    for view, key in (("al", "paper_ar5_msd_al40"), ("ag", "paper_ar5_msd_ag35")):
        model = fit_paper_model(
            ar5[key][train], snr=20.0, copies=20,
            mc_trials=args.mc_trials, alpha=0.01,
            seed=args.seed + (0 if view == "al" else 10007),
        )
        paper_scores[view] = model.normalized_score(ar5[key][test])
        reference = robust_reference(model.raw_score(ar5[key][train]))
        test_z_ar[view] = standardize(model.raw_score(ar5[key][test]), reference)

    oof_methods = {
        "ar5_al_calibrated": oof["ar5_al"],
        "ar5_ag_calibrated": oof["ar5_ag"],
        PRIMARY: (oof["ar5_al"] + oof["ar5_ag"]) / 2,
        "ar5_modal_ag_fusion": (oof["ar5_al"] + oof["ar5_ag"] + oof["modal_ag"]) / 3,
    }
    test_methods = {
        "paper_ar5_al": paper_scores["al"],
        "paper_ar5_ag": paper_scores["ag"],
        "ar5_al_calibrated": test_z_ar["al"],
        "ar5_ag_calibrated": test_z_ar["ag"],
        PRIMARY: (test_z_ar["al"] + test_z_ar["ag"]) / 2,
        "ar5_modal_ag_fusion": (test_z_ar["al"] + test_z_ar["ag"] + test_z["modal_ag"]) / 3,
    }
    thresholds = {"paper_ar5_al": 1.0, "paper_ar5_ag": 1.0}
    calibration_rows = []
    for method, values in oof_methods.items():
        threshold, realized = threshold_at_specificity(values, args.specificity_target)
        thresholds[method] = threshold
        calibration_rows.append({
            "method": method,
            "threshold": threshold,
            "target_specificity": args.specificity_target,
            "realized_oof_specificity": realized,
            "oof_healthy_samples": len(values),
            "oof_score_min": float(np.min(values)),
            "oof_score_max": float(np.max(values)),
            "oof_unique_scores": int(np.unique(values).size),
            "threshold_above_oof_max": bool(threshold > np.max(values)),
        })

    test_manifest = manifest.loc[test].reset_index(drop=True)
    labels = test_manifest.label.to_numpy(int)
    metric_rows, score_rows, state_rows = [], [], []
    for method in METHODS:
        scores, threshold = test_methods[method], thresholds[method]
        metric_rows.append({
            "method": method,
            "primary": method == PRIMARY,
            "threshold": threshold,
            "test_samples": len(labels),
            **metrics(labels, scores, threshold),
        })
        prediction = scores >= threshold
        for row, value, alarm in zip(test_manifest.itertuples(index=False), scores, prediction):
            score_rows.append({
                "sample_id": row.sample_id,
                "record_id": row.record_id,
                "start": int(row.start),
                "state": row.state,
                "label": int(row.label),
                "method": method,
                "score": float(value),
                "threshold": threshold,
                "prediction": int(alarm),
            })
        for state in [f"DS{i}" for i in range(1, 9)]:
            damage = test_manifest.state.eq(state).to_numpy()
            cohort = test_manifest.state.isin(["UDS", state]).to_numpy()
            state_rows.append({
                "method": method,
                "state": state,
                "damage_samples": int(damage.sum()),
                "damage_recall": float(prediction[damage].mean()),
                "auroc_vs_uds": float(roc_auc_score(labels[cohort], scores[cohort])),
            })

    summary = pd.DataFrame(metric_rows)
    score_table = pd.DataFrame(score_rows)
    states = pd.DataFrame(state_rows)
    primary_frame = score_table[score_table.method == PRIMARY].reset_index(drop=True)
    primary_frame.index = np.arange(len(primary_frame))
    primary_ci = block_bootstrap(
        primary_frame, args.bootstrap_samples, args.seed + 70001
    )
    paired = pd.concat([
        paired_block_bootstrap(
            score_table, baseline, args.bootstrap_samples,
            args.seed + 80001 + index * 1009,
        )
        for index, baseline in enumerate(("paper_ar5_al", "paper_ar5_ag"))
    ], ignore_index=True)

    summary.to_csv(args.output_dir / "metrics.csv", index=False)
    score_table.to_csv(args.output_dir / "sample_scores.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.output_dir / "calibration.csv", index=False)
    primary_ci.to_csv(args.output_dir / "primary_block_bootstrap_ci.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_paper_baselines.csv", index=False)
    (args.output_dir / "anchors.json").write_text(json.dumps(
        {"modal_ag": final_anchors_ag.tolist()}, indent=2
    ), encoding="utf-8")
    make_figure(summary, states, args.output_dir / "fig_c10_paper_unseen_fusion.png")

    protocol = {
        "protocol": "hellbridge-c10-paper-aligned-unseen-damage-fusion",
        "train": "80 healthy windows: 40 from each UDS repeat",
        "test": "80 healthy plus 640 unseen damage windows",
        "damage_labels_used_for_fit_anchor_calibration_fusion_or_threshold": False,
        "primary_method": PRIMARY,
        "primary_fusion": "equal average of robust standardized AR5-AL and AR5-AG scores",
        "healthy_calibration": "five blocked folds, each holding 8 windows per UDS repeat",
        "specificity_target": args.specificity_target,
        "score_normalization": "log1p raw score followed by healthy median/MAD; no upper clipping",
        "paper_baselines": ["paper_ar5_al", "paper_ar5_ag"],
        "repeat_holdout_c9_status": "retained as strict domain-shift stress test",
        "test_labels_used_only_for_final_metrics": True,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("\n=== C10 METRICS ===")
    print(summary.to_string(index=False))
    print("\n=== PRIMARY PAIRED VS PAPER BASELINES ===")
    print(paired.to_string(index=False))
    print("\n=== PRIMARY STATE METRICS ===")
    print(states[states.method == PRIMARY].to_string(index=False))
    print(f"HELLBRIDGE_C10_PAPER_UNSEEN_FUSION_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

