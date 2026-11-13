from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_hellbridge_c6_paper_ar5_msd import read_record_channels
from run_hellbridge_c7_z24_m1_transfer import modal_from_components, modal_spectrum
from run_z24_supervised_consistency_c1 import feature_from_signals as c1_feature


METHODS = ("m1_al40", "m1_ag35", "m1_al_ag_early_fusion")


def blocked_split(manifest: pd.DataFrame) -> pd.DataFrame:
    """Leakage-reduced chronological split with two four-window guard bands."""
    output = manifest.sort_values(["record_id", "start"]).reset_index(drop=True).copy()
    output["within_record_rank"] = output.groupby("record_id").cumcount()
    counts = output.groupby("record_id").size()
    if (counts < 80).any():
        raise ValueError(f"Every record needs >=80 windows, got {counts.to_dict()}")
    output = output[output.within_record_rank < 80].copy()
    rank = output.within_record_rank
    output["supervised_split"] = np.select(
        [rank < 40, (rank >= 44) & (rank < 60), rank >= 64],
        ["train", "validation", "test"],
        default="guard",
    )
    return output.reset_index(drop=True)


def learn_anchors(
    spectra: np.ndarray,
    frequency: np.ndarray,
    count: int = 5,
) -> np.ndarray:
    reference = np.median(np.asarray(spectra, float), axis=0)
    spacing = max(1, int(round(0.60 / (frequency[1] - frequency[0]))))
    peaks, properties = find_peaks(reference, distance=spacing, prominence=0)
    prominence = properties.get("prominences", reference[peaks])
    ordered = peaks[np.argsort(prominence)[::-1]] if len(peaks) else np.asarray([], int)
    selected: list[int] = []
    for index in ordered:
        if all(abs(frequency[index] - frequency[prior]) >= 0.60 for prior in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        ordered_all = np.argsort(reference)[::-1]
        for index in ordered_all:
            if all(abs(frequency[index] - frequency[prior]) >= 0.60 for prior in selected):
                selected.append(int(index))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"Could learn only {len(selected)} modal anchors")
    return np.sort(frequency[selected]).astype(float)


def extract_base(
    h5_path: Path,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    shapes = {name: [None] * len(manifest) for name in ("al40", "ag35")}
    spectra = {name: [None] * len(manifest) for name in ("al40", "ag35")}
    frequency: np.ndarray | None = None
    with h5py.File(h5_path, "r") as source:
        total = manifest.record_id.nunique()
        for record_position, (record_id, rows) in enumerate(
            manifest.groupby("record_id", sort=False), start=1
        ):
            channels = read_record_channels(source[record_id])
            for row in rows.itertuples():
                start, stop = int(row.start), int(row.stop)
                for group_name in ("al40", "ag35"):
                    signals = channels[group_name][:, start:stop]
                    shapes[group_name][row.Index] = c1_feature(signals, "avt")
                    current_frequency, current_spectrum = modal_spectrum(signals)
                    if frequency is None:
                        frequency = current_frequency
                    elif not np.allclose(frequency, current_frequency):
                        raise RuntimeError("Inconsistent frequency grids")
                    spectra[group_name][row.Index] = current_spectrum
            print(
                f"base-features record={record_position}/{total} "
                f"name={record_id} windows={len(rows)}",
                flush=True,
            )
    assert frequency is not None
    return (
        frequency,
        {name: np.stack(values) for name, values in shapes.items()},
        {name: np.stack(values) for name, values in spectra.items()},
    )


def build_features(
    manifest: pd.DataFrame,
    frequency: np.ndarray,
    shapes: dict[str, np.ndarray],
    spectra: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    healthy_train = (
        manifest.supervised_split.eq("train") & manifest.label.eq(0)
    ).to_numpy()
    anchors, group_features = {}, {}
    for group_name in ("al40", "ag35"):
        anchors[group_name] = learn_anchors(
            spectra[group_name][healthy_train], frequency, count=5
        )
        modal = np.stack([
            modal_from_components(frequency, spectrum, anchors[group_name])
            for spectrum in spectra[group_name]
        ])
        group_features[group_name] = np.concatenate(
            [shapes[group_name], modal], axis=1
        )
        print(
            f"anchors {group_name}={anchors[group_name].round(3).tolist()} "
            f"dimension={group_features[group_name].shape[1]}",
            flush=True,
        )
    return anchors, {
        "m1_al40": group_features["al40"],
        "m1_ag35": group_features["ag35"],
        "m1_al_ag_early_fusion": np.concatenate(
            [group_features["al40"], group_features["ag35"]], axis=1
        ),
    }


def fit_model(matrix: np.ndarray, labels: np.ndarray, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            solver="liblinear",
            random_state=seed,
        ),
    ).fit(matrix, labels)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
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


def choose_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    specificity_gate: float,
) -> tuple[float, dict[str, float], bool]:
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    evaluated = [(float(value), metrics(labels, scores, float(value))) for value in candidates]
    eligible = [item for item in evaluated if item[1]["specificity"] >= specificity_gate]
    gate_pass = bool(eligible)
    pool = eligible if eligible else evaluated
    threshold, result = max(
        pool,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["recall"],
            item[1]["specificity"],
        ),
    )
    return threshold, result, gate_pass


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    names = ("auroc", "recall", "specificity", "balanced_accuracy", "f1")
    draws = {name: [] for name in names}
    observed = metrics(labels, scores, threshold)
    for _ in range(samples):
        index = np.r_[
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ]
        result = metrics(labels[index], scores[index], threshold)
        for name in names:
            draws[name].append(result[name])
    return pd.DataFrame([
        {
            "metric": name,
            "estimate": observed[name],
            "ci_lower": float(np.quantile(draws[name], 0.025)),
            "ci_upper": float(np.quantile(draws[name], 0.975)),
            "bootstrap_samples": samples,
        }
        for name in names
    ])


def make_figure(summary: pd.DataFrame, states: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    measures = ["auroc", "recall", "specificity", "balanced_accuracy"]
    x = np.arange(len(summary))
    width = 0.19
    for index, measure in enumerate(measures):
        axes[0].bar(x + (index - 1.5) * width, summary[measure], width, label=measure)
    axes[0].set_xticks(x, summary.method.str.replace("m1_", ""), rotation=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("HellBridge validation-locked test metrics")
    axes[0].legend(ncol=2)
    selected = states[states.selected]
    axes[1].bar(selected.state, selected.damage_recall)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Selected model: test recall by damage state")
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
    parser.add_argument("--specificity-gate", type=float, default=0.80)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper_manifest = pd.read_csv(args.paper_split_dir / "sample_manifest.csv")
    manifest = blocked_split(paper_manifest)
    manifest.to_csv(args.output_dir / "supervised_manifest.csv", index=False)
    split_summary = manifest.groupby(
        ["supervised_split", "state", "label"], as_index=False
    ).size()
    split_summary.to_csv(args.output_dir / "split_summary.csv", index=False)
    print("\n=== SUPERVISED BLOCKED SPLIT ===")
    print(split_summary.to_string(index=False))

    frequency, shapes, spectra = extract_base(args.h5, manifest)
    anchors, feature_sets = build_features(
        manifest, frequency, shapes, spectra
    )
    (args.output_dir / "anchors.json").write_text(json.dumps(
        {name: values.tolist() for name, values in anchors.items()}, indent=2
    ), encoding="utf-8")

    train = manifest.supervised_split.eq("train").to_numpy()
    validation = manifest.supervised_split.eq("validation").to_numpy()
    test = manifest.supervised_split.eq("test").to_numpy()
    labels = manifest.label.to_numpy(int)
    models, validation_rows, test_rows, prediction_rows, state_rows = {}, [], [], [], []
    thresholds = {}
    for method_index, method in enumerate(METHODS):
        matrix = feature_sets[method]
        model = fit_model(matrix[train], labels[train], args.seed + method_index)
        models[method] = model
        validation_score = model.predict_proba(matrix[validation])[:, 1]
        threshold, validation_metric, gate_pass = choose_threshold(
            labels[validation], validation_score, args.specificity_gate
        )
        thresholds[method] = threshold
        validation_rows.append({
            "method": method,
            "threshold": threshold,
            "specificity_gate_pass": gate_pass,
            **validation_metric,
        })

    validation_table = pd.DataFrame(validation_rows)
    eligible = validation_table[validation_table.specificity_gate_pass]
    pool = eligible if not eligible.empty else validation_table
    selected_method = str(pool.sort_values(
        ["balanced_accuracy", "recall", "auroc", "specificity"],
        ascending=False,
    ).iloc[0].method)
    validation_table["selected"] = validation_table.method.eq(selected_method)

    test_manifest = manifest.loc[test].reset_index(drop=True)
    for method in METHODS:
        matrix = feature_sets[method]
        score = models[method].predict_proba(matrix[test])[:, 1]
        threshold = thresholds[method]
        result = metrics(labels[test], score, threshold)
        selected = method == selected_method
        test_rows.append({
            "method": method,
            "selected": selected,
            "threshold": threshold,
            "test_samples": int(test.sum()),
            **result,
        })
        for row, value in zip(test_manifest.itertuples(index=False), score):
            prediction_rows.append({
                "sample_id": row.sample_id,
                "record_id": row.record_id,
                "state": row.state,
                "label": int(row.label),
                "method": method,
                "selected": selected,
                "score": float(value),
                "threshold": threshold,
                "prediction": int(value >= threshold),
            })
        for state in [f"DS{i}" for i in range(1, 9)]:
            damage = test_manifest.state.eq(state).to_numpy()
            cohort = test_manifest.state.isin(["UDS", state]).to_numpy()
            state_rows.append({
                "method": method,
                "selected": selected,
                "state": state,
                "auroc_vs_uds": float(roc_auc_score(
                    test_manifest.loc[cohort, "label"], score[cohort]
                )),
                "damage_recall": float(np.mean(score[damage] >= threshold)),
                "damage_samples": int(damage.sum()),
            })

    test_table = pd.DataFrame(test_rows)
    predictions = pd.DataFrame(prediction_rows)
    states = pd.DataFrame(state_rows)
    selected_score = predictions[predictions.selected].score.to_numpy(float)
    selected_labels = predictions[predictions.selected].label.to_numpy(int)
    ci = bootstrap_ci(
        selected_labels,
        selected_score,
        thresholds[selected_method],
        args.bootstrap_samples,
        args.seed + 70001,
    )

    validation_table.to_csv(args.output_dir / "validation_selection.csv", index=False)
    test_table.to_csv(args.output_dir / "test_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)
    ci.to_csv(args.output_dir / "selected_bootstrap_ci.csv", index=False)
    joblib.dump({
        "selected_method": selected_method,
        "model": models[selected_method],
        "threshold": thresholds[selected_method],
        "anchors": anchors,
    }, args.output_dir / "selected_model.joblib")
    make_figure(test_table, states, args.output_dir / "fig_hellbridge_m1_from_scratch.png")

    run = {
        "protocol": "hellbridge-c8-m1-structure-trained-from-scratch",
        "source_weights_or_anchors_from_z24": False,
        "structure_from_z24": "C1 spectral shape + five-anchor modal vector + standardized balanced logistic",
        "split": {
            "train_ranks": "0:40",
            "guard_1_ranks": "40:44",
            "validation_ranks": "44:60",
            "guard_2_ranks": "60:64",
            "test_ranks": "64:80",
        },
        "damage_labels_used_for_training": True,
        "damage_labels_used_for_validation_selection": True,
        "test_labels_used_for_selection": False,
        "claim_scope": "within-HellBridge supervised generalization, not cross-bridge external validation",
        "selected_method": selected_method,
        "specificity_gate": args.specificity_gate,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("\n=== VALIDATION MODEL SELECTION ===")
    print(validation_table.to_string(index=False))
    print("\n=== LOCKED TEST RESULTS ===")
    print(test_table.to_string(index=False))
    print("\n=== SELECTED MODEL CI ===")
    print(ci.to_string(index=False))
    print(
        f"HELLBRIDGE_C8_M1_FROM_SCRATCH_PASS selected={selected_method} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

