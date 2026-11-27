from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import f1_score, roc_auc_score

from run_z24_b2 import (
    build_file_index,
    features,
    find_manifest,
    learn_anchors,
    load_vertical,
    normalized_sensor_spectra,
    resolve_source,
)


PRIMARY = "modal_frequency"
FUSION = "geometric_frequency_response"
METHODS = (PRIMARY, "spectral_response", FUSION)


def prepare_locked(root: Path) -> tuple[pd.DataFrame, Path]:
    manifest = find_manifest(root)
    frame = pd.read_csv(manifest)
    required = {"record_id", "state", "excitation", "setup", "split", "label"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Manifest missing {sorted(required.difference(frame.columns))}")
    frame = frame[frame.split.astype(str).str.lower().isin(["train", "val", "test"])].copy()
    frame["split"] = frame.split.astype(str).str.lower()
    frame["label"] = frame.label.astype(int)
    frame["state"] = frame.state.astype(int)
    frame["setup"] = frame.setup.astype(int)
    frame["excitation"] = frame.excitation.astype(str).str.lower()
    counts = (
        len(frame[(frame.split == "train") & (frame.label == 0)]),
        len(frame[(frame.split == "val") & (frame.label == 0)]),
        len(frame[frame.split == "test"]),
    )
    if counts != (24, 6, 132):
        raise RuntimeError(f"Unexpected locked split counts: {counts}")
    if frame.loc[frame.split == "train", "label"].ne(0).any():
        raise RuntimeError("Damage leakage in training")
    root_dir = manifest.parent.parent
    file_index = build_file_index(root_dir)
    frame["resolved_path"] = [
        str(resolve_source(row, file_index, root_dir)) for _, row in frame.iterrows()
    ]
    # State 01 is the pre-damage healthy campaign; state 08 is the restored
    # healthy campaign. This is chronological metadata, not a damage label.
    frame["campaign"] = np.where(frame.state < 8, "pre", "post")
    return frame.reset_index(drop=True), manifest


def record_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frequency, sensor_psd = normalized_sensor_spectra(load_vertical(path))
    return frequency, np.median(sensor_psd, axis=0)


def robust_location_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    mad = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    scale = np.maximum(mad, np.maximum(0.20 * np.std(matrix, axis=0), 1e-5))
    return center, scale


class HealthyView:
    def __init__(self, center: np.ndarray, scale: np.ndarray, model: LedoitWolf):
        self.center, self.scale, self.model = center, scale, model

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "HealthyView":
        center, scale = robust_location_scale(matrix)
        normalized = (matrix - center) / scale
        return cls(center, scale, LedoitWolf().fit(normalized))

    def score(self, vector: np.ndarray) -> float:
        normalized = (vector - self.center) / self.scale
        delta = normalized - self.model.location_
        return float(np.sqrt(max(delta @ self.model.precision_ @ delta, 0.0)))


def higher_threshold(values: np.ndarray, specificity: float) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, float))
    rank = min(len(ordered) - 1, max(0, int(np.ceil(specificity * len(ordered))) - 1))
    threshold = float(np.nextafter(ordered[rank], np.inf))
    return threshold, float(np.mean(ordered < threshold))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    prediction = scores >= 1.0
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


def bootstrap(frame: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = ("auroc", "recall", "specificity", "balanced_accuracy", "f1")
    rows = []
    for method, group in frame.groupby("method"):
        healthy = group[group.label == 0].reset_index(drop=True)
        damage = group[group.label == 1].reset_index(drop=True)
        observed = metrics(group.label, group.score)
        distribution = {name: [] for name in names}
        for _ in range(samples):
            sampled = pd.concat([
                healthy.iloc[rng.integers(0, len(healthy), len(healthy))],
                damage.iloc[rng.integers(0, len(damage), len(damage))],
            ])
            result = metrics(sampled.label, sampled.score)
            for name in names:
                distribution[name].append(result[name])
        for name in names:
            rows.append({
                "method": method, "metric": name, "estimate": observed[name],
                "ci_lower": float(np.quantile(distribution[name], 0.025)),
                "ci_upper": float(np.quantile(distribution[name], 0.975)),
                "bootstrap_samples": samples,
                "healthy_records": len(healthy), "damage_records": len(damage),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # With six independent calibration records, 0.90 would round to 6/6 and
    # reproduce the unrealistically conservative maximum-score threshold.
    parser.add_argument("--specificity-target", type=float, default=0.80)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, manifest = prepare_locked(args.input_root)
    # Validation damage is deliberately never loaded.
    used = frame[(frame.split == "train") | ((frame.split == "val") & (frame.label == 0)) | (frame.split == "test")].copy()
    spectra, frequencies = {}, {}
    for position, row in enumerate(used.itertuples(), start=1):
        frequency, spectrum = record_spectrum(Path(row.resolved_path))
        spectra[row.record_id] = spectrum
        frequencies[row.excitation] = frequency
        if position % 36 == 0:
            print(f"spectra={position}/{len(used)}", flush=True)

    feature_rows, anchors_by_excitation = [], {}
    for excitation in ("avt", "fvt"):
        train_exc = used[(used.split == "train") & (used.excitation == excitation)]
        frequency = frequencies[excitation]
        anchors = learn_anchors([spectra[item] for item in train_exc.record_id], frequency)
        anchors_by_excitation[excitation] = anchors
        subset = used[used.excitation == excitation]
        for row in subset.itertuples():
            vector = features(spectra[row.record_id], frequency, anchors)
            feature_rows.append({
                "record_id": row.record_id, "state": int(row.state), "setup": int(row.setup),
                "excitation": excitation, "campaign": row.campaign, "split": row.split,
                "label": int(row.label), "vector": vector,
            })
    table = pd.DataFrame(feature_rows)

    models, shifts, raw_rows = {}, {}, []
    frequency_indices = np.arange(0, 8, 2)
    response_indices = np.r_[np.arange(1, 8, 2), np.arange(8, 13)]
    for excitation in ("avt", "fvt"):
        train = table[(table.split == "train") & (table.excitation == excitation)].copy()
        paired_differences = []
        for setup in (4, 5, 6):
            pre = train[(train.setup == setup) & (train.campaign == "pre")]
            post = train[(train.setup == setup) & (train.campaign == "post")]
            if len(pre) != 1 or len(post) != 1:
                raise RuntimeError(f"Missing paired healthy campaigns for {excitation} setup {setup}")
            paired_differences.append(post.vector.iloc[0] - pre.vector.iloc[0])
        shift = np.median(np.stack(paired_differences), axis=0)
        shifts[excitation] = shift

        aligned_train = np.stack([
            row.vector - (shift if row.campaign == "post" else 0.0)
            for row in train.itertuples()
        ])
        frequency_model = HealthyView.fit(aligned_train[:, frequency_indices])
        response_model = HealthyView.fit(aligned_train[:, response_indices])
        models[excitation] = (frequency_model, response_model)

        for row in table[table.excitation == excitation].itertuples():
            aligned = row.vector - (shift if row.campaign == "post" else 0.0)
            raw_rows.append({
                "record_id": row.record_id, "state": row.state, "setup": row.setup,
                "excitation": excitation, "campaign": row.campaign, "split": row.split,
                "label": row.label,
                "frequency_raw": frequency_model.score(aligned[frequency_indices]),
                "response_raw": response_model.score(aligned[response_indices]),
            })
    raw = pd.DataFrame(raw_rows)
    calibration = raw[(raw.split == "val") & (raw.label == 0)]
    frequency_threshold, frequency_cal_spec = higher_threshold(
        calibration.frequency_raw, args.specificity_target
    )
    response_threshold, response_cal_spec = higher_threshold(
        calibration.response_raw, args.specificity_target
    )
    raw[PRIMARY] = raw.frequency_raw / frequency_threshold
    raw["spectral_response"] = raw.response_raw / response_threshold
    raw[FUSION] = np.sqrt(
        np.maximum(raw[PRIMARY], 0.0) * np.maximum(raw.spectral_response, 0.0)
    )

    rows = []
    for row in raw.itertuples():
        role = "train_healthy" if row.split == "train" else (
            "calibration_healthy" if row.split == "val" else "locked_test"
        )
        for method in METHODS:
            score = float(getattr(row, method))
            rows.append({
                "record_id": row.record_id, "role": role, "split": row.split,
                "label": row.label, "state": row.state, "setup": row.setup,
                "excitation": row.excitation, "campaign": row.campaign,
                "method": method, "score": score, "threshold": 1.0,
                "prediction": int(score >= 1.0),
            })
    scores = pd.DataFrame(rows)
    test = scores[scores.role == "locked_test"].reset_index(drop=True)
    summary = pd.DataFrame([
        {"method": method, "records": len(group), **metrics(group.label, group.score)}
        for method, group in test.groupby("method", sort=False)
    ])
    healthy = test[test.label == 0]
    state_rows = []
    for method in METHODS:
        healthy_method = healthy[healthy.method == method]
        for state, damage in test[(test.method == method) & (test.label == 1)].groupby("state"):
            cohort = pd.concat([healthy_method, damage])
            state_rows.append({
                "method": method, "state": int(state), "damage_records": len(damage),
                "damage_recall": float(damage.prediction.mean()),
                "auroc_vs_test_healthy": float(roc_auc_score(cohort.label, cohort.score)),
                "median_damage_score": float(damage.score.median()),
            })
    states = pd.DataFrame(state_rows)
    ci = bootstrap(test, args.bootstrap_samples, args.seed)

    raw.drop(columns=[]).to_csv(args.output_dir / "raw_scores.csv", index=False)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    summary.to_csv(args.output_dir / "metrics.csv", index=False)
    states.to_csv(args.output_dir / "state_metrics.csv", index=False)
    ci.to_csv(args.output_dir / "bootstrap_ci.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    measures = ("auroc", "recall", "specificity", "balanced_accuracy")
    x = np.arange(len(summary)); width = 0.19
    for index, name in enumerate(measures):
        axes[0].bar(x + (index - 1.5) * width, summary[name], width, label=name)
    axes[0].set_xticks(x, summary.method, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.05); axes[0].legend(ncol=2); axes[0].grid(axis="y", alpha=.25)
    primary_states = states[states.method == PRIMARY]
    axes[1].bar(primary_states.state.astype(str), primary_states.damage_recall)
    axes[1].set_ylim(0, 1.05); axes[1].set_xlabel("Locked unseen damage state")
    axes[1].set_ylabel("Recall"); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "z24_c14_results.png", dpi=220)
    plt.close(fig)

    protocol = {
        "protocol": "z24-c1.4-healthy-only-environment-normalized-multiview",
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "train_healthy": 24, "calibration_healthy": 6,
        "locked_test_healthy": 6, "locked_test_damage": 126,
        "locked_test_damage_states": [3, 7, 9, 11, 12, 13, 15],
        "validation_damage_loaded": False,
        "damage_used_for_training": False,
        "damage_used_for_calibration": False,
        "damage_used_for_selection": False,
        "test_used_for_model_fitting_or_threshold_calibration": False,
        "benchmark_status": "development benchmark; this public test split was observed during prior iterations",
        "primary_for_this_run": PRIMARY,
        "environment_shift_source": "paired healthy state01/state08, setups04-06",
        "fusion_sensitivity_method": FUSION,
        "fusion": "sqrt(normalized_frequency * normalized_response)",
        "specificity_target": args.specificity_target,
        "frequency_threshold": frequency_threshold,
        "response_threshold": response_threshold,
        "frequency_calibration_specificity": frequency_cal_spec,
        "response_calibration_specificity": response_cal_spec,
        "healthy_test_limitation": "Only six independent healthy test records; report CI.",
        "anchors": {key: [float(value) for value in values] for key, values in anchors_by_excitation.items()},
        "environment_shifts": {key: value.tolist() for key, value in shifts.items()},
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("\n=== Z24 C1.4 LOCKED HEALTHY-ONLY TEST ===")
    print(summary.to_string(index=False))
    print("\n=== PRIMARY BY DAMAGE STATE ===")
    print(primary_states.to_string(index=False))
    print("\n=== PRIMARY 95% CI ===")
    print(ci[ci.method == PRIMARY].to_string(index=False))
    print(f"Z24_C14_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()


