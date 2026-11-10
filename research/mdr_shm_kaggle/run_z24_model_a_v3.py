from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_z24_b2 import (
    Model,
    build_file_index,
    excitation,
    features,
    find_manifest,
    learn_anchors,
    load_vertical,
    metric_dict,
    normalized_sensor_spectra,
    resolve_source,
    stable_rng,
)


def prepare_frame(root: Path) -> tuple[pd.DataFrame, Path]:
    manifest = find_manifest(root)
    frame = pd.read_csv(manifest)
    frame["label"] = frame["label"].astype(int)
    if "role" not in frame:
        frame["role"] = frame["split"]
    frame["role"] = frame["role"].astype(str).str.lower().replace({"val": "calibration"})
    frame.loc[frame.label == 1, "role"] = "test"
    frame = frame[frame.role.isin(["train", "calibration", "test"])].copy()
    if frame.loc[frame.role.isin(["train", "calibration"]), "label"].ne(0).any():
        raise RuntimeError("Damage leakage into train/calibration")
    dataset_root = manifest.parent.parent
    index = build_file_index(dataset_root)
    frame["resolved_path"] = [
        str(resolve_source(row, index, dataset_root)) for _, row in frame.iterrows()
    ]
    frame["excitation_resolved"] = [excitation(row) for _, row in frame.iterrows()]
    return frame, manifest


def fit_b2(rows: pd.DataFrame) -> Model:
    train_spectra = []
    frequencies = None
    for _, row in rows[rows.role == "train"].iterrows():
        frequencies, sensor_psd = normalized_sensor_spectra(
            load_vertical(Path(row.resolved_path))
        )
        train_spectra.append(np.median(sensor_psd, axis=0))
    assert frequencies is not None
    anchors = learn_anchors(train_spectra, frequencies)
    matrix = np.stack([
        features(spectrum, frequencies, anchors) for spectrum in train_spectra
    ])
    model = Model.fit(anchors, matrix)
    calibration_scores = []
    for _, row in rows[rows.role == "calibration"].iterrows():
        freq, sensor_psd = normalized_sensor_spectra(
            load_vertical(Path(row.resolved_path))
        )
        calibration_scores.append(
            model.raw_score(features(np.median(sensor_psd, axis=0), freq, anchors))
        )
    model.threshold = float(max(calibration_scores))
    return model


def b2_score(
    model: Model,
    frequencies: np.ndarray,
    sensor_psd: np.ndarray,
    availability: np.ndarray,
) -> float:
    spectrum = np.median(sensor_psd[availability], axis=0)
    return model.score(features(spectrum, frequencies, model.anchors))


@dataclass
class ReliabilityCalibration:
    center: np.ndarray
    scale: np.ndarray
    threshold: float


def reliability_features(
    model: Model,
    frequencies: np.ndarray,
    sensor_psd: np.ndarray,
    availability: np.ndarray,
    quality: np.ndarray,
) -> tuple[np.ndarray, dict]:
    present = np.flatnonzero(availability)
    base = b2_score(model, frequencies, sensor_psd, availability)
    jackknife = []
    if len(present) > 2:
        for sensor in present:
            keep = availability.copy()
            keep[sensor] = False
            jackknife.append(b2_score(model, frequencies, sensor_psd, keep))
    else:
        jackknife = [base]
    jackknife_values = np.asarray(jackknife)
    jackknife_instability = float(
        np.std(jackknife_values) / (abs(np.mean(jackknife_values)) + 0.10)
    )

    present_psd = sensor_psd[availability]
    log_psd = np.log10(present_psd + 1e-16)
    high_band = frequencies >= 20.0
    high_frequency_share = np.trapezoid(
        present_psd[:, high_band], frequencies[high_band], axis=1
    )
    hf_center = float(np.median(high_frequency_share))
    hf_scale = float(
        1.4826 * np.median(np.abs(high_frequency_share - hf_center))
    )
    high_frequency_outlier = float(
        max(0.0, np.quantile(high_frequency_share, 0.90) - hf_center)
        / max(hf_scale, 1e-4)
    )
    roughness = np.median(np.abs(np.diff(log_psd, axis=1)), axis=1)
    rough_center = float(np.median(roughness))
    rough_scale = float(1.4826 * np.median(np.abs(roughness - rough_center)))
    spectral_roughness_outlier = float(
        max(0.0, np.quantile(roughness, 0.90) - rough_center)
        / max(rough_scale, 1e-3)
    )

    coverage = float(availability.mean())
    observed_quality = float(np.mean(quality[availability]))
    vector = np.asarray(
        [
            jackknife_instability,
            1.0 - coverage,
            1.0 - observed_quality,
            high_frequency_outlier,
            spectral_roughness_outlier,
        ],
        dtype=float,
    )
    return vector, {
        "jackknife_instability": jackknife_instability,
        "sensor_coverage": coverage,
        "observed_quality": observed_quality,
        "high_frequency_outlier": high_frequency_outlier,
        "spectral_roughness_outlier": spectral_roughness_outlier,
        "jackknife_score_min": float(jackknife_values.min()),
        "jackknife_score_max": float(jackknife_values.max()),
    }


RELIABILITY_WEIGHTS = np.asarray([1.0, 0.25, 0.50, 1.0, 1.0])


def normalized_reliability(
    vector: np.ndarray, calibration: ReliabilityCalibration
) -> float:
    z = np.maximum((vector - calibration.center) / calibration.scale, 0.0)
    raw = float(
        np.sum(z * RELIABILITY_WEIGHTS) / RELIABILITY_WEIGHTS.sum()
    )
    return raw / max(calibration.threshold, 1e-12)


def corrupt(
    signals: np.ndarray,
    condition: str,
    record_id: str,
    replicate: int,
    model: Model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = signals.copy()
    sensors, samples = output.shape
    availability = np.ones(sensors, dtype=bool)
    quality = np.ones(sensors, dtype=np.float32)
    if condition == "clean":
        return output, availability, quality

    rate = float(condition.split("_")[-1])
    rng = stable_rng(record_id, f"selective_b2_{condition}", replicate)
    affected_count = max(1, int(round(0.30 * sensors)))
    affected = rng.choice(sensors, affected_count, replace=False)

    if condition.startswith("missing"):
        length = int(round(rate * samples))
        coordinates = np.arange(samples)
        for sensor in affected:
            start = int(rng.integers(0, samples - length + 1))
            keep = np.ones(samples, dtype=bool)
            keep[start : start + length] = False
            output[sensor, ~keep] = np.interp(
                coordinates[~keep], coordinates[keep], output[sensor, keep]
            )
            quality[sensor] = 1.0 - rate
    elif condition.startswith("noise"):
        for sensor in affected:
            centered = output[sensor] - output[sensor].mean()
            power = float(np.mean(centered ** 2))
            sigma = np.sqrt(max(power, 1e-30) / 10 ** (rate / 10))
            output[sensor] += rng.normal(0.0, sigma, samples).astype(np.float32)
    elif condition.startswith("dropout"):
        count = min(sensors - 1, int(round(rate * sensors)))
        availability[rng.choice(sensors, count, replace=False)] = False
    elif condition.startswith("targeted"):
        frequencies, psd = normalized_sensor_spectra(output)
        evidence = np.asarray([
            model.raw_score(features(item, frequencies, model.anchors))
            for item in psd
        ])
        count = min(sensors - 1, int(round(rate * sensors)))
        availability[np.argsort(evidence)[-count:]] = False
    else:
        raise ValueError(condition)
    return output, availability, quality


def calibrate_reliability(
    model: Model,
    rows: pd.DataFrame,
) -> ReliabilityCalibration:
    vectors = []
    # Only clean and mild corruption are declared reliable for calibration.
    # Severe conditions remain unseen stress tests.
    calibration_conditions = (
        "clean", "missing_0.10", "noise_20", "dropout_0.10"
    )
    for _, row in rows[rows.role == "calibration"].iterrows():
        signals = load_vertical(Path(row.resolved_path))
        for condition in calibration_conditions:
            for replicate in range(3):
                changed, availability, quality = corrupt(
                    signals, condition, str(row.record_id), replicate, model
                )
                frequencies, psd = normalized_sensor_spectra(changed)
                vector, _ = reliability_features(
                    model, frequencies, psd, availability, quality
                )
                vectors.append(vector)
    matrix = np.stack(vectors)
    center = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    floors = np.asarray([0.01, 0.02, 0.01, 0.10, 0.10])
    scale = np.maximum(scale, floors)
    positive_z = np.maximum((matrix - center) / scale, 0.0)
    raw_scores = np.sum(
        positive_z * RELIABILITY_WEIGHTS[None, :], axis=1
    ) / RELIABILITY_WEIGHTS.sum()
    # Maximum healthy-mild calibration score: no damage label is consumed.
    threshold = max(float(np.max(raw_scores)), 1e-6)
    return ReliabilityCalibration(center, scale, threshold)


def selective_curve(group: pd.DataFrame) -> list[dict]:
    output = []
    ordered = group.sort_values("reliability")
    for requested in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(2, int(np.ceil(len(ordered) * requested)))
        kept = ordered.iloc[:count]
        if kept.label.nunique() < 2:
            continue
        output.append({
            "coverage": requested,
            "records": len(kept),
            **metric_dict(kept.label.tolist(), kept.score.tolist()),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/model_a_v3_selective_b2"),
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    args = parser.parse_args()

    frame, manifest = prepare_frame(args.input_root)
    models = {}
    reliability_calibrations = {}
    for excitation_name in ("avt", "fvt"):
        rows = frame[frame.excitation_resolved == excitation_name]
        model = fit_b2(rows)
        models[excitation_name] = model
        reliability_calibrations[excitation_name] = calibrate_reliability(model, rows)
        print(
            f"{excitation_name}: score_threshold={model.threshold:.6f} "
            f"reliability_threshold="
            f"{reliability_calibrations[excitation_name].threshold:.6f}",
            flush=True,
        )

    conditions = (
        ["clean", "missing_0.30", "noise_5", "dropout_0.30", "targeted_0.30"]
        if args.mode == "quick"
        else [
            "clean",
            "missing_0.10", "missing_0.30", "missing_0.50",
            "noise_20", "noise_10", "noise_5",
            "dropout_0.10", "dropout_0.30", "dropout_0.50",
            "targeted_0.10", "targeted_0.30", "targeted_0.50",
        ]
    )
    output = []
    test = frame[frame.role == "test"]
    for index, (_, row) in enumerate(test.iterrows(), 1):
        signals = load_vertical(Path(row.resolved_path))
        model = models[row.excitation_resolved]
        reliability_calibration = reliability_calibrations[row.excitation_resolved]
        for condition in conditions:
            repeats = 1 if args.mode == "quick" or condition == "clean" else 3
            for replicate in range(repeats):
                changed, availability, quality = corrupt(
                    signals, condition, str(row.record_id), replicate, model
                )
                frequencies, psd = normalized_sensor_spectra(changed)
                score = b2_score(model, frequencies, psd, availability)
                reliability_vector, details = reliability_features(
                    model, frequencies, psd, availability, quality
                )
                reliability = normalized_reliability(
                    reliability_vector, reliability_calibration
                )
                output.append({
                    "record_id": row.record_id,
                    "label": int(row.label),
                    "excitation": row.excitation_resolved,
                    "condition": condition,
                    "replicate": replicate,
                    "score": score,
                    "reliability": reliability,
                    "accepted": reliability <= 1.0,
                    **details,
                })
        if index % 36 == 0:
            print(f"test={index}/{len(test)}", flush=True)

    scores = pd.DataFrame(output)
    metrics = []
    risk_coverage = {}
    for (condition, replicate), group in scores.groupby(["condition", "replicate"]):
        accepted = group[group.accepted]
        entry = {
            "condition": condition,
            "replicate": int(replicate),
            **metric_dict(group.label.tolist(), group.score.tolist()),
            "acceptance_rate": float(group.accepted.mean()),
        }
        if accepted.label.nunique() == 2:
            accepted_metrics = metric_dict(
                accepted.label.tolist(), accepted.score.tolist()
            )
            entry.update({
                f"accepted_{key}": value
                for key, value in accepted_metrics.items()
            })
        metrics.append(entry)
        risk_coverage[f"{condition}_rep{replicate}"] = selective_curve(group)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "risk_coverage.json").write_text(
        json.dumps(risk_coverage, indent=2)
    )
    artifact = {
        "protocol": "model-a-v3-selective-b2",
        "manifest": str(manifest),
        "mode": args.mode,
        "selection_uses_damage": False,
        "models": {key: value.as_dict() for key, value in models.items()},
        "reliability_calibrations": {
            key: {
                "center": value.center.tolist(),
                "scale": value.scale.tolist(),
                "threshold": value.threshold,
            }
            for key, value in reliability_calibrations.items()
        },
    }
    (args.output_dir / "model.json").write_text(json.dumps(artifact, indent=2))
    summary = pd.DataFrame(metrics)
    print(summary.to_string(index=False))
    print(f"MODEL_A_V3_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()
