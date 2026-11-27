from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_z24_b2 import load_vertical, normalized_sensor_spectra
from run_z24_model_a_v3 import b2_score, fit_b2, prepare_frame
from run_z24_model_a_v31 import corrupt_v31, safe_metric_dict


GRID = np.linspace(0.5, 30.0, 96)
CONDITIONS = (
    "clean", "dropout_0.30", "missing_0.30", "noise_5",
    "targeted_0.30", "targeted_0.50",
)


def record_metadata(record_id):
    state = re.search(r"z24_s(\d+)", record_id, flags=re.IGNORECASE)
    setup = re.search(r"setup(\d+)", record_id, flags=re.IGNORECASE)
    return (int(state.group(1)) if state else -1, int(setup.group(1)) if setup else -1)


def shape_feature(frequencies, sensor_psd, availability):
    present = sensor_psd[np.asarray(availability, dtype=bool)]
    if present.shape[0] == 0:
        raise RuntimeError("No observed sensor remains")
    spectrum = np.median(present, axis=0)
    log_shape = np.interp(GRID, frequencies, np.log10(spectrum + 1e-18))
    log_shape -= np.mean(log_shape)
    return log_shape.astype(np.float64)


@dataclass
class ShapeModel:
    center: np.ndarray
    components: np.ndarray
    latent_center: np.ndarray
    latent_scale: np.ndarray
    residual_center: float
    residual_scale: float
    threshold: float = 0.0

    @classmethod
    def fit(cls, matrix):
        center = np.median(matrix, axis=0)
        centered = matrix - center
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2
        cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
        components_needed = int(np.searchsorted(cumulative, 0.95) + 1)
        count = max(2, min(6, matrix.shape[0] - 2, components_needed))
        components = vt[:count]
        latent = centered @ components.T
        latent_center = np.median(latent, axis=0)
        latent_scale = np.maximum(
            1.4826 * np.median(np.abs(latent - latent_center), axis=0), 1e-4
        )
        reconstructed = latent @ components
        residual = np.sqrt(np.mean((centered - reconstructed) ** 2, axis=1))
        residual_center = float(np.median(residual))
        residual_scale = float(
            max(1.4826 * np.median(np.abs(residual - residual_center)), 1e-4)
        )
        return cls(
            center, components, latent_center, latent_scale,
            residual_center, residual_scale,
        )

    def raw_score(self, feature):
        centered = feature - self.center
        latent = centered @ self.components.T
        latent_distance = np.sqrt(
            np.mean(((latent - self.latent_center) / self.latent_scale) ** 2)
        )
        residual_vector = centered - latent @ self.components
        residual = float(np.sqrt(np.mean(residual_vector**2)))
        residual_excess = max(
            0.0, (residual - self.residual_center) / self.residual_scale
        )
        return float(np.sqrt(latent_distance**2 + residual_excess**2))

    def score(self, feature):
        if self.threshold <= 0:
            raise RuntimeError("Shape model has not been calibrated")
        return self.raw_score(feature) / self.threshold

    def as_dict(self):
        return {
            "center": self.center.tolist(),
            "components": self.components.tolist(),
            "latent_center": self.latent_center.tolist(),
            "latent_scale": self.latent_scale.tolist(),
            "residual_center": self.residual_center,
            "residual_scale": self.residual_scale,
            "threshold": self.threshold,
        }


def load_shape(signals, availability=None):
    frequencies, psd = normalized_sensor_spectra(signals)
    if availability is None:
        availability = np.ones(psd.shape[0], dtype=bool)
    return shape_feature(frequencies, psd, availability)


def fit_shape(rows):
    train = rows[rows.role == "train"]
    calibration = rows[rows.role == "calibration"]
    matrix = np.stack([
        load_shape(load_vertical(Path(row.resolved_path)))
        for row in train.itertuples()
    ])
    model = ShapeModel.fit(matrix)
    calibration_scores = [
        model.raw_score(load_shape(load_vertical(Path(row.resolved_path))))
        for row in calibration.itertuples()
    ]
    model.threshold = float(max(calibration_scores))
    return model


def method_metrics(frame):
    rows = []
    columns = ["method", "condition", "replicate"]
    for key, group in frame.groupby(columns, sort=True):
        rows.append({
            **dict(zip(columns, key)),
            "scope": "overall", "excitation": "all", "records": len(group),
            **safe_metric_dict(group.label.to_numpy(), group.score.to_numpy()),
        })
        for excitation, subset in group.groupby("excitation"):
            rows.append({
                **dict(zip(columns, key)),
                "scope": "excitation", "excitation": excitation,
                "records": len(subset),
                **safe_metric_dict(subset.label.to_numpy(), subset.score.to_numpy()),
            })
    return pd.DataFrame(rows)


def damage_groups(frame):
    rows = []
    damage = frame[frame.label == 1]
    columns = ["method", "condition", "replicate", "state", "excitation"]
    for key, group in damage.groupby(columns, sort=True):
        prediction = group.score.to_numpy() > 1.0
        rows.append({
            **dict(zip(columns, key)),
            "records": len(group),
            "recall": float(np.mean(prediction)),
            "score_mean": float(group.score.mean()),
            "score_median": float(group.score.median()),
            "false_negatives": int(np.sum(~prediction)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/kaggle/working/z24_shape_fusion_b1"),
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, manifest = prepare_frame(args.input_root)
    b2_models, shape_models = {}, {}
    for excitation in ("avt", "fvt"):
        subset = frame[frame.excitation_resolved == excitation]
        b2_models[excitation] = fit_b2(subset)
        shape_models[excitation] = fit_shape(subset)
        print(
            f"{excitation}: b2_threshold={b2_models[excitation].threshold:.6f} "
            f"shape_threshold={shape_models[excitation].threshold:.6f} "
            f"shape_components={shape_models[excitation].components.shape[0]}",
            flush=True,
        )

    test = frame[frame.role == "test"].copy()
    output = []
    for index, row in enumerate(test.itertuples(), start=1):
        signals = load_vertical(Path(row.resolved_path))
        excitation = row.excitation_resolved
        b2_model = b2_models[excitation]
        shape_model = shape_models[excitation]
        state, setup = record_metadata(str(row.record_id))
        for condition in CONDITIONS:
            repeats = 1 if condition == "clean" or args.mode == "quick" else 3
            for replicate in range(repeats):
                changed, availability, _, signature = corrupt_v31(
                    signals, condition, str(row.record_id), replicate, b2_model
                )
                frequencies, psd = normalized_sensor_spectra(changed)
                modal_score = b2_score(b2_model, frequencies, psd, availability)
                shape_score = shape_model.score(
                    shape_feature(frequencies, psd, availability)
                )
                fusion_score = float(
                    np.sqrt(max(modal_score, 0) * max(shape_score, 0))
                )
                common = {
                    "record_id": row.record_id, "label": int(row.label),
                    "excitation": excitation, "state": state, "setup": setup,
                    "condition": condition, "replicate": replicate,
                    "corrupted_sensors": signature,
                }
                for method, score in (
                    ("modal_b2", modal_score),
                    ("shape_b1", shape_score),
                    ("equal_fusion", fusion_score),
                ):
                    output.append({**common, "method": method, "score": score})
        if index % 36 == 0:
            print(f"records={index}/{len(test)}", flush=True)

    scores = pd.DataFrame(output)
    metrics = method_metrics(scores)
    groups = damage_groups(scores)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    groups.to_csv(args.output_dir / "damage_group_metrics.csv", index=False)

    artifact = {
        "protocol": "z24-healthy-only-shape-fusion-b1",
        "manifest": str(manifest),
        "selection_uses_damage": False,
        "threshold_uses_damage": False,
        "fusion_weight_selected_with_damage": False,
        "methods_are_reported_without_posthoc_selection": True,
        "mode": args.mode,
        "b2_models": {key: value.as_dict() for key, value in b2_models.items()},
        "shape_models": {key: value.as_dict() for key, value in shape_models.items()},
    }
    (args.output_dir / "model.json").write_text(json.dumps(artifact, indent=2))

    clean = metrics[
        (metrics.condition == "clean") & (metrics.scope == "overall")
    ].sort_values("balanced_accuracy", ascending=False)
    excitation_clean = metrics[
        (metrics.condition == "clean") & (metrics.scope == "excitation")
    ].sort_values(["excitation", "recall"], ascending=[True, False])
    targeted = metrics[
        metrics.condition.isin(["targeted_0.30", "targeted_0.50"])
        & (metrics.scope == "overall")
    ].sort_values(["condition", "method", "replicate"])
    weakest = groups.sort_values(
        ["recall", "score_median"], ascending=[True, True]
    ).head(30)
    print("\n=== CLEAN OVERALL ===")
    print(clean.to_string(index=False))
    print("\n=== CLEAN BY EXCITATION ===")
    print(excitation_clean.to_string(index=False))
    print("\n=== TARGETED LOSS ===")
    print(targeted.to_string(index=False))
    print("\n=== WEAKEST DAMAGE GROUPS ===")
    print(weakest.to_string(index=False))
    print(f"SHAPE_FUSION_B1_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()
