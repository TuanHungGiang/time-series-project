from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multiview_v4_core import (
    MultiViewModel, conformal_threshold, extract_views, learn_anchors,
    metric_dict, normalized_score, sensor_spectra, signal_quality,
)
from run_z24_b2 import build_file_index, excitation, find_manifest, load_vertical, resolve_source


def stable_rng(*tokens: object) -> np.random.Generator:
    payload = "|".join(map(str, tokens)).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def corrupt(signals: np.ndarray, condition: str, record_id: str, replicate: int) -> np.ndarray:
    if condition == "clean":
        return signals.copy()
    rng = stable_rng("multiview-v4", record_id, condition, replicate)
    output = signals.copy()
    count, length = output.shape
    affected_count = max(1, int(round(0.30 * count)))
    affected = rng.choice(count, affected_count, replace=False)
    if condition == "dropout_0.30":
        keep = np.ones(count, dtype=bool)
        keep[affected] = False
        return output[keep]
    if condition == "missing_0.30":
        coordinates = np.arange(length)
        block = int(round(0.30 * length))
        for sensor in affected:
            start = int(rng.integers(0, length - block + 1))
            observed = np.ones(length, dtype=bool)
            observed[start : start + block] = False
            output[sensor, ~observed] = np.interp(coordinates[~observed], coordinates[observed], output[sensor, observed])
        return output
    if condition == "noise_5":
        for sensor in affected:
            centered = output[sensor] - output[sensor].mean()
            sigma = np.sqrt(np.mean(centered**2) / (10 ** 0.5))
            output[sensor] += rng.normal(0.0, sigma, length)
        return output
    raise ValueError(condition)


def load_protocol(root: Path) -> tuple[pd.DataFrame, Path]:
    manifest_path = find_manifest(root)
    frame = pd.read_csv(manifest_path)
    frame["label"] = frame.label.astype(int)
    if "role" not in frame:
        frame["role"] = frame["split"]
    frame["role"] = frame.role.astype(str).str.lower().replace({"val": "calibration"})
    frame.loc[frame.label == 1, "role"] = "test"
    frame = frame[frame.role.isin(["train", "calibration", "test"])].copy()
    if frame.loc[frame.role.isin(["train", "calibration"]), "label"].ne(0).any():
        raise RuntimeError("Leakage: damage in train/calibration")
    dataset_root = manifest_path.parent.parent
    index = build_file_index(dataset_root)
    frame["resolved_path"] = [str(resolve_source(row, index, dataset_root)) for _, row in frame.iterrows()]
    frame["excitation_resolved"] = [excitation(row) for _, row in frame.iterrows()]
    return frame, manifest_path


def prepare(signals: np.ndarray, anchors: list[float]) -> dict[str, np.ndarray]:
    frequency, psd = sensor_spectra(signals)
    return extract_views(frequency, psd, anchors)


def fit_group(frame: pd.DataFrame, signal_cache: dict[str, np.ndarray], alpha: float) -> dict:
    train = frame[frame.role == "train"]
    calibration = frame[frame.role == "calibration"]
    if len(train) < 8 or calibration.empty:
        raise RuntimeError(f"Insufficient healthy records: train={len(train)}, calibration={len(calibration)}")

    spectra = {}
    for row in pd.concat([train, calibration]).itertuples():
        frequency, psd = sensor_spectra(signal_cache[row.record_id])
        spectra[row.record_id] = (frequency, psd)

    train_global = [np.median(spectra[row.record_id][1], axis=0) for row in train.itertuples()]
    frequency = spectra[train.iloc[0].record_id][0]
    anchors = learn_anchors(train_global, frequency)
    final_views = [extract_views(*spectra[row.record_id], anchors) for row in train.itertuples()]
    model = MultiViewModel.fit(final_views)

    # Cross-fitted train scores enlarge the healthy calibration support without
    # ever scoring a record with a model trained on that record.
    healthy_scores = []
    healthy_domain = []
    train_rows = list(train.itertuples())
    for heldout_index, heldout in enumerate(train_rows):
        fit_rows = [row for index, row in enumerate(train_rows) if index != heldout_index]
        fit_global = [np.median(spectra[row.record_id][1], axis=0) for row in fit_rows]
        fold_anchors = learn_anchors(fit_global, frequency)
        fold_views = [extract_views(*spectra[row.record_id], fold_anchors) for row in fit_rows]
        fold_model = MultiViewModel.fit(fold_views)
        value = fold_model.components(extract_views(*spectra[heldout.record_id], fold_anchors))
        healthy_scores.append(value["score_raw"])
        healthy_domain.append(value["domain_error"])
    for row in calibration.itertuples():
        value = model.components(extract_views(*spectra[row.record_id], anchors))
        healthy_scores.append(value["score_raw"])
        healthy_domain.append(value["domain_error"])

    healthy_scores = np.asarray(healthy_scores)
    healthy_domain = np.asarray(healthy_domain)
    return {
        "anchors": anchors,
        "model": model,
        "score_calibration": healthy_scores,
        "score_threshold": conformal_threshold(healthy_scores, alpha),
        "domain_calibration": healthy_domain,
        "domain_threshold": conformal_threshold(healthy_domain, alpha),
        "calibration_records": len(healthy_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/z24_multiview_v4"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, manifest_path = load_protocol(args.input_root)
    signal_cache = {row.record_id: load_vertical(Path(row.resolved_path)) for row in frame.itertuples()}
    groups = {}
    for excitation_name in ("avt", "fvt"):
        subset = frame[(frame.excitation_resolved == excitation_name) & (frame.label == 0)]
        groups[excitation_name] = fit_group(subset, signal_cache, args.alpha)
        print(
            f"{excitation_name}: calibration={groups[excitation_name]['calibration_records']} "
            f"threshold={groups[excitation_name]['score_threshold']:.5f}", flush=True,
        )

    conditions = ["clean", "missing_0.30", "noise_5", "dropout_0.30"]
    replicates = 1 if args.mode == "quick" else 3
    rows = []
    test = frame[frame.role == "test"]
    for row in test.itertuples():
        group = groups[row.excitation_resolved]
        for condition in conditions:
            for replicate in range(replicates if condition != "clean" else 1):
                modified = corrupt(signal_cache[row.record_id], condition, row.record_id, replicate)
                views = prepare(modified, group["anchors"])
                components = group["model"].components(views)
                quality = signal_quality(modified)
                sensor_coverage = modified.shape[0] / signal_cache[row.record_id].shape[0]
                time_coverage = 0.91 if condition == "missing_0.30" else 1.0
                quality["observed_fraction"] = float(sensor_coverage * time_coverage)
                quality["integrity_error"] += float(-5.0 * np.log(max(quality["observed_fraction"], 1e-6)))
                score = normalized_score(components["score_raw"], group["score_threshold"], group["score_calibration"])
                domain_error = normalized_score(
                    components["domain_error"], group["domain_threshold"], group["domain_calibration"]
                )
                rows.append({
                    "dataset": "Z24", "record_id": row.record_id, "label": int(row.label),
                    "excitation": row.excitation_resolved, "condition": condition, "replicate": replicate,
                    "score": score, "prediction": score > 1.0, "domain_error": domain_error,
                    **quality, **components,
                })
    scores = pd.DataFrame(rows)
    metric_rows = []
    for (condition, replicate), subset in scores.groupby(["condition", "replicate"]):
        metric_rows.append({"condition": condition, "replicate": replicate, **metric_dict(subset.label, subset.score)})
    metrics = pd.DataFrame(metric_rows)
    subgroup_rows = []
    clean = scores[scores.condition == "clean"]
    for excitation_name, subset in clean.groupby("excitation"):
        subgroup_rows.append({"excitation": excitation_name, "records": len(subset), **metric_dict(subset.label, subset.score)})

    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    protocol = {
        "version": "multiview-v4", "dataset": "Z24", "alpha": args.alpha,
        "fit_labels": "healthy only", "damage_labels_used_for_selection": False,
        "views": ["global modal/band", "permutation-invariant inter-sensor disagreement"],
        "calibration": "excitation-conditioned cross-fitted healthy records plus held-out healthy calibration",
        "manifest": str(manifest_path),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(pd.DataFrame(subgroup_rows).to_string(index=False))
    print(f"Z24_MULTIVIEW_V4_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

