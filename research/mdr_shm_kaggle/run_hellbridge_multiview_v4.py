from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from multiview_v4_core import (
    MultiViewModel, conformal_threshold, extract_views, learn_anchors,
    metric_dict, normalized_score, sensor_spectra, signal_quality,
)


NAME_PATTERN = re.compile(r"MVS_(P[12])_(UDS|DS\d+)_(SM|NM)_([YZ])_(\d+)")


def parse_name(name: str) -> dict[str, object]:
    match = NAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected HellBridge recording: {name}")
    position, state, excitation, axis, repeat = match.groups()
    return {
        "record_id": name, "position": position, "state": state,
        "excitation": excitation, "axis": axis, "repeat": int(repeat),
        "regime": f"{position}_{excitation}_{axis}", "label": int(state != "UDS"),
    }


def read_signals(group, axis: str) -> np.ndarray:
    axis_name = axis.lower()
    sensor_names = [f"AG{index:02d}" for index in range(1, 19)]
    available = [name for name in sensor_names if name in group["acceleration"] and axis_name in group["acceleration"][name]]
    if len(available) < 12:
        raise ValueError(f"Only {len(available)} sensors provide axis {axis}")
    return np.stack([np.asarray(group["acceleration"][name][axis_name], dtype=np.float32) for name in available])


def window_psds(signals: np.ndarray, seconds: int) -> tuple[np.ndarray, list[np.ndarray]]:
    width = 100 * seconds
    result = []
    frequency = None
    for start in range(0, signals.shape[1] - width + 1, width):
        frequency, psd = sensor_spectra(signals[:, start : start + width])
        result.append(psd)
    if len(result) < 8:
        raise ValueError(f"Need >=8 windows, found {len(result)}")
    assert frequency is not None
    return frequency, result


def stable_rng(*tokens: object) -> np.random.Generator:
    payload = "|".join(map(str, tokens)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))


def corrupt(signals: np.ndarray, condition: str, record_id: str, replicate: int) -> np.ndarray:
    if condition == "clean":
        return signals.copy()
    rng = stable_rng("hell-multiview-v4", record_id, condition, replicate)
    output = signals.copy()
    count, length = output.shape
    affected = rng.choice(count, max(1, int(round(0.30 * count))), replace=False)
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


def fit_fold(frequency: np.ndarray, psds: list[np.ndarray], alpha: float, seed: int) -> dict:
    split = max(5, int(np.floor(0.65 * len(psds))))
    fit_psds, calibration_psds = psds[:split], psds[split:]
    if len(calibration_psds) < 3:
        calibration_psds = psds[-3:]
        fit_psds = psds[:-3]
    anchors = learn_anchors([np.median(psd, axis=0) for psd in fit_psds], frequency)
    fit_views = [extract_views(frequency, psd, anchors) for psd in fit_psds]
    model = MultiViewModel.fit(fit_views)
    calibration_components = [model.components(extract_views(frequency, psd, anchors)) for psd in calibration_psds]
    calibration_scores = np.asarray([item["score_raw"] for item in calibration_components])
    calibration_domain = np.asarray([item["domain_error"] for item in calibration_components])

    # Calibrate the same median statistic used at inference. Moving-block data are
    # unavailable with one UDS record, so this explicitly labelled bootstrap is
    # a sensitivity analysis rather than an independent sample-size claim.
    rng = np.random.default_rng(seed)
    pseudo_score = np.median(rng.choice(calibration_scores, (1000, len(psds)), replace=True), axis=1)
    pseudo_domain = np.median(rng.choice(calibration_domain, (1000, len(psds)), replace=True), axis=1)
    return {
        "anchors": anchors, "model": model,
        "score_calibration": pseudo_score,
        "score_threshold": conformal_threshold(pseudo_score, alpha),
        "domain_calibration": pseudo_domain,
        "domain_threshold": conformal_threshold(pseudo_domain, alpha),
        "fit_windows": len(fit_psds), "calibration_windows": len(calibration_psds),
    }


def score_record(fold: dict, frequency: np.ndarray, psds: list[np.ndarray]) -> dict[str, float]:
    components = [fold["model"].components(extract_views(frequency, psd, fold["anchors"])) for psd in psds]
    raw = float(np.median([item["score_raw"] for item in components]))
    domain = float(np.median([item["domain_error"] for item in components]))
    return {
        "score": normalized_score(raw, fold["score_threshold"], fold["score_calibration"]),
        "score_raw": raw,
        "domain_error": normalized_score(domain, fold["domain_threshold"], fold["domain_calibration"]),
        "global_z": float(np.median([item["global_z"] for item in components])),
        "relational_z": float(np.median([item["relational_z"] for item in components])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hellbridge_multiview_v4"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--stress-test", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    uds_cache = {}
    with h5py.File(args.h5, "r") as source:
        for name in sorted(source.keys()):
            item = parse_name(name)
            if args.mode == "quick" and item["state"] not in ("UDS", "DS1", "DS4"):
                continue
            metadata[name] = item
            if item["state"] == "UDS":
                signals = read_signals(source[name], str(item["axis"]))
                uds_cache[name] = window_psds(signals, args.window_seconds)

    folds: dict[str, list[dict]] = {}
    fold_rows = []
    for regime in sorted({str(item["regime"]) for item in metadata.values()}):
        healthy = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] == "UDS")
        if len(healthy) != 2:
            raise ValueError(f"Expected two UDS records in {regime}, found {healthy}")
        folds[regime] = []
        for fold_index, train_name in enumerate(healthy):
            frequency, psds = uds_cache[train_name]
            fold = fit_fold(frequency, psds, args.alpha, 20261112 + 31 * fold_index + len(fold_rows))
            fold["train_name"] = train_name
            fold["heldout_name"] = healthy[1 - fold_index]
            folds[regime].append(fold)
            fold_rows.append({
                "regime": regime, "fold": fold_index, "train_uds": train_name,
                "heldout_uds": healthy[1 - fold_index], "fit_windows": fold["fit_windows"],
                "calibration_windows": fold["calibration_windows"],
            })

    conditions = ["clean"]
    if args.stress_test:
        conditions += ["missing_0.30", "noise_5", "dropout_0.30"]
    replicates = 1 if args.mode == "quick" else 3
    rows = []
    with h5py.File(args.h5, "r") as source:
        for index, (name, item) in enumerate(sorted(metadata.items())):
            signals = read_signals(source[name], str(item["axis"]))
            relevant_folds = folds[str(item["regime"])]
            if item["state"] == "UDS":
                relevant_folds = [fold for fold in relevant_folds if fold["heldout_name"] == name]
            for condition in conditions:
                for replicate in range(replicates if condition != "clean" else 1):
                    modified = corrupt(signals, condition, name, replicate)
                    frequency, psds = window_psds(modified, args.window_seconds)
                    values = [score_record(fold, frequency, psds) for fold in relevant_folds]
                    quality = signal_quality(modified)
                    rows.append({
                        **item, "dataset": "HellBridge", "condition": condition, "replicate": replicate,
                        "fold_scores": len(values),
                        "score": float(np.mean([value["score"] for value in values])),
                        "score_raw": float(np.mean([value["score_raw"] for value in values])),
                        "domain_error": float(np.mean([value["domain_error"] for value in values])),
                        "global_z": float(np.mean([value["global_z"] for value in values])),
                        "relational_z": float(np.mean([value["relational_z"] for value in values])),
                        **quality,
                    })
            print(f"scored {index + 1}/{len(metadata)} {name}", flush=True)

    scores = pd.DataFrame(rows)
    scores["prediction"] = scores.score > 1.0
    metric_rows = []
    for (condition, replicate), subset in scores.groupby(["condition", "replicate"]):
        metric_rows.append({"condition": condition, "replicate": replicate, **metric_dict(subset.label, subset.score)})
    metrics = pd.DataFrame(metric_rows)
    subgroup_rows = []
    clean = scores[scores.condition == "clean"]
    for regime, subset in clean.groupby("regime"):
        subgroup_rows.append({"group_type": "regime", "group": regime, "records": len(subset), **metric_dict(subset.label, subset.score)})
    for state in sorted(set(clean.state) - {"UDS"}):
        subset = clean[clean.state.isin(["UDS", state])]
        subgroup_rows.append({"group_type": "state", "group": state, "records": len(subset), **metric_dict(subset.label, subset.score)})

    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    protocol = {
        "version": "multiview-v4", "dataset": "HellBridge", "alpha": args.alpha,
        "fit_labels": "UDS only", "damage_labels_used_for_selection": False,
        "protocol": "regime-matched two-fold leave-one-UDS-record-out",
        "record_calibration": "nested pseudo-record median bootstrap from calibration windows",
        "warning": "pseudo-record bootstrap does not increase the number of independent healthy bridge records",
        "stress_test": args.stress_test,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(pd.DataFrame(subgroup_rows).to_string(index=False))
    print(f"HELLBRIDGE_MULTIVIEW_V4_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

