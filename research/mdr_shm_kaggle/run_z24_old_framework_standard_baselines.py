from __future__ import annotations

"""Z24 detector baselines using the original B2 spectral representation.

Protocol:
* train healthy records fit preprocessing and each detector;
* complete validation records (healthy + damage) select one threshold per
  excitation using balanced accuracy;
* complete test records are evaluated once and never select a parameter.

The representation is identical across detectors: Welch PSD in 0.5--30 Hz,
per-sensor unit-area normalization, sensor-median fusion, and healthy modal
anchors learned independently for AVT and FVT.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import publication_metrics  # noqa: E402
from run_z24_b2 import (  # noqa: E402
    Model,
    features,
    learn_anchors,
    load_vertical,
    normalized_sensor_spectra,
)
from run_z24_detector_baselines import (  # noqa: E402
    CLASSICAL_MODELS,
    DEEP_MODELS,
    SpectralNormalizer,
    ae_sensor_scores,
    augmented_record_spectra,
    fit_classical,
    record_log_spectrum,
    set_seed,
    svdd_sensor_scores,
    train_autoencoder,
    train_deep_svdd,
)
from run_z24_source_baselines import load_manifest  # noqa: E402


MODELS = ("b2",) + CLASSICAL_MODELS + DEEP_MODELS


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("Every validation excitation must contain both classes")
    unique = np.unique(scores)
    epsilon = max(float(np.ptp(unique)), 1.0) * 1e-9
    candidates = np.concatenate([
        [unique[0] - epsilon],
        (unique[:-1] + unique[1:]) / 2.0,
        [unique[-1] + epsilon],
    ])
    best = None
    for threshold in candidates:
        metric = publication_metrics(labels, scores, float(threshold))
        key = (metric["ba"], metric["f1"], metric["specificity"])
        if best is None or key > best[0]:
            best = (key, float(threshold), metric)
    return best[1], best[2]


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    center = np.median(values)
    return float(max(1.4826 * np.median(np.abs(values - center)), np.std(values), 1e-6))


def build_cache(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for index, row in enumerate(frame.itertuples(), 1):
        output[str(row.record_id)] = normalized_sensor_spectra(
            load_vertical(Path(row.path))
        )
        if index % 30 == 0 or index == len(frame):
            print(f"spectra={index}/{len(frame)}", flush=True)
    return output


def fit_b2(train: pd.DataFrame, cache: dict) -> Model:
    spectra = [
        np.median(cache[str(record_id)][1], axis=0)
        for record_id in train.record_id
    ]
    frequencies = cache[str(train.iloc[0].record_id)][0]
    anchors = learn_anchors(spectra, frequencies)
    matrix = np.stack([
        features(spectrum, frequencies, anchors) for spectrum in spectra
    ])
    return Model.fit(anchors, matrix)


def feature_matrix(rows: pd.DataFrame, model: Model, cache: dict) -> np.ndarray:
    output = []
    for record_id in rows.record_id:
        frequencies, psd = cache[str(record_id)]
        output.append(features(np.median(psd, axis=0), frequencies, model.anchors))
    return np.stack(output)


def deep_record_scores(name: str, model, normalizer: SpectralNormalizer,
                       rows: pd.DataFrame, cache: dict, device) -> np.ndarray:
    matrix = np.stack([
        record_log_spectrum(cache[str(record_id)][1], np.ones(
            cache[str(record_id)][1].shape[0], dtype=bool
        ))
        for record_id in rows.record_id
    ])
    matrix = normalizer.transform(matrix)
    if name == "deep_svdd":
        return svdd_sensor_scores(model, matrix, device)
    return ae_sensor_scores(model, matrix, device)


def run_one(model_name: str, seed: int, frame: pd.DataFrame, cache: dict,
            epochs: int, batch_size: int, latent_dim: int, device):
    validation_parts, test_parts, threshold_rows = [], [], []
    for excitation in ("avt", "fvt"):
        group = frame[frame.excitation == excitation]
        train = group[(group.split == "train") & (group.label == 0)].copy()
        validation = group[group.split == "val"].copy()
        test = group[group.split == "test"].copy()
        b2 = fit_b2(train, cache)
        train_features = feature_matrix(train, b2, cache)
        validation_features = feature_matrix(validation, b2, cache)
        test_features = feature_matrix(test, b2, cache)

        if model_name == "b2":
            validation_raw = np.asarray([b2.raw_score(x) for x in validation_features])
            test_raw = np.asarray([b2.raw_score(x) for x in test_features])
        elif model_name in CLASSICAL_MODELS:
            fitted = fit_classical(model_name, train_features, seed)
            validation_raw = fitted.score(validation_features)
            test_raw = fitted.score(test_features)
        else:
            augmented = [
                augmented_record_spectra(
                    cache[str(record_id)][1], str(record_id), count=12
                )
                for record_id in train.record_id
            ]
            normalizer = SpectralNormalizer.fit(augmented)
            deep_train = normalizer.transform(np.concatenate(augmented, axis=0))
            set_seed(seed)
            if model_name == "deep_svdd":
                fitted = train_deep_svdd(
                    deep_train, device, seed, max(4, epochs // 3), epochs,
                    batch_size, latent_dim,
                )
            else:
                fitted = train_autoencoder(
                    deep_train, device, seed, epochs, batch_size, latent_dim,
                    model_name == "masked_cnn_ae",
                )
            validation_raw = deep_record_scores(
                model_name, fitted, normalizer, validation, cache, device
            )
            test_raw = deep_record_scores(
                model_name, fitted, normalizer, test, cache, device
            )

        threshold, local_metric = select_threshold(
            validation.label.to_numpy(), validation_raw
        )
        scale = robust_scale(validation_raw)
        validation_score = (validation_raw - threshold) / scale
        test_score = (test_raw - threshold) / scale
        threshold_rows.append({
            "model": model_name, "seed": seed, "excitation": excitation,
            "threshold_raw": threshold, "validation_scale": scale,
            "validation_ba_local": local_metric["ba"],
            "validation_auc_local": local_metric["auc"],
            "train_healthy_records": len(train),
            "validation_records": len(validation), "test_records": len(test),
        })
        for rows, raw, score, destination, split in (
            (validation, validation_raw, validation_score, validation_parts, "val"),
            (test, test_raw, test_score, test_parts, "test"),
        ):
            current = rows[[
                "record_id", "label", "state", "setup", "excitation"
            ]].copy()
            current["source_split"] = split
            current["raw_score"] = raw
            current["score"] = score
            current["threshold"] = 0.0
            current["model"] = model_name
            current["seed"] = seed
            destination.append(current)
    validation_scores = pd.concat(validation_parts, ignore_index=True)
    test_scores = pd.concat(test_parts, ignore_index=True)
    validation_metric = publication_metrics(
        validation_scores.label, validation_scores.score, 0.0
    )
    test_metric = publication_metrics(test_scores.label, test_scores.score, 0.0)
    metric = {
        "model": model_name,
        "seed": seed,
        **test_metric,
        **{f"validation_{key}": value for key, value in validation_metric.items()
           if key in {"auc", "ap", "ba", "recall", "fpr", "precision", "f1", "specificity"}},
        "train_healthy_records": int(((frame.split == "train") & (frame.label == 0)).sum()),
        "validation_healthy_records": int(((frame.split == "val") & (frame.label == 0)).sum()),
        "validation_damage_records": int(((frame.split == "val") & (frame.label == 1)).sum()),
        "test_healthy_records": int(((frame.split == "test") & (frame.label == 0)).sum()),
        "test_damage_records": int(((frame.split == "test") & (frame.label == 1)).sum()),
    }
    return metric, validation_scores, test_scores, pd.DataFrame(threshold_rows)


def refresh(output: Path) -> None:
    metric_paths = sorted((output / "runs").glob("*/metrics.json"))
    if not metric_paths:
        return
    metrics = pd.DataFrame([
        json.loads(path.read_text(encoding="utf-8")) for path in metric_paths
    ])
    metrics.to_csv(output / "metrics.csv", index=False)
    aggregate = metrics.groupby("model", as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        specificity_mean=("specificity", "mean"), runs=("seed", "size"),
    )
    aggregate.to_csv(output / "aggregate.csv", index=False)
    test_scores = pd.concat([
        pd.read_csv(path.parent / "test_scores.csv") for path in metric_paths
    ], ignore_index=True)
    test_scores.to_csv(output / "record_scores.csv", index=False)
    axis = aggregate.set_index("model")[[
        "auc_mean", "ba_mean", "recall_mean", "specificity_mean"
    ]].plot.bar(figsize=(12, 6), rot=25, ylim=(0, 1.02), grid=True)
    axis.set_title("Z24 old-framework baselines: locked test")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "baseline_comparison.png", dpi=220)
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261113])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_manifest(args.input_root)
    expected = {
        "train_healthy": 24,
        "val_healthy": 6, "val_damage": 126,
        "test_healthy": 6, "test_damage": 126,
    }
    observed = {
        "train_healthy": int(((frame.split == "train") & (frame.label == 0)).sum()),
        "val_healthy": int(((frame.split == "val") & (frame.label == 0)).sum()),
        "val_damage": int(((frame.split == "val") & (frame.label == 1)).sum()),
        "test_healthy": int(((frame.split == "test") & (frame.label == 0)).sum()),
        "test_damage": int(((frame.split == "test") & (frame.label == 1)).sum()),
    }
    if observed != expected:
        raise RuntimeError(f"Unexpected three-way split: {observed}")
    cache = build_cache(frame)
    protocol = {
        "dataset": "Z24", "protocol": "standard_train_val_test",
        "counts": observed, "fit": "train healthy only",
        "model_selection": "labeled validation; BA primary, F1 then specificity ties",
        "final_evaluation": "locked test only",
        "conditioning": "independent AVT/FVT models and validation thresholds",
        "representation": (
            "linear detrend; Welch nperseg=4096/noverlap=2048; 0.5-30 Hz; "
            "unit-area PSD per sensor; median sensor fusion; healthy modal anchors"
        ),
        "models": args.models, "seeds": args.seeds, "epochs": args.epochs,
        "damage_used_for_detector_fit": False,
        "damage_labels_used_for_validation_selection": True,
        "test_used_for_selection": False,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    for model_name in args.models:
        model_seeds = args.seeds if model_name in DEEP_MODELS else args.seeds[:1]
        for seed in model_seeds:
            run_dir = args.output_dir / "runs" / f"{model_name}__{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "metrics.json").exists():
                print(f"SKIP model={model_name} seed={seed}", flush=True)
                continue
            print(f"\nmodel={model_name} seed={seed}", flush=True)
            started = time.time()
            metric, validation, test, thresholds = run_one(
                model_name, seed, frame, cache, args.epochs, args.batch_size,
                args.latent_dim, device,
            )
            metric["seconds"] = time.time() - started
            validation.to_csv(run_dir / "validation_scores.csv", index=False)
            test.to_csv(run_dir / "test_scores.csv", index=False)
            thresholds.to_csv(run_dir / "thresholds.csv", index=False)
            (run_dir / "metrics.json").write_text(
                json.dumps(metric, indent=2), encoding="utf-8"
            )
            print(json.dumps(metric, indent=2), flush=True)
            refresh(args.output_dir)
    refresh(args.output_dir)
    print(pd.read_csv(args.output_dir / "aggregate.csv").to_string(index=False))
    print(f"Z24_OLD_FRAMEWORK_STANDARD_BASELINES_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

