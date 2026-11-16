from __future__ import annotations

"""One-seed Z24 observed spectral-relation graph diagnostic.

Each temporal window is represented by pairwise cross-spectral coherence,
phase and log-transmissibility features.  A missing sensor removes its incident
edges; no value is imputed.  Robust healthy-only edge-feature references turn
the observed subgraph into mean and top-k anomaly scores.  This deliberately
tests the representation before introducing a learned graph encoder.
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax, Normalizer, publication_metrics, seed_everything,
)
from run_z24_mir_usad_v0 import scenarios, sensor_corrupt  # noqa: E402
from run_z24_recovery_kd_v3 import (  # noqa: E402
    count_parameters, fit_compact_usad, score_compact,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)

METHODS = ("compact_raw", "spectral_edge_mean", "spectral_edge_topk")


def edge_index(channels: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(channels, k=1)


def spectral_relation_features(
    values: np.ndarray,
    segment_length: int,
    segment_stride: int,
    bands: int,
) -> np.ndarray:
    """Return [windows, undirected_edges, 4 * bands] relational features."""
    values = np.asarray(values, dtype=np.float32)
    windows, channels, length = values.shape
    if length < segment_length:
        raise RuntimeError(f"Window length {length} < spectral segment {segment_length}")
    starts = list(range(0, length - segment_length + 1, segment_stride))
    if len(starts) < 2:
        raise RuntimeError("At least two spectral segments are required")
    taper = np.hanning(segment_length).astype(np.float32)
    segments = np.stack(
        [values[:, :, start:start + segment_length] for start in starts], axis=2,
    )
    segments = segments - segments.mean(axis=-1, keepdims=True)
    spectra = np.fft.rfft(segments * taper[None, None, None, :], axis=-1)
    # DC carries offsets rather than structural dynamics.
    frequency_bins = np.arange(1, spectra.shape[-1])
    band_bins = [part for part in np.array_split(frequency_bins, bands) if len(part)]
    left, right = edge_index(channels)
    output = []
    eps = 1e-10
    for selected in band_bins:
        current = spectra[..., selected]
        # Average over overlapping segments and all bins in the band.
        cross = np.einsum(
            "ncsk,ndsk->ncd", current, current.conj(), optimize=True,
        ) / float(current.shape[2] * current.shape[3])
        power = np.maximum(np.real(np.diagonal(cross, axis1=1, axis2=2)), eps)
        pair_cross = cross[:, left, right]
        denominator = power[:, left] * power[:, right]
        coherence = np.clip(np.abs(pair_cross) ** 2 / denominator, 0.0, 1.0)
        magnitude = np.maximum(np.abs(pair_cross), eps)
        phase_cos = np.real(pair_cross) / magnitude
        phase_sin = np.imag(pair_cross) / magnitude
        log_transmissibility = 0.5 * (
            np.log(power[:, left]) - np.log(power[:, right])
        )
        output.extend([coherence, phase_cos, phase_sin, log_transmissibility])
    return np.stack(output, axis=-1).astype(np.float32)


def fit_edge_reference(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(features, axis=0)
    mad = 1.4826 * np.median(np.abs(features - center[None, :, :]), axis=0)
    standard = np.std(features, axis=0)
    global_scale = max(float(np.median(mad)), 0.25 * float(np.std(features)), 1e-5)
    raw_scale = np.maximum.reduce([
        mad, 0.25 * standard, np.full_like(mad, 1e-5),
    ])
    # Partial pooling stabilizes references fitted from only 46-93 windows.
    scale = 0.75 * raw_scale + 0.25 * global_scale
    return center.astype(np.float32), scale.astype(np.float32)


def observed_edge_mask(available: np.ndarray) -> np.ndarray:
    left, right = edge_index(available.shape[1])
    return (available[:, left] > 0.5) & (available[:, right] > 0.5)


def relation_window_score(
    features: np.ndarray,
    available: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray],
    mode: str,
    top_fraction: float,
) -> np.ndarray:
    deviation = np.abs((features - reference[0][None]) / reference[1][None])
    edge_available = observed_edge_mask(available)
    rows = []
    for index in range(len(features)):
        observed = deviation[index, edge_available[index], :].reshape(-1)
        if len(observed) == 0:
            raise RuntimeError("A window has fewer than two observed sensors")
        if mode == "mean":
            rows.append(float(np.mean(observed)))
        elif mode == "topk":
            count = max(1, int(math.ceil(top_fraction * len(observed))))
            rows.append(float(np.mean(np.partition(observed, -count)[-count:])))
        else:
            raise ValueError(mode)
    return np.asarray(rows, dtype=float)


def all_available(windows: int, channels: int) -> np.ndarray:
    return np.ones((windows, channels), dtype=np.float32)


def build_values(records, cache, condition, kind, rate, replicate, importance):
    arrays, masks, counts, metadata = [], [], [], []
    for row in records.itertuples():
        values, available = sensor_corrupt(
            cache[str(row.record_id)], kind, rate, str(row.record_id),
            replicate, importance,
        )
        arrays.append(values)
        masks.append(available)
        counts.append(len(values))
        metadata.append({
            "record_id": row.record_id, "split": row.split,
            "label": int(row.label), "state": row.state,
            "setup": row.setup, "excitation": row.excitation,
            "condition": condition, "replicate": replicate, "rate": rate,
        })
    return np.concatenate(arrays), np.concatenate(masks), counts, metadata


def make_frame(method, upper, two_sided, counts, metadata, available, parameters):
    frame = pd.DataFrame(metadata)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two_sided
    frame["windows"] = counts
    frame["parameters"] = parameters
    coverage_by_window = available.mean(1)
    start, coverage = 0, []
    for count in counts:
        coverage.append(float(coverage_by_window[start:start + count].mean()))
        start += count
    frame["coverage"] = coverage
    return frame


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def extract(args, values):
    return spectral_relation_features(
        values, args.segment_length, args.segment_stride, args.bands,
    )


def run_group(group, labels, seed, group_index, args, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_by_record = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_by_record)
    normalizer = Normalizer.fit(fit_raw)
    fit_normalized = normalizer.transform(fit_raw).astype(np.float32)
    internal_normalized = normalizer.transform(internal_raw).astype(np.float32)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scaler = FitMinMax(fit_normalized)
    fit_scaled = scaler.transform(fit_normalized).astype(np.float32)
    internal_scaled = scaler.transform(internal_normalized).astype(np.float32)
    print(
        f"  fit={tuple(fit_scaled.shape)} internal={tuple(internal_scaled.shape)} "
        f"edges={len(edge_index(len(labels))[0])}", flush=True,
    )

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)
    # Whole-sensor loss does not alter any edge between sensors that remain
    # observed.  Extract each clean record once and only change the edge mask
    # across stress conditions; this avoids eight redundant FFT passes.
    relation_cache = {
        record_id: extract(args, values) for record_id, values in cache.items()
    }

    seed_everything(seed + group_index * 1009)
    compact = fit_compact_usad(
        fit_scaled, args.compact_epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    compact_internal = score_compact(
        compact, internal_scaled, args.batch_size, device,
    )
    compact_reference = robust_score_reference(compact_internal)

    fit_features = extract(args, fit_normalized)
    internal_features = extract(args, internal_normalized)
    edge_reference = fit_edge_reference(fit_features)
    internal_available = all_available(len(internal_features), len(labels))
    relation_references = {}
    for method, mode in (
        ("spectral_edge_mean", "mean"),
        ("spectral_edge_topk", "topk"),
    ):
        score = relation_window_score(
            internal_features, internal_available, edge_reference,
            mode, args.top_fraction,
        )
        relation_references[method] = robust_score_reference(score)

    diagnostic = {
        "setup": int(group.setup.iloc[0]),
        "excitation": str(group.excitation.iloc[0]),
        "sensors": len(labels),
        "edges": int(len(edge_index(len(labels))[0])),
        "feature_dimensions_per_edge": int(fit_features.shape[2]),
        "fit_windows": int(len(fit_features)),
        "internal_windows": int(len(internal_features)),
    }
    output.with_name(output.stem + "_spectral.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8",
    )
    parameters = {
        "compact_raw": count_parameters(compact),
        "spectral_edge_mean": 0,
        "spectral_edge_topk": 0,
    }
    print(f"  spectral={diagnostic} parameters={parameters}", flush=True)

    cohorts = [(validation, "clean", "clean", 0.0, 0)]
    cohorts.extend([
        (test, condition, kind, rate, replicate)
        for condition, kind, rate, replicate in scenarios()
    ])
    frames = []
    for records, condition, kind, rate, replicate in cohorts:
        values, available, counts, metadata = build_values(
            records, cache, condition, kind, rate, replicate, importance,
        )
        compact_score = score_compact(
            compact, scaler.transform(values).astype(np.float32),
            args.batch_size, device,
        )
        compact_upper, compact_two = aggregate_record_scores(
            compact_score, counts, compact_reference,
        )
        frames.append(make_frame(
            "compact_raw", compact_upper, compact_two, counts, metadata,
            available, parameters["compact_raw"],
        ))

        relation_features = np.concatenate([
            relation_cache[str(row.record_id)] for row in records.itertuples()
        ])
        for method, mode in (
            ("spectral_edge_mean", "mean"),
            ("spectral_edge_topk", "topk"),
        ):
            window_score = relation_window_score(
                relation_features, available, edge_reference,
                mode, args.top_fraction,
            )
            upper, two_sided = aggregate_record_scores(
                window_score, counts, relation_references[method],
            )
            frames.append(make_frame(
                method, upper, two_sided, counts, metadata,
                available, parameters[method],
            ))
    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del compact
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finish(output, frame):
    metric_rows, scored = [], []
    for method in METHODS:
        validation = frame[(frame.method == method) & (frame.split == "val")].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                validation.label, validation[f"score_{tail}"],
            )
            candidates.append(((metric["ba"], metric["auc"], metric["f1"]), tail, threshold))
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        test["score"] = test[f"score_{tail}"]
        test["threshold"] = threshold
        test["score_tail"] = tail
        scored.append(test)
        rows, clean = [], None
        for (condition, replicate), current in test.groupby(["condition", "replicate"]):
            row = {
                "method": method, "condition": condition,
                "replicate": int(replicate), "score_tail": tail,
                "threshold": threshold,
                **publication_metrics(current.label, current.score, threshold),
                "coverage_mean": float(current.coverage.mean()),
                "parameters": int(current.parameters.iloc[0]),
            }
            rows.append(row)
            if condition == "clean":
                clean = row
        for row in rows:
            row["delta_auc"] = row["auc"] - clean["auc"]
            row["delta_ba"] = row["ba"] - clean["ba"]
            row["delta_fpr"] = row["fpr"] - clean["fpr"]
        metric_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(scored, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"), parameters=("parameters", "mean"),
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(12, 5.5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 observed spectral-relation graph diagnostic")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "spectral_graph_auc.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== ONE-SEED SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary


def decision(summary):
    table = summary.set_index(["method", "condition"])
    corruptions = ["sensor_0.30", "sensor_0.50", "targeted_0.50"]
    comparisons = {}
    candidates = []
    for method in ("spectral_edge_mean", "spectral_edge_topk"):
        effects = {}
        for condition in ["clean", *corruptions]:
            proposed = table.loc[(method, condition)]
            baseline = table.loc[("compact_raw", condition)]
            effects[condition] = {
                "delta_auc": float(proposed.auc_mean - baseline.auc_mean),
                "delta_ba": float(proposed.ba_mean - baseline.ba_mean),
                "delta_fpr": float(proposed.fpr_mean - baseline.fpr_mean),
                "delta_recall": float(proposed.recall_mean - baseline.recall_mean),
            }
        comparisons[method] = effects
        average_auc = float(np.mean([effects[c]["delta_auc"] for c in corruptions]))
        average_ba = float(np.mean([effects[c]["delta_ba"] for c in corruptions]))
        promote = bool(
            average_auc > 0.0 and average_ba > 0.0
            and effects["targeted_0.50"]["delta_auc"] >= 0.0
            and effects["clean"]["delta_auc"] >= -0.02
        )
        candidates.append((promote, average_auc + average_ba, method))
    promote, _, best = max(candidates)
    return {
        "effects_vs_compact_raw": comparisons,
        "best_spectral_relation": best,
        "promote_to_graph_encoder": bool(promote),
        "decision": (
            f"PROMOTE {best}; relational features justify a learned graph encoder."
            if promote else
            "DO NOT BUILD a graph encoder yet; relational features did not beat the compact baseline."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--compact-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--segment-stride", type=int, default=64)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={METHODS} seed={args.seed} "
        f"segment={args.segment_length}/{args.segment_stride} bands={args.bands}",
        flush=True,
    )
    started = time.time()
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, *key)
        if path.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(group, labels, args.seed, group_index, args, device, path)

    paths = [group_file(args.output_dir, *key) for key in sorted(sensor_map)]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(METHODS) * (132 + 132 * len(scenarios()))
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/record/condition keys")
    summary = finish(args.output_dir, frame)
    screening = decision(summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 observed spectral-relation graph diagnostic v8",
        "seed": args.seed, "fit": "healthy train only",
        "selection": "clean labeled validation only",
        "test_used_for_fit_or_selection": False,
        "missing_sensor_handling": "remove incident edges; no imputation",
        "edge_features": ["coherence", "phase_cos", "phase_sin", "log_transmissibility"],
        "spectral_config": {
            "segment_length": args.segment_length,
            "segment_stride": args.segment_stride,
            "bands": args.bands,
            "top_fraction": args.top_fraction,
        },
        "methods": list(METHODS), "conditions": scenarios(),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(screening, indent=2), flush=True)
    print(f"SPECTRAL_GRAPH_V8_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

