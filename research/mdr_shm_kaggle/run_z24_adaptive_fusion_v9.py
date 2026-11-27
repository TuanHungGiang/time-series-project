from __future__ import annotations

"""Validation-locked adaptive fusion for Z24 missing-sensor detection.

The three branches are the global Compact USAD, observed-sensor mean evidence,
and observed-sensor top-k evidence from V7.  Fusion weights are deterministic
functions of mask coverage and healthy-train sensor criticality.  Only two
coefficients are selected on clean/corrupted validation records; all test
scores remain unseen until the policy and branch thresholds are locked.
"""

import argparse
import gc
import itertools
import json
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
from run_z24_sensor_pool_v7 import (  # noqa: E402
    aggregate_windows, fit_sensor_reference, fit_sensorwise,
    pool_window_scores, sensor_errors, standardized_sensor_scores,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)

BRANCHES = ("compact_raw", "sensorwise_mean", "sensorwise_topk")
METHODS = (*BRANCHES, "adaptive_fusion")


def build_values(records, cache, condition, kind, rate, replicate, importance):
    arrays, masks, counts, metadata = [], [], [], []
    weight = importance / max(float(importance.sum()), 1e-12)
    for row in records.itertuples():
        values, available = sensor_corrupt(
            cache[str(row.record_id)], kind, rate, str(row.record_id),
            replicate, importance,
        )
        arrays.append(values)
        masks.append(available)
        counts.append(len(values))
        missing_fraction = 1.0 - available.mean(1)
        criticality = ((1.0 - available) * weight[None, :]).sum(1)
        metadata.append({
            "record_id": row.record_id, "split": row.split,
            "label": int(row.label), "state": row.state,
            "setup": row.setup, "excitation": row.excitation,
            "condition": condition, "replicate": replicate, "rate": rate,
            "missing_fraction": float(missing_fraction.mean()),
            "missing_criticality": float(criticality.mean()),
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
    window_coverage = available.mean(1)
    start, coverage = 0, []
    for count in counts:
        coverage.append(float(window_coverage[start:start + count].mean()))
        start += count
    frame["coverage"] = coverage
    return frame


def group_file(output, setup, excitation):
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


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
    fit_normalized = normalizer.transform(fit_raw)
    internal_normalized = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scaler = FitMinMax(fit_normalized)
    fit = scaler.transform(fit_normalized).astype(np.float32)
    internal = scaler.transform(internal_normalized).astype(np.float32)
    print(f"  fit={tuple(fit.shape)} internal={tuple(internal.shape)}", flush=True)

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    base_seed = seed + group_index * 1009
    seed_everything(base_seed)
    compact = fit_compact_usad(
        fit, args.compact_epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    compact_internal = score_compact(compact, internal, args.batch_size, device)
    compact_reference = robust_score_reference(compact_internal)

    seed_everything(base_seed + 97)
    sensorwise, selected_epoch, internal_loss = fit_sensorwise(
        fit, internal, args.sensor_epochs, args.batch_size,
        args.sensor_latent, args.patience, device,
    )
    internal_errors = sensor_errors(sensorwise, internal, args.batch_size, device)
    sensor_reference = fit_sensor_reference(internal_errors)
    parameters = {
        "compact_raw": count_parameters(compact),
        "sensorwise_mean": count_parameters(sensorwise),
        "sensorwise_topk": count_parameters(sensorwise),
    }
    print(
        f"  sensor_epoch={selected_epoch} internal_loss={internal_loss:.6f} "
        f"parameters={parameters}", flush=True,
    )

    cohorts = []
    for split_records in (validation, test):
        cohorts.extend([
            (split_records, condition, kind, rate, replicate)
            for condition, kind, rate, replicate in scenarios()
        ])
    frames = []
    for records, condition, kind, rate, replicate in cohorts:
        values, available, counts, metadata = build_values(
            records, cache, condition, kind, rate, replicate, importance,
        )
        scaled = scaler.transform(values).astype(np.float32)
        compact_score = score_compact(compact, scaled, args.batch_size, device)
        compact_upper, compact_two = aggregate_record_scores(
            compact_score, counts, compact_reference,
        )
        frames.append(make_frame(
            "compact_raw", compact_upper, compact_two, counts, metadata,
            available, parameters["compact_raw"],
        ))

        errors = sensor_errors(sensorwise, scaled, args.batch_size, device)
        sensor_z = standardized_sensor_scores(errors, sensor_reference)
        for method, mode in (("sensorwise_mean", "mean"), ("sensorwise_topk", "topk")):
            upper_windows, two_windows = pool_window_scores(
                sensor_z, available, mode, args.top_fraction,
            )
            upper, two_sided = aggregate_windows(upper_windows, two_windows, counts)
            frames.append(make_frame(
                method, upper, two_sided, counts, metadata,
                available, parameters[method],
            ))

    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    result["top_fraction"] = args.top_fraction
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del compact, sensorwise
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def branch_calibration(frame):
    decisions = {}
    transformed = []
    for method in BRANCHES:
        current = frame[frame.method == method].copy()
        clean_validation = current[
            (current.split == "val") & (current.condition == "clean")
        ]
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                clean_validation.label, clean_validation[f"score_{tail}"],
            )
            candidates.append(((metric["ba"], metric["auc"], metric["f1"]), tail, threshold))
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        validation_scores = current.loc[current.split == "val", f"score_{tail}"].to_numpy()
        center = float(np.median(validation_scores))
        scale = max(
            1.4826 * float(np.median(np.abs(validation_scores - center))),
            0.25 * float(np.std(validation_scores)), 1e-6,
        )
        # A zero margin is exactly the branch's locked clean-validation boundary.
        current["branch_score"] = current[f"score_{tail}"]
        current["margin"] = (current.branch_score - threshold) / scale
        current["score_tail"] = tail
        current["threshold"] = threshold
        decisions[method] = {
            "tail": tail, "threshold": threshold, "scale": scale,
        }
        transformed.append(current)
    return pd.concat(transformed, ignore_index=True), decisions


def aligned_wide(frame):
    keys = [
        "record_id", "split", "label", "state", "setup", "excitation",
        "condition", "replicate", "rate", "missing_fraction",
        "missing_criticality", "coverage", "windows", "seed",
    ]
    wide = None
    for method in BRANCHES:
        current = frame[frame.method == method][keys + ["margin"]].copy()
        current = current.rename(columns={"margin": f"margin_{method}"})
        wide = current if wide is None else wide.merge(current, on=keys, validate="one_to_one")
    return wide


def fusion_score(frame, alpha, beta):
    missing = frame.missing_fraction.to_numpy(float)
    criticality = frame.missing_criticality.to_numpy(float)
    level = np.clip(missing / 0.50, 0.0, 1.0)
    relative = criticality / np.maximum(missing, 1e-8)
    # Random loss has relative criticality near one; preferential loss is >1.
    targetedness = np.clip(relative - 1.0, 0.0, 1.0)
    mean_weight = alpha * level * (1.0 - targetedness)
    top_weight = beta * level * targetedness
    compact_weight = 1.0 - mean_weight - top_weight
    return (
        compact_weight * frame.margin_compact_raw.to_numpy(float)
        + mean_weight * frame.margin_sensorwise_mean.to_numpy(float)
        + top_weight * frame.margin_sensorwise_topk.to_numpy(float)
    ), compact_weight, mean_weight, top_weight


def condition_metrics(frame, score, threshold=0.0):
    current = frame.copy()
    current["candidate_score"] = score
    rows = []
    for (condition, replicate), cohort in current.groupby(["condition", "replicate"]):
        rows.append({
            "condition": condition, "replicate": int(replicate),
            **publication_metrics(cohort.label, cohort.candidate_score, threshold),
        })
    return pd.DataFrame(rows)


def select_policy(validation, grid):
    compact_metrics = condition_metrics(
        validation, validation.margin_compact_raw.to_numpy(), 0.0,
    )
    compact_corrupted = compact_metrics[compact_metrics.condition != "clean"]
    compact_recall = float(compact_corrupted.recall.mean())
    compact_targeted = float(
        compact_metrics.loc[compact_metrics.condition == "targeted_0.50", "recall"].mean()
    )
    rows = []
    for alpha, beta in itertools.product(grid, repeat=2):
        score, _, _, _ = fusion_score(validation, alpha, beta)
        metric = condition_metrics(validation, score, 0.0)
        corrupted = metric[metric.condition != "clean"]
        targeted_recall = float(
            metric.loc[metric.condition == "targeted_0.50", "recall"].mean()
        )
        mean_recall = float(corrupted.recall.mean())
        feasible = bool(
            mean_recall >= compact_recall - 0.03
            and targeted_recall >= compact_targeted - 0.03
        )
        rows.append({
            "alpha_mean": alpha, "beta_topk": beta,
            "feasible_recall": feasible,
            "corrupted_auc": float(corrupted.auc.mean()),
            "corrupted_ba": float(corrupted.ba.mean()),
            "corrupted_recall": mean_recall,
            "corrupted_fpr": float(corrupted.fpr.mean()),
            "targeted_recall": targeted_recall,
        })
    search = pd.DataFrame(rows)
    search["objective"] = search.corrupted_auc + search.corrupted_ba - 0.10 * search.corrupted_fpr
    feasible = search[search.feasible_recall]
    pool = feasible if len(feasible) else search
    selected = pool.sort_values(
        ["objective", "corrupted_ba", "corrupted_auc"], ascending=False,
    ).iloc[0]
    return selected, search


def finish(output, raw_frame, grid):
    calibrated, branch_decisions = branch_calibration(raw_frame)
    wide = aligned_wide(calibrated)
    validation = wide[wide.split == "val"].copy()
    test = wide[wide.split == "test"].copy()
    selected, search = select_policy(validation, grid)
    alpha, beta = float(selected.alpha_mean), float(selected.beta_topk)
    search.to_csv(output / "fusion_weight_search.csv", index=False)

    fusion, wc, wm, wk = fusion_score(test, alpha, beta)
    test["fusion_score"] = fusion
    test["weight_compact"] = wc
    test["weight_mean"] = wm
    test["weight_topk"] = wk

    metric_rows, score_frames = [], []
    for method in BRANCHES:
        score = test[f"margin_{method}"].to_numpy()
        current = test.copy()
        current["method"] = method
        current["score"] = score
        current["threshold"] = 0.0
        score_frames.append(current)
        metric = condition_metrics(test, score, 0.0)
        metric["method"] = method
        metric_rows.append(metric)
    fusion_frame = test.copy()
    fusion_frame["method"] = "adaptive_fusion"
    fusion_frame["score"] = fusion
    fusion_frame["threshold"] = 0.0
    score_frames.append(fusion_frame)
    fusion_metrics = condition_metrics(test, fusion, 0.0)
    fusion_metrics["method"] = "adaptive_fusion"
    metric_rows.append(fusion_metrics)

    metrics = pd.concat(metric_rows, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    clean_lookup = metrics[metrics.condition == "clean"].set_index("method")
    metrics["delta_auc"] = metrics.apply(
        lambda row: row.auc - clean_lookup.loc[row.method, "auc"], axis=1,
    )
    metrics["delta_ba"] = metrics.apply(
        lambda row: row.ba - clean_lookup.loc[row.method, "ba"], axis=1,
    )
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"), delta_ba_mean=("delta_ba", "mean"),
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    selection = {
        "alpha_mean": alpha, "beta_topk": beta,
        "validation_objective": float(selected.objective),
        "validation_feasible_recall": bool(selected.feasible_recall),
        "branch_calibration": branch_decisions,
    }
    (output / "fusion_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8",
    )
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(13, 5.5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 validation-locked adaptive evidence fusion")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "adaptive_fusion_auc.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== FUSION SELECTION ===", flush=True)
    print(json.dumps(selection, indent=2), flush=True)
    print("\n=== ONE-SEED SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary, selection


def decision(summary, selection):
    table = summary.set_index(["method", "condition"])
    corruptions = ["sensor_0.30", "sensor_0.50", "targeted_0.50"]
    effects = {}
    for condition in ["clean", *corruptions]:
        fusion = table.loc[("adaptive_fusion", condition)]
        baseline = table.loc[("compact_raw", condition)]
        effects[condition] = {
            "delta_auc": float(fusion.auc_mean - baseline.auc_mean),
            "delta_ba": float(fusion.ba_mean - baseline.ba_mean),
            "delta_fpr": float(fusion.fpr_mean - baseline.fpr_mean),
            "delta_recall": float(fusion.recall_mean - baseline.recall_mean),
        }
    promote = bool(
        np.mean([effects[c]["delta_auc"] for c in corruptions]) > 0.0
        and np.mean([effects[c]["delta_ba"] for c in corruptions]) > 0.0
        and np.mean([effects[c]["delta_recall"] for c in corruptions]) >= -0.03
        and effects["targeted_0.50"]["delta_auc"] >= -0.01
    )
    return {
        "selected_policy": selection,
        "effects_fusion_minus_compact": effects,
        "promote_to_three_seeds": promote,
        "decision": (
            "PROMOTE adaptive fusion to a three-seed pilot."
            if promote else
            "STOP after one seed; validation-locked fusion did not improve the test trade-off."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--compact-epochs", type=int, default=5)
    parser.add_argument("--sensor-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--sensor-latent", type=int, default=32)
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument("--weight-grid-step", type=float, default=0.25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={METHODS} seed={args.seed}",
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
    expected = len(BRANCHES) * 2 * 132 * len(scenarios())
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate branch/record/split/condition keys")
    grid = np.arange(0.0, 1.0 + args.weight_grid_step / 2, args.weight_grid_step)
    summary, selection = finish(args.output_dir, frame, grid)
    screening = decision(summary, selection)
    (args.output_dir / "decision.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 validation-locked adaptive fusion v9",
        "seed": args.seed, "fit": "healthy train only",
        "fusion_selection": "clean/corrupted labeled validation only",
        "test_used_for_fit_or_selection": False,
        "policy_features": ["coverage", "healthy-train missing criticality"],
        "recall_constraint": "corrupted and targeted validation recall within 0.03 of compact",
        "branches": list(BRANCHES), "conditions": scenarios(),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(screening, indent=2), flush=True)
    print(f"ADAPTIVE_FUSION_V9_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

