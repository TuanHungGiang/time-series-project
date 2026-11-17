from __future__ import annotations

"""MDR-SHM V15: V14 model with mask-conditional validation calibration.

The representation and fusion policy are unchanged from V14.  V15 evaluates
every missing-sensor condition on the validation split and locks one lower-tail
threshold per method/condition before test evaluation.  It additionally reports
standardized partial AUROC (FPR <= 0.20) and recall at a matched test FPR of
1/6, clearly labelled as evaluation-only ROC quantities.

Training remains healthy-only, but operational threshold calibration uses
validation damage labels.  Test labels never fit the model, fusion, or deployed
thresholds.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_mdr_v12 as v12  # noqa: E402
import run_z24_mdr_v13 as v13  # noqa: E402
import run_z24_mdr_v14 as v14  # noqa: E402
from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax, Normalizer, publication_metrics, seed_everything,
)
from run_z24_recovery_kd_v3 import count_parameters, score_compact  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)


BASE_METHODS = v14.BASE_METHODS
ALL_METHODS = v14.ALL_METHODS
MATCHED_FPR = 1.0 / 6.0


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def run_group(group, labels, seed, group_index, args, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]

    train_by_record = [
        record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window, args.window_stride,
            "raw_temporal", args.downsample, 128,
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
            read_stable_vertical(Path(row.path), labels),
            args.raw_window, args.window_stride,
            "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    base_seed = seed + group_index * 1009
    seed_everything(base_seed)
    compact = v14.fit_compact_and_capture(
        fit, args.epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    compact_internal = score_compact(
        compact, internal, args.batch_size, device,
    )
    compact_reference = robust_score_reference(compact_internal)

    seed_everything(base_seed + 97)
    anchored = v14.fit_anchored_adapter(
        fit, importance, args.epochs, args.batch_size,
        args.recovery_hidden, args.hidden1, args.hidden2, args.latent,
        args.recovery_weight, args.consistency_weight, device,
    )
    internal_mask = np.ones(internal.shape[:2], dtype=np.float32)
    anchored_internal = v13.observed_only_maskaware_score(
        anchored, internal, internal_mask, args.batch_size, device,
    )
    anchored_reference = robust_score_reference(anchored_internal)
    parameters = {
        "compact_raw": count_parameters(compact),
        "maskaware_consistency": count_parameters(anchored),
    }
    print(f"  parameters={parameters}", flush=True)

    # Unlike V14, every condition is generated on validation and test.
    cohorts = []
    for split_records in (validation, test):
        cohorts.extend([
            (split_records, condition, kind, rate, replicate)
            for condition, kind, rate, replicate in v12.scenarios_v12()
        ])

    frames = []
    for records, condition, kind, rate, replicate in cohorts:
        values, available, counts, metadata = v14.build_values_with_criticality(
            records, cache, condition, kind, rate, replicate, importance,
        )
        scaled = scaler.transform(values).astype(np.float32)

        raw_scores = score_compact(compact, scaled, args.batch_size, device)
        raw_upper, raw_two = aggregate_record_scores(
            raw_scores, counts, compact_reference,
        )
        frames.append(v12.core.make_frame(
            "compact_raw", raw_scores, counts, metadata,
            compact_reference, available, parameters["compact_raw"],
        ))

        anchored_scores = v13.observed_only_maskaware_score(
            anchored, scaled, available.astype(np.float32),
            args.batch_size, device,
        )
        frames.append(v12.core.make_frame(
            "maskaware_consistency", anchored_scores, counts, metadata,
            anchored_reference, available,
            parameters["maskaware_consistency"],
        ))

    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del compact, anchored
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def matched_recall(labels, scores, target_fpr=MATCHED_FPR):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    healthy = scores[labels == 0]
    damage = scores[labels == 1]
    if len(healthy) == 0 or len(damage) == 0:
        return float("nan"), float("nan")
    span = max(float(np.ptp(scores)), 1.0)
    candidates = np.concatenate([
        [scores.max() + span * 1e-9],
        np.unique(scores)[::-1],
        [scores.min() - span * 1e-9],
    ])
    best_recall, best_fpr = 0.0, 0.0
    for threshold in candidates:
        fpr = float(np.mean(healthy >= threshold))
        if fpr <= target_fpr + 1e-12:
            recall = float(np.mean(damage >= threshold))
            if (recall, -fpr) > (best_recall, -best_fpr):
                best_recall, best_fpr = recall, fpr
    return best_recall, best_fpr


def low_fpr_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=0.20))


def finish(output: Path, frame: pd.DataFrame):
    frame = v14.add_criticality_fusion(frame)
    metric_rows, scored_frames = [], []

    for method in ALL_METHODS:
        current = frame[frame.method == method]
        for condition in [
            "clean", "sensor_0.10", "sensor_0.30", "sensor_0.50",
            "targeted_0.30", "targeted_0.50",
        ]:
            validation = current[
                (current.split == "val") & (current.condition == condition)
            ].copy()
            test = current[
                (current.split == "test") & (current.condition == condition)
            ].copy()
            threshold, validation_metric = threshold_from_labeled_validation(
                validation.label.to_numpy(), validation.score_lower.to_numpy(),
            )
            test["score"] = test.score_lower
            test["threshold"] = threshold
            test["score_tail"] = "lower_fixed"
            test["calibration"] = "condition_matched_labeled_validation"
            scored_frames.append(test)

            for replicate, cohort in test.groupby("replicate", sort=False):
                matched, realized = matched_recall(
                    cohort.label, cohort.score, MATCHED_FPR,
                )
                metric_rows.append({
                    "method": method,
                    "condition": condition,
                    "replicate": int(replicate),
                    "score_tail": "lower_fixed",
                    "threshold": threshold,
                    "validation_ba": validation_metric["ba"],
                    **publication_metrics(cohort.label, cohort.score, threshold),
                    "partial_auc_fpr_0.20": low_fpr_auc(
                        cohort.label, cohort.score,
                    ),
                    "matched_recall_fpr_1_over_6": matched,
                    "matched_realized_fpr": realized,
                    "coverage_mean": float(cohort.coverage.mean()),
                    "parameters": int(cohort.parameters.iloc[0]),
                })

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(scored_frames, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ba_mean=("ba", "mean"),
        ba_std=("ba", "std"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        partial_auc_fpr_0_20=("partial_auc_fpr_0.20", "mean"),
        matched_recall_fpr_1_over_6=("matched_recall_fpr_1_over_6", "mean"),
        matched_realized_fpr=("matched_realized_fpr", "mean"),
        coverage_mean=("coverage_mean", "mean"),
        parameters=("parameters", "mean"), runs=("replicate", "size"),
    )

    baseline = summary[summary.method == "compact_raw"].set_index("condition")
    effect_rows = []
    for method in ("maskaware_consistency", "criticality_fusion"):
        proposed = summary[summary.method == method].set_index("condition")
        columns = [
            "auc_mean", "ba_mean", "recall_mean", "fpr_mean",
            "partial_auc_fpr_0_20", "matched_recall_fpr_1_over_6",
        ]
        effects = proposed[columns].subtract(baseline[columns])
        effects.columns = [
            "delta_auc", "delta_ba", "delta_recall", "delta_fpr",
            "delta_partial_auc_fpr_0_20", "delta_matched_recall_fpr_1_over_6",
        ]
        effects["method"] = method
        effect_rows.append(effects.reset_index())
    effects = pd.concat(effect_rows, ignore_index=True)

    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)

    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(13, 5.5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("MDR-SHM V15: mask-conditional calibration")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v15_auc.png", dpi=220)
    plt.close(axis.figure)

    print("\n=== V15 SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== EFFECTS VS BEST USAD BASELINE ===", flush=True)
    print(effects.to_string(index=False), flush=True)
    return summary


def decision(summary: pd.DataFrame):
    table = summary.set_index(["method", "condition"])
    effects = {}
    for condition in (
        "clean", "sensor_0.10", "sensor_0.30", "sensor_0.50",
        "targeted_0.30", "targeted_0.50",
    ):
        proposed = table.loc[("criticality_fusion", condition)]
        baseline = table.loc[("compact_raw", condition)]
        effects[condition] = {
            "delta_auc": float(proposed.auc_mean - baseline.auc_mean),
            "delta_ba": float(proposed.ba_mean - baseline.ba_mean),
            "delta_recall": float(proposed.recall_mean - baseline.recall_mean),
            "delta_fpr": float(proposed.fpr_mean - baseline.fpr_mean),
            "delta_partial_auc_fpr_0_20": float(
                proposed.partial_auc_fpr_0_20 - baseline.partial_auc_fpr_0_20
            ),
            "delta_matched_recall_fpr_1_over_6": float(
                proposed.matched_recall_fpr_1_over_6
                - baseline.matched_recall_fpr_1_over_6
            ),
        }
    random_conditions = ("sensor_0.10", "sensor_0.30", "sensor_0.50")
    mean_random_auc = float(np.mean([
        effects[c]["delta_auc"] for c in random_conditions
    ]))
    promote = bool(
        effects["clean"]["delta_auc"] >= -0.005
        and mean_random_auc > 0.01
        and effects["sensor_0.50"]["delta_auc"] >= 0.02
        and effects["sensor_0.50"]["delta_ba"] >= 0.0
        and effects["sensor_0.50"]["delta_recall"] >= -0.02
        and effects["sensor_0.50"]["delta_matched_recall_fpr_1_over_6"] >= 0.0
        and effects["targeted_0.50"]["delta_auc"] >= -0.005
        and effects["targeted_0.50"]["delta_ba"] >= 0.0
    )
    return {
        "effects_criticality_fusion_minus_best_usad": effects,
        "mean_random_loss_delta_auc": mean_random_auc,
        "promote_to_three_seeds": promote,
        "decision": (
            "PROMOTE V15 to three seeds."
            if promote else
            "STOP after one seed; at least one conditional-calibration gate failed."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--recovery-hidden", type=int, default=48)
    parser.add_argument("--recovery-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={ALL_METHODS} "
        f"seed={args.seed} epochs={args.epochs}", flush=True,
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
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/record/split/condition keys")
    expected = len(BASE_METHODS) * 2 * 132 * len(v12.scenarios_v12())
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} base rows, got {len(frame)}")

    summary = finish(args.output_dir, frame)
    screening = decision(summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "MDR-SHM V15 mask-conditional calibration",
        "seed": args.seed,
        "epochs": args.epochs,
        "model_fit": "healthy train only",
        "backbone": "frozen Compact USAD",
        "adapter_training_masks": "random complete-sensor 10/30/50%; no targeted masks",
        "fusion": "coverage and healthy-train criticality; fixed before test",
        "score_orientation": "fixed lower-tail",
        "operational_threshold": "method/condition-specific labeled validation BA",
        "threshold_calibration_uses_damage_validation_labels": True,
        "test_used_for_fit_calibration_or_selection": False,
        "matched_fpr_metric": "evaluation only; threshold not deployable",
        "matched_fpr": MATCHED_FPR,
        "conditions": v12.scenarios_v12(),
        "methods": list(ALL_METHODS),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(screening, indent=2), flush=True)
    print(f"MDR_V15_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

