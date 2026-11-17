from __future__ import annotations

"""CLMM V20: held-out healthy record calibration for Z24.

V19 established a positive random-50 ranking effect but calibrated the clean
alarm with only six healthy validation records.  V20 fixes that protocol
defect without using damage labels: the final quarter of every healthy
training record is separated by a guard window, never used to fit USAD or the
CLMM adapter, and retained as the record-level calibration cohort.

The detector branch and lower-tail orientation remain locked before
evaluation.  Damage validation records are ignored.  Test labels are used
only after all predictions have been produced.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_care_usad_v18 as care  # noqa: E402
import run_z24_clmm_v19 as v19  # noqa: E402
from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax,
    Normalizer,
    fit_usad,
    score_usad,
    seed_everything,
)
from run_z24_source_baselines import (  # noqa: E402
    load_manifest,
    read_stable_vertical,
    record_windows,
    robust_score_reference,
    stable_sensor_map,
)


CONDITIONS = care.CONDITIONS
SCENARIOS = care.SCENARIOS
PRIMARY_PROTOCOL = v19.PRIMARY_PROTOCOL
STRESS_PROTOCOL = v19.STRESS_PROTOCOL


def chronological_fit_calibration(windows: np.ndarray):
    """Split one healthy record with an explicit one-window guard band."""
    boundary = int(np.floor(0.75 * len(windows)))
    fit_stop = max(1, boundary - 1)
    fit = windows[:fit_stop]
    calibration = windows[boundary:]
    if len(calibration) == 0:
        raise RuntimeError(
            f"Healthy record has no held-out windows: total={len(windows)}"
        )
    return fit, calibration


def run_group(args, group, labels, group_index, device, output):
    train = group[(group.split == "train") & (group.label == 0)].copy()
    validation = group[group.split == "val"].copy()
    test = group[group.split == "test"].copy()

    fit_parts = []
    calibration_raw = {}
    for row in train.itertuples():
        windows = record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        fit, heldout = chronological_fit_calibration(windows)
        fit_parts.append(fit)
        calibration_raw[str(row.record_id)] = heldout

    fit_raw = np.concatenate(fit_parts)
    normalizer = Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw).astype(np.float32)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scale = FitMinMax(fit)
    fit_scaled = scale.transform(fit)

    calibration_records = train.copy()
    calibration_records["split"] = "calibration"
    calibration_records["label"] = 0

    cache = {
        record_id: normalizer.transform(values).astype(np.float32)
        for record_id, values in calibration_raw.items()
    }
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    seed = args.seed + group_index * 1009
    seed_everything(seed)
    source = fit_usad(
        fit_scaled, args.epochs, args.batch_size, device,
    )

    # The nonconformity function is fitted on the proper training cohort.
    # Held-out calibration records remain untouched until inference below.
    source_reference = robust_score_reference(
        score_usad(source, fit_scaled, args.batch_size, device),
    )
    training_score_reference = (
        source_reference[0],
        max(float(source_reference[1]), args.score_scale_floor),
    )

    seed_everything(seed + 97)
    adapter = v19.fit_adapter(
        fit_scaled,
        importance,
        source,
        training_score_reference,
        args.adapter_epochs,
        args.batch_size,
        args.embedding,
        args.heads,
        args.hard_candidates,
        args.latent_weight,
        args.score_weight,
        device,
        seed + 97,
    )
    conditional_usad, conditional_consistency = care.conditional_reference(
        adapter,
        source,
        fit,
        scale,
        importance,
        args.batch_size,
        device,
        seed + 193,
    )
    source_parameters = care.count_parameters(source)
    proposed_parameters = source_parameters + care.count_parameters(adapter)

    frames = []
    for records in (calibration_records, validation, test):
        for condition, kind, rate, replicate in SCENARIOS:
            values, available, counts, rows = care.build_cohort(
                records,
                cache,
                condition,
                kind,
                rate,
                replicate,
                importance,
            )
            scaled_values = scale.transform(values)
            baseline_scores = score_usad(
                source, scaled_values, args.batch_size, device,
            )
            frames.append(care.source_frame(
                rows,
                baseline_scores,
                counts,
                source_reference,
                source_parameters,
            ))

            completed, consistency = care.complete(
                adapter,
                scaled_values,
                available,
                args.batch_size,
                device,
            )
            imputed_scores = score_usad(
                source, completed, args.batch_size, device,
            )
            reference_u = (
                source_reference if condition == "clean"
                else conditional_usad
            )
            reference_c = (
                robust_score_reference(np.zeros(max(2, len(consistency))))
                if condition == "clean" else conditional_consistency
            )
            frames.extend(care.proposed_frames(
                rows,
                imputed_scores,
                consistency,
                counts,
                reference_u,
                reference_c,
                proposed_parameters,
                args.fusion_weight,
            ))

    result = pd.concat(frames, ignore_index=True)
    result["policy_parameters"] = proposed_parameters
    result["heldout_calibration"] = result.split.eq("calibration")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    del source, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluation_frame(frame: pd.DataFrame):
    """Expose only held-out training records to V19's calibration routine."""
    result = frame.copy()
    result.loc[result.split == "val", "split"] = "unused_validation"
    result.loc[result.split == "calibration", "split"] = "val"
    if not result.loc[result.split == "val", "label"].eq(0).all():
        raise RuntimeError("Held-out calibration cohort contains damage labels")
    return result


def finish(args, frame):
    calibrated = evaluation_frame(frame)
    all_metrics, all_scores, all_calibration = [], [], []
    for protocol in (PRIMARY_PROTOCOL, STRESS_PROTOCOL):
        metrics, scores, calibration = v19.evaluate_protocol(
            calibrated, protocol, args.conformal_alpha,
        )
        all_metrics.append(metrics)
        all_scores.append(scores)
        all_calibration.append(calibration)

    metrics = pd.concat(all_metrics, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    summary = v19.summarize(metrics)
    effects = v19.paired_effects(metrics)
    effect_summary = effects.groupby(
        ["protocol", "condition"], as_index=False,
    ).agg(
        delta_auc_mean=("delta_auc", "mean"),
        delta_auc_median=("delta_auc", "median"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_recall_mean=("delta_recall", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"),
        positive_auc_pairs=("delta_auc", lambda x: int((x > 0).sum())),
        positive_ba_pairs=("delta_ba", lambda x: int((x > 0).sum())),
        pairs=("delta_auc", "count"),
    )

    primary_summary = summary[summary.protocol == PRIMARY_PROTOCOL]
    clean = primary_summary[
        (primary_summary.method == "clmm_v19")
        & (primary_summary.condition == "clean")
    ].iloc[0]
    random_effect = effect_summary[
        (effect_summary.protocol == PRIMARY_PROTOCOL)
        & (effect_summary.condition == "sensor_0.50")
    ].iloc[0]
    criteria = {
        "clean_ba_at_least_0_75": bool(clean.ba_mean >= 0.75),
        "clean_recall_at_least_0_55": bool(clean.recall_mean >= 0.55),
        "clean_fpr_at_most_0_15": bool(clean.fpr_mean <= 0.15),
        "random50_delta_auc_at_least_0_03": bool(
            random_effect.delta_auc_mean >= 0.03
        ),
        "random50_delta_ba_at_least_0_05": bool(
            random_effect.delta_ba_mean >= 0.05
        ),
        "random50_delta_recall_at_least_0_10": bool(
            random_effect.delta_recall_mean >= 0.10
        ),
        "random50_positive_auc_masks_at_least_3_of_5": bool(
            random_effect.positive_auc_pairs >= 3
        ),
    }
    decision = {
        "promote_to_three_seed": bool(all(criteria.values())),
        "criteria": criteria,
        "primary_protocol": PRIMARY_PROTOCOL,
        "fixed_branch": "imputed_usad",
        "fixed_tail": "lower",
        "calibration_source": (
            "chronologically held-out quarters of healthy training records"
        ),
        "damage_validation_used_for_selection": False,
        "test_used_for_selection": False,
        "targeted50_role": "stress_test_only",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    calibration.to_csv(
        args.output_dir / "healthy_calibration_audit.csv", index=False,
    )
    frame.loc[frame.split == "calibration", [
        "record_id", "state", "setup", "excitation", "condition",
        "replicate", "method", "coverage",
    ]].drop_duplicates().to_csv(
        args.output_dir / "heldout_calibration_manifest.csv", index=False,
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    effect_summary.to_csv(
        args.output_dir / "effect_summary.csv", index=False,
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    pivot = primary_summary.pivot(
        index="condition", columns="method", values="ba_mean",
    )
    axis = pivot.plot.bar(
        figsize=(10, 5), ylim=(0, 1.02), rot=15, grid=True,
    )
    axis.set_ylabel("Record-level balanced accuracy")
    axis.set_title("CLMM V20 — held-out healthy record calibration")
    axis.figure.tight_layout()
    axis.figure.savefig(
        args.output_dir / "clmm_v20_screening.png", dpi=220,
    )
    plt.close(axis.figure)

    print("\n=== CLMM V20 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== CLMM V20 PAIRED EFFECTS ===")
    print(effects.to_string(index=False))
    print("\n=== CLMM V20 EFFECT SUMMARY ===")
    print(effect_summary.to_string(index=False))
    print("\n=== HELD-OUT HEALTHY CALIBRATION AUDIT ===")
    print(calibration.to_string(index=False))
    print("\n=== CLMM V20 SCREENING DECISION ===")
    print(json.dumps(decision, indent=2))
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270717)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hard-candidates", type=int, default=4)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--latent-weight", type=float, default=0.20)
    parser.add_argument("--score-weight", type=float, default=0.10)
    parser.add_argument("--score-scale-floor", type=float, default=0.02)
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    args = parser.parse_args()

    if not 0.0 < args.conformal_alpha < 1.0:
        raise ValueError("conformal-alpha must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} seed={args.seed} "
        f"source_epochs={args.epochs} adapter_epochs={args.adapter_epochs} "
        f"alpha={args.conformal_alpha}",
        flush=True,
    )

    for index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = care.group_file(args.output_dir, key[0], key[1])
        if path.exists():
            print(f"SKIP cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[
            (used.setup == key[0])
            & (used.excitation == key[1])
        ]
        run_group(args, group, labels, index, device, path)

    paths = sorted((args.output_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 group files, got {len(paths)}")
    frame = pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True,
    )
    decision = finish(args, frame)
    protocol = {
        "dataset": "Z24",
        "method": "CLMM V20",
        "full_name": "Counteracting Loss from Missing Measurements",
        "baseline": "vendored official EURECOM USAD",
        "proper_training": "first 75% minus one guard window per healthy train record",
        "calibration": "final 25% of each healthy train record",
        "calibration_labels": "healthy-only",
        "conformal_alpha": args.conformal_alpha,
        "fixed_branch": "imputed_usad",
        "fixed_tail": "lower",
        "clean_branch": "untouched source USAD",
        "damage_validation_used_for_selection": False,
        "test_used_for_selection": False,
        "targeted50_role": "stress_test_only",
        "seed": args.seed,
        "decision": decision,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"CLMM_V20_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

