from __future__ import annotations

"""Publication baseline run for Z24 random 50% sensor loss.

Every detector is fitted on clean healthy training windows.  All methods use
the exact deterministic record-level random sensor masks used by CLMM-RS.
Score orientation is fixed to the lower tail, and one condition-matched
threshold per model/condition is calibrated on labelled validation records
under the same FPR cap used by CLMM V21/V23.  Test labels are evaluation-only.
The run is resumable by model, seed, and setup/excitation group.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_missingness_baselines as missing  # noqa: E402
from run_hellbridge_source_baselines import (  # noqa: E402
    MODELS,
    Normalizer,
    publication_metrics,
)
from run_z24_mdr_v16 import threshold_under_fpr_cap  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores,
    load_manifest,
    read_stable_vertical,
    record_windows,
    robust_score_reference,
    split_training_windows,
    stable_sensor_map,
)


DEFAULT_MODELS = (
    "pca_spe",
    "lstm_ae",
    "usad",
    "tranad",
    "timesnet",
    "moderntcn",
    "scatterad",
)
DEFAULT_SEEDS = (
    20270901,
    20270913,
    20270929,
    20271011,
    20271027,
    20271109,
    20271123,
    20271207,
    20271219,
    20280103,
)
SCENARIOS = (
    ("clean", "clean", 0.0, 0),
    *(("sensor_0.50", "sensor", 0.50, replicate) for replicate in range(5)),
)
METRICS = (
    "auc", "ap", "ba", "recall", "specificity", "fpr", "precision", "f1",
)


def group_path(run_dir: Path, setup: int, excitation: str) -> Path:
    return run_dir / "groups" / f"setup{setup:02d}_{excitation}.csv"


def score_records(scorer, values, counts, reference):
    window_scores = scorer(values)
    upper, two_sided = aggregate_record_scores(
        window_scores, counts, reference,
    )
    return np.asarray(upper), -np.asarray(upper), np.asarray(two_sided)


def build_split(
    records,
    cache,
    scenarios,
    scorer,
    reference,
    sensor_importance,
):
    frames = []
    for condition, kind, rate, replicate in scenarios:
        arrays, counts, rows = [], [], []
        for row in records.itertuples():
            clean = cache[str(row.record_id)]
            current = missing.corrupt(
                clean,
                kind,
                rate,
                str(row.record_id),
                condition,
                replicate,
                sensor_importance,
            )
            observed = ~np.all(np.isclose(current, 0.0, atol=1e-8), axis=2)
            arrays.append(current)
            counts.append(len(current))
            rows.append({
                "record_id": row.record_id,
                "split": row.split,
                "label": int(row.label),
                "state": row.state,
                "setup": row.setup,
                "excitation": row.excitation,
                "condition": condition,
                "kind": kind,
                "rate": rate,
                "replicate": replicate,
                "coverage": float(observed.mean()),
                "windows": len(current),
            })
        values = np.concatenate(arrays)
        upper, lower, two_sided = score_records(
            scorer, values, counts, reference,
        )
        frame = pd.DataFrame(rows)
        frame["score_upper"] = upper
        frame["score_lower"] = lower
        frame["score_two_sided"] = two_sided
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_group(
    args,
    model_name,
    seed,
    group_index,
    group,
    labels,
    device,
    output,
):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_windows = [
        record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_windows)
    normalizer = Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    sensor_importance = np.mean(np.square(fit_raw), axis=(0, 2))

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw)

    missing.seed_everything(seed + 1009 * group_index)
    scorer = missing.fit_scorer(
        model_name,
        fit,
        args.epochs,
        args.batch_size,
        device,
    )
    reference = robust_score_reference(scorer(internal))
    validation_frame = build_split(
        validation,
        cache,
        SCENARIOS,
        scorer,
        reference,
        sensor_importance,
    )
    test_frame = build_split(
        test,
        cache,
        SCENARIOS,
        scorer,
        reference,
        sensor_importance,
    )
    result = pd.concat([validation_frame, test_frame], ignore_index=True)
    result["model"] = model_name
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect(run_dir: Path, sensor_map):
    paths = [
        group_path(run_dir, setup, excitation)
        for setup, excitation in sorted(sensor_map)
    ]
    if not all(path.exists() for path in paths):
        raise RuntimeError("Incomplete setup/excitation groups")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = 132 * len(SCENARIOS) * 2
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate record/split/condition/replicate keys")
    return frame


def finish_run(args, model_name, seed, run_dir, frame):
    validation = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    policy_rows, metric_rows, score_rows = [], [], []
    for condition in ("clean", "sensor_0.50"):
        calibration = validation[validation.condition == condition].copy()
        threshold, validation_metric = threshold_under_fpr_cap(
            calibration.label.to_numpy(),
            calibration.score_lower.to_numpy(),
            args.fpr_cap,
        )
        policy_rows.append({
            "model": model_name,
            "seed": seed,
            "condition": condition,
            "score_tail": "lower",
            "threshold": threshold,
            "fpr_cap": args.fpr_cap,
            "validation_records": len(calibration),
            "validation_healthy_records": int((calibration.label == 0).sum()),
            "validation_damage_records": int((calibration.label == 1).sum()),
            **{f"validation_{key}": value for key, value in validation_metric.items()},
        })
        selected = test[test.condition == condition].copy()
        for replicate, cohort in selected.groupby("replicate"):
            metric = publication_metrics(
                cohort.label,
                cohort.score_lower,
                threshold,
            )
            metric_rows.append({
                "model": model_name,
                "seed": seed,
                "condition": condition,
                "replicate": int(replicate),
                "score_tail": "lower",
                "threshold": threshold,
                "fpr_cap": args.fpr_cap,
                **metric,
                "records": len(cohort),
                "healthy_records": int((cohort.label == 0).sum()),
                "damage_records": int((cohort.label == 1).sum()),
            })
            scored = cohort.copy()
            scored["score"] = scored.score_lower
            scored["threshold"] = threshold
            scored["prediction"] = (scored.score >= threshold).astype(int)
            score_rows.append(scored)

    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(
        run_dir / "record_scores.csv", index=False,
    )
    pd.DataFrame(policy_rows).to_csv(
        run_dir / "validation_policy.csv", index=False,
    )
    (run_dir / "complete.json").write_text(
        json.dumps({
            "model": model_name,
            "seed": seed,
            "fixed_tail": "lower",
            "threshold_calibration": "condition-matched validation under FPR cap",
        }, indent=2),
        encoding="utf-8",
    )


def refresh(output: Path):
    complete = sorted((output / "runs").glob("*/complete.json"))
    if not complete:
        return
    run_dirs = [path.parent for path in complete]
    metrics = pd.concat(
        [pd.read_csv(path / "metrics.csv") for path in run_dirs],
        ignore_index=True,
    )
    scores = pd.concat(
        [pd.read_csv(path / "record_scores.csv") for path in run_dirs],
        ignore_index=True,
    )
    policies = pd.concat(
        [pd.read_csv(path / "validation_policy.csv") for path in run_dirs],
        ignore_index=True,
    )
    metrics.to_csv(output / "all_metrics.csv", index=False)
    scores.to_csv(output / "all_record_scores.csv", index=False)
    policies.to_csv(output / "all_validation_policies.csv", index=False)

    summary = metrics.groupby(["model", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"),
        ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        observations=("replicate", "size"), seeds=("seed", "nunique"),
    )
    summary.to_csv(output / "publication_summary.csv", index=False)

    seed_summary = metrics.groupby(
        ["model", "seed", "condition"], as_index=False,
    )[list(METRICS)].mean()
    seed_summary.to_csv(output / "per_seed_summary.csv", index=False)

    # Publication variability is computed across independent training seeds.
    # For sensor_0.50, each seed is first averaged over its five masks; the 50
    # seed-mask observations are not treated as 50 independent training runs.
    aggregation = {}
    for metric in METRICS:
        aggregation[f"{metric}_mean"] = (metric, "mean")
        aggregation[f"{metric}_seed_sd"] = (metric, "std")
    aggregation["seeds"] = ("seed", "nunique")
    seed_level = seed_summary.groupby(
        ["model", "condition"], as_index=False,
    ).agg(**aggregation)
    seed_level.to_csv(
        output / "seed_level_publication_summary.csv", index=False,
    )

    formatted = seed_level.copy()
    for metric in METRICS:
        mean_column = f"{metric}_mean"
        sd_column = f"{metric}_seed_sd"
        formatted[f"{metric}_mean_pm_seed_sd"] = [
            (
                f"{mean:.4f} +/- {sd:.4f}"
                if np.isfinite(sd)
                else f"{mean:.4f} (deterministic)"
            )
            for mean, sd in zip(
                formatted[mean_column].to_numpy(dtype=float),
                formatted[sd_column].to_numpy(dtype=float),
            )
        ]
    formatted_columns = ["model", "condition", "seeds"] + [
        f"{metric}_mean_pm_seed_sd" for metric in METRICS
    ]
    formatted[formatted_columns].to_csv(
        output / "publication_table_mean_pm_sd.csv", index=False,
    )

    sensor = summary[summary.condition == "sensor_0.50"].sort_values(
        "ba_mean", ascending=False,
    )
    axis = sensor.plot.bar(
        x="model",
        y=["auc_mean", "ba_mean", "recall_mean"],
        figsize=(11, 5),
        ylim=(0, 1.02),
        rot=20,
        grid=True,
    )
    axis.set_ylabel("Record-level metric")
    axis.set_title("Z24 random sensor-50% baseline comparison")
    axis.figure.tight_layout()
    axis.figure.savefig(
        output / "sensor50_baselines.png", dpi=220, bbox_inches="tight",
    )
    plt.close(axis.figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--fpr-cap", type=float, default=0.20)
    args = parser.parse_args()

    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("Exactly ten unique seeds are required")
    if not 0 <= args.fpr_cap < 1:
        raise ValueError("fpr-cap must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    missing.assert_nested_masks()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    protocol = {
        "dataset": "Z24",
        "experiment": "canonical baselines under random 50% sensor loss",
        "training": "clean healthy-only",
        "preprocessing": "canonical detrend, polyphase downsample, train-only channel z-score",
        "input": "[batch,C_cell,512]",
        "conditions": [list(value) for value in SCENARIOS],
        "mask_identity": "same run_z24_missingness_baselines.corrupt function used by CLMM-RS",
        "random_masks_per_seed": 5,
        "fixed_tail": "lower",
        "threshold_calibration": "condition-matched labelled validation under FPR cap",
        "fpr_cap": args.fpr_cap,
        "test_used_for_training_selection_or_threshold": False,
        "models": args.models,
        "seeds": args.seeds,
        "pca_seed_policy": "one deterministic fit",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(
        f"device={device} models={args.models} seeds={args.seeds} "
        f"cells={len(sensor_map)}",
        flush=True,
    )

    for model_name in args.models:
        model_seeds = args.seeds[:1] if model_name.startswith("pca_") else args.seeds
        for seed in model_seeds:
            run_dir = args.output_dir / "runs" / f"{model_name}__{seed}"
            if (run_dir / "complete.json").exists():
                print(f"SKIP complete model={model_name} seed={seed}", flush=True)
                continue
            started = time.time()
            for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
                path = group_path(run_dir, key[0], key[1])
                if path.exists():
                    print(
                        f"SKIP group model={model_name} seed={seed} cell={key}",
                        flush=True,
                    )
                    continue
                print(
                    f"model={model_name} seed={seed} cell={key} sensors={len(labels)}",
                    flush=True,
                )
                group = used[
                    (used.setup == key[0]) & (used.excitation == key[1])
                ]
                run_group(
                    args, model_name, seed, group_index, group, labels, device, path,
                )
            frame = collect(run_dir, sensor_map)
            finish_run(args, model_name, seed, run_dir, frame)
            info_path = run_dir / "complete.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["seconds"] = time.time() - started
            info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            refresh(args.output_dir)
            current = pd.read_csv(run_dir / "metrics.csv")
            compact = current.groupby("condition", as_index=False).agg(
                auc=("auc", "mean"), ba=("ba", "mean"),
                recall=("recall", "mean"), fpr=("fpr", "mean"),
                f1=("f1", "mean"), replicates=("replicate", "size"),
            )
            print(compact.to_string(index=False), flush=True)

    refresh(args.output_dir)
    summary = pd.read_csv(args.output_dir / "publication_summary.csv")
    print("\n=== Z24 SENSOR-50 BASELINE PUBLICATION SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== SEED-LEVEL MEAN +/- SD (FIVE MASKS AVERAGED WITHIN SEED) ===")
    seed_level = pd.read_csv(
        args.output_dir / "seed_level_publication_summary.csv"
    )
    print(seed_level.to_string(index=False))
    print("\n=== FORMATTED PUBLICATION TABLE ===")
    formatted = pd.read_csv(
        args.output_dir / "publication_table_mean_pm_sd.csv"
    )
    print(formatted.to_string(index=False))
    print(f"Z24_SENSOR50_BASELINES_V1_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

