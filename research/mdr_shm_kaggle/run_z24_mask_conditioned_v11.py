from __future__ import annotations

"""One-seed Z24 screening for mask-conditioned residual anomaly scoring.

The detector is fitted once on complete healthy training windows.  At test
time the source score uses a complete-data healthy reference, while the
proposed score uses a healthy reference generated under the exact same sensor
mask as each query record.  No damage test label is used for fitting,
reference estimation, tail selection, or threshold selection.
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

from run_hellbridge_source_baselines import (  # noqa: E402
    Normalizer, publication_metrics, seed_everything,
)
from run_z24_missingness_baselines import corrupt, fit_scorer  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)


METHODS = ("clean_reference", "mask_conditioned")


def scenarios():
    rows = [("clean", "clean", 0.0, 0)]
    for rate in (0.10, 0.30, 0.50):
        for replicate in range(5):
            rows.append((f"sensor_{rate:.2f}", "sensor", rate, replicate))
        rows.append((f"targeted_{rate:.2f}", "targeted", rate, 0))
    return rows


def group_path(output, model, setup, excitation):
    return output / "groups" / model / f"setup{setup:02d}_{excitation}.csv"


def mask_signature(mask):
    # Complete-sensor loss is constant over time for the protocol in this file.
    available = mask[0, :, 0] > 0.5
    return tuple(np.flatnonzero(available).tolist())


def exact_mask(shape, kind, rate, record_id, condition, replicate, importance):
    ones = np.ones(shape, dtype=np.float32)
    return corrupt(
        ones, kind, rate, record_id, condition, replicate, importance,
    )


def aggregate_single(scores, reference):
    upper, two_sided = aggregate_record_scores(scores, [len(scores)], reference)
    return float(upper[0]), float(two_sided[0])


def run_group(model_name, group, labels, seed, group_index, args, device, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_by_record = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        ) for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_by_record)
    normalizer = Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw)

    seed_everything(seed + group_index * 1009)
    scorer = fit_scorer(
        model_name, fit, args.epochs, args.batch_size, device,
    )
    clean_reference = robust_score_reference(scorer(internal))
    reference_cache = {}

    def conditioned_reference(mask):
        signature = mask_signature(mask)
        if signature not in reference_cache:
            masked_internal = internal * mask[:1]
            reference_cache[signature] = robust_score_reference(
                scorer(masked_internal),
            )
        return reference_cache[signature]

    rows = []
    cohorts = [(validation, "clean", "clean", 0.0, 0, "val")]
    cohorts += [
        (test, condition, kind, rate, replicate, "test")
        for condition, kind, rate, replicate in scenarios()
    ]
    for records, condition, kind, rate, replicate, split in cohorts:
        for row in records.itertuples():
            record_id = str(row.record_id)
            clean_values = cache[record_id]
            mask = exact_mask(
                clean_values.shape, kind, rate, record_id, condition,
                replicate, importance,
            )
            values = clean_values * mask
            scores = scorer(values)
            source_upper, source_two = aggregate_single(
                scores, clean_reference,
            )
            conditional_upper, conditional_two = aggregate_single(
                scores, conditioned_reference(mask),
            )
            common = {
                "record_id": record_id, "split": split,
                "label": int(row.label), "state": row.state,
                "setup": row.setup, "excitation": row.excitation,
                "condition": condition, "kind": kind, "rate": rate,
                "replicate": replicate,
                "coverage": float(mask.mean()),
                "mask_signature": ",".join(map(str, mask_signature(mask))),
                "model": model_name, "seed": seed,
            }
            for method, upper, two_sided in (
                ("clean_reference", source_upper, source_two),
                ("mask_conditioned", conditional_upper, conditional_two),
            ):
                rows.append({
                    **common, "method": method,
                    "score_upper": upper, "score_lower": -upper,
                    "score_two_sided": two_sided,
                })
    pd.DataFrame(rows).to_csv(output, index=False)
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calibrate_and_evaluate(frame):
    decisions, metrics, score_frames = {}, [], []
    for (model, method), current in frame.groupby(["model", "method"]):
        validation = current[
            (current.split == "val") & (current.condition == "clean")
        ]
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            column = f"score_{tail}"
            threshold, metric = threshold_from_labeled_validation(
                validation.label, validation[column],
            )
            candidates.append(
                ((metric["ba"], metric["auc"], metric["f1"]), tail, threshold)
            )
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        decisions[f"{model}:{method}"] = {
            "tail": tail, "threshold": float(threshold),
        }
        test = current[current.split == "test"].copy()
        test["score_tail"] = tail
        test["score"] = test[f"score_{tail}"]
        test["threshold"] = threshold
        score_frames.append(test)
        for (condition, replicate), cohort in test.groupby(
            ["condition", "replicate"], sort=False,
        ):
            metrics.append({
                "model": model, "method": method,
                "condition": condition, "replicate": int(replicate),
                **publication_metrics(cohort.label, cohort.score, threshold),
            })
    return pd.DataFrame(metrics), pd.concat(score_frames), decisions


def finish(output, frame):
    metrics, scores, decisions = calibrate_and_evaluate(frame)
    clean = metrics[metrics.condition == "clean"].set_index(["model", "method"])
    for metric in ("auc", "ba", "recall", "fpr", "f1"):
        metrics[f"delta_{metric}"] = metrics.apply(
            lambda row: row[metric] - clean.loc[(row.model, row.method), metric],
            axis=1,
        )
    summary = metrics.groupby(["model", "method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_mask_std=("auc", "std"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_ba_mean=("delta_ba", "mean"),
        masks=("replicate", "size"),
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "calibration.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8",
    )
    comparison = summary.pivot_table(
        index=["model", "condition"], columns="method",
        values=["auc_mean", "ba_mean", "recall_mean", "fpr_mean"],
    ).reset_index()
    comparison.columns = [
        "_".join(value).strip("_") if isinstance(value, tuple) else value
        for value in comparison.columns
    ]
    for metric in ("auc_mean", "ba_mean", "recall_mean", "fpr_mean"):
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_mask_conditioned"]
            - comparison[f"{metric}_clean_reference"]
        )
    comparison.to_csv(output / "comparison.csv", index=False)

    corruptions = comparison[comparison.condition != "clean"]
    targeted50 = comparison[comparison.condition == "targeted_0.50"]
    promote = bool(
        corruptions.delta_auc_mean.mean() >= 0.03
        and corruptions.delta_ba_mean.mean() >= 0.03
        and corruptions.delta_fpr_mean.mean() <= -0.05
        and corruptions.delta_recall_mean.mean() >= -0.05
        and (len(targeted50) == 0 or targeted50.delta_auc_mean.mean() >= 0.0)
    )
    decision = {
        "promote_to_three_seeds": promote,
        "mean_corrupted_delta_auc": float(corruptions.delta_auc_mean.mean()),
        "mean_corrupted_delta_ba": float(corruptions.delta_ba_mean.mean()),
        "mean_corrupted_delta_recall": float(corruptions.delta_recall_mean.mean()),
        "mean_corrupted_delta_fpr": float(corruptions.delta_fpr_mean.mean()),
        "decision": (
            "PROMOTE mask-conditioned scoring to three seeds."
            if promote else
            "STOP: mask-conditioned scoring did not pass the one-seed gate."
        ),
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )
    pivot = summary.pivot_table(
        index="condition", columns=["model", "method"], values="auc_mean",
    )
    axis = pivot.plot.bar(figsize=(14, 6), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 V11 mask-conditioned healthy-reference screening")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v11_auc.png", dpi=220, bbox_inches="tight")
    plt.close(axis.figure)
    print("\n=== V11 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== V11 COMPARISON ===")
    print(comparison.to_string(index=False))
    print("\n=== V11 DECISION ===")
    print(json.dumps(decision, indent=2))
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("pca_spe", "usad"),
                        default=["pca_spe", "usad"])
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensors = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    started = time.time()
    print(
        f"device={device} models={args.models} cells={len(sensors)} "
        f"conditions={len(scenarios())}", flush=True,
    )
    for model in args.models:
        for group_index, (key, labels) in enumerate(sorted(sensors.items())):
            output = group_path(args.output_dir, model, *key)
            if output.exists():
                print(f"SKIP model={model} cell={key}", flush=True)
                continue
            print(f"model={model} cell={key} sensors={len(labels)}", flush=True)
            group = used[(used.setup == key[0]) & (used.excitation == key[1])]
            run_group(
                model, group, labels, args.seed, group_index, args, device, output,
            )
    paths = [
        group_path(args.output_dir, model, *key)
        for model in args.models for key in sorted(sensors)
    ]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["model", "method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate model/method/record/condition keys")
    decision = finish(args.output_dir, frame)
    protocol = {
        "experiment": "Z24 mask-conditioned residual scoring V11",
        "fit": "complete healthy train windows only",
        "mask_reference": "same exact query sensor mask applied to internal healthy windows",
        "selection": "clean labeled validation only",
        "test_used_for_fit_reference_or_selection": False,
        "models": args.models, "seed": args.seed,
        "conditions": scenarios(), "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"Z24_MASK_CONDITIONED_V11_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

