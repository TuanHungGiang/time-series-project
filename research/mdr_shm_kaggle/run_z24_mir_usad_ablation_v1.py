from __future__ import annotations

"""One-seed ablation for sensor-loss-aware USAD blocks on Z24."""

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
    FitMinMax, Normalizer, fit_usad, publication_metrics, score_usad,
    seed_everything,
)
from run_z24_mir_usad_v0 import (  # noqa: E402
    fit_mir_usad, mir_components, scenarios, sensor_corrupt,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)


VARIANTS = {
    "usad": None,
    "mask_observed": {"recovery": 0.0, "consistency": 0.0},
    "mask_recovery": {"recovery": 0.10, "consistency": 0.0},
    "mask_recovery_cons": {"recovery": 0.10, "consistency": 0.02},
}


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def build_values(
    records: pd.DataFrame,
    cache: dict[str, np.ndarray],
    condition: str,
    kind: str,
    rate: float,
    replicate: int,
    importance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]]:
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


def make_record_frame(
    method: str,
    window_scores: np.ndarray,
    counts: list[int],
    metadata: list[dict],
    reference: dict,
    reliability: np.ndarray,
) -> pd.DataFrame:
    upper, two_sided = aggregate_record_scores(window_scores, counts, reference)
    frame = pd.DataFrame(metadata)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two_sided
    frame["windows"] = counts
    start, record_reliability = 0, []
    for count in counts:
        record_reliability.append(float(np.mean(reliability[start:start + count])))
        start += count
    frame["reliability"] = record_reliability
    return frame


def run_group(
    group: pd.DataFrame,
    labels: list[str],
    seed: int,
    group_index: int,
    args,
    device: torch.device,
    output: Path,
) -> None:
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
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scale = FitMinMax(fit)
    fit_scaled = scale.transform(fit)
    internal_scaled = scale.transform(internal)

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw)

    cohorts = [("val", validation, "clean", "clean", 0.0, 0)]
    cohorts.extend(
        [("test", test, condition, kind, rate, replicate)
         for condition, kind, rate, replicate in scenarios()]
    )
    frames = []

    for variant_index, (method, config) in enumerate(VARIANTS.items()):
        seed_everything(seed + group_index * 1009)
        if method == "usad":
            model = fit_usad(fit_scaled, args.epochs, args.batch_size, device)
            internal_scores = score_usad(
                model, internal_scaled, args.batch_size, device,
            )

            def scorer(values, available):
                scores = score_usad(
                    model, scale.transform(values), args.batch_size, device,
                )
                return scores, np.ones(len(values), dtype=float)
        else:
            model = fit_mir_usad(
                fit_scaled, args.epochs, args.batch_size, device,
                config["recovery"], config["consistency"],
            )
            internal_mask = np.ones(internal.shape[:2], dtype=np.float32)
            internal_errors, _ = mir_components(
                model, internal_scaled, internal_mask, args.batch_size, device,
            )
            # Uniform observed-sensor fusion isolates training blocks from the
            # inverse-error weighting that harmed MIR-USAD-v0 at 50% loss.
            internal_scores = internal_errors.mean(1)

            def scorer(values, available):
                errors, _ = mir_components(
                    model, scale.transform(values), available,
                    args.batch_size, device,
                )
                denominator = available.sum(1).clip(1.0)
                scores = (errors * available).sum(1) / denominator
                reliability = available.mean(1)
                return scores, reliability

        reference = robust_score_reference(internal_scores)
        print(f"  scoring variant={method}", flush=True)
        for split, records, condition, kind, rate, replicate in cohorts:
            values, available, counts, metadata = build_values(
                records, cache, condition, kind, rate, replicate, importance,
            )
            window_scores, reliability = scorer(values, available)
            frames.append(make_record_frame(
                method, window_scores, counts, metadata, reference, reliability,
            ))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def finish(output: Path, frame: pd.DataFrame) -> None:
    metric_rows, scored = [], []
    for method in VARIANTS:
        validation = frame[(frame.method == method) & (frame.split == "val")].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                validation.label, validation[f"score_{tail}"],
            )
            candidates.append(
                ((metric["ba"], metric["auc"], metric["f1"]), tail, threshold)
            )
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
                "reliability_mean": float(current.reliability.mean()),
            }
            rows.append(row)
            if condition == "clean":
                clean = row
        for row in rows:
            row["delta_auc"] = row["auc"] - clean["auc"]
            row["delta_ba"] = row["ba"] - clean["ba"]
        metric_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(scored, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_ba_mean=("delta_ba", "mean"),
        reliability_mean=("reliability_mean", "mean"),
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(14, 6), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("MIR-USAD block ablation (one seed)")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "ablation_auc.png", dpi=220)
    plt.close(axis.figure)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
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
    expected = len(VARIANTS) * (132 + 132 * len(scenarios()))
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate ablation record keys")
    finish(args.output_dir, frame)
    protocol = {
        "experiment": "MIR-USAD-v1 block ablation", "seed": args.seed,
        "epochs": args.epochs, "fit": "healthy train only",
        "selection": "clean labeled validation", "variants": VARIANTS,
        "fusion": "uniform observed-sensor mean for all MIR variants",
        "reliability": "retained sensor fraction; no gate applied",
        "conditions": scenarios(), "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"MIR_USAD_ABLATION_V1_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

