"""Z24 trials of residual pooling and healthy reference under sensor loss."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.hellbridge import Normalizer
from .data.masking import sensor_mask, stable_rng
from .data.z24 import load_manifest, read_stable_vertical, record_windows, split_training_windows, stable_sensor_map
from .experiment import BATCH, anchor_periods, score_rows
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .score_calibration import fit_mask_aware, location_scale


METHODS = ("mean", "trimmed_20")
RATE = 0.50
VAL_REPS = tuple(range(100, 120))
TEST_REPS = tuple(range(5))
REFERENCE_METHODS = ("clean_reference", "matched_reference")
REFERENCE_TEST_REPS = tuple(range(200, 210))


def pool(residual):
    ordered = np.sort(residual, axis=1)
    trim = int(0.20 * residual.shape[1])
    middle = ordered[:, trim:residual.shape[1] - trim]
    return {"mean": np.log1p(residual.mean(1)), "trimmed_20": np.log1p(middle.mean(1))}


def load(row, channels):
    return record_windows(read_stable_vertical(Path(row.path), channels), 1024, 1024, "raw_temporal", 2, 128)


@torch.no_grad()
def channel_errors(model, values, kept, device):
    mask = torch.as_tensor(kept, device=device).view(1, 1, -1)
    x = torch.as_tensor(values, device=device).transpose(1, 2)
    errors = np.concatenate([
        (batch - model(batch, mask=mask)).square().mean(1).cpu().numpy()
        for batch in x.split(BATCH)
    ]).astype(np.float64)
    return errors[:, kept]


def validate_and_test(scores, out, methods=METHODS, test_reps=TEST_REPS):
    keys = ["method", "split", "condition", "replicate", "sample_id"]
    if scores.duplicated(keys).any() or not np.isfinite(scores.score).all():
        raise ValueError("Invalid or duplicate scores")
    expected = {
        (method, split, condition, rep)
        for method in methods
        for split, reps in (("val", VAL_REPS), ("test", test_reps))
        for condition, rep in [("clean", 0), *(("sensor_0.50", r) for r in reps)]
    }
    groups = scores.groupby(keys[:-1])
    if set(groups.groups) != expected:
        raise ValueError("Unexpected condition, split, method or replicate")
    for key, cohort in groups:
        if cohort.label.value_counts().to_dict() != {0: 6, 1: 126}:
            raise ValueError(f"Expected 6 healthy and 126 damage records: {key}")
    val = scores[scores.split.eq("val")].copy()
    signs = {}
    validation_rows = []
    for method, group in val.groupby("method"):
        clean = group[group.condition.eq("clean")]
        upper = roc_auc_score(clean.label, clean.score)
        signs[method] = -1 if upper < 0.5 else 1
        for rep, trial in group[group.condition.eq("sensor_0.50")].groupby("replicate"):
            oriented = trial.score * signs[method]
            validation_rows.append(dict(method=method, replicate=rep,
                                        auc=roc_auc_score(trial.label, oriented),
                                        recall_at_zero_fp=recall_at_fpr(trial.label, oriented, 0.0)))
    validation = pd.DataFrame(validation_rows)
    ranking = validation.groupby("method", as_index=False)[["auc", "recall_at_zero_fp"]].mean()
    ranking = ranking.sort_values(["recall_at_zero_fp", "auc", "method"], ascending=[False, False, True])
    selected = ranking.iloc[0].method
    ranking.assign(selected=ranking.method.eq(selected)).to_csv(out / "validation_selection.csv", index=False)
    validation.to_csv(out / "validation_by_mask.csv", index=False)

    # Pool healthy scores across validation-only masks for one fixed 50% threshold.
    val["oriented_score"] = val.score * val.method.map(signs)
    healthy = val[val.label.eq(0)]
    test_rows = []
    for (method, condition, rep), trial in scores[scores.split.eq("test")].groupby(["method", "condition", "replicate"]):
        calibration = healthy[healthy.method.eq(method) & healthy.condition.eq(condition)]
        threshold = threshold_from_healthy(calibration.oriented_score.to_numpy(), 0.90)
        y, oriented = trial.label.to_numpy(), trial.score.to_numpy() * signs[method]
        test_rows.append(dict(method=method, condition=condition, replicate=rep, selected=method == selected,
                              threshold=threshold, fp=int(((oriented >= threshold) & (y == 0)).sum()),
                              recall_at_zero_fp=recall_at_fpr(y, oriented, 0.0),
                              **publication_metrics(y, oriented, threshold)))
    results = pd.DataFrame(test_rows)
    results.to_csv(out / "test_by_mask.csv", index=False)
    results.groupby(["condition", "method", "selected"], as_index=False).mean(numeric_only=True).drop(
        columns=["replicate"]
    ).to_csv(out / "test_summary.csv", index=False)
    if methods == REFERENCE_METHODS:
        metrics = ["auc", "ba", "recall", "fpr", "recall_at_zero_fp"]
        paired = results.pivot(index=["condition", "replicate"], columns="method", values=metrics)
        control = paired.xs(REFERENCE_METHODS[0], axis=1, level="method")
        matched = paired.xs(REFERENCE_METHODS[1], axis=1, level="method")
        (matched - control).add_prefix("delta_").to_csv(out / "paired_test_deltas.csv")
    scores.assign(oriented_score=scores.score * scores.method.map(signs)).to_csv(
        out / "record_scores.csv.gz", index=False,
    )
    (out / "selection.json").write_text(json.dumps(dict(
        selected=selected, signs=signs, validation_masks=VAL_REPS, test_masks=test_reps,
        threshold="Q90 pooled across validation-only masks within condition",
    ), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270905)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--matched-reference", action="store_true")
    args = parser.parse_args()
    test_reps = REFERENCE_TEST_REPS if args.matched_reference else TEST_REPS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.z24_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, sensor_rows = [], []
    for index, ((setup, excitation), channels) in enumerate(sorted(stable_sensor_map(manifest).items())):
        print(f"Z24 {setup=} {excitation=}", flush=True)
        group = manifest[manifest.setup.eq(setup) & manifest.excitation.eq(excitation)]
        fit, ref = split_training_windows([load(row, channels) for row in group[group.split.eq("train")].itertuples()])
        scaler = Normalizer.fit(fit)
        fit = scaler.transform(fit)
        model = fit_mask_aware(fit, device, args.seed + 1009 * index, args.epochs)
        anchor_periods(model, torch.as_tensor(fit, device=device))
        full = np.ones(len(channels), dtype=bool)
        reference_data = scaler.transform(ref)
        reference_errors = channel_errors(model, reference_data, full, device)
        references = {method: location_scale(s) for method, s in pool(reference_errors).items()}
        mask_references = {full.tobytes(): references["mean"]}
        for row in group[group.split.isin(["val", "test"])].itertuples():
            data = scaler.transform(load(row, channels))
            reps = VAL_REPS if row.split == "val" else test_reps
            cases = [("clean", 0, full)]
            for rep in reps:
                order = stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                cases.append(("sensor_0.50", rep, sensor_mask(len(channels), RATE, order)))
            for condition, rep, kept in cases:
                residual = channel_errors(model, data, kept, device)
                if args.matched_reference:
                    mask_key = kept.tobytes()
                    if mask_key not in mask_references:
                        healthy_errors = channel_errors(model, reference_data, kept, device)
                        mask_references[mask_key] = location_scale(np.log1p(healthy_errors.mean(axis=1)))
                    pooled = dict.fromkeys(REFERENCE_METHODS, np.log1p(residual.mean(axis=1)))
                    score_references = dict(zip(REFERENCE_METHODS, (references["mean"], mask_references[mask_key])))
                else:
                    pooled = pool(residual)
                    score_references = references
                if row.split == "val" and row.label == 0:
                    sensor_rows.extend(dict(
                        record_id=row.record_id, setup=setup, excitation=excitation,
                        condition=condition, replicate=rep, sensor=sensor, median_residual=value,
                    ) for sensor, value in zip(np.asarray(channels)[kept], np.median(residual, axis=0), strict=True))
                standardized = {
                    method: [float(np.median((score - score_references[method][0]) / score_references[method][1]))]
                    for method, score in pooled.items()
                }
                score_rows(rows, standardized, [row.record_id], [row.label], row.split, condition, rep)
        del model
    pd.DataFrame(sensor_rows).to_csv(args.output_dir / "validation_healthy_sensor_residuals.csv", index=False)
    methods = REFERENCE_METHODS if args.matched_reference else METHODS
    validate_and_test(pd.DataFrame(rows), args.output_dir, methods=methods, test_reps=test_reps)
    print("Z24_MATCHED_REFERENCE_PASS" if args.matched_reference else "Z24_ROBUST_POOLING_PASS", flush=True)


if __name__ == "__main__":
    main()
