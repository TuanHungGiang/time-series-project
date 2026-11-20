"""Compare clean and same-mask healthy references on HellBridge."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.hellbridge import Normalizer, development_split, load_windows
from .data.masking import sensor_mask
from .experiment import anchor_periods, score_rows
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .robust_pooling import REFERENCE_METHODS, channel_errors
from .score_calibration import fit_mask_aware, location_scale


METHODS = REFERENCE_METHODS
MASK_RATE = 0.50
VAL_REPS = range(100, 120)
TEST_REPS = range(200, 210)
CONDITION = "sensor_0.50"


def score_cases(channels):
    rng = np.random.default_rng(20261120)
    used = set()
    yield "clean", 0, np.ones(channels, dtype=bool)
    for rep in (*VAL_REPS, *TEST_REPS):
        kept = sensor_mask(channels, MASK_RATE, rng.permutation(channels))
        key = kept.tobytes()
        if key in used:
            raise ValueError("Repeated calibration or test mask")
        used.add(key)
        yield CONDITION, rep, kept


def evaluate(scores, out):
    keys = ["method", "split", "condition", "replicate"]
    if scores.duplicated(keys + ["sample_id"]).any() or not np.isfinite(scores.score).all():
        raise ValueError("Invalid or duplicate scores")
    expected = {
        (method, split, condition, rep)
        for method in METHODS
        for split, reps in (("val", VAL_REPS), ("test", TEST_REPS))
        for condition, rep in [("clean", 0), *((CONDITION, r) for r in reps)]
    }
    groups = scores.groupby(keys)
    if set(groups.groups) != expected:
        raise ValueError("Missing calibration or test cohort")
    for (_, split, _, _), cohort in groups:
        counts = {0: 10} if split == "val" else {0: 80, 1: 640}
        if cohort.label.value_counts().to_dict() != counts:
            raise ValueError(f"Unexpected label counts in {split}")

    healthy = scores[scores.split.eq("val")]
    test = scores[scores.split.eq("test")]
    rows = []
    for (method, condition, rep), trial in test.groupby(["method", "condition", "replicate"]):
        calibration = healthy[healthy.method.eq(method) & healthy.condition.eq(condition)]
        threshold = threshold_from_healthy(calibration.score.to_numpy(), 0.90)
        y, s = trial.label.to_numpy(), trial.score.to_numpy()
        rows.append(dict(method=method, condition=condition, replicate=rep, threshold=threshold,
                         fp=int(((s >= threshold) & (y == 0)).sum()),
                         recall_at_fpr_05=recall_at_fpr(y, s, 0.05),
                         **publication_metrics(y, s, threshold)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "test_by_mask.csv", index=False)
    fields = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "fp", "recall_at_fpr_05"]
    metrics.groupby(["condition", "method"], as_index=False)[fields].mean().to_csv(
        out / "test_summary.csv", index=False,
    )
    paired = metrics.pivot(index=["condition", "replicate"], columns="method", values=fields)
    delta = paired.xs(METHODS[1], axis=1, level="method") - paired.xs(METHODS[0], axis=1, level="method")
    if not np.allclose(delta[["auc", "recall_at_fpr_05"]], 0):
        raise ValueError("A common affine reference should preserve within-mask ranking")
    delta.add_prefix("delta_").to_csv(out / "paired_test_deltas.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270905)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split = out / "hellbridge_split"
    subprocess.run([sys.executable, "-m", "clmm_timesnet.data.prepare_hellbridge", "--h5", str(args.hell_h5),
                    "--output-dir", str(split), "--profile", "svendsen2022", "--regime", "P2_NM_Z",
                    "--seed", "20261113", "--strict-paper-counts"], check=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, held_out, test = development_split(manifest)
    # One of each held-out pair is used as reference, the other to fit Q90.
    pairs = held_out.reshape(-1, 2)
    ref, cal = pairs[:, 0], pairs[:, 1]
    if len(np.unique(np.concatenate([fit, ref, cal, test]))) != len(manifest):
        raise ValueError("Overlapping window assignments")
    raw = load_windows(args.hell_h5, manifest, 128)
    scaler = Normalizer.fit(raw[fit])
    values = scaler.transform(raw)
    del raw
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fit={len(fit)} reference={len(ref)} calibration={len(cal)} test={len(test)} {device=}", flush=True)
    model = fit_mask_aware(values[fit], device, args.seed, args.epochs)
    anchor_periods(model, torch.as_tensor(values[fit], device=device))

    def residual_scores(ids, kept):
        return np.log1p(channel_errors(model, values[ids], kept, device).mean(axis=1))

    clean_reference = location_scale(residual_scores(ref, np.ones(values.shape[1], dtype=bool)))
    rows, masks = [], []
    for condition, rep, kept in score_cases(values.shape[1]):
        matched_reference = clean_reference if condition == "clean" else location_scale(residual_scores(ref, kept))
        roles = (("val", cal), ("test", test)) if condition == "clean" else (
            (("val", cal),) if rep in VAL_REPS else (("test", test),)
        )
        for role, ids in roles:
            raw_score = residual_scores(ids, kept)
            scores = {method: (raw_score - center[0]) / scale[0]
                      for method, (center, scale) in zip(METHODS, (clean_reference, matched_reference), strict=True)}
            score_rows(rows, scores, manifest.loc[ids, "sample_id"], manifest.loc[ids, "label"], role, condition, rep)
            masks.append(dict(split=role, condition=condition, replicate=rep,
                              dropped_sensors=" ".join(f"AL{i+1:02d}" for i in np.flatnonzero(~kept))))
        print(f"Scored {condition} replicate={rep}", flush=True)
    scores = pd.DataFrame(rows)
    evaluate(scores, out)
    scores.to_csv(out / "window_scores.csv.gz", index=False)
    pd.DataFrame(masks).to_csv(out / "masks.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(dict(
        seed=args.seed, epochs=args.epochs, healthy_fit=60, healthy_reference=10, healthy_calibration=10,
        healthy_test=80, damage_test=640, calibration_masks=list(VAL_REPS), test_masks=list(TEST_REPS),
        orientation="upper", threshold="Q90 within condition, pooling calibration mask scores",
    ), indent=2), encoding="utf-8")
    print("HELLBRIDGE_MATCHED_REFERENCE_PASS", flush=True)


if __name__ == "__main__":
    main()
