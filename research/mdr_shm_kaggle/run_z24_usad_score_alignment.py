"""Compare fit-reference and held-out healthy score alignment on Z24."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_hellbridge_source_baselines as metrics
from run_clmm_multi_detector import recall_at_fpr, sensor_mask
from run_clmm_shared_configuration import BATCH, LENGTH, RATE, SPECIFICITY, spectral_features
from run_usad_clean_two_bridges import train_compact_usad
from run_z24_missingness_baselines import stable_rng
from run_z24_source_baselines import (
    load_manifest, read_stable_vertical, split_training_windows, stable_sensor_map,
)


def reference(values):
    center = float(np.median(values))
    spread = max(1.4826 * np.median(np.abs(values - center)), 0.25 * values.std(), 1e-4)
    return center, float(spread)


def fit_scorer(raw_fit, seed, device):
    scaler = metrics.FitMinMax(raw_fit)
    fit = scaler.transform(raw_fit)
    median = torch.as_tensor(np.median(fit, axis=0), device=device)
    metrics.seed_everything(seed)
    model = train_compact_usad(fit, 30, BATCH, device, metrics.UsadModel)

    def residual(x):
        flat = x.flatten(1)
        first = model.decoder1(model.encoder(flat))
        cycle = model.decoder2(model.encoder(first))
        return (0.5 * ((flat - first).square() + (flat - cycle).square())).reshape_as(x)

    fit_values = torch.as_tensor(fit, device=device)
    with torch.no_grad():
        errors = torch.cat([residual(batch) for batch in fit_values.split(BATCH)]).cpu().numpy().astype(np.float64)
    scale = np.median(errors, axis=0)
    floor = np.maximum(np.quantile(scale, 0.25, axis=1, keepdims=True), 1e-6)
    scale = torch.as_tensor(np.maximum(scale, floor), device=device)

    @torch.no_grad()
    def score(values, available):
        mask = torch.as_tensor(available, device=device).view(1, -1, 1)
        indices = torch.as_tensor(np.flatnonzero(available), device=device)
        values = torch.where(mask, values, median)
        return np.concatenate([
            torch.log1p((residual(batch) / scale).index_select(1, indices).mean((1, 2))).cpu().numpy()
            for batch in values.split(BATCH)
        ])

    full = np.ones(fit.shape[1], dtype=bool)
    original = reference(score(fit_values, full))

    def predict(raw, available):
        return score(torch.as_tensor(scaler.transform(raw), device=device), available)

    return predict, original


def evaluate(scores, column):
    healthy = scores.loc[scores.split.eq("val"), column].to_numpy()
    if len(healthy) != 6:
        raise ValueError(f"Expected 6 healthy validation records, got {len(healthy)}")
    threshold = metrics.threshold_from_healthy(healthy, SPECIFICITY)
    rows = []
    for (condition, replicate), group in scores[scores.split.eq("test")].groupby(["condition", "replicate"]):
        y = group.label.to_numpy()
        s = group[column].to_numpy()
        if len(np.unique(group.record_id)) != 132 or (y == 0).sum() != 6 or (y == 1).sum() != 126:
            raise ValueError(f"Unexpected test cohort: {condition}/{replicate}")
        rows.append({
            "method": column, "condition": condition, "replicate": replicate,
            "threshold": threshold, **metrics.publication_metrics(y, s, threshold),
            "false_positives": int(((y == 0) & (s >= threshold)).sum()),
            "true_positives": int(((y == 1) & (s >= threshold)).sum()),
            "recall_at_fpr_05": recall_at_fpr(y, s, 0.05),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270901)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.z24_root)
    rows, references = [], []
    for index, ((setup, excitation), channels) in enumerate(sorted(stable_sensor_map(manifest).items())):
        print(f"Z24 {setup=} {excitation=} sensors={len(channels)}", flush=True)
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            raw = read_stable_vertical(Path(row.path), channels)
            length = raw.shape[-1] // LENGTH * LENGTH
            if length < LENGTH:
                raise ValueError(f"No complete windows: {row.record_id}")
            return spectral_features(raw[:, :length].reshape(len(channels), -1, LENGTH).transpose(1, 0, 2))

        train = group[group.split.eq("train") & group.label.eq(0)]
        fit, held_out = split_training_windows([load(row) for row in train.itertuples()])
        predict, (fit_center, fit_scale) = fit_scorer(fit, args.seed + index * 1009, device)
        full = np.ones(len(channels), dtype=bool)
        hold_center, hold_scale = reference(predict(held_out, full))
        references.append({
            "setup": setup, "excitation": excitation, "train_records": len(train),
            "held_out_windows": len(held_out), "fit_center": fit_center, "fit_scale": fit_scale,
            "hold_center": hold_center, "hold_scale": hold_scale,
        })
        for row in group[group.split.eq("test") | (group.split.eq("val") & group.label.eq(0))].itertuples():
            features = load(row)
            scenarios = [("clean", 0, full)]
            if row.split == "test":
                scenarios.extend(
                    ("sensor_0.30", rep, sensor_mask(
                        len(channels), RATE,
                        stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels)),
                    )) for rep in range(5)
                )
            for condition, rep, mask in scenarios:
                rows.append({
                    "record_id": row.record_id, "setup": setup, "excitation": excitation,
                    "split": row.split, "label": row.label, "state": row.state,
                    "condition": condition, "replicate": rep, "raw_score": float(np.median(predict(features, mask))),
                })
        del predict

    refs = pd.DataFrame(references)
    pooled = refs.groupby("excitation")["hold_scale"].transform("median")
    # Equal-weight log-scale pooling is fixed before evaluating validation/test scores.
    refs["aligned_scale"] = np.sqrt(refs.hold_scale * pooled)
    refs.to_csv(out / "healthy_group_reference.csv", index=False)
    scores = pd.DataFrame(rows).merge(refs, on=["setup", "excitation"], validate="many_to_one")
    scores["A_fit_reference"] = (scores.raw_score - scores.fit_center) / scores.fit_scale
    scores["B_heldout_alignment"] = (scores.raw_score - scores.hold_center) / scores.aligned_scale
    scores.to_csv(out / "record_scores.csv", index=False)

    methods = ("A_fit_reference", "B_heldout_alignment")
    per_mask = pd.DataFrame([row for method in methods for row in evaluate(scores, method)])
    per_mask.to_csv(out / "metrics_per_mask.csv", index=False)
    fields = ["auc", "ap", "ba", "recall", "fpr", "false_positives", "true_positives", "recall_at_fpr_05"]
    summary = per_mask.groupby(["condition", "method"], as_index=False)[fields].mean()
    summary.to_csv(out / "summary.csv", index=False)
    paired = per_mask.pivot(index=["condition", "replicate"], columns="method", values=fields)
    (paired.xs(methods[1], level="method", axis=1) - paired.xs(methods[0], level="method", axis=1)).to_csv(
        out / "paired_deltas.csv"
    )
    (out / "protocol.json").write_text(json.dumps({
        "seed": args.seed, "fit": "healthy training records, first 75% minus guard window",
        "alignment": "healthy training records, held-out final windows, per setup/excitation",
        "alignment_scale": "geometric mean of group scale and median group scale within excitation",
        "calibration": "shared clean empirical q90 on 6 healthy validation records, fixed across masks",
        "evaluation_unit": "record", "missing_sensor_rate": RATE, "mask_replicates": 5,
    }, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print("Z24_SCORE_ALIGNMENT_PASS", flush=True)


if __name__ == "__main__":
    main()
