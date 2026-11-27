"""One-seed paired CLMM spectral trial on existing HellBridge and Z24 splits."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from clmm_observed_spectrum import ObservedSpectrum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--hell-h5", required=True, type=Path)
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270901)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    import run_hellbridge_source_baselines as hb
    import run_z24_clmm_v21 as clmm
    import run_z24_source_baselines as z24
    from run_z24_care_usad_v18 import complete
    from run_z24_missingness_baselines import stable_rng
    from run_z24_mdr_v16 import threshold_under_fpr_cap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rates = (0.10, 0.20, 0.30)
    subsets = [("clean", 0.0, 0)] + [
        (f"sensor_{rate:.2f}", rate, rep) for rate in rates for rep in range(5)
    ]
    print(f"device={device} seed={args.seed}", flush=True)

    def fit_group(raw_fit, raw_reference, seed, batch, direction):
        normalizer = hb.Normalizer.fit(raw_fit)
        fit = normalizer.transform(raw_fit)
        scaler = hb.FitMinMax(fit)
        scaled = scaler.transform(fit)
        hb.seed_everything(seed)
        source = hb.fit_usad(scaled, 30, batch, device)

        def score(values):
            return hb.score_usad(source, values, batch, device)

        reference = z24.robust_score_reference(score(scaler.transform(normalizer.transform(raw_reference))))
        hb.seed_everything(seed + 97)
        adapter = clmm.fit_adapter(
            scaled, np.square(raw_fit).mean(axis=(0, 2)), source, reference,
            20, 16, 64, 4, 4, 0.0, 0.10, device, seed + 97,
        )
        spectrum = ObservedSpectrum(fit)
        cache = {}

        def imputed(values, available):
            if available.all():
                return score(values)
            mask = np.broadcast_to(available, (len(values), len(available)))
            filled, _ = complete(adapter, values, mask, batch, device)
            return score(filled)

        def score_methods(raw, available):
            values = normalizer.transform(raw)
            transformed = scaler.transform(values)
            zeroed = values.copy()
            zeroed[:, ~available] = 0
            unadapted = score(scaler.transform(zeroed))
            adapted = unadapted if available.all() else imputed(transformed, available)
            key = available.tobytes()
            if key not in cache:
                cache[key] = imputed(scaled, available)
            extra = spectrum.score_pair(
                values, available, cache[key], adapted, usad_direction=direction,
            )
            center, scale = reference
            return {
                "usad_zero_fill": direction * (np.log1p(unadapted.astype(float)) - center) / scale,
                "clmm_rs": direction * (np.log1p(adapted.astype(float)) - center) / scale,
                **extra,
            }

        return score_methods

    def sensor_mask(channels, rate, order):
        result = np.ones(channels, dtype=bool)
        if rate:
            result[order[:min(channels - 1, max(1, round(channels * rate)))]] = False
        return result

    def evaluate(dataset, scores):
        rows = []
        for (condition, method), frame in scores.groupby(["condition", "method"]):
            calibration = scores[(scores.split == "val") & (scores.method == method)]
            if dataset == "HellBridge":
                calibration = calibration[calibration.condition == "clean"]
                threshold = hb.threshold_from_healthy(calibration.score.to_numpy(), 0.90)
            else:
                calibration = calibration[calibration.condition == condition]
                threshold, _ = threshold_under_fpr_cap(
                    calibration.label.to_numpy(), calibration.score.to_numpy(), 0.20,
                )
            for rep, test in frame[frame.split == "test"].groupby("replicate"):
                rows.append({
                    "dataset": dataset, "seed": args.seed, "condition": condition,
                    "method": method, "replicate": rep, "threshold": threshold,
                    **hb.publication_metrics(test.label, test.score, threshold),
                })
        scores.to_csv(out / f"{dataset}_scores.csv", index=False)
        return pd.DataFrame(rows)

    split = out / "hellbridge_split"
    subprocess.run([
        sys.executable, str(args.source_dir / "prepare_hellbridge_paper_split.py"),
        "--h5", str(args.hell_h5), "--output-dir", str(split),
        "--profile", "svendsen2022", "--regime", "P2_NM_Z",
        "--seed", "20261113", "--strict-paper-counts",
    ], check=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(
        ["record_id", "start"]
    ).reset_index(drop=True)
    fit_idx, cal_idx, test_idx = hb.development_split(manifest)
    raw = hb.load_windows(args.hell_h5, manifest, 128)
    # The healthy reference for adapter score scaling comes from fit only.
    hell_score = fit_group(raw[fit_idx], raw[fit_idx], args.seed, 16, 1)
    rng = np.random.default_rng(20261119)
    orders = [rng.permutation(40) for _ in range(5)]
    blocks = []
    for indices, role in ((cal_idx, "val"), (test_idx, "test")):
        for condition, rate, rep in subsets:
            if role == "val" and rate:
                continue
            available = sensor_mask(40, rate, orders[rep])
            for method, values in hell_score(raw[indices], available).items():
                blocks.append(pd.DataFrame({
                    "sample_id": manifest.loc[indices, "sample_id"].to_numpy(),
                    "label": manifest.loc[indices, "label"].to_numpy(),
                    "split": role, "method": method, "condition": condition,
                    "replicate": rep, "score": values,
                }))
    hell = evaluate("HellBridge", pd.concat(blocks, ignore_index=True))
    hell.to_csv(out / "HellBridge_metrics_per_mask.csv", index=False)
    del raw, hell_score, blocks

    manifest = z24.load_manifest(args.z24_root)
    rows = []
    for index, ((setup, excitation), channels) in enumerate(sorted(z24.stable_sensor_map(manifest).items())):
        print(f"Z24 group {index + 1} setup={setup} excitation={excitation}", flush=True)
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            signal = z24.read_stable_vertical(Path(row.path), channels)
            return z24.record_windows(signal, 1024, 1024, "raw_temporal", 2, 128)

        fit, reference = z24.split_training_windows([
            load(row) for row in group[group.split == "train"].itertuples()
        ])
        score_group = fit_group(fit, reference, args.seed + index * 1009, 32, -1)
        for row in group[group.split.isin(["val", "test"])].itertuples():
            raw = load(row)
            for condition, rate, rep in subsets:
                order = stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                available = sensor_mask(len(channels), rate, order)
                for method, values in score_group(raw, available).items():
                    rows.append({
                        "record_id": row.record_id, "label": row.label,
                        "split": row.split, "method": method, "condition": condition,
                        "replicate": rep, "score": float(np.median(values)),
                    })
        del score_group
    z24_result = evaluate("Z24", pd.DataFrame(rows))
    metrics = pd.concat([hell, z24_result], ignore_index=True)
    metrics.to_csv(out / "metrics_per_mask.csv", index=False)
    names = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "specificity"]
    summary = metrics.groupby(["dataset", "condition", "method"])[names].mean()
    summary.to_csv(out / "summary.csv")
    paired = metrics.pivot(index=["dataset", "condition", "replicate"], columns="method", values=names)
    deltas = paired.xs("fused", axis=1, level="method") - paired.xs("clmm_rs", axis=1, level="method")
    deltas.to_csv(out / "fused_minus_clmm_per_mask.csv")
    print(summary[["auc", "ba", "recall", "fpr"]].round(4).to_string())
    print("CLMM_TWO_BRIDGE_TRIAL_PASS", flush=True)


if __name__ == "__main__":
    main()
