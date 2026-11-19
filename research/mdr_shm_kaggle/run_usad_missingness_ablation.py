"""One-seed ablation of mask training, observed scores, and calibration."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_clmm_shared_configuration import (
    BATCH, LENGTH, RATE, SPECIFICITY, append_scores, hell_features, spectral_features,
)
from run_clmm_multi_detector import recall_at_fpr, sensor_mask
from run_usad_clean_two_bridges import train_compact_usad


def weighted_mean(residual, scale, observed):
    weights = observed.unsqueeze(-1)
    return ((residual / scale) * weights).sum((1, 2)) / (
        observed.sum(1) * residual.shape[-1]
    )


def train_masked_usad(fit, median, seed, device, mask_input, epochs=30):
    _, channels, length = fit.shape
    width = channels * length
    source = nn.Module()
    source.encoder = nn.Sequential(
        nn.Linear(width + channels, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
    )

    def decoder():
        return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 512),
                             nn.ReLU(), nn.Linear(512, width), nn.Sigmoid())

    source.decoder1, source.decoder2 = decoder(), decoder()
    source.to(device)
    values = torch.as_tensor(fit, device=device)
    first = torch.optim.Adam([*source.encoder.parameters(), *source.decoder1.parameters()], lr=1e-3)
    second = torch.optim.Adam([*source.encoder.parameters(), *source.decoder2.parameters()], lr=1e-3)
    rng = np.random.default_rng(seed)

    def encode(x, observed):
        x = torch.where(observed.unsqueeze(-1).bool(), x, median).flatten(1)
        mask_features = observed if mask_input else torch.zeros_like(observed)
        return source.encoder(torch.cat([x, mask_features], dim=1))

    def forward(x, observed, include_second=False):
        z = encode(x, observed)
        initial = source.decoder1(z).reshape_as(x)
        cycle = source.decoder2(encode(initial, observed)).reshape_as(x)
        alternate = source.decoder2(z).reshape_as(x) if include_second else None
        return initial, cycle, alternate

    def observed_mse(x, output, mask):
        return weighted_mean((x - output).square(), 1.0, mask).mean()

    for epoch in range(1, epochs + 1):
        losses = []
        order = rng.permutation(len(fit))
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH].tolist()
            x = values[ids]
            masks = np.stack([
                sensor_mask(channels, float(rng.choice([0.0, 0.10, 0.20, 0.30])), rng.permutation(channels))
                for _ in ids
            ])
            mask = torch.as_tensor(masks, dtype=x.dtype, device=device)
            source.zero_grad(set_to_none=True)
            initial, cycle, _ = forward(x, mask)
            loss1 = observed_mse(x, initial, mask) / epoch + (1 - 1 / epoch) * observed_mse(x, cycle, mask)
            loss1.backward()
            first.step()

            source.zero_grad(set_to_none=True)
            _, cycle, alternate = forward(x, mask, include_second=True)
            loss2 = observed_mse(x, alternate, mask) / epoch - (1 - 1 / epoch) * observed_mse(x, cycle, mask)
            loss2.backward()
            second.step()
            losses.append(float((loss1 + loss2).detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  mask_input={mask_input} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    source.eval().requires_grad_(False)
    return forward


def fit_group(raw_fit, seed, device, hb):
    scaler = hb.FitMinMax(raw_fit)
    fit = scaler.transform(raw_fit)
    median = torch.as_tensor(np.median(fit, axis=0), device=device)
    fit_t = torch.as_tensor(fit, device=device)
    channels = fit.shape[1]
    full_mask = np.ones(channels, dtype=bool)

    hb.seed_everything(seed)
    native = train_compact_usad(fit, 30, BATCH, device, hb.UsadModel)

    def native_forward(x, observed):
        flat = torch.where(observed.unsqueeze(-1).bool(), x, median).flatten(1)
        first = native.decoder1(native.encoder(flat))
        cycle = native.decoder2(native.encoder(first))
        return first.reshape_as(x), cycle.reshape_as(x), None

    hb.seed_everything(seed)
    masked = train_masked_usad(fit, median, seed + 31, device, False)
    hb.seed_everything(seed)
    aware = train_masked_usad(fit, median, seed + 31, device, True)
    forwards = {"median_observed": native_forward, "masked_train": masked, "mask_aware": aware}

    @torch.no_grad()
    def residuals(forward, windows, available):
        m = torch.as_tensor(available, dtype=windows.dtype, device=device)
        blocks = []
        for x in windows.split(BATCH):
            initial, cycle, _ = forward(x, m.expand(len(x), -1))
            blocks.append(0.5 * ((x - initial).square() + (x - cycle).square()))
        return torch.cat(blocks).to(torch.float64)

    def scored(residual, scale, available):
        mask = torch.as_tensor(available, dtype=residual.dtype, device=device)
        return torch.log1p(weighted_mean(residual, scale, mask.expand(len(residual), -1))).cpu().numpy()

    def reference(scores):
        center = np.median(scores)
        spread = max(1.4826 * np.median(np.abs(scores - center)), 0.25 * scores.std(), 1e-4)
        return center, spread

    scales, anchors, local = {}, {}, {}
    for name, forward in forwards.items():
        errors = residuals(forward, fit_t, full_mask)
        coordinate_median = np.median(errors.cpu().numpy(), axis=0)
        floor = np.maximum(np.quantile(coordinate_median, 0.25, axis=1, keepdims=True), 1e-6)
        scales[name] = torch.as_tensor(np.maximum(coordinate_median, floor), device=device)
        anchors[name] = reference(scored(errors, scales[name], full_mask))

    def predict(raw, available):
        x = torch.as_tensor(scaler.transform(raw), device=device)
        result = {}
        for name, forward in forwards.items():
            errors = residuals(forward, x, available)
            score = scored(errors, scales[name], available)
            center, spread = anchors[name]
            result[name] = (score - center) / spread
            if name == "masked_train":
                continue
            key = (name, available.tobytes())
            if key not in local:
                if available.all():
                    local[key] = anchors[name]
                else:
                    fit_errors = residuals(forward, fit_t, available)
                    local[key] = reference(scored(fit_errors, scales[name], available))
            location, scale = local[key]
            result[f"{name}_conditional_score"] = (score - location) / scale
        return result

    return predict


def evaluate(dataset, rows, hb, seed, out):
    frame = pd.DataFrame(rows)
    if frame.duplicated(["sample_id", "split", "method", "condition", "replicate"]).any():
        raise ValueError(f"{dataset}: duplicate score key")
    if not np.isfinite(frame.score).all():
        raise ValueError(f"{dataset}: nonfinite score")
    frame.to_csv(out / f"{dataset}_scores.csv.gz", index=False, compression="gzip")
    metrics = []
    for method, method_frame in frame.groupby("method"):
        cal = method_frame[method_frame.split.eq("val")]
        clean = cal[cal.condition.eq("clean")]
        for (condition, rep), test in method_frame[method_frame.split.eq("test")].groupby(["condition", "replicate"]):
            matched = cal[cal.condition.eq(condition) & cal.replicate.eq(rep)]
            ensure = 20 if dataset == "HellBridge" else 6
            if any(len(group) != ensure or not group.label.eq(0).all() for group in (clean, matched)):
                raise ValueError(f"{dataset}: invalid healthy calibration cohort")
            y, scores = test.label.to_numpy(), test.score.to_numpy()
            r05 = recall_at_fpr(y, scores, 0.05)
            for policy, reference in (("clean_fixed", clean), ("condition_matched", matched)):
                threshold = hb.threshold_from_healthy(reference.score.to_numpy(), SPECIFICITY)
                metrics.append({
                    "dataset": dataset, "seed": seed, "method": method,
                    "condition": condition, "replicate": rep, "threshold_mode": policy,
                    "threshold": threshold, **hb.publication_metrics(y, scores, threshold),
                    "recall_at_fpr_05": r05,
                })
    result = pd.DataFrame(metrics)
    if len(result) != 60:
        raise ValueError(f"{dataset}: expected 5 methods x 6 conditions/masks x 2 policies")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hell-h5", required=True, type=Path)
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270901)
    args = parser.parse_args()
    import run_hellbridge_source_baselines as hb
    import run_z24_source_baselines as z24
    from run_hellbridge_tsad_baselines import read_al40
    from run_z24_missingness_baselines import stable_rng

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = out / "hellbridge_split"
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().with_name("prepare_hellbridge_paper_split.py")),
        "--h5", str(args.hell_h5), "--output-dir", str(split), "--profile", "svendsen2022",
        "--regime", "P2_NM_Z", "--seed", "20261113", "--strict-paper-counts",
    ], check=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, cal, test = hb.development_split(manifest)
    raw = hell_features(args.hell_h5, manifest, read_al40)
    print(f"HellBridge seed={args.seed} device={device}", flush=True)
    predict = fit_group(raw[fit], args.seed, device, hb)
    rng = np.random.default_rng(20261119)
    scenarios = [("clean", 0, np.ones(40, dtype=bool))] + [
        ("sensor_0.30", rep, sensor_mask(40, RATE, rng.permutation(40))) for rep in range(1, 6)
    ]
    rows = []
    for ids, role in ((cal, "val"), (test, "test")):
        for condition, rep, mask in scenarios:
            append_scores(rows, predict(raw[ids], mask), manifest.loc[ids, "sample_id"],
                          manifest.loc[ids, "label"], role, condition, rep)
    hell = evaluate("HellBridge", rows, hb, args.seed, out)
    hell.to_csv(out / "HellBridge_metrics_per_mask.csv", index=False)
    del predict, raw, rows

    manifest = z24.load_manifest(args.z24_root)
    rows = []
    for index, ((setup, excitation), channels) in enumerate(sorted(z24.stable_sensor_map(manifest).items())):
        print(f"Z24 setup={setup} excitation={excitation}", flush=True)
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            signal = z24.read_stable_vertical(Path(row.path), channels)
            usable = signal.shape[-1] // LENGTH * LENGTH
            windows = signal[:, :usable].reshape(len(channels), -1, LENGTH).transpose(1, 0, 2)
            return spectral_features(windows)

        fit, _ = z24.split_training_windows([
            load(row) for row in group[group.split.eq("train") & group.label.eq(0)].itertuples()
        ])
        predict = fit_group(fit, args.seed + index * 1009, device, hb)
        for row in group[group.split.eq("test") | (group.split.eq("val") & group.label.eq(0))].itertuples():
            features = load(row)
            clean = [("clean", 0, np.ones(len(channels), dtype=bool))]
            missing = [
                ("sensor_0.30", rep, sensor_mask(
                    len(channels), RATE, stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                )) for rep in range(5)
            ]
            for condition, rep, mask in clean + missing:
                scores = {name: [np.median(score)] for name, score in predict(features, mask).items()}
                append_scores(rows, scores, [row.record_id], [row.label], row.split, condition, rep)
        del predict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.concat([hell, evaluate("Z24", rows, hb, args.seed, out)], ignore_index=True)
    fields = ["auc", "ba", "recall", "fpr", "f1", "recall_at_fpr_05"]
    groups = ["dataset", "condition", "threshold_mode", "method"]
    summary = metrics.groupby(groups, as_index=False)[fields].mean()
    summary.to_csv(out / "summary.csv", index=False)
    metrics.to_csv(out / "metrics_per_mask.csv", index=False)
    pairs = metrics.pivot(index=["dataset", "condition", "threshold_mode", "replicate"], columns="method", values=fields)
    comparisons = []
    for candidate, control in (
        ("masked_train", "median_observed"),
        ("mask_aware", "masked_train"),
        ("mask_aware", "median_observed"),
        ("median_observed_conditional_score", "median_observed"),
        ("mask_aware_conditional_score", "mask_aware"),
        ("mask_aware_conditional_score", "median_observed_conditional_score"),
    ):
        delta = (pairs.xs(candidate, axis=1, level="method")
                 - pairs.xs(control, axis=1, level="method")).reset_index()
        comparisons.append(delta.assign(candidate=candidate, control=control))
    delta = pd.concat(comparisons, ignore_index=True)
    delta.to_csv(out / "paired_deltas_per_mask.csv", index=False)
    delta.groupby(["dataset", "condition", "threshold_mode", "candidate", "control"])[fields].mean().to_csv(
        out / "paired_deltas_summary.csv"
    )
    (out / "protocol.json").write_text(json.dumps({
        "seed": args.seed, "input": "128 shared log-power bins, 0--30 Hz",
        "training_mask_rates": [0, 0.10, 0.20, 0.30], "test_mask_rate": RATE,
        "fit_labels": "healthy only", "score": "upper-tail weighted observed USAD residual",
        "conditional_score": "center and scale from same-mask healthy fit scores",
        "condition_matched_threshold": "healthy validation at same missing rate and replicate",
        "calibration_masks": {
            "HellBridge": "same channel set as test for each replicate",
            "Z24": "same rate and replicate, record-specific channel set",
        },
        "nominal_specificity": SPECIFICITY,
    }, indent=2), encoding="utf-8")
    print(summary[summary.condition.eq("sensor_0.30")].round(4).to_string(index=False))
    print("USAD_MISSINGNESS_ABLATION_PASS", flush=True)


if __name__ == "__main__":
    main()
