"""Paired healthy-only CLMM ablations for HellBridge and Z24."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
import torch
import torch.nn.functional as F


RATES = (0.10, 0.20, 0.30)
SCENARIOS = [("clean", 0, 0.0)] + [
    (f"sensor_{rate:.2f}", rep, rate) for rate in RATES for rep in range(5)
]
OBJECTIVES = {
    "recon_only": (),
    "usad_consistency": ("usad",),
    "multi_consistency": ("usad", "pca_spe"),
}


def sensor_mask(channels, rate, order):
    available = np.ones(channels, dtype=bool)
    if rate:
        count = min(channels - 1, max(1, round(channels * rate)))
        available[order[:count]] = False
    return available


def read_hell_spectrum(h5_path, manifest, read_al40):
    result = np.empty((len(manifest), 40, 128), dtype=np.float32)
    taper = np.hanning(1024).astype(np.float32)
    with h5py.File(h5_path, "r") as h5:
        for record, rows in manifest.groupby("record_id", sort=False):
            signal = read_al40(h5[record])
            for row in rows.itertuples():
                window = signal[:, int(row.start):int(row.stop)]
                if window.shape != (40, 1024):
                    raise ValueError(f"Unexpected window shape {window.shape}: {row.sample_id}")
                fft = np.fft.rfft(detrend(window, axis=-1, type="linear") * taper, axis=-1)
                power = np.abs(fft[:, 1:]) ** 2 / 1024
                result[row.Index] = np.log1p(power).reshape(40, 128, 4).mean(axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite spectral inputs")
    return result


def train_adapter(adapter_class, fit, score_functions, objectives, seed, epochs,
                  device, seed_all, nested_weight=0.0):
    seed_all(seed)
    _, channels, length = fit.shape
    adapter = adapter_class(channels, length, 64, 4).to(device)
    median = torch.from_numpy(np.median(fit, axis=0).astype(np.float32)).to(device)
    values = torch.from_numpy(fit).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-5)
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        targets = {
            name: torch.cat([torch.log1p(score_functions[name](x)) for x in values.split(16)])
            for name in objectives
        }
        scales = {
            name: (torch.quantile(score, 0.75) - torch.quantile(score, 0.25)).clamp_min(1e-4)
            for name, score in targets.items()
        }

    for epoch in range(epochs):
        adapter.train()
        stats = []
        permutation = rng.permutation(len(fit))
        for offset in range(0, len(fit), 16):
            indices = permutation[offset:offset + 16].tolist()
            x = values[indices]
            rates_and_orders = [
                (float(rng.choice(RATES)), rng.permutation(channels)) for _ in indices
            ]
            available = np.stack([sensor_mask(channels, rate, order)
                                  for rate, order in rates_and_orders])
            mask = torch.as_tensor(available, dtype=x.dtype, device=device)
            missing = 1 - mask.unsqueeze(-1)
            residual = adapter((x - median) * mask.unsqueeze(-1), mask)
            prediction = median + residual
            completed = torch.where(mask.unsqueeze(-1).bool(), x, prediction.clamp(0, 1))
            error = F.smooth_l1_loss(prediction, x, reduction="none", beta=0.05)
            reconstruction = (error * missing).sum() / (missing.sum() * length)
            consistency = x.new_zeros(())
            nested = x.new_zeros(())
            if objectives:
                completed_scores = {
                    name: torch.log1p(score_functions[name](completed)) for name in objectives
                }

                def difference(name, target):
                    gap = (completed_scores[name] - target) / scales[name]
                    return F.smooth_l1_loss(gap, torch.zeros_like(gap))

                consistency = torch.stack([difference(name, targets[name][indices])
                                           for name in objectives]).mean()
                if nested_weight:
                    milder = np.stack([
                        sensor_mask(channels, rate / 2 if round(rate * channels) > 1 else 0, order)
                        for rate, order in rates_and_orders
                    ])
                    mild_mask = torch.as_tensor(milder, dtype=x.dtype, device=device)
                    mild_mask = mild_mask.unsqueeze(-1)
                    with torch.no_grad():
                        mild_prediction = (
                            median + adapter((x - median) * mild_mask, mild_mask.squeeze(-1))
                        ).clamp(0, 1)
                        mild_completed = torch.where(mild_mask.bool(), x, mild_prediction)
                        mild_scores = {
                            name: torch.log1p(score_functions[name](mild_completed))
                            for name in objectives
                        }
                    nested = torch.stack([difference(name, mild_scores[name])
                                          for name in objectives]).mean()
            loss = reconstruction + 0.10 * consistency + nested_weight * nested
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            stats.append([float(reconstruction.detach()), float(consistency.detach()),
                          float(nested.detach())])
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            recon, score, nested_score = np.mean(stats, axis=0)
            print(f"  epoch={epoch + 1}/{epochs} recon={recon:.6f} "
                  f"score={score:.6f} nested={nested_score:.6f}", flush=True)
    return adapter.eval(), median


def recall_at_fpr(labels, scores, limit):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return float(tpr[fpr <= limit + 1e-12].max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--hell-h5", required=True, type=Path)
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270901)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    parser.add_argument("--nested-weight", type=float, default=0.05)
    parser.add_argument("--timesnet-z24", action="store_true")
    args = parser.parse_args()
    if args.nested_weight < 0:
        parser.error("--nested-weight must be nonnegative")
    sys.path.insert(0, str(args.source_dir.resolve()))
    import run_hellbridge_source_baselines as hb
    import run_z24_clmm_v21 as v21
    import run_z24_source_baselines as z24
    from run_hellbridge_tsad_baselines import read_al40
    from run_z24_care_usad_v18 import CrossSensorAdapter, complete as legacy_complete
    from run_z24_mdr_v16 import threshold_under_fpr_cap
    from run_z24_missingness_baselines import stable_rng

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_group(raw_fit, raw_reference, seed, batch, hellbridge, use_timesnet):
        normalizer = None if hellbridge else hb.Normalizer.fit(raw_fit)
        normalize = (lambda x: x) if normalizer is None else normalizer.transform
        scaler = hb.FitMinMax(normalize(raw_fit))
        fit = scaler.transform(normalize(raw_fit))
        pca = PCA(n_components=0.95, svd_solver="full").fit(fit.reshape(len(fit), -1))
        mean = torch.as_tensor(pca.mean_, device=device, dtype=torch.float32)
        basis = torch.as_tensor(pca.components_, device=device, dtype=torch.float32)

        def pca_score(x):
            centered = x.flatten(1) - mean
            return (centered - (centered @ basis.T) @ basis).square().mean(1)

        hb.seed_everything(seed)
        usad = hb.fit_usad(fit, 30, batch, device).eval().requires_grad_(False)

        def usad_score(x):
            x = x.flatten(1)
            first = usad.decoder1(usad.encoder(x))
            cycle = usad.decoder2(usad.encoder(first))
            return (0.5 * (x - first).square() + 0.5 * (x - cycle).square()).mean(1)

        scores = {"pca_spe": pca_score, "usad": usad_score}
        if use_timesnet:
            hb.seed_everything(seed)
            model = hb.fit_reconstructor("timesnet", fit, 30, batch, device)
            model.eval().requires_grad_(False)

            def timesnet_score(x):
                return (x - hb.model_forward("timesnet", model, x)).square().mean((1, 2))

            scores["timesnet"] = timesnet_score

        def score_windows(function, windows):
            with torch.no_grad():
                return np.concatenate([
                    function(x.to(device)).cpu().numpy()
                    for x in torch.from_numpy(windows).split(batch)
                ])

        reference = z24.robust_score_reference(score_windows(usad_score, scaler.transform(normalize(raw_reference))))
        hb.seed_everything(seed + 97)
        original = v21.fit_adapter(
            fit, np.square(raw_fit).mean(axis=(0, 2)), usad, reference, 20,
            16, 64, 4, 4, 0.0, 0.10, device, seed + 97,
        )
        adapters = {}
        objectives_list = dict(OBJECTIVES)
        nested_name = "nested_multi" if hellbridge else "nested_usad"
        if args.nested_weight > 0:
            objectives_list[nested_name] = OBJECTIVES[
                "multi_consistency" if hellbridge else "usad_consistency"
            ]
        for name, objectives in objectives_list.items():
            print(f"  FIT {name} seed={seed}", flush=True)
            adapters[name] = train_adapter(
                CrossSensorAdapter, fit, scores, objectives, seed + 97, args.adapter_epochs,
                device, hb.seed_everything,
                nested_weight=args.nested_weight if name == nested_name else 0.0,
            )
        median = np.median(fit, axis=0)
        spec_spread = np.maximum(
            np.maximum(1.4826 * np.median(np.abs(fit - median), axis=0),
                       0.25 * fit.std(axis=0)), 0.05,
        )

        def predict(raw, available):
            x = scaler.transform(normalize(raw))
            inputs = {}
            if available.all():
                inputs = {name: x for name in ("baseline", "clmm_rs", *adapters)}
            else:
                baseline = x.copy()
                # Keep the published Z24 baseline's zero fill in z-score space.
                if hellbridge:
                    baseline[:, ~available] = median[~available]
                else:
                    zeroed = normalize(raw).copy()
                    zeroed[:, ~available] = 0
                    baseline = scaler.transform(zeroed)
                inputs["baseline"] = baseline
                availability = np.broadcast_to(available, x.shape[:2])
                inputs["clmm_rs"], _ = legacy_complete(original, x, availability, batch, device)
                mask = torch.as_tensor(available, dtype=torch.float32, device=device)
                with torch.no_grad():
                    for name, (adapter, origin) in adapters.items():
                        chunks = []
                        for values in torch.from_numpy(x).split(batch):
                            values = values.to(device)
                            m = mask.expand(len(values), -1)
                            prediction = (origin + adapter((values - origin) * m.unsqueeze(-1), m)).clamp(0, 1)
                            chunks.append(torch.where(m.unsqueeze(-1).bool(), values, prediction).cpu().numpy())
                        inputs[name] = np.concatenate(chunks)

            result = {}
            for detector, function in scores.items():
                for variant in inputs:
                    if variant == "clmm_rs" and detector != "usad":
                        continue
                    raw_score = score_windows(function, inputs[variant]).astype(np.float64)
                    if hellbridge:
                        score = raw_score
                    else:
                        # Match the native lower-tail USAD score. For other
                        # detectors, use a fixed two-sided healthy-reference score.
                        key = detector
                        center, spread = score_references[key]
                        normalized = (np.log1p(raw_score) - center) / spread
                        score = -normalized if detector == "usad" else np.abs(normalized)
                    result[detector, variant] = score
            if hellbridge:
                deviation = (x[:, available] - median[available]) / spec_spread[available]
                result["spectrum", "observed_only"] = np.abs(deviation).mean((1, 2))
            return result

        reference_x = scaler.transform(normalize(raw_reference))
        score_references = {} if hellbridge else {
            name: z24.robust_score_reference(score_windows(function, reference_x))
            for name, function in scores.items()
        }
        return predict

    def evaluate(dataset, rows):
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"{dataset}_scores.csv.gz", index=False, compression="gzip")
        metrics = []
        for (detector, variant, condition), cohort in frame.groupby(["detector", "variant", "condition"]):
            calibration = frame[(frame.split == "val") & (frame.detector == detector)]
            calibration = calibration[calibration.variant == variant]
            if dataset == "Z24":
                calibration = calibration[calibration.condition == condition]
                threshold, _ = threshold_under_fpr_cap(
                    calibration.label.to_numpy(), calibration.score.to_numpy(), 0.20,
                )
            for rep, test in cohort[cohort.split == "test"].groupby("replicate"):
                labels, scores = test.label.to_numpy(), test.score.to_numpy()
                if dataset == "HellBridge":
                    clean = calibration[calibration.condition == "clean"]
                    masked = calibration[(calibration.condition == condition) &
                                         (calibration.replicate == rep)]
                    if any(len(block) != 20 or not block.label.eq(0).all()
                           for block in (clean, masked)):
                        raise ValueError("Expected 20 healthy calibration scores per mask")
                    policies = {
                        "clean_fixed": hb.threshold_from_healthy(clean.score.to_numpy(), 0.90),
                        "healthy_maskcal": hb.threshold_from_healthy(masked.score.to_numpy(), 0.90),
                    }
                else:
                    policies = {"labeled_val_fpr20": threshold}
                recall05 = recall_at_fpr(labels, scores, 0.05)
                recall10 = recall_at_fpr(labels, scores, 0.10)
                for policy, cutoff in policies.items():
                    metrics.append({
                        "dataset": dataset, "seed": args.seed, "detector": detector,
                        "variant": variant, "condition": condition, "replicate": rep,
                        "threshold_mode": policy, "threshold": cutoff,
                        **hb.publication_metrics(labels, scores, cutoff),
                        "recall_at_fpr_05": recall05, "recall_at_fpr_10": recall10,
                    })
        return pd.DataFrame(metrics)

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
    fit, cal, test = hb.development_split(manifest)
    print(f"HellBridge seed={args.seed} device={device}", flush=True)
    raw = read_hell_spectrum(args.hell_h5, manifest, read_al40)
    predict = fit_group(raw[fit], raw[fit], args.seed, 16, True, True)
    rng = np.random.default_rng(20261119)
    orders = [rng.permutation(40) for _ in range(5)]
    rows = []
    for indices, role in ((cal, "val"), (test, "test")):
        ids = manifest.loc[indices, "sample_id"].to_numpy()
        labels = manifest.loc[indices, "label"].to_numpy()
        for condition, rep, rate in SCENARIOS:
            available = sensor_mask(40, rate, orders[rep])
            for (detector, variant), scores in predict(raw[indices], available).items():
                rows.extend(
                    {"sample_id": key, "label": int(label), "split": role, "condition": condition,
                     "replicate": rep, "detector": detector, "variant": variant, "score": float(score)}
                    for key, label, score in zip(ids, labels, scores, strict=True)
                )
    hell = evaluate("HellBridge", rows)
    hell.to_csv(out / "HellBridge_metrics.csv", index=False)
    del rows, raw, predict

    manifest = z24.load_manifest(args.z24_root)
    rows = []
    for index, ((setup, excitation), channels) in enumerate(sorted(z24.stable_sensor_map(manifest).items())):
        print(f"Z24 setup={setup} excitation={excitation}", flush=True)
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            signal = z24.read_stable_vertical(Path(row.path), channels)
            return z24.record_windows(signal, 1024, 1024, "raw_temporal", 2, 128)

        healthy = group[(group.split == "train") & (group.label == 0)]
        fit, reference = z24.split_training_windows([load(row) for row in healthy.itertuples()])
        predict = fit_group(fit, reference, args.seed + index * 1009, 32, False, args.timesnet_z24)
        for row in group[group.split.isin(["val", "test"])].itertuples():
            raw = load(row)
            for condition, rep, rate in SCENARIOS:
                order = stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                available = sensor_mask(len(channels), rate, order)
                for (detector, variant), scores in predict(raw, available).items():
                    rows.append({
                        "sample_id": row.record_id, "label": row.label, "split": row.split,
                        "condition": condition, "replicate": rep, "detector": detector,
                        "variant": variant, "score": float(np.median(scores)),
                    })
        del predict
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.concat([hell, evaluate("Z24", rows)], ignore_index=True)
    result.to_csv(out / "metrics_per_mask.csv", index=False)
    fields = ["auc", "ap", "ba", "recall", "fpr", "f1", "recall_at_fpr_05", "recall_at_fpr_10"]
    keys = ["dataset", "detector", "condition", "replicate", "threshold_mode"]
    reference = result[result.variant.eq("baseline")][keys + fields]
    paired = result.merge(reference, on=keys, suffixes=("", "_baseline"), validate="many_to_one")
    deltas = [f"delta_{field}" for field in fields]
    for metric, delta in zip(fields, deltas):
        paired[delta] = paired[metric] - paired[f"{metric}_baseline"]
    paired.to_csv(out / "paired_vs_baseline_per_mask.csv", index=False)
    groups = ["dataset", "detector", "condition", "variant", "threshold_mode"]
    summary = result.groupby(groups)[fields].mean()
    summary.to_csv(out / "summary.csv")
    paired.groupby(groups)[deltas].mean().to_csv(out / "paired_delta_summary.csv")
    parent_names = {"nested_multi": "multi_consistency", "nested_usad": "usad_consistency"}
    comparisons = []
    for nested, parent in parent_names.items():
        candidate = result[result.variant.eq(nested)][keys + fields]
        if candidate.empty:
            continue
        control = result[result.variant.eq(parent)][keys + fields]
        matched = candidate.merge(control, on=keys, suffixes=("", "_parent"), validate="one_to_one")
        if len(matched) != len(candidate):
            raise ValueError(f"Incomplete paired comparison for {nested}")
        comparison = matched[keys].copy()
        comparison["variant"] = nested
        comparison["parent"] = parent
        for metric, delta in zip(fields, deltas):
            comparison[delta] = matched[metric] - matched[f"{metric}_parent"]
        comparisons.append(comparison)
    if comparisons:
        comparison = pd.concat(comparisons, ignore_index=True)
        comparison.to_csv(out / "nested_vs_parent_per_mask.csv", index=False)
        comparison.groupby(groups + ["parent"])[deltas].mean().to_csv(out / "nested_vs_parent_summary.csv")
    (out / "protocol.json").write_text(json.dumps({
        "seed": args.seed, "adapter_epochs": args.adapter_epochs,
        "adapter_objectives": OBJECTIVES, "loss_weight": 0.10,
        "nested_weight": args.nested_weight,
        "nested_hellbridge": "USAD and PCA-SPE, same healthy window under nested masks",
        "nested_z24": "USAD, same healthy window under nested masks",
        "training_sensor_loss": RATES,
        "threshold_hellbridge": "clean-fixed and mask-matched healthy calibration, specificity target 0.90",
        "threshold_z24": "condition-matched labeled validation, FPR cap 0.20",
        "z24_usad_score": "lower-tail", "z24_other_scores": "two-sided",
        "hellbridge_input": "1024-sample log-power spectrum pooled to 128 bins",
        "z24_input": "native temporal 1024 samples downsampled to 512",
    }, indent=2), encoding="utf-8")
    print(summary.xs("sensor_0.30", level="condition")[["auc", "ba", "recall", "fpr", "recall_at_fpr_05"]].round(4))
    print("CLMM_MULTI_DETECTOR_PASS", flush=True)


if __name__ == "__main__":
    main()
