"""One locked CLMM configuration for two bridge benchmarks."""

import argparse
from functools import partial
import json
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch
from sklearn.decomposition import PCA
import torch

from run_clmm_multi_detector import recall_at_fpr, sensor_mask, train_adapter


FS = 100.0
LENGTH = 1024
FREQUENCY_LIMIT = 30.0
BINS = 128
BATCH = 16
RATE = 0.30
SPECIFICITY = 0.90
BACKBONES = ("usad", "pca_spe")
FREQUENCIES = np.fft.rfftfreq(LENGTH, d=1 / FS)[1:]
EDGES = np.searchsorted(FREQUENCIES, np.linspace(0, FREQUENCY_LIMIT, BINS + 1))
TAPER = np.hanning(LENGTH).astype(np.float32)
REPRESENTATIONS = ("log_power", "relative_power", "welch_log_power", "welch_relative_power")


def spectral_features(windows, representation="log_power"):
    """Same physical frequency bins and transform for both bridges."""
    if windows.ndim != 3 or windows.shape[-1] != LENGTH:
        raise ValueError("Expected [window, channel, 1024]")
    windows = detrend(windows, axis=-1, type="linear")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown spectral representation: {representation}")
    if representation.startswith("welch_"):
        _, density = welch(windows, fs=FS, nperseg=512, noverlap=256,
                           nfft=LENGTH, detrend=False, axis=-1)
        # Convert positive-frequency density to the original periodogram scale.
        power = density[..., 1:EDGES[-1] + 1] * (FS * np.sum(TAPER**2) / (2 * LENGTH))
    else:
        fft = np.fft.rfft(windows * TAPER, axis=-1)[..., 1:EDGES[-1] + 1]
        power = np.abs(fft) ** 2 / LENGTH
    if representation.endswith("relative_power"):
        # Unit total in-band energy per channel. Scaling keeps mean bin power near one.
        energy = power.sum(axis=-1, keepdims=True)
        power = np.divide(power, energy, out=np.zeros_like(power), where=energy > 0)
        power *= power.shape[-1]
    log_power = np.log1p(power)
    sums = np.concatenate([
        np.zeros((*log_power.shape[:-1], 1), dtype=log_power.dtype),
        np.cumsum(log_power, axis=-1),
    ], axis=-1)
    pooled = np.diff(sums[..., EDGES], axis=-1) / np.diff(EDGES)
    if not np.isfinite(pooled).all():
        raise ValueError("Nonfinite common spectral features")
    return pooled.astype(np.float32)


def hell_features(h5_path, manifest, read_al40, transform=spectral_features):
    features = np.empty((len(manifest), 40, BINS), dtype=np.float32)
    with h5py.File(h5_path, "r") as h5:
        for record, rows in manifest.groupby("record_id", sort=False):
            values = read_al40(h5[record])
            windows = np.stack([values[:, int(r.start):int(r.stop)] for r in rows.itertuples()])
            features[rows.index] = transform(windows)
    return features


def fit_ppca(fit, device):
    """Fit a low-rank healthy Gaussian and condition it on observed sensors."""
    flat = fit.reshape(len(fit), -1).astype(np.float64)
    if len(flat) < 2:
        raise ValueError("PPCA requires at least two healthy fit windows")
    rank = min(8, len(flat) - 1, flat.shape[1] - 1)
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=0).fit(flat)
    total_variance = flat.var(axis=0, ddof=1).sum()
    noise = max((total_variance - pca.explained_variance_.sum()) / (flat.shape[1] - rank), 1e-6)
    loadings = pca.components_.T * np.sqrt(np.maximum(pca.explained_variance_ - noise, 0))
    mean = pca.mean_
    identity = np.eye(rank, dtype=flat.dtype)

    @torch.no_grad()
    def infer(windows, available):
        # Keep the small covariance solve on CPU so it does not depend on CUDA solver kernels.
        observed = np.repeat(available, windows.shape[-1])
        if not observed.any():
            raise ValueError("PPCA requires at least one observed channel")
        values = windows.flatten(1).cpu().numpy().astype(flat.dtype)
        observed_basis = loadings[observed]
        delta = values[:, observed] - mean[observed]
        precision = observed_basis.T @ observed_basis + noise * identity
        latent = np.linalg.solve(precision, (delta @ observed_basis).T).T
        # Observed marginal Mahalanobis distance via the Woodbury identity.
        residual = delta - latent @ observed_basis.T
        distance = (np.square(residual).sum(1) / noise + np.square(latent).sum(1)) / delta.shape[1]
        values[:, ~observed] = np.clip(mean[~observed] + latent @ loadings[~observed].T, 0, 1)
        filled = torch.as_tensor(values.reshape(windows.shape), dtype=windows.dtype, device=device)
        return distance, filled

    return infer


def fit_group(raw_fit, seed, device, hb, adapter_class, backbone,
              weighted_upper=False, observed_residual=False, use_ppca=False):
    scaler = hb.FitMinMax(raw_fit)
    fit = scaler.transform(raw_fit)
    median = torch.as_tensor(np.median(fit, axis=0), dtype=torch.float32, device=device)
    hb.seed_everything(seed)
    if backbone == "pca_spe":
        pca = PCA(n_components=0.95, svd_solver="full").fit(fit.reshape(len(fit), -1))
        mean = torch.as_tensor(pca.mean_, dtype=torch.float32, device=device)
        basis = torch.as_tensor(pca.components_, dtype=torch.float32, device=device)

        def detector(x):
            centered = x.flatten(1) - mean
            return (centered - (centered @ basis.T) @ basis).square().mean(1)

    else:
        if weighted_upper:
            # Import lazily: the clean runner imports our spectral transform.
            from run_usad_clean_two_bridges import train_compact_usad
            source = train_compact_usad(fit, 30, BATCH, device, hb.UsadModel)
        else:
            source = hb.fit_usad(fit, 30, BATCH, device).eval().requires_grad_(False)

        def usad_residual(x):
            flat = x.flatten(1)
            first = source.decoder1(source.encoder(flat))
            cycle = source.decoder2(source.encoder(first))
            return (0.5 * ((flat - first).square() + (flat - cycle).square())).reshape_as(x)

        if weighted_upper:
            with torch.no_grad():
                residual = np.concatenate([
                    usad_residual(batch.to(device)).cpu().numpy()
                    for batch in torch.from_numpy(fit).split(BATCH)
                ]).astype(np.float64)
            scale = np.median(residual, axis=0)
            floor = np.maximum(np.quantile(scale, 0.25, axis=1, keepdims=True), 1e-6)
            coordinate_scale = torch.as_tensor(np.maximum(scale, floor), device=device)

            def weighted_errors(x):
                return usad_residual(x) / coordinate_scale

            def detector(x):
                return weighted_errors(x).mean((1, 2))

        else:
            def detector(x):
                return usad_residual(x).mean((1, 2))

    adapter, _ = train_adapter(
        adapter_class, fit, {backbone: detector}, (backbone,), seed + 97,
        20, device, hb.seed_everything,
    )

    @torch.no_grad()
    def score_raw(values):
        return np.concatenate([
            torch.log1p(detector(batch)).cpu().numpy() for batch in values.split(BATCH)
        ]).astype(float)

    reference = score_raw(torch.as_tensor(fit, device=device))
    center = float(np.median(reference))
    spread = max(1.4826 * np.median(np.abs(reference - center)), 0.25 * reference.std(), 1e-4)

    def orient(score):
        normalized = (score - center) / spread
        return normalized if weighted_upper else np.abs(normalized)

    @torch.no_grad()
    def score_all_and_observed(values, available):
        indices = torch.as_tensor(np.flatnonzero(available), dtype=torch.long, device=device)
        if not indices.numel():
            raise ValueError("Observed-only scoring requires at least one channel")
        full, observed = [], []
        for batch in values.split(BATCH):
            errors = weighted_errors(batch)
            full.append(torch.log1p(errors.mean((1, 2))).cpu().numpy())
            observed.append(torch.log1p(errors.index_select(1, indices).mean((1, 2))).cpu().numpy())
        return orient(np.concatenate(full)), orient(np.concatenate(observed))

    if use_ppca:
        ppca = fit_ppca(fit, device)
        full = np.ones(fit.shape[1], dtype=bool)
        reference, _ = ppca(torch.as_tensor(fit, device=device), full)
        reference = np.log1p(reference)
        ppca_center = np.median(reference)
        ppca_spread = max(1.4826 * np.median(np.abs(reference - ppca_center)),
                          0.25 * reference.std(), 1e-4)

    @torch.no_grad()
    def predict(raw, available):
        x = torch.as_tensor(scaler.transform(raw), device=device)
        mask = torch.as_tensor(available, device=device).view(1, -1, 1)
        median_filled = torch.where(mask, x, median)
        if available.all():
            clmm_filled = x
        else:
            chunks = []
            for batch in x.split(BATCH):
                observed = mask.squeeze(-1).expand(len(batch), -1).to(batch.dtype)
                residual = adapter((batch - median) * mask, observed)
                candidate = (median + residual).clamp(0, 1)
                chunks.append(torch.where(mask, batch, candidate))
            clmm_filled = torch.cat(chunks)
        if observed_residual:
            if available.all():
                clean = orient(score_raw(x))
                result = {name: clean.copy() for name in (
                    "median", "clmm", "median_observed", "clmm_observed",
                )}
            else:
                median_full, median_observed = score_all_and_observed(median_filled, available)
                clmm_full, clmm_observed = score_all_and_observed(clmm_filled, available)
                result = {
                    "median": median_full, "clmm": clmm_full,
                    "median_observed": median_observed, "clmm_observed": clmm_observed,
                }
            if use_ppca:
                distance, filled = ppca(x, available)
                result["ppca_observed"] = (np.log1p(distance) - ppca_center) / ppca_spread
                result["ppca_usad_observed"] = (
                    clean.copy() if available.all() else score_all_and_observed(filled, available)[1]
                )
            return result
        control = orient(score_raw(median_filled))
        proposed = control.copy() if available.all() else orient(score_raw(clmm_filled))
        return {"median": control, "clmm": proposed}

    return predict


def robust_healthy_threshold(calibration, specificity):
    center = np.median(calibration)
    spread = 1.4826 * np.median(np.abs(calibration - center))
    if spread <= 0:
        return None
    return float(np.nextafter(center + NormalDist().inv_cdf(specificity) * spread, np.inf))


def evaluate(dataset, rows, hb, seed, out, observed_residual=False, use_ppca=False):
    scores = pd.DataFrame(rows)
    keys = ["sample_id", "condition", "replicate", "method"]
    if scores.duplicated(keys).any() or not np.isfinite(scores.score).all():
        raise ValueError(f"{dataset}: duplicated or nonfinite scores")
    scores.to_csv(out / f"{dataset}_scores.csv.gz", compression="gzip", index=False)
    results, calibration_results, threshold_audit = [], [], []
    for method, frame in scores.groupby("method"):
        calibration = frame.loc[
            frame.split.eq("val") & frame.label.eq(0) & frame.condition.eq("clean"), "score"
        ].to_numpy(dtype=float)
        if len(calibration) != (20 if dataset == "HellBridge" else 6):
            raise ValueError(f"{dataset}: unexpected healthy calibration size")
        threshold = hb.threshold_from_healthy(calibration, SPECIFICITY)
        robust = robust_healthy_threshold(calibration, SPECIFICITY) if use_ppca else None
        if use_ppca:
            cal_rows = frame.loc[frame.split.eq("val") & frame.label.eq(0) & frame.condition.eq("clean")]
            for index, row in enumerate(cal_rows.itertuples()):
                remaining = np.delete(calibration, index)
                threshold_audit.append({
                    "dataset": dataset, "method": method, "sample_id": row.sample_id,
                    "score": row.score, "quantile": threshold,
                    "quantile_without_record": hb.threshold_from_healthy(remaining, SPECIFICITY),
                    "robust": robust,
                    "robust_without_record": robust_healthy_threshold(remaining, SPECIFICITY),
                })
        for (condition, rep), test in frame[frame.split.eq("test")].groupby(["condition", "replicate"]):
            labels = test.label.to_numpy(dtype=int)
            values = test.score.to_numpy(dtype=float)
            results.append({
                "dataset": dataset, "seed": seed, "method": method,
                "condition": condition, "replicate": rep, "threshold": threshold,
                **hb.publication_metrics(labels, values, threshold),
                "recall_at_fpr_05": recall_at_fpr(labels, values, 0.05),
                "recall_at_fpr_10": recall_at_fpr(labels, values, 0.10),
            })
            if use_ppca:
                calibration_results.append({
                    **results[-1], "threshold_mode": "empirical_q90",
                    "calibration_fpr": float((calibration >= threshold).mean()),
                })
                if robust is not None:
                    calibration_results.append({
                        **results[-1], **hb.publication_metrics(labels, values, robust),
                        "threshold": robust, "threshold_mode": "median_mad",
                        "calibration_fpr": float((calibration >= robust).mean()),
                    })
    metrics = pd.DataFrame(results)
    expected = 36 if use_ppca else (24 if observed_residual else 12)
    if len(metrics) != expected:
        raise ValueError(f"{dataset}: expected {expected} metric rows")
    metrics.to_csv(out / f"{dataset}_metrics_per_mask.csv", index=False)
    if use_ppca:
        pd.DataFrame(calibration_results).to_csv(out / f"{dataset}_threshold_comparison.csv", index=False)
        pd.DataFrame(threshold_audit).to_csv(out / f"{dataset}_healthy_threshold_audit.csv", index=False)
    return metrics


def append_scores(rows, results, ids, labels, role, condition, rep):
    for method, scores in results.items():
        rows.extend(
            {"sample_id": uid, "label": int(label), "split": role, "method": method,
             "condition": condition, "replicate": rep, "score": float(score)}
            for uid, label, score in zip(ids, labels, scores, strict=True)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hell-h5", required=True, type=Path)
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270901)
    parser.add_argument("--detector", choices=BACKBONES, default="usad")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="log_power")
    parser.add_argument("--weighted-upper", action="store_true",
                        help="Use compact USAD with fit-only weighted residual and upper-tail scoring")
    parser.add_argument("--observed-residual", action="store_true",
                        help="Add observed-channel scoring for both median and CLMM completion")
    parser.add_argument("--ppca", action="store_true",
                        help="Add observed PPCA scoring and PPCA completion before USAD")
    args = parser.parse_args()
    if args.weighted_upper and args.detector != "usad":
        parser.error("--weighted-upper requires --detector usad")
    if args.observed_residual and not args.weighted_upper:
        parser.error("--observed-residual requires --weighted-upper")
    if args.ppca and not args.observed_residual:
        parser.error("--ppca requires --observed-residual")
    import run_hellbridge_source_baselines as hb
    import run_z24_source_baselines as z24
    from run_hellbridge_tsad_baselines import read_al40
    from run_z24_care_usad_v18 import CrossSensorAdapter
    from run_z24_missingness_baselines import stable_rng

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rule = "weighted_upper" if args.weighted_upper else "native_two_sided"
    transform = partial(spectral_features, representation=args.representation)
    print(f"Shared configuration: {args.detector}/{rule} | 0--30 Hz / 128 bins | seed={args.seed} | {device}", flush=True)
    print(f"Representation: {args.representation}", flush=True)
    print(f"Observed-residual ablation: {args.observed_residual}", flush=True)

    split = out / "hellbridge_split"
    prepare = Path(__file__).resolve().with_name("prepare_hellbridge_paper_split.py")
    subprocess.run([
        sys.executable, str(prepare), "--h5", str(args.hell_h5), "--output-dir", str(split),
        "--profile", "svendsen2022", "--regime", "P2_NM_Z", "--seed", "20261113",
        "--strict-paper-counts",
    ], check=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(
        ["record_id", "start"]
    ).reset_index(drop=True)
    fit, cal, test = hb.development_split(manifest)
    raw = hell_features(args.hell_h5, manifest, read_al40, transform)
    predictor = fit_group(raw[fit], args.seed, device, hb, CrossSensorAdapter,
                          args.detector, args.weighted_upper, args.observed_residual, args.ppca)
    rng = np.random.default_rng(20261119)
    hell_masks = [sensor_mask(40, RATE, rng.permutation(40)) for _ in range(5)]
    rows = []
    for indices, role in ((cal, "val"), (test, "test")):
        ids = manifest.loc[indices, "sample_id"].to_numpy()
        labels = manifest.loc[indices, "label"].to_numpy()
        append_scores(rows, predictor(raw[indices], np.ones(40, dtype=bool)), ids, labels, role, "clean", 0)
        if role == "test":
            for rep, mask in enumerate(hell_masks, 1):
                append_scores(rows, predictor(raw[indices], mask), ids, labels, role, "sensor_0.30", rep)
    hell = evaluate("HellBridge", rows, hb, args.seed, out, args.observed_residual, args.ppca)
    del predictor, raw, rows

    manifest = z24.load_manifest(args.z24_root)
    rows = []
    for index, ((setup, excitation), channels) in enumerate(sorted(z24.stable_sensor_map(manifest).items())):
        print(f"Z24 setup={setup} excitation={excitation}", flush=True)
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            signal = z24.read_stable_vertical(Path(row.path), channels)
            usable = signal.shape[-1] // LENGTH * LENGTH
            if usable < LENGTH:
                raise ValueError(f"No complete windows: {row.record_id}")
            windows = signal[:, :usable].reshape(len(channels), -1, LENGTH).transpose(1, 0, 2)
            return transform(windows)

        healthy = group[(group.split == "train") & group.label.eq(0)]
        fit, _ = z24.split_training_windows([load(row) for row in healthy.itertuples()])
        predictor = fit_group(fit, args.seed + index * 1009, device, hb, CrossSensorAdapter,
                              args.detector, args.weighted_upper, args.observed_residual, args.ppca)
        for row in group[group.split.eq("test") | (group.split.eq("val") & group.label.eq(0))].itertuples():
            features = load(row)
            scenarios = [("clean", 0, np.ones(len(channels), dtype=bool))]
            if row.split == "test":
                scenarios.extend(
                    ("sensor_0.30", rep, sensor_mask(
                        len(channels), RATE, stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                    )) for rep in range(5)
                )
            for condition, rep, mask in scenarios:
                record_scores = {name: [np.median(scores)] for name, scores in predictor(features, mask).items()}
                append_scores(rows, record_scores, [row.record_id], [row.label], row.split, condition, rep)
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined = pd.concat([
        hell, evaluate("Z24", rows, hb, args.seed, out, args.observed_residual, args.ppca),
    ], ignore_index=True)
    combined.insert(1, "detector", args.detector)
    combined.to_csv(out / "metrics_per_mask.csv", index=False)
    fields = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_at_fpr_05", "recall_at_fpr_10"]
    groups = ["dataset", "detector", "condition", "method"]
    summary = combined.groupby(groups)[fields].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    pairs = combined.pivot(index=["dataset", "condition", "replicate"], columns="method", values=fields)
    delta = pairs.xs("clmm", axis=1, level="method") - pairs.xs("median", axis=1, level="method")
    delta.to_csv(out / "clmm_delta_per_mask.csv")
    delta.groupby(level=["dataset", "condition"]).mean().to_csv(out / "clmm_delta_summary.csv")
    if args.observed_residual:
        comparisons = []
        for candidate, control in (
            ("median_observed", "median"),
            ("clmm_observed", "clmm"),
            ("clmm_observed", "median_observed"),
            ("clmm_observed", "median"),
        ):
            change = pairs.xs(candidate, axis=1, level="method") - pairs.xs(control, axis=1, level="method")
            change = change.reset_index()
            change["candidate"] = candidate
            change["control"] = control
            comparisons.append(change)
        observed_delta = pd.concat(comparisons, ignore_index=True)
        observed_delta.to_csv(out / "observed_delta_per_mask.csv", index=False)
        observed_delta.groupby(["dataset", "condition", "candidate", "control"])[
            fields
        ].mean().to_csv(out / "observed_delta_summary.csv")
    (out / "protocol.json").write_text(json.dumps({
        "seed": args.seed, "sampling_rate_hz": FS, "window_samples": LENGTH,
        "frequency_limit_hz": FREQUENCY_LIMIT, "frequency_bins": BINS,
        "representation": args.representation,
        "fit": "healthy-only", "detector": args.detector,
        "score_rule": rule,
        "observed_residual_ablation": args.observed_residual,
        "ppca": args.ppca,
        "ppca_rank_limit": 8 if args.ppca else None,
        "ppca_solver": "NumPy CPU" if args.ppca else None,
        "threshold_sensitivity": "healthy clean median + Gaussian q90 times scaled MAD" if args.ppca else None,
        "observed_residual_rule": "mean weighted residual over observed channels, same clean reference",
        "score_direction": "upper-tail" if args.weighted_upper else "two-sided",
        "usad_hidden_widths": [512, 256, 64] if args.weighted_upper else None,
        "threshold": "healthy-only clean, fixed across test masks",
        "target_calibration_specificity": SPECIFICITY,
        "evaluation_units": {"HellBridge": "window", "Z24": "record median"},
    }, indent=2), encoding="utf-8")
    print(summary[["dataset", "condition", "method", "auc", "ba", "recall", "fpr"]].round(4).to_string(index=False))
    print("CLMM_SHARED_CONFIGURATION_PASS", flush=True)


if __name__ == "__main__":
    main()
