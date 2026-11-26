"""One-seed native-resolution HellBridge spectral scoring and threshold pilot."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from .mask_pair_block import evaluate, run_hellbridge
from .metrics import publication_metrics, recall_at_fpr
from .post_block_head import GainHeadAudit, gain_deviation
from .timesnet_native_1024 import LENGTH, SEED


METHODS = ("timesnet_native", "adapter_mse", "clmm_gain_old", "spectral_residual", "observed_spectrum", "clmm_spectral")
SEGMENT_METHODS = ("clmm_spectral_bounded_signed", "clmm_spectral_segment_consensus")
WHITENING_METHODS = (
    "clmm_gain_spectral", "clmm_spectral_bounded_positive",
    "clmm_diagonal_residual", "clmm_whitened_residual",
)
BANDS_HZ = ((2, 6), (6, 12), (12, 20), (20, 32), (32, 45))
SAMPLE_HZ = 100
SEGMENT, STRIDE = 512, 256
SPECTRAL_WEIGHT = .25
MASKS = tuple(range(300, 305))
OUTPUT = Path("/kaggle/working/clmm_hellbridge_spectral_1024_seed20270905")
SEGMENT_OUTPUT = OUTPUT.with_name("clmm_hellbridge_segment_consensus_1024_seed20270905")
WHITENING_OUTPUT = OUTPUT.with_name("clmm_hellbridge_whitened_1024_seed20270905")
INPUT = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")
CORRELATION_SHRINKAGE = .50


def scoring_methods(segment_consensus, residual_whitening):
    methods = (*METHODS, *SEGMENT_METHODS) if segment_consensus else METHODS
    return (*methods, *WHITENING_METHODS) if residual_whitening else methods


def segment_spectral_power(values, hann):
    segments = values.unfold(-1, SEGMENT, STRIDE) * hann
    return torch.fft.rfft(segments, dim=-1).abs().square()


def band_shape(power, in_band):
    bands = (power @ in_band).clamp_min(1e-10)
    return bands / bands.sum(-1, keepdim=True)


def spectral_distance(a, b):
    midpoint = (a + b) / 2
    return ((a * (a / midpoint).log() + b * (b / midpoint).log()).sum(-1) / 2).clamp_min(0)


def spectrum_statistics(observed, predicted, hann, in_band, segment_consensus):
    actual, fitted = (segment_spectral_power(x, hann) for x in (observed, predicted))
    mean_psd, mean_pred = (band_shape(x.mean(-2), in_band) for x in (actual, fitted))
    conventional = spectral_distance(mean_psd, mean_pred).mean(-1)
    if not segment_consensus:
        return conventional, mean_psd, None
    actual_bands, fitted_bands = (band_shape(x, in_band) for x in (actual, fitted))
    per_segment = spectral_distance(actual_bands, fitted_bands)
    # Take the median over segments per sensor before averaging retained sensors.
    persistent = per_segment.median(dim=-1).values.mean(-1)
    return conventional, mean_psd, persistent


def healthy_scale(values):
    center = np.median(values)
    scale = max(1.4826 * np.median(np.abs(values - center)), .25 * values.std(), 1e-5)
    return center, scale


def fit_residual_whitening(residual):
    """Fit a regularized cross-sensor metric to healthy, masked reconstruction residuals."""
    samples = residual.transpose(1, 2).reshape(-1, residual.size(1))
    center = samples.mean(0)
    centered = samples - center
    deviation = centered.square().mean(0).clamp_min(1e-8).sqrt()
    standardized = centered / deviation
    corr = standardized.T @ standardized / len(samples)
    identity = torch.eye(len(center), device=corr.device, dtype=corr.dtype)
    regularized = (1 - CORRELATION_SHRINKAGE) * corr + CORRELATION_SHRINKAGE * identity
    return center, deviation, torch.linalg.cholesky(regularized)


def residual_energies(residual, whitening):
    center, deviation, factor = whitening
    standardized = (residual.transpose(1, 2) - center) / deviation
    flat = standardized.reshape(-1, len(center))
    whitened = torch.linalg.solve_triangular(factor, flat.T.contiguous(), upper=False).T
    return torch.stack((
        standardized.square().mean((1, 2)),
        whitened.reshape_as(standardized).square().mean((1, 2)),
    ), dim=-1).log1p()


class SpectralGainAudit(GainHeadAudit):
    columns = METHODS

    def fit(self, raw, seed, args, group):
        super().fit(raw, seed, args, group)
        self.residual_whitening = getattr(args, "residual_whitening", False)
        if self.residual_whitening:
            (healthy_fit,) = self.prepare(raw)
            self.healthy_fit = torch.as_tensor(healthy_fit, device=self.detector.device)

    def reference_audit(self, reference, group, args):
        super().reference_audit(reference, group, args)
        self.segment_consensus = args.segment_consensus
        self.columns = scoring_methods(self.segment_consensus, self.residual_whitening)
        device = self.detector.device
        self.hann = torch.hann_window(SEGMENT, device=device)
        f = torch.fft.rfftfreq(SEGMENT, d=1 / SAMPLE_HZ, device=device)
        self.in_band = torch.stack([(f >= lo) & (f < hi) for lo, hi in BANDS_HZ], dim=1).float()
        (healthy,) = self.prepare(reference)
        self.healthy = torch.as_tensor(healthy, device=device)
        self.spectral_references = {}
        self.residual_references = {}

    def components(self, values, mask):
        observed = values[:, mask]
        off, _ = self.predict_observed(values, mask, enabled=False)
        on, _ = self.predict_observed(values, mask, enabled=True) if not mask.all().item() else (off, ())
        e_off, e_on = (observed.sub(reconstruction).square() for reconstruction in (off, on))
        raw_off = e_off.mean((1, 2)).cpu().numpy().astype(np.float64)
        raw_on = e_on.mean((1, 2)).cpu().numpy().astype(np.float64)
        gain = (np.log1p(e_off.mean(-1).cpu().numpy().astype(np.float64))
                - np.log1p(e_on.mean(-1).cpu().numpy().astype(np.float64)))
        spectral, psd, persistent = spectrum_statistics(observed, on, self.hann, self.in_band, self.segment_consensus)
        spectral = spectral.cpu().numpy().astype(np.float64)
        consensus = persistent.cpu().numpy().astype(np.float64) if persistent is not None else None
        residual = observed - on if self.residual_whitening and not mask.all().item() else None
        return raw_off, raw_on, gain, spectral, psd, consensus, residual

    def matched_residual_reference(self, key, mask, traces):
        if key not in self.residual_references:
            errors = []
            for chunk in self.healthy_fit.split(self.detector.batch_size):
                predicted, _ = self.predict_observed(chunk, mask, enabled=True)
                errors.append(chunk[:, mask] - predicted)
            whitening = fit_residual_whitening(torch.cat(errors))
            energies = torch.cat([residual_energies(trace[6], whitening) for trace in traces]).cpu().numpy()
            centers, scales = zip(*(healthy_scale(column) for column in energies.T), strict=True)
            self.residual_references[key] = (whitening, np.array(centers), np.array(scales))
        return self.residual_references[key]

    def matched_reference(self, kept, mask):
        key = tuple(kept)
        if key not in self.spectral_references:
            traces = [self.components(chunk, mask) for chunk in self.healthy.split(self.detector.batch_size)]
            distance = np.concatenate([x[3] for x in traces])
            template = torch.cat([x[4] for x in traces]).mean(0)
            consensus_ref = healthy_scale(np.concatenate([x[5] for x in traces])) if self.segment_consensus else None
            if self.residual_whitening and not all(key):
                self.matched_residual_reference(key, mask, traces)
            self.spectral_references[key] = (*healthy_scale(distance), template, consensus_ref)
        return self.spectral_references[key]

    @torch.no_grad()
    def scores(self, data, kept):
        device = self.detector.device
        values = torch.as_tensor(data, device=device)
        mask = torch.as_tensor(kept, device=device)
        missing = int((~kept).sum())
        center, scale = self.detector.reference_center, self.detector.reference_scale
        spectral_center, spectral_scale, template, consensus_ref = self.matched_reference(kept, mask)
        result = []
        for chunk in values.split(self.detector.batch_size):
            off, on, gain, spectral, psd, consensus, residual = self.components(chunk, mask)
            base_off, base_on = (np.log1p(x) / scale - center / scale for x in (off, on))
            previous_gain = (gain_deviation(gain, self.centers[missing][kept], self.scale[kept])
                             if missing else np.zeros(len(chunk)))
            spectrum_only = spectral_distance(psd, template).mean(-1).cpu().numpy()
            z_spectral = (spectral - spectral_center) / spectral_scale
            combined = base_on + SPECTRAL_WEIGHT * z_spectral
            legacy = base_on + self.weight * previous_gain
            scores = [base_off, base_on, legacy, spectral, spectrum_only, combined]
            if self.segment_consensus:
                consensus_center, consensus_scale = consensus_ref
                z_consensus = (consensus - consensus_center) / consensus_scale
                for z in (z_spectral, z_consensus):
                    scores.append(legacy + SPECTRAL_WEIGHT * np.tanh(z) if missing else legacy)
            if self.residual_whitening:
                if missing:
                    whitening, ref_center, ref_scale = self.residual_references[tuple(kept)]
                    energies = residual_energies(residual, whitening).cpu().numpy()
                    calibrated = (energies - ref_center) / ref_scale
                    gain_bonus = self.weight * previous_gain
                    scores.extend((
                        legacy + SPECTRAL_WEIGHT * z_spectral,
                        legacy + SPECTRAL_WEIGHT * np.tanh(np.maximum(z_spectral, 0)),
                        calibrated[:, 0] + gain_bonus,
                        calibrated[:, 1] + gain_bonus,
                    ))
                else:
                    scores.extend((legacy,) * len(WHITENING_METHODS))
            result.append(np.column_stack(scores))
        return np.concatenate(result)


def calibration_cutoff(healthy, alpha):
    values = np.sort(np.asarray(healthy, dtype=float))
    count = int(np.floor((len(values) + 1) * alpha))
    return float(np.nextafter(values[-count], np.inf)) if count else float("inf")


def calibration_report(saved_scores, methods=METHODS):
    scores = pd.read_csv(saved_scores, float_precision="round_trip")
    grouped = ["condition", "replicate"]
    clean = scores.loc[scores.split.eq("val") & scores.condition.eq("clean")]
    if len(clean) != 20 or not clean.label.eq(0).all():
        raise ValueError("Expected 20 healthy clean calibration windows")
    alerts, state_rows = [], []
    for (condition, replicate), part in scores.groupby(grouped):
        cal = part.loc[part.split.eq("val")]
        test = part.loc[part.split.eq("test")]
        if (set(cal.sample_id) != set(clean.sample_id) or len(cal) != 20 or not cal.label.eq(0).all()
                or test.label.value_counts().to_dict() != {0: 80, 1: 640}):
            raise ValueError(f"Invalid calibration or test cohort for {condition}/{replicate}")
        y = test.label.to_numpy()
        for method in methods:
            values = test[method].to_numpy()
            for policy, ref in (("clean_calibration", clean), ("same_mask_calibration", cal)):
                for alpha in (.05, .10):
                    threshold = calibration_cutoff(ref[method], alpha)
                    meta = dict(condition=condition, replicate=replicate, method=method, policy=policy,
                                alpha=alpha, threshold=threshold)
                    alerts.append(dict(**meta, recall_at_fpr_05=recall_at_fpr(y, values, .05),
                                       **publication_metrics(y, values, threshold)))
                    for state, subset in test.groupby("state"):
                        state_rows.append(dict(**meta, state=state, windows=len(subset),
                                               alarm_rate=subset[method].ge(threshold).mean()))
    return pd.DataFrame(alerts), pd.DataFrame(state_rows)


def write_calibration_reports(saved_scores, out, methods=METHODS, candidate="clmm_spectral"):
    alerts, states = calibration_report(saved_scores, methods)
    alerts.to_csv(out / "calibration_by_mask.csv", index=False)
    states.to_csv(out / "alarms_by_damage_state.csv", index=False)
    metrics = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_at_fpr_05"]
    keys = ["condition", "method", "policy", "alpha"]
    summary = alerts.groupby(keys)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(out / "calibration_summary.csv", index=False)
    index = ["condition", "replicate", "policy", "alpha"]
    paired = alerts.set_index([*index, "method"])
    proposed = paired.xs(candidate, level="method")[metrics]
    deltas = pd.concat([
        (proposed - paired.xs(method, level="method")[metrics]).add_prefix("delta_")
        .assign(baseline=method).reset_index()
        for method in methods if method != candidate
    ], ignore_index=True)
    deltas.to_csv(out / "paired_deltas.csv", index=False)


def write_baseline_comparison(out, path):
    current = pd.read_csv(out / "calibration_summary.csv")
    current = current.loc[current.alpha.eq(.05) & current.policy.eq("same_mask_calibration")].copy()
    current["dataset"] = "HellBridge"
    current["protocol"] = "native_1024_paired"
    current["design"] = "nested_random"
    current["seeds"] = 1
    current["sd_over"] = "masks"
    current["source"] = "calibration_summary.csv"
    current["masks"] = np.where(current.condition.eq("clean"), 1, len(MASKS))
    history = pd.read_csv(path).drop(columns="alpha", errors="ignore")
    if history.empty or history.protocol.eq("native_1024_paired").any():
        raise ValueError("Historical baseline rows must declare their separate protocols")
    # Keep reported precision and metric availability when enlarging the reference table.
    combined = pd.concat([current, history], ignore_index=True, sort=False)
    columns = ["dataset", "condition", "method", "protocol", "design", "seeds", "masks", "sd_over", "policy",
               "auc_mean", "auc_std", "recall_at_fpr_05_mean", "recall_at_fpr_05_std", "recall_mean", "fpr_mean"]
    tail = [col for col in combined if col not in columns]
    combined.reindex(columns=[*columns, *tail]).to_csv(out / "baseline_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--segment-consensus", action="store_true")
    parser.add_argument("--residual-whitening", action="store_true")
    parser.add_argument("--baseline-reference", type=Path)
    cli = parser.parse_args()
    if not cli.hell_h5.is_file():
        raise FileNotFoundError(f"Attach HellBridge: {cli.hell_h5}")
    if cli.baseline_reference is not None and not cli.baseline_reference.is_file():
        raise FileNotFoundError(cli.baseline_reference)
    segment_consensus = cli.segment_consensus or cli.residual_whitening
    default_output = WHITENING_OUTPUT if cli.residual_whitening else SEGMENT_OUTPUT if segment_consensus else OUTPUT
    out = cli.output_dir or default_output
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(hell_h5=cli.hell_h5, seed=SEED, epochs=5, adapter_epochs=8, batch_size=8, strength=.50,
                              gain_weight=.25, replicates=MASKS, layer_probe=False, train_log=[], reference_log=[],
                              segment_consensus=segment_consensus, residual_whitening=cli.residual_whitening)
    methods = scoring_methods(segment_consensus, cli.residual_whitening)
    (out / "protocol.json").write_text(json.dumps(dict(
        seed=SEED, dataset="HellBridge P2_NM_Z", input=str(cli.hell_h5), timesnet_epochs=args.epochs,
        adapter_epochs=args.adapter_epochs, batch_size=args.batch_size, native_points=LENGTH, sample_rate_hz=SAMPLE_HZ,
        bands_hz=BANDS_HZ, segment_length=SEGMENT, segment_stride=STRIDE, spectral_weight=SPECTRAL_WEIGHT,
        methods=methods, nested_replicates=MASKS, threshold_alpha=(.05, .10),
        segment_consensus=segment_consensus, residual_whitening=cli.residual_whitening,
        residual_correlation_shrinkage=CORRELATION_SHRINKAGE if cli.residual_whitening else None,
        residual_covariance_source="50 healthy fit windows masked before reconstruction" if cli.residual_whitening else None,
        split="50 healthy fit, 10 healthy reference, 20 healthy threshold calibration, 80 healthy and 640 damaged test windows",
        dependence="healthy development and test windows share two source recordings; calibration targets are nominal",
    ), indent=2), encoding="utf-8")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Enable a GPU accelerator on Kaggle for the 1024-point pilot")
    print(f"device={device} seed={SEED} | points={LENGTH} | masks={list(MASKS)}", flush=True)
    rows, masks, fits = run_hellbridge(args, device, out, make_block=SpectralGainAudit)
    evaluate("HellBridge", rows, masks, fits, out, methods=methods)
    candidate = WHITENING_METHODS[-1] if cli.residual_whitening else SEGMENT_METHODS[-1] if segment_consensus else "clmm_spectral"
    write_calibration_reports(out / "scores.csv.gz", out, methods, candidate)
    if cli.baseline_reference is not None:
        write_baseline_comparison(out, cli.baseline_reference)
    pd.DataFrame(args.train_log).to_csv(out / "adapter_training.csv", index=False)
    bundle = shutil.make_archive(str(out) + "_report", "zip", root_dir=out)
    marker = ({WHITENING_METHODS[-1]: "HELLBRIDGE_RESIDUAL_WHITENING_PASS",
               SEGMENT_METHODS[-1]: "HELLBRIDGE_SEGMENT_CONSENSUS_PASS"}
              .get(candidate, "HELLBRIDGE_SPECTRAL_PASS"))
    print(f"{marker} reports={out} zip={bundle}", flush=True)


if __name__ == "__main__":
    main()
