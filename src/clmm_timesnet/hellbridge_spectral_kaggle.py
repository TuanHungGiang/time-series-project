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
BANDS_HZ = ((2, 6), (6, 12), (12, 20), (20, 32), (32, 45))
SAMPLE_HZ = 100
SEGMENT, STRIDE = 512, 256
SPECTRAL_WEIGHT = .25
MASKS = tuple(range(300, 305))
OUTPUT = Path("/kaggle/working/clmm_hellbridge_spectral_1024_seed20270905")
SEGMENT_OUTPUT = OUTPUT.with_name("clmm_hellbridge_segment_consensus_1024_seed20270905")
INPUT = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")


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


class SpectralGainAudit(GainHeadAudit):
    columns = METHODS

    def reference_audit(self, reference, group, args):
        super().reference_audit(reference, group, args)
        self.segment_consensus = args.segment_consensus
        self.columns = (*METHODS, *SEGMENT_METHODS) if self.segment_consensus else METHODS
        device = self.detector.device
        self.hann = torch.hann_window(SEGMENT, device=device)
        f = torch.fft.rfftfreq(SEGMENT, d=1 / SAMPLE_HZ, device=device)
        self.in_band = torch.stack([(f >= lo) & (f < hi) for lo, hi in BANDS_HZ], dim=1).float()
        (healthy,) = self.prepare(reference)
        self.healthy = torch.as_tensor(healthy, device=device)
        self.spectral_references = {}

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
        return raw_off, raw_on, gain, spectral, psd, consensus

    def matched_reference(self, kept, mask):
        key = tuple(kept)
        if key not in self.spectral_references:
            traces = [self.components(chunk, mask) for chunk in self.healthy.split(self.detector.batch_size)]
            distance = np.concatenate([x[3] for x in traces])
            template = torch.cat([x[4] for x in traces]).mean(0)
            consensus_ref = healthy_scale(np.concatenate([x[5] for x in traces])) if self.segment_consensus else None
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
            off, on, gain, spectral, psd, consensus = self.components(chunk, mask)
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
    metrics = ["auc", "ap", "ba", "recall", "fpr", "f1", "recall_at_fpr_05"]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--segment-consensus", action="store_true")
    cli = parser.parse_args()
    if not cli.hell_h5.is_file():
        raise FileNotFoundError(f"Attach HellBridge: {cli.hell_h5}")
    out = cli.output_dir or (SEGMENT_OUTPUT if cli.segment_consensus else OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(hell_h5=cli.hell_h5, seed=SEED, epochs=5, adapter_epochs=8, batch_size=8, strength=.50,
                              gain_weight=.25, replicates=MASKS, layer_probe=False, train_log=[], reference_log=[],
                              segment_consensus=cli.segment_consensus)
    methods = (*METHODS, *SEGMENT_METHODS) if cli.segment_consensus else METHODS
    (out / "protocol.json").write_text(json.dumps(dict(
        seed=SEED, dataset="HellBridge P2_NM_Z", input=str(cli.hell_h5), timesnet_epochs=args.epochs,
        adapter_epochs=args.adapter_epochs, batch_size=args.batch_size, native_points=LENGTH, sample_rate_hz=SAMPLE_HZ,
        bands_hz=BANDS_HZ, segment_length=SEGMENT, segment_stride=STRIDE, spectral_weight=SPECTRAL_WEIGHT,
        methods=methods, nested_replicates=MASKS, threshold_alpha=(.05, .10),
        segment_consensus=cli.segment_consensus,
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
    candidate = SEGMENT_METHODS[-1] if cli.segment_consensus else "clmm_spectral"
    write_calibration_reports(out / "scores.csv.gz", out, methods, candidate)
    pd.DataFrame(args.train_log).to_csv(out / "adapter_training.csv", index=False)
    bundle = shutil.make_archive(str(out) + "_report", "zip", root_dir=out)
    marker = "HELLBRIDGE_SEGMENT_CONSENSUS_PASS" if cli.segment_consensus else "HELLBRIDGE_SPECTRAL_PASS"
    print(f"{marker} reports={out} zip={bundle}", flush=True)


if __name__ == "__main__":
    main()
