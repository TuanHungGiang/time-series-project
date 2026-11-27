"""Run CLMM and its reference baselines on antialiased, 128-point bridge windows."""

import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly


RAW_POINTS = 1024
POINTS = 128
RAW_HZ = 100
SEED = 20270905
NATIVE = "timesnet_native"
CLMM = "clmm_whitened_residual"
BASELINES = ("pca_spe", "lstm_ae")
METHODS = (NATIVE, CLMM, *BASELINES)
METRICS = ("auc", "ap", "ba", "recall", "fpr", "precision", "f1", "diagnostic_recall")


def to_model_points(windows):
    if windows.ndim != 3 or windows.shape[-1] != RAW_POINTS:
        raise ValueError(f"Expected raw windows [N, C, {RAW_POINTS}], got {windows.shape}")
    return resample_poly(windows, up=1, down=RAW_POINTS // POINTS, axis=-1).astype(np.float32)


def load_reduced(loader):
    def read(*args):
        return to_model_points(loader(*args))
    return read


def run_bridge(bridge, config, device, source, out):
    from clmm_timesnet import hellbridge_spectral_kaggle as spectral
    from clmm_timesnet import mask_pair_block as window_protocol
    from clmm_timesnet.matched_baselines_kaggle import BaselineBlock
    from clmm_timesnet.whitening_two_bridges_kaggle import Z24WhiteningAudit, orient_z24

    z24 = bridge == "Z24"
    audit_type = Z24WhiteningAudit if z24 else spectral.SpectralGainAudit
    config.active_models = BASELINES

    class CombinedBlock:
        def __init__(self, fit, reference, seed, args, accelerator, group):
            print(f"  {group} | fit {fit.shape} | reference {reference.shape}", flush=True)
            self.clmm = audit_type(fit, reference, seed, args, accelerator, group)
            self.baselines = BaselineBlock(fit, reference, seed, args, accelerator, group, source)
            self.columns = (*self.clmm.columns, *self.baselines.columns)

        def prepare(self, raw):
            (prepared,) = self.clmm.prepare(raw)
            return (prepared,)

        def scores(self, data, kept):
            return np.column_stack((self.clmm.scores(data, kept), self.baselines.scores(data, kept)))

    out.mkdir(parents=True, exist_ok=True)
    # Keep the original window starts. Resample only after the native detrend.
    # The reused implementation also computes unused spectral controls internally.
    # Scale their FFT length and sample rate to keep those operations well defined.
    factor = RAW_POINTS // POINTS
    with patch.multiple(
        window_protocol,
        load_z24=load_reduced(window_protocol.load_z24),
        load_hellbridge=load_reduced(window_protocol.load_hellbridge),
    ), patch.multiple(
        spectral, SEGMENT=spectral.SEGMENT // factor, STRIDE=spectral.STRIDE // factor,
        SAMPLE_HZ=RAW_HZ / factor,
    ):
        if z24:
            rows, masks, fits = window_protocol.run_z24(config, device, make_block=CombinedBlock)
        else:
            rows, masks, fits = window_protocol.run_hellbridge(config, device, out, make_block=CombinedBlock)

    if z24:
        sign = orient_z24(rows)
        raw = pd.DataFrame(rows)
        signs = {model: window_protocol.native_score_direction(bridge, raw, baseline_method=model)
                 for model in BASELINES}
        signs[CLMM] = sign
        for row in rows:
            for model, direction in signs.items():
                row[model] *= direction
        metrics = window_protocol.evaluate(bridge, rows, masks, fits, out, methods=METHODS)
        selected = metrics.loc[metrics.policy.eq("masked_q90")]
        selected = selected.rename(columns={"recall_low_fpr": "diagnostic_recall"})
    else:
        scores = pd.DataFrame(rows)
        if not np.isfinite(scores[list(METHODS)]).all().all():
            raise ValueError("Non-finite HellBridge scores")
        scores_path = out / "scores.csv.gz"
        scores.to_csv(scores_path, index=False)
        pd.DataFrame(masks).to_csv(out / "masks.csv", index=False)
        (out / "fit_groups.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
        metrics, by_state = spectral.calibration_report(scores_path, methods=METHODS)
        by_state.to_csv(out / "alarms_by_damage_state.csv", index=False)
        metrics.to_csv(out / "metrics_by_mask.csv", index=False)
        selected = metrics.loc[metrics.alpha.eq(.05) & metrics.policy.eq("same_mask_calibration")]
        selected = selected.rename(columns={"recall_at_fpr_05": "diagnostic_recall"})
    return selected.assign(dataset=bridge, seed=config.seed, points=POINTS)


def main(argv=None):
    from clmm_timesnet.whitening_two_bridges_kaggle import MASKS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.z24_root / "metadata/records_manifest.csv", args.hell_h5,
                 args.baseline_dir / "run_hellbridge_source_baselines.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU under Kaggle Settings → Accelerator.")
    torch.set_num_threads(2)
    device = torch.device("cuda")
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    source = importlib.import_module("run_hellbridge_source_baselines")
    config = argparse.Namespace(
        z24_root=args.z24_root, hell_h5=args.hell_h5, seed=SEED, epochs=5, adapter_epochs=8,
        batch_size=8, z24_windows=8, strength=.50, gain_weight=.25, replicates=MASKS,
        layer_probe=False, segment_consensus=False, residual_whitening=True,
        train_log=[], reference_log=[],
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(
        seed=SEED, raw_points=RAW_POINTS, model_points=POINTS, raw_hz=RAW_HZ,
        model_hz=RAW_HZ * POINTS / RAW_POINTS, duration_seconds=RAW_POINTS / RAW_HZ,
        resampling="linear detrend then scipy.signal.resample_poly(up=1, down=8)",
        models=METHODS, masks=MASKS, epochs=config.epochs, adapter_epochs=config.adapter_epochs,
        batch_size=config.batch_size, diagnostic_fpr_limit={"Z24": 0.0, "HellBridge": .05},
        calibration="healthy mask-matched", summary="mean across masks of one training seed",
    )
    (output / "protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tables = []
    for bridge in ("Z24", "HellBridge"):
        print(f"\n=== {bridge} | {RAW_POINTS} → {POINTS} | seed {SEED} ===", flush=True)
        tables.append(run_bridge(bridge, config, device, source, output / bridge))
    per_mask = pd.concat(tables, ignore_index=True)
    keys = ["dataset", "seed", "points", "method", "condition"]
    conditions = {"clean": (0,), "sensor_0.10": MASKS, "sensor_0.30": MASKS, "sensor_0.50": MASKS}
    expected = {(d, m, c, r) for d in ("Z24", "HellBridge") for m in METHODS
                for c, rep in conditions.items() for r in rep}
    found = set(per_mask[["dataset", "method", "condition", "replicate"]].itertuples(index=False, name=None))
    if expected != found or len(per_mask) != len(expected) or not np.isfinite(per_mask[list(METRICS)]).all().all():
        raise ValueError("Incomplete or invalid result matrix")
    per_mask.to_csv(output / "per_mask_metrics.csv", index=False)
    summary = per_mask.groupby(keys, as_index=False)[list(METRICS)].mean()
    clean = summary.loc[summary.condition.eq("clean")].set_index(["dataset", "method"])
    for metric in ("auc", "diagnostic_recall"):
        summary[f"delta_{metric}_vs_clean"] = summary[metric] - clean[metric].reindex(
            pd.MultiIndex.from_frame(summary[["dataset", "method"]])
        ).to_numpy()
    summary.to_csv(output / "one_seed_summary.csv", index=False)
    pd.DataFrame(config.train_log).to_csv(output / "adapter_training.csv", index=False)
    print(f"CLMM_128_ONE_SEED_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
