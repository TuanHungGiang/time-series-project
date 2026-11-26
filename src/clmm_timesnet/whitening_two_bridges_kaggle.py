"""Apply the HellBridge whitening head to Z24 with its existing validation score tail."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from .hellbridge_spectral_kaggle import CORRELATION_SHRINKAGE, INPUT, MASKS, SpectralGainAudit
from .hellbridge_spectral_kaggle import main as run_hellbridge_pilot
from .mask_pair_block import FIELDS, evaluate, method_deltas, native_score_direction, run_z24
from .timesnet_native_1024 import LENGTH, SEED


Z24_INPUT = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
OUTPUT = Path("/kaggle/working/clmm_whitening_two_bridges_seed20270905")
CANDIDATE = "clmm_whitened_residual"
METHODS = ("timesnet_native", "adapter_mse", "clmm_gain_old", "clmm_diagonal_residual", CANDIDATE)
GAIN_HEADS = METHODS[2:]
LOWER_COLUMNS = tuple(f"{method}_lower_raw" for method in GAIN_HEADS)


class Z24WhiteningAudit(SpectralGainAudit):
    def reference_audit(self, reference, group, args):
        super().reference_audit(reference, group, args)
        self.method_indices = [self.columns.index(method) for method in METHODS]
        self.columns = (*METHODS, *LOWER_COLUMNS)

    def scores(self, data, kept):
        upper = super().scores(data, kept)[:, self.method_indices]
        # The same unsigned gain deviation is added to anomaly evidence
        # in either tail. Construct both raw tails before record pooling.
        gain_bonus = upper[:, 2] - upper[:, 1]
        lower = upper[:, 2:] - 2 * gain_bonus[:, None]
        return np.column_stack((upper, lower))


def orient_z24(rows):
    sign = native_score_direction("Z24", pd.DataFrame(rows))
    for row in rows:
        for method, column in zip(GAIN_HEADS, LOWER_COLUMNS, strict=True):
            lower = row.pop(column)
            if sign == -1:
                row[method] = lower
    return sign


def run_z24_pilot(input_root, out, device):
    args = argparse.Namespace(
        z24_root=input_root, seed=SEED, epochs=5, adapter_epochs=8, batch_size=8, z24_windows=8,
        strength=.50, gain_weight=.25, replicates=MASKS, layer_probe=False, train_log=[], reference_log=[],
        segment_consensus=False, residual_whitening=True,
    )
    print(f"Z24 device={device} seed={SEED} points={LENGTH} masks={list(MASKS)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    rows, masks, fits = run_z24(args, device, make_block=Z24WhiteningAudit)
    sign = orient_z24(rows)
    print(f"Z24 shared tail from clean validation: {'upper' if sign == 1 else 'lower'}", flush=True)
    metrics = evaluate("Z24", rows, masks, fits, out, methods=METHODS, share_native_direction=True)
    method_deltas(metrics, CANDIDATE, METHODS[:-1]).to_csv(out / "paired_deltas.csv", index=False)
    pd.DataFrame(args.train_log).to_csv(out / "adapter_training.csv", index=False)
    keys = ["dataset", "condition", "method", "policy"]
    summary = metrics.groupby(keys)[list(FIELDS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    information = {
        "seed": SEED, "points_per_window": LENGTH, "masks": MASKS, "window_cap_per_record": args.z24_windows,
        "training_epochs": args.epochs, "adapter_epochs": args.adapter_epochs, "batch_size": args.batch_size,
        "residual_correlation_shrinkage": CORRELATION_SHRINKAGE,
        "score_tail": "upper" if sign == 1 else "lower", "score_tail_source": "native scores on labeled clean validation",
        "fit": "healthy train only by setup and excitation", "calibration": "healthy validation Q90",
        "test_unit": "record with median window score", "test_healthy": 6, "test_damage": 126,
        "roc_fpr_limit": 0.0,
    }
    (out / "protocol.json").write_text(json.dumps(information, indent=2), encoding="utf-8")
    return summary.loc[summary.policy.eq("masked_q90")].copy()


def tag_current(frame, protocol):
    return frame.assign(
        protocol=protocol, design="nested_random", seeds=1, sd_over="masks",
        masks=np.where(frame.condition.eq("clean"), 1, len(MASKS)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, default=Z24_INPUT)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    cli = parser.parse_args()
    for path in (cli.hell_h5, cli.z24_root / "metadata/records_manifest.csv", cli.baseline_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU for the native 1024-point comparison")
    torch.set_num_threads(2)
    out = cli.output_dir
    hell_out = out / "HellBridge"
    run_hellbridge_pilot([
        "--hell-h5", str(cli.hell_h5), "--residual-whitening", "--output-dir", str(hell_out),
    ])
    hell = pd.read_csv(hell_out / "calibration_summary.csv")
    hell = hell.loc[hell.alpha.eq(.05) & hell.policy.eq("same_mask_calibration")].assign(dataset="HellBridge")
    hell = tag_current(hell, "hellbridge_native_1024_paired").drop(columns="alpha")
    z24 = tag_current(run_z24_pilot(cli.z24_root, out / "Z24", torch.device("cuda")), "z24_native_1024_paired")
    paired = pd.concat([hell, z24], ignore_index=True, sort=False)
    paired["diagnostic_recall_mean"] = paired["recall_at_fpr_05_mean"].fillna(paired["recall_low_fpr_mean"])
    paired["diagnostic_fpr_limit"] = np.where(paired.dataset.eq("Z24"), .0, .05)
    paired.to_csv(out / "paired_summary.csv", index=False)
    historical = pd.read_csv(cli.baseline_reference)
    pd.concat([paired, historical], ignore_index=True, sort=False).to_csv(out / "baseline_comparison.csv", index=False)
    columns = ["dataset", "condition", "method", "auc_mean", "diagnostic_recall_mean", "recall_mean", "fpr_mean"]
    print(paired.loc[paired.method.isin(METHODS), columns].round(4).to_string(index=False), flush=True)
    archive = shutil.make_archive(f"{out}_report", "zip", root_dir=out)
    print(f"CLMM_WHITENING_TWO_BRIDGES_PASS reports={out} zip={archive}", flush=True)


if __name__ == "__main__":
    main()
