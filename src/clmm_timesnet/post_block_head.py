"""Compare a correction-aware score with the frozen TimesNet reconstruction head."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.masking import sensor_mask, stable_rng
from .mask_pair_block import FIELDS, RATES, evaluate, location_scale, method_deltas
from .mask_pair_block import native_score_direction, run_hellbridge, run_z24
from .post_block_audit import METHODS as AUDIT_METHODS, ResidualAudit
from .timesnet_native_1024 import LENGTH, SEED

NEW_METHOD = "post_block_gain_head"
METHODS = (*AUDIT_METHODS, NEW_METHOD)
REFERENCE_MASKS = 5
SCALE_FLOOR = 0.005
TOP_FRACTION = 0.25
DEVIATION_CAP = 3.0


def gain_deviation(gain, center, scale):
    distance = np.minimum(np.abs(gain - center) / scale, DEVIATION_CAP)
    count = max(1, int(np.ceil(distance.shape[1] * TOP_FRACTION)))
    return np.partition(distance, -count, axis=1)[:, -count:].mean(axis=1)


class GainHeadAudit(ResidualAudit):
    """Use the same fitted adapter and reconstruction map for both scoring heads."""

    columns = (*AUDIT_METHODS, "head_upper_raw", "head_lower_raw", "gain_deviation")

    def batch_signals(self, values, kept):
        off, _ = self.residual(values, kept, enabled=False)
        on, _ = self.residual(values, kept, enabled=True)
        global_off = off.mean((1, 2)).cpu().numpy().astype(np.float64)
        global_on = on.mean((1, 2)).cpu().numpy().astype(np.float64)
        channel_off = off.mean(-1).cpu().numpy().astype(np.float64)
        channel_on = on.mean(-1).cpu().numpy().astype(np.float64)
        gain = np.log1p(channel_off) - np.log1p(channel_on)
        return global_off, global_on, gain

    @torch.no_grad()
    def reference_audit(self, reference, group, args):
        self.weight = args.gain_weight
        device = self.detector.device
        (data,) = self.prepare(reference)
        channels = data.shape[1]
        values = torch.as_tensor(data, device=device)
        order = stable_rng(group, "post-block-gain-reference", 0).permutation(channels)
        # Spread masks evenly around a randomized sensor order so each sensor
        # has healthy reference samples even at the 50% loss level.
        rolls = np.arange(REFERENCE_MASKS) * channels // REFERENCE_MASKS
        by_count = {}
        for rate in RATES:
            gains_by_sensor = [[] for _ in range(channels)]
            for roll in rolls:
                kept = sensor_mask(channels, rate, np.roll(order, roll))
                mask = torch.as_tensor(kept, device=device)
                gain = np.concatenate([
                    self.batch_signals(chunk, mask)[2] for chunk in values.split(args.batch_size)
                ])
                for index, column in zip(np.flatnonzero(kept), gain.T, strict=True):
                    gains_by_sensor[index].append(column)
            if not all(gains_by_sensor):
                raise RuntimeError(f"Incomplete healthy sensor references for {group}/{rate}")
            by_count[int((~kept).sum())] = [np.concatenate(parts) for parts in gains_by_sensor]

        self.centers = {count: np.array([np.median(x) for x in gains]) for count, gains in by_count.items()}
        self.scale = np.array([
            location_scale(np.concatenate([
                gains[c] - self.centers[count][c] for count, gains in by_count.items()
            ]), floor=SCALE_FLOOR)[1]
            for c in range(channels)
        ])
        for count, gains in by_count.items():
            for channel, reference_gains in enumerate(gains):
                args.reference_log.append(dict(
                    group=group, missing_sensors=count, sensor_index=channel, windows=len(reference_gains),
                    gain_center=self.centers[count][channel], gain_scale=self.scale[channel],
                ))

    @torch.no_grad()
    def scores(self, data, kept):
        device = self.detector.device
        values = torch.as_tensor(data, device=device)
        mask = torch.as_tensor(kept, device=device)
        missing = int((~kept).sum())
        center, scale = self.detector.reference_center, self.detector.reference_scale
        rows = []
        for chunk in values.split(self.detector.batch_size):
            off, on, gain = self.batch_signals(chunk, mask)
            off_score = (np.log1p(off) - center) / scale
            on_score = (np.log1p(on) - center) / scale
            if missing:
                deviation = gain_deviation(gain, self.centers[missing][kept], self.scale[kept])
            else:
                if not np.array_equal(off, on):
                    raise RuntimeError("Clean identity failed for the post-block adapter")
                deviation = np.zeros(len(on_score))
            bonus = self.weight * deviation
            rows.append(np.stack((
                off_score, on_score, on_score + bonus, on_score - bonus, deviation,
            ), axis=1))
        return np.concatenate(rows)


def select_raw_direction(dataset, rows, baseline_method=AUDIT_METHODS[0], method=NEW_METHOD):
    sign = native_score_direction(dataset, pd.DataFrame(rows), baseline_method)
    for row in rows:
        upper = row.pop("head_upper_raw")
        lower = row.pop("head_lower_raw")
        row[method] = upper if sign == 1 else lower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--adapter-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--z24-windows", type=int, default=8)
    parser.add_argument("--mask-reps", type=int, default=5)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--gain-weight", type=float, default=0.25)
    args = parser.parse_args()
    if min(args.epochs, args.adapter_epochs, args.batch_size, args.z24_windows, args.mask_reps) < 1:
        parser.error("epochs, adapter-epochs, batch-size, z24-windows and mask-reps must be positive")
    if not 0 < args.strength <= 1 or not 0 < args.gain_weight <= 1:
        parser.error("strength and gain-weight must be in (0, 1]")
    args.replicates = list(range(300, 300 + args.mask_reps))
    args.layer_probe = False
    args.train_log, args.reference_log = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        z24_root=str(args.z24_root), hell_h5=str(args.hell_h5), seed=args.seed, device=str(device),
        native_epochs=args.epochs, adapter_epochs=args.adapter_epochs, batch_size=args.batch_size,
        window_points=LENGTH, z24_windows=args.z24_windows, methods=METHODS,
        rates=(0.0, *RATES), nested_mask_reps=args.replicates, strength=args.strength,
        reference_masks_per_rate=REFERENCE_MASKS, gain_floor=SCALE_FLOOR, gain_weight=args.gain_weight,
        gain_top_fraction=TOP_FRACTION, gain_cap=DEVIATION_CAP,
        gain_reference="held-back healthy, per group/sensor/loss-count center, pooled scale across loss rates",
        score="oriented adapter reconstruction score plus bounded deviation of per-sensor reconstruction gain",
        score_direction="shared native direction; Z24 from native clean validation only",
        z24_protocol="locked record split; median of final window scores; mask varies by record",
        hellbridge_protocol="P2_NM_Z, 50 fit/10 reference/20 calibration/720 test windows; "
                            "healthy test windows share source recordings with development",
        calibration="healthy Q90, clean-frozen and mask-specific",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"device={device} seed={args.seed} mask_reps={args.replicates}", flush=True)
    metrics, gain_audit = [], []
    for dataset in ("HellBridge", "Z24"):
        bridge_out = out / dataset
        args.train_log.clear()
        args.reference_log.clear()
        if dataset == "HellBridge":
            rows, masks, fits = run_hellbridge(args, device, bridge_out, make_block=GainHeadAudit)
        else:
            rows, masks, fits = run_z24(args, device, make_block=GainHeadAudit)
        select_raw_direction(dataset, rows)
        metrics.append(evaluate(dataset, rows, masks, fits, bridge_out,
                                methods=METHODS, share_native_direction=True))
        pd.DataFrame(args.train_log).to_csv(bridge_out / "adapter_train.csv", index=False)
        pd.DataFrame(args.reference_log).to_csv(bridge_out / "gain_references.csv", index=False)
        scores = pd.read_csv(bridge_out / "scores.csv.gz")
        test = scores.loc[scores.split.eq("test")]
        gain_audit.append(
            test.groupby(["condition", "replicate", "label"], as_index=False)
            .agg(samples=("sample_id", "size"), median_gain_deviation=("gain_deviation", "median"),
                 gain_deviation_q90=("gain_deviation", lambda values: values.quantile(0.9)))
            .assign(dataset=dataset)
        )
    all_metrics = pd.concat(metrics, ignore_index=True)
    group = ["dataset", "condition", "method", "policy"]
    summary = all_metrics.groupby(group, as_index=False)[list(FIELDS)].mean()
    summary.to_csv(out / "summary.csv", index=False)
    method_deltas(all_metrics, NEW_METHOD, AUDIT_METHODS).to_csv(out / "paired_deltas.csv", index=False)
    pd.concat(gain_audit, ignore_index=True).to_csv(out / "gain_by_class.csv", index=False)
    print(summary.loc[summary.policy.eq("clean_q90")].round(4).to_string(index=False), flush=True)
    print(f"POST_BLOCK_HEAD_PASS output={out}", flush=True)


if __name__ == "__main__":
    main()
