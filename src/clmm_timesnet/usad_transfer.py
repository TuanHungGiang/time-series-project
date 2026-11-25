"""Paired transfer of the CLMM residual adapter and score to reference USAD."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.masking import sensor_mask
from .mask_pair_block import FIELDS, RATES, evaluate, location_scale, method_deltas, run_hellbridge, run_z24
from .models.backbone import seed_everything
from .models.usad import UsadModel
from .post_block_audit import PostBlockAdapter
from .post_block_head import GainHeadAudit, select_raw_direction
from .timesnet_native_1024 import LENGTH, SEED

BASELINE = "usad_native"
ADAPTER = "usad_clmm_adapter"
NEW_HEAD = "usad_clmm_gain_head"
METHODS = (BASELINE, ADAPTER, NEW_HEAD)


class USADDetector:
    def __init__(self, raw, reference, seed, args, device):
        self.device = device
        self.batch_size = args.batch_size
        self.patch = args.patch
        self.low = raw.min(axis=(0, 2), keepdims=True)
        self.span = np.maximum(raw.max(axis=(0, 2), keepdims=True) - self.low, 1e-6)
        fit = self.transform(raw)
        self.fill = torch.as_tensor(fit.mean(axis=(0, 2)), device=device).view(1, -1, 1)
        dimension = raw.shape[1] * self.patch
        self.latent_width = max(5, dimension // 8)
        seed_everything(seed)
        self.model = UsadModel(dimension, self.latent_width).to(device)
        self.train_native(fit, args.epochs)
        self.model.eval().requires_grad_(False)
        self.reference_center, self.reference_scale = self.clean_reference(reference)

    def transform(self, raw):
        return np.clip((raw - self.low) / self.span, 0, 1).astype(np.float32)

    def patches(self, values):
        batch, channels, points = values.shape
        return values.reshape(batch, channels, points // self.patch, self.patch).transpose(1, 2).flatten(0, 1)

    def train_native(self, fit, epochs):
        patches = self.patches(torch.as_tensor(fit, device=self.device)).flatten(1)
        batch_size = self.batch_size * (LENGTH // self.patch)
        parameters = list(self.model.encoder.parameters())
        first = torch.optim.Adam(parameters + list(self.model.decoder1.parameters()), lr=1e-3)
        second = torch.optim.Adam(parameters + list(self.model.decoder2.parameters()), lr=1e-3)
        for epoch in range(1, epochs + 1):
            losses = []
            for indices in torch.randperm(len(patches), device=self.device).split(batch_size):
                batch = patches[indices]
                self.model.zero_grad(set_to_none=True)
                loss1, _ = self.model.training_step(batch, epoch)
                loss1.backward()
                first.step()
                self.model.zero_grad(set_to_none=True)
                _, loss2 = self.model.training_step(batch, epoch)
                loss2.backward()
                second.step()
                losses.append(float((loss1 + loss2).detach()))
            if epoch in (1, epochs):
                print(f"  USAD epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
        self.model.zero_grad(set_to_none=True)

    def per_sensor_error(self, values, kept, adapter=None):
        masked = torch.where(kept.view(1, -1, 1), values, self.fill)
        x = self.patches(masked).flatten(1)
        target = self.patches(values)
        latent = self.model.encoder(x)
        if adapter is not None:
            # Correct only the first encoding. The re-encoding cycle is native USAD.
            latent = adapter(latent.unsqueeze(1), kept).squeeze(1)
        first = self.model.decoder1(latent)
        cycle = self.model.decoder2(self.model.encoder(first))
        first_error = (target[:, kept] - first.reshape_as(target)[:, kept]).square()
        cycle_error = (target[:, kept] - cycle.reshape_as(target)[:, kept]).square()
        error = 0.5 * (first_error + cycle_error)
        return error.reshape(len(values), -1, error.shape[1], self.patch).mean((1, 3))

    @torch.no_grad()
    def clean_reference(self, raw):
        values = torch.as_tensor(self.transform(raw), device=self.device)
        kept = torch.ones(raw.shape[1], dtype=torch.bool, device=self.device)
        residuals = torch.cat([
            self.per_sensor_error(chunk, kept).mean(1) for chunk in values.split(self.batch_size)
        ]).cpu().numpy().astype(np.float64)
        return location_scale(np.log1p(residuals), floor=1e-4)


class USADTransfer(GainHeadAudit):
    columns = (BASELINE, ADAPTER, "head_upper_raw", "head_lower_raw", "gain_deviation")

    def __init__(self, fit, reference, seed, args, device, group):
        self.detector = USADDetector(fit, reference, seed, args, device)
        seed_everything(seed + 7001)
        self.adapter = PostBlockAdapter(self.detector.latent_width, fit.shape[1], args.strength).to(device)
        self.train_adapter(fit, seed, args, group)
        self.reference_audit(reference, group, args)

    def prepare(self, raw):
        return (self.detector.transform(raw),)

    def train_adapter(self, fit, seed, args, group):
        values = torch.as_tensor(self.detector.transform(fit), device=self.detector.device)
        rng = np.random.default_rng(seed + 9613)
        channels = values.shape[1]
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=1e-3, weight_decay=1e-5)
        for epoch in range(1, args.adapter_epochs + 1):
            for rate in RATES:
                losses = []
                for batch in values[torch.as_tensor(rng.permutation(len(values)), device=values.device)].split(args.batch_size):
                    kept = torch.as_tensor(sensor_mask(channels, rate, rng.permutation(channels)), device=values.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.detector.per_sensor_error(batch, kept, self.adapter).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                args.train_log.append(dict(group=group, epoch=epoch, rate=rate, observed_mse=np.mean(losses)))
            if epoch in (1, args.adapter_epochs):
                print(f"  CLMM adapter epoch={epoch}/{args.adapter_epochs}", flush=True)
        self.adapter.eval().requires_grad_(False)

    def batch_signals(self, values, kept):
        off = self.detector.per_sensor_error(values, kept).cpu().numpy().astype(np.float64)
        on = self.detector.per_sensor_error(values, kept, self.adapter).cpu().numpy().astype(np.float64)
        return off.mean(1), on.mean(1), np.log1p(off) - np.log1p(on)


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
    parser.add_argument("--patch", type=int, default=128, choices=(32, 64, 128))
    args = parser.parse_args()
    if min(args.epochs, args.adapter_epochs, args.batch_size, args.z24_windows, args.mask_reps) < 1:
        parser.error("epochs, adapter-epochs, batch-size, z24-windows and mask-reps must be positive")
    args.replicates = list(range(300, 300 + args.mask_reps))
    args.strength, args.gain_weight = 0.5, 0.25
    args.train_log, args.reference_log = [], []
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = dict(
        z24_root=str(args.z24_root), hell_h5=str(args.hell_h5), seed=args.seed, device=str(device),
        usad_epochs=args.epochs, adapter_epochs=args.adapter_epochs, batch_size=args.batch_size,
        window_points=LENGTH, patch_points=args.patch, sampling_hz=100, z24_windows=args.z24_windows,
        preprocess="shared detrending; no downsampling; per-sensor min-max fitted on healthy fit only",
        missing_input="per-sensor healthy fit mean in scaled units, identical for all methods",
        baseline="EURECOM encoder and two decoders, original alternating objectives, equal error weights; "
                 "errors averaged over observed sensors and all contiguous patches in each 1024-point window",
        adapter="uses the first encoding only; native re-encoding cycle; only adapter trained on masked healthy fit",
        methods=METHODS, rates=(0.0, *RATES), nested_mask_reps=args.replicates,
        adapter_strength=args.strength, gain_weight=args.gain_weight,
        score_direction="shared native direction; Z24 from clean validation, HellBridge upper",
        calibration="healthy Q90, clean-frozen and mask-specific",
        z24_protocol="locked record split; median of final window scores; mask varies by record",
        hellbridge_protocol="P2_NM_Z, 50 fit/10 reference/20 calibration/720 test windows; "
                            "healthy test windows share source recordings with development",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"device={device} seed={args.seed} patch={args.patch} mask_reps={args.replicates}", flush=True)
    metrics = []
    for dataset in ("HellBridge", "Z24"):
        bridge_out = out / dataset
        args.train_log.clear()
        args.reference_log.clear()
        if dataset == "HellBridge":
            rows, masks, fits = run_hellbridge(args, device, bridge_out, make_block=USADTransfer)
        else:
            rows, masks, fits = run_z24(args, device, make_block=USADTransfer)
        select_raw_direction(dataset, rows, BASELINE, NEW_HEAD)
        metrics.append(evaluate(dataset, rows, masks, fits, bridge_out, methods=METHODS,
                                share_native_direction=True, baseline_method=BASELINE))
        pd.DataFrame(args.train_log).to_csv(bridge_out / "adapter_train.csv", index=False)
        pd.DataFrame(args.reference_log).to_csv(bridge_out / "gain_references.csv", index=False)
    all_metrics = pd.concat(metrics, ignore_index=True)
    group = ["dataset", "condition", "method", "policy"]
    summary = all_metrics.groupby(group, as_index=False)[list(FIELDS)].mean()
    summary.to_csv(out / "summary.csv", index=False)
    deltas = pd.concat([
        method_deltas(all_metrics, ADAPTER, (BASELINE,)).assign(method=ADAPTER),
        method_deltas(all_metrics, NEW_HEAD, (BASELINE, ADAPTER)).assign(method=NEW_HEAD),
    ], ignore_index=True)
    deltas.to_csv(out / "paired_deltas.csv", index=False)
    print(summary.loc[summary.policy.eq("clean_q90")].round(4).to_string(index=False), flush=True)
    print(f"USAD_TRANSFER_PASS output={out}", flush=True)


if __name__ == "__main__":
    main()
