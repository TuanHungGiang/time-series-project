"""Audit a learned residual adapter immediately after TimesBlock 2."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data.masking import sensor_mask, stable_rng
from .mask_pair_block import FIELDS, RATES, cases, evaluate, method_deltas, run_hellbridge, run_z24
from .models.backbone import reconstruct, seed_everything
from .timesnet_native_1024 import Detector, LENGTH, SEED

METHODS = ("timesnet_native", "post_block_residual")
AUDIT_COLUMNS = (
    "mse_off", "mse_on", "relative_mse_change",
    "relative_activation_shift", "relative_after_norm_shift",
)


def relative_rms(before, after):
    change = (after - before).square().mean((1, 2)).sqrt()
    return change / before.square().mean((1, 2)).sqrt().clamp_min(1e-8)


class PostBlockAdapter(nn.Module):
    def __init__(self, width, channels, strength):
        super().__init__()
        self.strength = strength
        self.net = nn.Sequential(
            nn.Linear(width + channels, 32), nn.GELU(), nn.Linear(32, width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden, kept):
        batch, points, width = hidden.shape
        missing = (~kept).to(dtype=hidden.dtype)
        context = missing.view(1, 1, -1).expand(batch, points, -1)
        normalized = nn.functional.layer_norm(hidden, (width,))
        scale = hidden.detach().square().mean((1, 2), keepdim=True).sqrt()
        return hidden + self.strength * missing.mean() * scale * self.net(
            torch.cat([normalized, context], dim=-1),
        ).tanh()


class ResidualAudit:
    columns = (*METHODS, *AUDIT_COLUMNS)

    def __init__(self, fit, reference, seed, args, device, group):
        self.detector = Detector(fit, reference, seed, args, device)
        seed_everything(seed + 7001)
        width = self.detector.model.configs.d_model
        self.adapter = PostBlockAdapter(width, fit.shape[1], args.strength).to(device)
        self.fit(fit, seed, args, group)
        self.reference_audit(reference, group, args)

    def prepare(self, raw):
        return (self.detector.scaler.transform(raw),)

    def predict_observed(self, values, kept, enabled, monitor=False):
        mask = kept.view(1, -1, 1)
        masked = torch.where(mask, values, 0.0)
        shifts = ()
        model = self.detector.model
        if enabled:
            def after_block(_layer, _args, hidden):
                nonlocal shifts
                corrected = self.adapter(hidden, kept)
                if monitor:
                    shifts = (
                        relative_rms(hidden, corrected),
                        relative_rms(model.layer_norm(hidden), model.layer_norm(corrected)),
                    )
                return corrected

            hook = model.model[1].register_forward_hook(after_block)
            try:
                predicted = reconstruct(model, masked)
            finally:
                hook.remove()
        else:
            predicted = reconstruct(model, masked)
        return predicted[:, kept], shifts

    def residual(self, values, kept, enabled, monitor=False):
        prediction, shifts = self.predict_observed(values, kept, enabled, monitor)
        return (prediction - values[:, kept]).square(), shifts

    def error(self, values, kept, enabled, monitor=False):
        squared, shifts = self.residual(values, kept, enabled, monitor)
        return squared.mean((1, 2)), shifts

    def fit(self, raw, seed, args, group):
        (data,) = self.prepare(raw)
        device = self.detector.device
        values = torch.as_tensor(data, device=device)
        rng = np.random.default_rng(seed + 9613)
        channels = values.shape[1]
        training_mask = getattr(args, "training_sensor_mask", None)
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=1e-3, weight_decay=1e-5)
        self.adapter.train()
        for epoch in range(1, args.adapter_epochs + 1):
            for rate in RATES:
                losses = []
                for ids in torch.as_tensor(rng.permutation(len(values)), device=device).split(args.batch_size):
                    available = (training_mask(rng, rate) if training_mask is not None
                                 else sensor_mask(channels, rate, rng.permutation(channels)))
                    kept = torch.as_tensor(available, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    error, _ = self.error(values[ids], kept, enabled=True)
                    loss = error.mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.adapter.parameters(), 5.0)
                    optimizer.step()
                    losses.extend(error.detach().cpu().tolist())
                args.train_log.append(dict(group=group, epoch=epoch, rate=rate, observed_mse=np.mean(losses)))
            if epoch in (1, args.adapter_epochs):
                print(f"  residual adapter epoch={epoch}/{args.adapter_epochs}", flush=True)
        self.adapter.eval().requires_grad_(False)

    @torch.no_grad()
    def scores(self, data, kept):
        device = self.detector.device
        values = torch.as_tensor(data, device=device)
        mask = torch.as_tensor(kept, device=device)
        results = []
        for chunk in values.split(self.detector.batch_size):
            off, _ = self.error(chunk, mask, enabled=False)
            on, shifts = self.error(chunk, mask, enabled=True, monitor=True)
            off, on, activation_shift, after_norm_shift = (
                x.cpu().numpy().astype(np.float64) for x in (off, on, *shifts)
            )
            center, scale = self.detector.reference_center, self.detector.reference_scale
            if kept.all() and not np.array_equal(off, on):
                raise RuntimeError(f"Clean identity failed; maximum MSE change is {np.abs(off - on).max()}")
            results.append(np.stack((
                (np.log1p(off) - center) / scale,
                (np.log1p(on) - center) / scale,
                off, on, (on - off) / np.maximum(off, 1e-8), activation_shift, after_norm_shift,
            ), axis=1))
        return np.concatenate(results)

    def reference_audit(self, reference, group, args):
        (data,) = self.prepare(reference)
        for condition, rep, kept in cases(data.shape[1], f"{group}/healthy-reference", [400]):
            rows = self.scores(data, kept)
            args.reference_log.append(dict(
                group=group, condition=condition, replicate=rep, windows=len(rows),
                **dict(zip(AUDIT_COLUMNS, np.median(rows[:, 2:], axis=0), strict=True)),
            ))


def log_test_diagnostics(dataset, out):
    scores = pd.read_csv(out / "scores.csv.gz")
    test = scores[scores.split.eq("test")].copy()
    test["score_delta"] = test.post_block_residual - test.timesnet_native
    test["class"] = test.label.map({0: "healthy", 1: "damage"})
    keys = ["condition", "replicate"]
    grouped = test.groupby([*keys, "class"])
    class_summary = grouped.agg(
        samples=("sample_id", "size"),
        median_score_delta=("score_delta", "median"),
        median_mse_off=("mse_off", "median"),
        median_mse_on=("mse_on", "median"),
        median_relative_mse_change=("relative_mse_change", "median"),
        median_activation_shift=("relative_activation_shift", "median"),
        median_after_norm_shift=("relative_after_norm_shift", "median"),
        share_score_decreased=("score_delta", lambda col: col.lt(0).mean()),
    ).reset_index().assign(dataset=dataset)

    damage_states = test.loc[test.label.eq(1)].groupby([*keys, "state"]).agg(
        samples=("sample_id", "size"),
        median_score_delta=("score_delta", "median"),
        median_relative_mse_change=("relative_mse_change", "median"),
        median_activation_shift=("relative_activation_shift", "median"),
    ).reset_index().assign(dataset=dataset)
    damage_states.to_csv(out / "damage_states.csv", index=False)

    separation = []
    for (condition, rep), trial in test.groupby(keys):
        healthy = trial.loc[trial.label.eq(0), list(METHODS)].to_numpy()
        damage = trial.loc[trial.label.eq(1), list(METHODS)].to_numpy()
        healthy_tail = np.quantile(healthy, 0.90, axis=0)
        damage_middle = np.median(damage, axis=0)
        margin = damage_middle - healthy_tail
        ordered_off = damage[:, 0, None] > healthy[:, 0][None, :]
        ordered_on = damage[:, 1, None] > healthy[:, 1][None, :]
        separation.append(dict(
            dataset=dataset, condition=condition, replicate=rep,
            delta_healthy_q90=healthy_tail[1] - healthy_tail[0],
            delta_damage_median=damage_middle[1] - damage_middle[0],
            delta_margin=margin[1] - margin[0],
            lost_pairs=int(np.count_nonzero(ordered_off & ~ordered_on)),
            gained_pairs=int(np.count_nonzero(~ordered_off & ordered_on)),
            pairs_total=ordered_off.size,
        ))
    return class_summary, pd.DataFrame(separation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--adapter-epochs", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--z24-windows", type=int, default=8)
    parser.add_argument("--mask-reps", type=int, default=5)
    args = parser.parse_args()
    if min(args.epochs, args.adapter_epochs, args.batch_size, args.z24_windows, args.mask_reps) < 1:
        parser.error("epochs, batch-size, z24-windows and mask-reps must be positive")
    if not 0 < args.strength <= 1:
        parser.error("strength must be in (0, 1]")
    args.replicates = list(range(300, 300 + args.mask_reps))
    args.layer_probe = False
    args.train_log, args.reference_log = [], []
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = dict(
        z24_root=str(args.z24_root), hell_h5=str(args.hell_h5),
        native_epochs=args.epochs, adapter_epochs=args.adapter_epochs, seed=args.seed,
        batch_size=args.batch_size, window_points=LENGTH, z24_windows=args.z24_windows,
        device=str(device), adapter_location="TimesBlock 2 output before the original LayerNorm",
        training="freeze TimesNet; fit adapter on healthy fit windows with observed-channel MSE",
        adapter_mask_rates=RATES, test_mask_rates=(0.0, *RATES), nested_mask_reps=args.replicates,
        gate="strength times actual missing fraction, with a bounded tanh residual",
        strength=args.strength, calibration="healthy Q90, clean-frozen and mask-specific",
        score_direction="shared between off and on; Z24 direction from native clean validation only",
        z24_protocol="locked records; median across windows; mask varies by record",
        hellbridge_protocol="P2_NM_Z, 50 fit/10 reference/20 calibration/720 test windows; "
                            "healthy test windows share source recordings with development",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    metrics, classes, separation = [], [], []
    for dataset in ("HellBridge", "Z24"):
        bridge_out = out / dataset
        args.train_log.clear()
        args.reference_log.clear()
        if dataset == "HellBridge":
            rows, masks, fits = run_hellbridge(args, device, bridge_out, make_block=ResidualAudit)
        else:
            rows, masks, fits = run_z24(args, device, make_block=ResidualAudit)
        metrics.append(evaluate(dataset, rows, masks, fits, bridge_out, methods=METHODS, share_native_direction=True))
        pd.DataFrame(args.train_log).to_csv(bridge_out / "adapter_train.csv", index=False)
        pd.DataFrame(args.reference_log).to_csv(bridge_out / "healthy_reference.csv", index=False)
        by_class, ranking = log_test_diagnostics(dataset, bridge_out)
        classes.append(by_class)
        separation.append(ranking)

    all_metrics = pd.concat(metrics, ignore_index=True)
    group = ["dataset", "condition", "method", "policy"]
    all_metrics.groupby(group, as_index=False)[list(FIELDS)].mean().to_csv(out / "summary.csv", index=False)
    method_deltas(all_metrics, METHODS[1], METHODS[:1]).to_csv(out / "paired_deltas.csv", index=False)
    pd.concat(classes, ignore_index=True).to_csv(out / "class_impact_by_mask.csv", index=False)
    pd.concat(separation, ignore_index=True).to_csv(out / "separation_by_mask.csv", index=False)
    print(f"POST_BLOCK_AUDIT_PASS output={out}", flush=True)


if __name__ == "__main__":
    main()
