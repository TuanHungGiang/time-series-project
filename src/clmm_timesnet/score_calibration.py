"""Paired residual, mask-aware training, and calibration trials for TimesNet."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data.hellbridge import Normalizer, development_split, load_windows
from .data.masking import sensor_mask, stable_rng
from .data.z24 import load_manifest, read_stable_vertical, record_windows, split_training_windows, stable_sensor_map
from .experiment import BATCH, LOSS_RATES, MASK_REPLICATES, anchor_periods, score_rows, test_masks
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .models.backbone import TimesNet, fit_timesnet, reconstruct, seed_everything, timesnet_config


METHODS = ("observed_mean", "sensor_normalized")
MASK_METHODS = ("observed_mean", "mask_aware")
FIELDS = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_at_fpr_05", "fp", "tp"]


def location_scale(values, axis=None):
    center = np.median(values, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(values - center), axis=axis, keepdims=True)
    spread = np.maximum(np.maximum(mad, 0.25 * values.std(axis=axis, keepdims=True)), 1e-4)
    return center, spread


def pool_residual(residual, kept, center, spread):
    observed = residual[:, kept]
    return {
        METHODS[0]: np.log1p(observed.mean(axis=1)),
        METHODS[1]: ((np.log1p(observed) - center[:, kept]) / spread[:, kept]).mean(axis=1),
    }


class MaskAwareTimesNet(TimesNet):
    def __init__(self, config):
        super().__init__(config)
        self.mask_embedding = nn.Parameter(torch.zeros(config.d_model, config.enc_in))

    def forward(self, windows, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if mask is None:
            mask = torch.ones_like(windows[:, :1], dtype=torch.bool)
        visible = torch.where(mask, windows, 0.0)
        mean = visible.mean(dim=1, keepdim=True).detach()
        scale = (visible.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        # Missing channels have no window statistics. Predict in train-scaled units.
        scale = torch.where(mask, scale, 1.0)
        encoded = self.enc_embedding((visible - mean) / scale, None)
        encoded = encoded + nn.functional.linear((~mask).to(windows.dtype), self.mask_embedding)
        for layer in self.model:
            encoded = self.layer_norm(layer(encoded))
        return self.projection(encoded) * scale + mean


def fit_mask_aware(fit, device, seed, epochs):
    seed_everything(seed)
    channels, length = fit.shape[1:]
    model = MaskAwareTimesNet(timesnet_config(channels, length)).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(fit)), batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    rates = (0.0, *LOSS_RATES)
    for epoch in range(1, epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            available = np.stack([
                sensor_mask(channels, rng.choice(rates), rng.permutation(channels))
                for _ in range(len(batch))
            ])
            kept = torch.as_tensor(available, device=device).unsqueeze(1)
            error = (model(batch.transpose(1, 2), mask=kept) - batch.transpose(1, 2)).square().mean(dim=1)
            kept = kept.squeeze(1)
            observed = (error * kept).sum(1) / kept.sum(1)
            hidden = (error * ~kept).sum(1) / (~kept).sum(1).clamp_min(1)
            loss = (observed + 0.25 * hidden).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  mask-aware epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    return model.eval().requires_grad_(False)


def fit_detector(raw_fit, raw_reference, device, seed, epochs, mask_aware=False):
    scaler = Normalizer.fit(raw_fit)
    fit = scaler.transform(raw_fit)
    seed_everything(seed)
    model = fit_timesnet(fit, epochs, BATCH, device)
    anchor_periods(model, torch.as_tensor(fit, device=device))
    candidate = fit_mask_aware(fit, device, seed, epochs) if mask_aware else None
    if candidate is not None:
        anchor_periods(candidate, torch.as_tensor(fit, device=device))
    full = np.ones(fit.shape[1], dtype=bool)

    @torch.no_grad()
    def channel_residual(values, kept):
        x = torch.as_tensor(values, device=device)
        mask = torch.as_tensor(kept, device=device).view(1, -1, 1)
        masked = torch.where(mask, x, 0.0)
        return np.concatenate([
            (part - reconstruct(model, part)).square().mean(-1).cpu().numpy()
            for part in masked.split(BATCH)
        ]).astype(np.float64)

    @torch.no_grad()
    def candidate_score(values, kept):
        x = torch.as_tensor(values, device=device).transpose(1, 2)
        mask = torch.as_tensor(kept, device=device).view(1, 1, -1)
        residual = np.concatenate([
            (batch - candidate(batch, mask=mask)).square().mean(1).cpu().numpy()
            for batch in x.split(BATCH)
        ]).astype(np.float64)
        return np.log1p(residual[:, kept].mean(axis=1))

    if candidate is None:
        center, spread = location_scale(np.log1p(channel_residual(fit, full)), axis=0)

    def raw_scores(values, kept):
        error = channel_residual(values, kept)
        if candidate is None:
            return pool_residual(error, kept, center, spread)
        return {
            MASK_METHODS[0]: np.log1p(error[:, kept].mean(axis=1)),
            MASK_METHODS[1]: candidate_score(values, kept),
        }

    refs = {name: location_scale(values)
            for name, values in raw_scores(scaler.transform(raw_reference), full).items()}

    def predict(raw, kept):
        scores = raw_scores(scaler.transform(raw), kept)
        return {name: (s - refs[name][0]) / refs[name][1] for name, s in scores.items()}

    return predict


def evaluate(dataset, rows, out, mask_aware=False):
    frame = pd.DataFrame(rows)
    keys = ["sample_id", "split", "method", "condition", "replicate"]
    if frame.duplicated(keys).any() or not np.isfinite(frame.score).all():
        raise ValueError(f"Invalid scores for {dataset}")
    directions = frame[frame.split.eq("direction")]
    sign = 1
    direction_auc = None
    if dataset == "Z24":
        base = directions[directions.method.eq(METHODS[0])]
        if len(base) != 132 or base.label.value_counts().to_dict() != {0: 6, 1: 126}:
            raise ValueError("Expected 6 healthy and 126 damage validation records for score orientation")
        direction_auc = float(roc_auc_score(base.label, base.score))
        sign = -1 if direction_auc < 0.5 else 1
    signs = dict.fromkeys(MASK_METHODS if mask_aware else METHODS, sign)
    candidate_direction_auc = None
    if mask_aware and dataset == "Z24":
        candidate = directions[directions.method.eq(MASK_METHODS[1])]
        if candidate.label.value_counts().to_dict() != {0: 6, 1: 126}:
            raise ValueError("Invalid candidate validation cohort for score orientation")
        candidate_direction_auc = float(roc_auc_score(candidate.label, candidate.score))
        signs[MASK_METHODS[1]] = -1 if candidate_direction_auc < 0.5 else 1
    frame["oriented_score"] = frame.score * frame.method.map(signs)
    frame.to_csv(out / f"{dataset}_scores.csv.gz", index=False)
    results = []
    for method, group in frame.groupby("method"):
        val = group[group.split.eq("val")]
        clean = val[val.condition.eq("clean")]
        for (condition, rep), trial in group[group.split.eq("test")].groupby(["condition", "replicate"]):
            matching = val[val.condition.eq(condition) & val.replicate.eq(rep)]
            n_val, n_healthy, n_damage = (6, 6, 126) if dataset == "Z24" else (20, 80, 640)
            if any(len(cohort) != n_val or not cohort.label.eq(0).all() for cohort in (clean, matching)):
                raise ValueError(f"Invalid healthy calibration cohort for {dataset}/{condition}/{rep}")
            y, s = trial.label.to_numpy(), trial.oriented_score.to_numpy()
            if ((y == 0).sum(), (y == 1).sum()) != (n_healthy, n_damage):
                raise ValueError(f"Invalid test cohort for {dataset}/{condition}/{rep}")
            for policy, reference in (("clean_q90", clean), ("masked_q90", matching)):
                threshold = threshold_from_healthy(reference.oriented_score.to_numpy(), 0.90)
                binary = s >= threshold
                results.append(dict(dataset=dataset, method=method, condition=condition, replicate=rep,
                                    policy=policy, sign=signs[method], threshold=threshold,
                                    **publication_metrics(y, s, threshold), recall_at_fpr_05=recall_at_fpr(y, s, 0.05),
                                    fp=int((binary & (y == 0)).sum()), tp=int((binary & (y == 1)).sum())))
    if len(results) != 64:
        raise ValueError(f"Expected 64 metric rows on {dataset}, got {len(results)}")
    audit = dict(dataset=dataset, direction_sign=sign, validation_upper_auc=direction_auc)
    if mask_aware:
        audit.update(candidate_sign=signs[MASK_METHODS[1]], candidate_validation_upper_auc=candidate_direction_auc)
    return pd.DataFrame(results), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--hell-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270905)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mask-aware", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split = out / "hellbridge_split"
    subprocess.run([sys.executable, "-m", "clmm_timesnet.data.prepare_hellbridge", "--h5", str(args.hell_h5),
                    "--output-dir", str(split), "--profile", "svendsen2022", "--regime", "P2_NM_Z",
                    "--seed", "20261113", "--strict-paper-counts"], check=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HellBridge {args.seed=} {device=}", flush=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, cal, test = development_split(manifest)
    raw = load_windows(args.hell_h5, manifest, 128)
    predict = fit_detector(raw[fit], raw[cal], device, args.seed, args.epochs, args.mask_aware)
    rng = np.random.default_rng(20261119)
    orders = [rng.permutation(40) for _ in range(MASK_REPLICATES)]
    rows = []
    for condition, rep, kept in test_masks(40, orders):
        for role, ids in (("val", cal), ("test", test)):
            score_rows(rows, predict(raw[ids], kept), manifest.loc[ids, "sample_id"],
                       manifest.loc[ids, "label"], role, condition, rep)
    hell, hell_audit = evaluate("HellBridge", rows, out, args.mask_aware)
    del predict, raw, rows

    manifest = load_manifest(args.z24_root)
    rows = []
    for index, ((setup, excitation), channels) in enumerate(sorted(stable_sensor_map(manifest).items())):
        print(f"Z24 {setup=} {excitation=}", flush=True)
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            return record_windows(read_stable_vertical(Path(row.path), channels), 1024, 1024, "raw_temporal", 2, 128)

        fit, ref = split_training_windows([load(row) for row in group[group.split.eq("train")].itertuples()])
        predict = fit_detector(fit, ref, device, args.seed + index * 1009, args.epochs, args.mask_aware)
        full = np.ones(len(channels), dtype=bool)
        for row in group[group.split.isin(["val", "test"])].itertuples():
            raw = load(row)
            orders = [stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                      for rep in range(MASK_REPLICATES)]
            cases = test_masks(len(channels), orders) if row.split == "test" or row.label == 0 else [("clean", 0, full)]
            for condition, rep, kept in cases:
                scores = {name: [np.median(s)] for name, s in predict(raw, kept).items()}
                if row.split == "test" or row.label == 0:
                    score_rows(rows, scores, [row.record_id], [row.label], row.split, condition, rep)
                if row.split == "val" and condition == "clean":
                    score_rows(rows, scores, [row.record_id], [row.label], "direction", condition, rep)
        del predict
    z24, z24_audit = evaluate("Z24", rows, out, args.mask_aware)
    result = pd.concat([z24, hell], ignore_index=True).assign(seed=args.seed)
    result.to_csv(out / "metrics_per_mask.csv", index=False)
    summary = result.groupby(["dataset", "condition", "method", "policy"], as_index=False)[FIELDS].mean()
    summary.to_csv(out / "summary.csv", index=False)
    if args.mask_aware:
        keys = ["dataset", "condition", "replicate", "policy"]
        paired = result.pivot(index=keys, columns="method", values=["auc", "ba", "recall", "fpr", "recall_at_fpr_05"])
        deltas = paired.xs(MASK_METHODS[1], axis=1, level=1) - paired.xs(MASK_METHODS[0], axis=1, level=1)
        deltas.add_prefix("delta_").to_csv(out / "paired_deltas.csv")
    (out / "protocol.json").write_text(json.dumps({
        "seed": args.seed, "epochs": args.epochs, "orientation": [z24_audit, hell_audit],
        "mask_matching": {"HellBridge": "same exact sensor indices", "Z24": "same rate and replicate, record-specific indices"},
        "fit": "healthy-only", "mask_aware": args.mask_aware,
        "Z24_direction": ("same labeled clean validation rule applied separately per model"
                          if args.mask_aware else "labeled clean validation only, shared across scores"),
        "threshold_fit": "healthy validation only", "test_labels": "metrics only",
    }, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print("TIMESNET_SCORE_CALIBRATION_PASS", flush=True)


if __name__ == "__main__":
    main()
