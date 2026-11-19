"""One healthy-only TimesNet/CLMM ablation on Z24 and HellBridge."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from run_clmm_multi_detector import recall_at_fpr, sensor_mask
from run_hellbridge_source_baselines import (
    Normalizer, development_split, fit_reconstructor, load_windows, model_forward,
    publication_metrics, seed_everything, threshold_from_healthy,
)
from run_z24_care_usad_v18 import CrossSensorAdapter
from run_z24_missingness_baselines import stable_rng
from run_z24_source_baselines import (
    load_manifest, read_stable_vertical, record_windows, robust_score_reference,
    split_training_windows, stable_sensor_map,
)


METHODS = ("timesnet", "timesnet_anchored", "timesnet_observed", "clmm_fill", "clmm_cross_predict")
BATCH = 16
PROBES = 4


def probe_groups(available):
    observed = np.flatnonzero(available)
    return [observed[group::PROBES] for group in range(PROBES) if group < len(observed)]


def make_training_masks(rng, count, channels):
    kept, probes = [], []
    for _ in range(count):
        order = rng.permutation(channels)
        available = sensor_mask(channels, float(rng.choice([0.0, 0.10, 0.20, 0.30])), order)
        observed = order[available[order]]
        target = np.zeros(channels, dtype=bool)
        target[observed[int(rng.integers(PROBES))::PROBES]] = True
        kept.append(available)
        probes.append(target)
    return np.stack(kept), np.stack(probes)


def complete(adapter, x, visible):
    return torch.where(visible, x, adapter(x * visible, visible.squeeze(-1).to(x.dtype)))


def observed_mean(error, mask):
    return (error.mean(-1) * mask).sum(1) / mask.sum(1)


@torch.no_grad()
def anchor_periods(model, fit):
    anchors = []
    for layer in model.model:
        spectrum = None

        def collect(_layer, args):
            nonlocal spectrum
            total = torch.fft.rfft(args[0], dim=1).abs().mean(-1).sum(0)
            spectrum = total if spectrum is None else spectrum + total

        handle = layer.register_forward_pre_hook(collect)
        try:
            for batch in fit.split(BATCH):
                model_forward("timesnet", model, batch)
        finally:
            handle.remove()
        layer.period_bins = tuple((spectrum[1:].topk(layer.k).indices + 1).tolist())
        anchors.append(layer.period_bins)
    return anchors


def fit_adapter(mode, model, train, device, seed, epochs):
    seed_everything(seed)
    _, channels, length = train.shape
    adapter = CrossSensorAdapter(channels, length, 64, 4).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        losses = []
        for ids in np.array_split(rng.permutation(len(train)), max(1, (len(train) + BATCH - 1) // BATCH)):
            x = train[ids.tolist()]
            available, probe = (torch.as_tensor(mask, device=device) for mask in make_training_masks(rng, len(x), channels))
            if mode == "fill":
                filled = complete(adapter, x, available.unsqueeze(-1))
                hidden = ~available
                # Keep this control a plain healthy-only imputer.
                point_loss = F.smooth_l1_loss(filled, x, reduction="none")
                loss = (point_loss.mean(-1) * hidden).sum() / hidden.sum().clamp_min(1)
                if not hidden.any():
                    continue
            else:
                filled = complete(adapter, x, (available & ~probe).unsqueeze(-1))
                prediction = model_forward("timesnet", model, filled)
                loss = observed_mean(F.smooth_l1_loss(prediction, x, reduction="none"), probe).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  {mode=} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    return adapter.eval().requires_grad_(False)


def fit_group(raw_fit, raw_reference, device, seed, epochs, adapter_epochs):
    normalizer = Normalizer.fit(raw_fit)
    fit = normalizer.transform(raw_fit)
    seed_everything(seed)
    model = fit_reconstructor("timesnet", fit, epochs, BATCH, device).requires_grad_(False)
    train = torch.as_tensor(fit, device=device)
    reference = torch.as_tensor(normalizer.transform(raw_reference), device=device)
    audit_x = reference[:min(BATCH, len(reference))]

    def batch_error(x):
        return (x - model_forward("timesnet", model, x)).square().mean((1, 2))

    @torch.no_grad()
    def batch_delta():
        alone = torch.cat([batch_error(x) for x in audit_x.split(1)])
        grouped = batch_error(audit_x)
        return float((alone - grouped).abs().mean())

    original_delta = batch_delta()
    anchors = anchor_periods(model, train)
    anchored_delta = batch_delta()
    fill = fit_adapter("fill", model, train, device, seed + 97, adapter_epochs)
    cross = fit_adapter("cross", model, train, device, seed + 97, adapter_epochs)

    @torch.no_grad()
    def score(raw, available):
        x_all = torch.as_tensor(normalizer.transform(raw), device=device)
        chunks = {method: [] for method in METHODS}
        visible = torch.as_tensor(available, device=device).view(1, -1, 1)
        for x in x_all.split(BATCH):
            masked = torch.where(visible, x, 0)
            mask = visible.squeeze(-1).expand(len(x), -1)
            for layer in model.model:
                layer.period_bins = None
            chunks["timesnet"].append(batch_error(masked).cpu().numpy())
            for layer, bins in zip(model.model, anchors, strict=True):
                layer.period_bins = bins
            error = (masked - model_forward("timesnet", model, masked)).square()
            chunks["timesnet_anchored"].append(error.mean((1, 2)).cpu().numpy())
            chunks["timesnet_observed"].append(observed_mean(error, mask).cpu().numpy())
            filled = complete(fill, x, visible.expand(len(x), -1, -1)) if not available.all() else x
            residual = (x - model_forward("timesnet", model, filled)).square()
            chunks["clmm_fill"].append(observed_mean(residual, mask).cpu().numpy())
            probe_total = x.new_zeros(len(x))
            for channels in probe_groups(available):
                shown = visible.clone()
                shown[:, channels, :] = False
                input_values = complete(cross, x, shown.expand(len(x), -1, -1))
                prediction = model_forward("timesnet", model, input_values)
                probe_total += (prediction[:, channels] - x[:, channels]).square().mean(-1).sum(1)
            chunks["clmm_cross_predict"].append((probe_total / int(available.sum())).cpu().numpy())
        return {method: np.concatenate(values) for method, values in chunks.items()}

    full = np.ones(raw_fit.shape[1], dtype=bool)
    ref = {method: robust_score_reference(values) for method, values in score(raw_reference, full).items()}

    def predict(raw, available):
        return {method: (np.log1p(values) - ref[method][0]) / ref[method][1]
                for method, values in score(raw, available).items()}

    return predict, {"batch_delta_original": original_delta, "batch_delta_anchored": anchored_delta, "bins": anchors}


def score_rows(rows, predicted, ids, labels, role, condition, rep):
    for method, values in predicted.items():
        rows.extend(dict(sample_id=uid, label=int(y), split=role, condition=condition,
                         replicate=rep, method=method, score=float(s))
                    for uid, y, s in zip(ids, labels, values, strict=True))


def evaluate(dataset, rows, output):
    frame = pd.DataFrame(rows)
    keys = ["sample_id", "split", "condition", "replicate", "method"]
    if frame.duplicated(keys).any() or not np.isfinite(frame.score).all():
        raise ValueError(f"Invalid scores: {dataset}")
    frame.to_csv(output / f"{dataset}_scores.csv.gz", index=False)
    results = []
    expected = (6, 6, 126) if dataset == "Z24" else (20, 80, 640)
    for method, group in frame.groupby("method"):
        valid = group[group.split.eq("val")]
        if len(valid) != expected[0] or not valid.label.eq(0).all():
            raise ValueError(f"Incorrect calibration cohort: {dataset}/{method}")
        threshold = threshold_from_healthy(valid.score.to_numpy(), 0.90)
        for (condition, rep), trial in group[group.split.eq("test")].groupby(["condition", "replicate"]):
            y, s = trial.label.to_numpy(), trial.score.to_numpy()
            if ((y == 0).sum(), (y == 1).sum()) != expected[1:]:
                raise ValueError(f"Incorrect test cohort: {dataset}/{method}/{condition}/{rep}")
            results.append(dict(dataset=dataset, method=method, condition=condition, replicate=rep,
                                threshold=threshold, **publication_metrics(y, s, threshold),
                                recall_at_fpr_05=recall_at_fpr(y, s, 0.05),
                                fp=int(((s >= threshold) & (y == 0)).sum()),
                                tp=int(((s >= threshold) & (y == 1)).sum())))
    if len(results) != 6 * len(METHODS):
        raise ValueError(f"Expected 6 conditions/masks per method on {dataset}")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--hell-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270905)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit = []

    print(f"HellBridge {args.seed=} {device=}", flush=True)
    manifest = pd.read_csv(args.hell_split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, cal, test = development_split(manifest)
    raw = load_windows(args.hell_h5, manifest, 128)
    predict, details = fit_group(raw[fit], raw[cal], device, args.seed, args.epochs, args.adapter_epochs)
    audit.append(dict(dataset="HellBridge", **details))
    full = np.ones(40, dtype=bool)
    rows = []
    score_rows(rows, predict(raw[cal], full), manifest.loc[cal, "sample_id"], manifest.loc[cal, "label"], "val", "clean", 0)
    rng = np.random.default_rng(20261119)
    masks = [("clean", 0, full)] + [("sensor_0.30", r, sensor_mask(40, 0.30, rng.permutation(40))) for r in range(5)]
    for condition, rep, available in masks:
        score_rows(rows, predict(raw[test], available), manifest.loc[test, "sample_id"], manifest.loc[test, "label"],
                   "test", condition, rep)
    hell = evaluate("HellBridge", rows, out)
    del predict, raw

    manifest = load_manifest(args.z24_root)
    rows = []
    for index, ((setup, excitation), channels) in enumerate(sorted(stable_sensor_map(manifest).items())):
        print(f"Z24 {setup=} {excitation=}", flush=True)
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            raw = read_stable_vertical(Path(row.path), channels)
            return record_windows(raw, 1024, 1024, "raw_temporal", 2, 128)

        train = group[group.split.eq("train") & group.label.eq(0)]
        fit, reference = split_training_windows([load(row) for row in train.itertuples()])
        predict, details = fit_group(fit, reference, device, args.seed + index * 1009, args.epochs, args.adapter_epochs)
        audit.append(dict(dataset="Z24", setup=setup, excitation=excitation, **details))
        full = np.ones(len(channels), dtype=bool)
        for row in group[group.split.eq("test") | (group.split.eq("val") & group.label.eq(0))].itertuples():
            raw = load(row)
            masks = [("clean", 0, full)]
            if row.split == "test":
                masks += [("sensor_0.30", r, sensor_mask(len(channels), 0.30,
                           stable_rng(str(row.record_id), "sensor", r).permutation(len(channels)))) for r in range(5)]
            for condition, rep, available in masks:
                pooled = {name: [np.median(values)] for name, values in predict(raw, available).items()}
                score_rows(rows, pooled, [row.record_id], [row.label], row.split, condition, rep)
        del predict

    result = pd.concat([hell, evaluate("Z24", rows, out)], ignore_index=True)
    result.insert(1, "seed", args.seed)
    result.to_csv(out / "metrics_per_mask.csv", index=False)
    fields = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_at_fpr_05", "fp", "tp"]
    summary = result.groupby(["dataset", "condition", "method"], as_index=False)[fields].mean()
    summary.to_csv(out / "summary.csv", index=False)
    (out / "period_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print("CLMM_TIMESNET_TWO_BRIDGES_PASS", flush=True)


if __name__ == "__main__":
    main()
