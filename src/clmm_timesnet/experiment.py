"""One healthy-only TimesNet/CLMM ablation on Z24 and HellBridge."""

import argparse
from importlib.metadata import version
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from .data.hellbridge import Normalizer, development_split, load_windows
from .data.masking import sensor_mask, stable_rng
from .data.z24 import load_manifest, read_stable_vertical, record_windows, split_training_windows, stable_sensor_map
from .metrics import publication_metrics, recall_at_fpr, robust_score_reference, threshold_from_healthy
from .models.adapters import CrossSensorAdapter, ObservedAttentionAdapter
from .models.backbone import fit_timesnet, reconstruct, seed_everything


METHODS = (
    "timesnet", "timesnet_anchored", "timesnet_observed", "clmm_fill",
    "clmm_cross_predict", "clmm_observed_attention", "clmm_time_frequency",
)
BATCH = 16
PROBES = 4
LOSS_RATES = (0.10, 0.30, 0.50)
MASK_REPLICATES = 5
SPECTRAL_WEIGHT = 0.20


def test_masks(channels, orders):
    if len(orders) != MASK_REPLICATES:
        raise ValueError(f"Expected {MASK_REPLICATES} sensor orders")
    yield "clean", 0, np.ones(channels, dtype=bool)
    for rate in LOSS_RATES:
        for replicate, order in enumerate(orders):
            yield f"sensor_{rate:.2f}", replicate, sensor_mask(channels, rate, order)


def probe_groups(available):
    observed = np.flatnonzero(available)
    return [observed[group::PROBES] for group in range(PROBES) if group < len(observed)]


def make_training_masks(rng, count, channels):
    kept, probes = [], []
    for _ in range(count):
        order = rng.permutation(channels)
        available = sensor_mask(channels, float(rng.choice((0.0, *LOSS_RATES))), order)
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


def log_spectral_power(signal):
    spectrum = torch.fft.rfft(signal, dim=-1, norm="ortho")[..., 1:]
    return torch.log1p(spectrum.real.square() + spectrum.imag.square())


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
                reconstruct(model, batch)
        finally:
            handle.remove()
        layer.period_bins = tuple((spectrum[1:].topk(layer.k).indices + 1).tolist())
        anchors.append(layer.period_bins)
    return anchors


def fit_adapter(mode, model, train, device, seed, epochs, *, observed_attention=False, spectral_weight=0.0):
    seed_everything(seed)
    _, channels, length = train.shape
    adapter_type = ObservedAttentionAdapter if observed_attention else CrossSensorAdapter
    adapter = adapter_type(channels, length, 64, 4).to(device)
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
                prediction = reconstruct(model, filled)
                loss = observed_mean(F.smooth_l1_loss(prediction, x, reduction="none"), probe).mean()
                if spectral_weight:
                    frequency_error = F.smooth_l1_loss(
                        log_spectral_power(prediction), log_spectral_power(x), reduction="none",
                    )
                    loss = loss + spectral_weight * observed_mean(frequency_error, probe).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  {mode=} {observed_attention=} {spectral_weight=} "
                  f"epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    return adapter.eval().requires_grad_(False)


def fit_group(raw_fit, raw_reference, device, seed, epochs, adapter_epochs):
    normalizer = Normalizer.fit(raw_fit)
    fit = normalizer.transform(raw_fit)
    seed_everything(seed)
    model = fit_timesnet(fit, epochs, BATCH, device)
    train = torch.as_tensor(fit, device=device)
    reference = torch.as_tensor(normalizer.transform(raw_reference), device=device)
    audit_x = reference[:min(BATCH, len(reference))]

    def batch_error(x):
        return (x - reconstruct(model, x)).square().mean((1, 2))

    @torch.no_grad()
    def batch_delta():
        alone = torch.cat([batch_error(x) for x in audit_x.split(1)])
        grouped = batch_error(audit_x)
        return float((alone - grouped).abs().mean())

    original_delta = batch_delta()
    anchors = anchor_periods(model, train)
    anchored_delta = batch_delta()
    fill = fit_adapter("fill", model, train, device, seed + 97, adapter_epochs)
    cross_adapters = {
        "clmm_cross_predict": fit_adapter("cross", model, train, device, seed + 97, adapter_epochs),
        "clmm_observed_attention": fit_adapter("cross", model, train, device, seed + 97, adapter_epochs,
                                              observed_attention=True),
        "clmm_time_frequency": fit_adapter("cross", model, train, device, seed + 97, adapter_epochs,
                                           observed_attention=True, spectral_weight=SPECTRAL_WEIGHT),
    }

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
            error = (masked - reconstruct(model, masked)).square()
            chunks["timesnet_anchored"].append(error.mean((1, 2)).cpu().numpy())
            chunks["timesnet_observed"].append(observed_mean(error, mask).cpu().numpy())
            filled = complete(fill, x, visible.expand(len(x), -1, -1)) if not available.all() else x
            residual = (x - reconstruct(model, filled)).square()
            chunks["clmm_fill"].append(observed_mean(residual, mask).cpu().numpy())
            for method, adapter in cross_adapters.items():
                probe_total = x.new_zeros(len(x))
                for channels in probe_groups(available):
                    shown = visible.clone()
                    shown[:, channels, :] = False
                    input_values = complete(adapter, x, shown.expand(len(x), -1, -1))
                    prediction = reconstruct(model, input_values)
                    probe_total += (prediction[:, channels] - x[:, channels]).square().mean(-1).sum(1)
                chunks[method].append((probe_total / int(available.sum())).cpu().numpy())
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
    expected_keys = {("clean", 0)} | {
        (f"sensor_{rate:.2f}", rep) for rate in LOSS_RATES for rep in range(MASK_REPLICATES)
    }
    for method in METHODS:
        actual_keys = {(row["condition"], row["replicate"]) for row in results if row["method"] == method}
        if actual_keys != expected_keys:
            raise ValueError(f"Unexpected condition/mask keys: {dataset}/{method}")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.seed = int(config["seed"])
    args.epochs = int(config["timesnet_epochs"])
    args.adapter_epochs = int(config["adapter_epochs"])
    fixed = {
        "sensor_loss": list(LOSS_RATES), "mask_replicates": MASK_REPLICATES, "batch_size": BATCH,
        "probe_groups": PROBES, "spectral_weight": SPECTRAL_WEIGHT,
    }
    for name, value in fixed.items():
        if config.get(name) != value:
            raise ValueError(f"{name} is fixed at {value} in this protocol")
    if args.epochs < 1 or args.adapter_epochs < 1:
        raise ValueError("Epoch counts must be positive")
    if not (args.z24_root / "metadata" / "records_manifest.csv").is_file():
        raise FileNotFoundError(args.z24_root / "metadata" / "records_manifest.csv")
    if not args.hell_h5.is_file():
        raise FileNotFoundError(args.hell_h5)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new experiment output directory: {out}")
    hell_out, z24_out, summary_out = (out / name for name in ("hellbridge", "z24", "summary"))
    for directory in (hell_out, z24_out, summary_out):
        directory.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    args.hell_split = hell_out / "split"
    subprocess.run([
        sys.executable, "-m", "clmm_timesnet.data.prepare_hellbridge",
        "--h5", str(args.hell_h5), "--output-dir", str(args.hell_split),
        "--profile", "svendsen2022", "--regime", "P2_NM_Z", "--seed", "20261113",
        "--strict-paper-counts",
    ], check=True)
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
    orders = [rng.permutation(40) for _ in range(MASK_REPLICATES)]
    for condition, rep, available in test_masks(40, orders):
        score_rows(rows, predict(raw[test], available), manifest.loc[test, "sample_id"], manifest.loc[test, "label"],
                   "test", condition, rep)
    hell = evaluate("HellBridge", rows, hell_out)
    hell.assign(seed=args.seed).to_csv(hell_out / "metrics_per_mask.csv", index=False)
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
                orders = [stable_rng(str(row.record_id), "sensor", r).permutation(len(channels))
                          for r in range(MASK_REPLICATES)]
                masks = test_masks(len(channels), orders)
            for condition, rep, available in masks:
                pooled = {name: [np.median(values)] for name, values in predict(raw, available).items()}
                score_rows(rows, pooled, [row.record_id], [row.label], row.split, condition, rep)
        del predict

    z24 = evaluate("Z24", rows, z24_out)
    z24.assign(seed=args.seed).to_csv(z24_out / "metrics_per_mask.csv", index=False)
    result = pd.concat([hell, z24], ignore_index=True)
    result.insert(1, "seed", args.seed)
    result.to_csv(summary_out / "metrics_per_mask.csv", index=False)
    fields = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_at_fpr_05", "fp", "tp"]
    summary = result.groupby(["dataset", "condition", "method"], as_index=False)[fields].mean()
    summary.to_csv(summary_out / "summary.csv", index=False)
    pairs = result.pivot(index=["dataset", "condition", "replicate"], columns="method", values=fields)
    comparisons = [*zip(METHODS[1:], METHODS[:-1]), (METHODS[-1], METHODS[0])]
    deltas = pd.concat([
        (pairs.xs(candidate, axis=1, level="method") - pairs.xs(control, axis=1, level="method"))
        .reset_index().assign(candidate=candidate, control=control)
        for candidate, control in comparisons
    ], ignore_index=True)
    deltas.to_csv(summary_out / "paired_deltas_per_mask.csv", index=False)
    deltas.groupby(["dataset", "condition", "candidate", "control"])[fields].mean().to_csv(
        summary_out / "paired_deltas_summary.csv"
    )
    (summary_out / "period_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out / "protocol.json").write_text(json.dumps({
        "seed": args.seed, "timesnet_epochs": args.epochs, "adapter_epochs": args.adapter_epochs,
        "methods": METHODS, "training_sensor_loss": [0.0, *LOSS_RATES],
        "test_sensor_loss": [0.0, *LOSS_RATES], "mask_replicates": MASK_REPLICATES,
        "probe_groups": PROBES, "fit": "healthy-only", "threshold": "clean healthy validation q90",
        "observed_attention": "absent and intentionally hidden sensors excluded as keys/values; all queries preserved",
        "spectral_loss": "SmoothL1 on non-DC log1p FFT power of probed sensors",
        "spectral_loss_weight": SPECTRAL_WEIGHT, "score": "upper-tail time-domain prediction residual for all cross-prediction variants",
        "mask_rule": "one permutation per replicate, reused across rates and methods",
        "mask_unit": {"HellBridge": "shared across test windows", "Z24": "record"},
        "evaluation_unit": {"HellBridge": "window", "Z24": "record median"},
        "runtime": {"python": sys.version.split()[0], "device": str(device),
                    **{name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "h5py", "pyyaml")}},
    }, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print("CLMM_TIMESNET_TWO_BRIDGES_PASS", flush=True)


if __name__ == "__main__":
    main()
