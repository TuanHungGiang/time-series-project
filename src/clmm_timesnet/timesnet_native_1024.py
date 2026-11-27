"""Native-resolution TimesNet and a controlled availability-embedding ablation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import detrend
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .data.hellbridge import Normalizer, development_split, read_al40
from .data.masking import sensor_mask, stable_rng
from .data.z24 import load_manifest, read_stable_vertical, split_training_windows, stable_sensor_map
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .models.backbone import fit_timesnet, reconstruct, seed_everything, timesnet_config
from .score_calibration import MaskAwareTimesNet

LENGTH = 1024
MASK_REPS = range(3)
RATE = 0.50
SEED = 20270905
CONDITION = "sensor_0.50"
METHODS = ("native", "sensor_dropout", "sensor_dropout_mask")
TRAIN_RATES = (0.0, 0.10, 0.30, 0.50)
PROBE_STAGES = ("embedding", "timesblock_1", "timesblock_2", "projection")
METRICS = ["auc", "ap", "ba", "recall", "fpr", "precision", "f1", "fp", "recall_low_fpr", "rank_reversal"]


def load_z24(path, channels):
    signals = read_stable_vertical(path, channels)
    count = signals.shape[-1] // LENGTH
    if count < 16:
        raise ValueError(f"Too few Z24 windows in {path}")
    windows = signals[:, :count * LENGTH].reshape(len(channels), count, LENGTH).transpose(1, 0, 2)
    return detrend(windows, axis=-1, type="linear").astype(np.float32)


def load_hellbridge(h5, manifest):
    data = np.empty((len(manifest), 40, LENGTH), dtype=np.float32)
    with h5py.File(h5, "r") as source:
        for record, group in manifest.groupby("record_id"):
            signal = read_al40(source[record])
            for row in group.itertuples():
                values = signal[:, row.start:row.stop]
                if values.shape[-1] != LENGTH:
                    raise ValueError(f"Unexpected window length for {row.sample_id}")
                data[row.Index] = detrend(values, axis=-1, type="linear")
            print(f"  load {record} windows={len(group)}", flush=True)
    return data


def subset(data, count):
    return data[np.linspace(0, len(data) - 1, min(len(data), count), dtype=int)]


def fit_with_dropout(fit, args, seed, device, method):
    channels, length = fit.shape[1:]
    model = MaskAwareTimesNet(timesnet_config(channels, length)).to(device)
    model.mask_embedding.requires_grad_(method == "sensor_dropout_mask")
    loader = DataLoader(TensorDataset(torch.from_numpy(fit)), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            target = batch.to(device).transpose(1, 2)
            available = np.stack([
                sensor_mask(channels, rng.choice(TRAIN_RATES), rng.permutation(channels))
                for _ in batch
            ])
            kept = torch.as_tensor(available, device=device)
            prediction = model(target, mask=kept.unsqueeze(1))
            error = (target - prediction).square().mean(dim=1)
            loss = ((error * kept).sum(1) / kept.sum(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.detach().item())
        if epoch in (1, args.epochs) or epoch % 5 == 0:
            print(f"  {method} epoch={epoch}/{args.epochs} loss={np.mean(losses):.6f}", flush=True)
    return model.eval().requires_grad_(False)


class Detector:
    def __init__(self, fit, reference, seed, args, device, method="native", epoch_callback=None):
        self.method = method
        self.device = device
        self.batch_size = args.batch_size
        self.scaler = Normalizer.fit(fit)
        self.full = np.ones(fit.shape[1], dtype=bool)
        seed_everything(seed)
        fit = self.scaler.transform(fit)
        if method == "native":
            self.model = fit_timesnet(fit, args.epochs, self.batch_size, device, epoch_callback=epoch_callback)
        else:
            self.model = fit_with_dropout(fit, args, seed, device, method)
        self.reference_center = 0.0
        self.reference_scale = 1.0
        ref, activations = self.score(self.scaler.transform(reference), self.full, capture=args.layer_probe)
        self.probe_reference = {}
        for name in PROBE_STAGES:
            if name in activations:
                features = stage_features(activations[name])
                center = np.median(features, axis=0)
                mad = 1.4826 * np.median(np.abs(features - center), axis=0)
                scale = np.maximum(np.maximum(mad, 0.25 * features.std(axis=0)), 1e-3)
                self.probe_reference[name] = (center, scale)
        self.reference_center = np.median(ref)
        self.reference_scale = max(1.4826 * np.median(np.abs(ref - self.reference_center)),
                                   0.25 * ref.std(), 1e-4)

    @torch.no_grad()
    def score(self, data, kept, capture=False):
        x = torch.as_tensor(data, device=self.device)
        mask = torch.as_tensor(kept, device=self.device)
        logs = {}

        def save(name, fn):
            def hook(layer, args, output):
                logs.setdefault(name, []).append(fn(layer, args, output).detach().cpu().numpy())
            return hook

        def bins(layer, args, output):
            spectrum = torch.fft.rfft(args[0], dim=1).abs().mean(0).mean(-1)
            spectrum[0] = 0.0
            return spectrum.topk(layer.k).indices.expand(len(args[0]), -1)

        handles = []
        if capture:
            def embedding_input(_layer, inputs):
                logs.setdefault("embedding", []).append(inputs[0].detach().cpu().numpy())
            handles.append(self.model.model[0].register_forward_pre_hook(embedding_input))
            for i, block in enumerate(self.model.model, 1):
                handles.append(block.register_forward_hook(save(f"timesblock_{i}", lambda _, a, y: y)))
                handles.append(block.register_forward_hook(save(f"fft_bins_{i}", bins)))
            handles.append(self.model.projection.register_forward_hook(save("projection", lambda _, a, y: y)))

        errors = []
        try:
            for chunk in x.split(self.batch_size):
                visible = torch.where(mask.view(1, -1, 1), chunk, 0.0)
                if self.method == "native":
                    prediction = reconstruct(self.model, visible)
                else:
                    prediction = self.model(visible.transpose(1, 2), mask=mask.view(1, 1, -1)).transpose(1, 2)
                errors.append((prediction[:, mask] - chunk[:, mask]).square().mean((1, 2)).cpu().numpy())
        finally:
            for handle in handles:
                handle.remove()
        scores = np.log1p(np.concatenate(errors).astype(np.float64))
        return (scores - self.reference_center) / self.reference_scale, {
            name: np.concatenate(parts) for name, parts in logs.items()
        }


def stage_features(activation):
    mean = activation.mean(axis=1, dtype=np.float64)
    log_std = np.log(np.maximum(activation.std(axis=1, dtype=np.float64), 1e-6))
    return np.concatenate((mean, log_std), axis=1)


def probe_layer_scores(activations, reference):
    return {
        name: np.abs((stage_features(activations[name]) - center) / scale).mean(axis=1)
        for name, (center, scale) in reference.items()
    }


def evaluate_layer_probes(dataset, rows, out):
    table = pd.DataFrame(rows)
    keys = ["stage", "condition", "replicate", "sample_id"]
    if table.duplicated(keys).any() or not np.isfinite(table.score).all():
        raise ValueError("Invalid layer probe scores")
    measures = []
    expected = {0: 6, 1: 126} if dataset == "Z24" else {0: 80, 1: 640}
    low_fpr = 0.0 if dataset == "Z24" else 0.05
    for (stage, condition, rep), group in table.groupby(keys[:-1]):
        if group.label.value_counts().to_dict() != expected:
            raise ValueError(f"Unexpected layer probe cohort for {dataset}/{stage}/{condition}/{rep}")
        measures.append(dict(
            stage=stage, condition=condition, replicate=rep, auc=roc_auc_score(group.label, group.score),
            recall_low_fpr=recall_at_fpr(group.label, group.score, low_fpr),
        ))
    metrics = pd.DataFrame(measures)
    if len(metrics) != len(PROBE_STAGES) * (1 + len(MASK_REPS)):
        raise ValueError(f"Incomplete layer probe results for {dataset}")
    metrics.to_csv(out / "layer_probe_metrics_by_mask.csv", index=False)
    table.to_csv(out / "layer_probe_scores.csv.gz", index=False)
    fields = ["auc", "recall_low_fpr"]
    means = metrics.groupby(["stage", "condition"])[fields].mean().unstack("condition")
    summary = means.xs("clean", level="condition", axis=1).add_prefix("clean_").join(
        means.xs(CONDITION, level="condition", axis=1).add_prefix("masked_")
    ).reindex(PROBE_STAGES)
    for field in fields:
        summary[f"delta_{field}"] = summary[f"masked_{field}"] - summary[f"clean_{field}"]
    summary.to_csv(out / "layer_probe_summary.csv")
    print(f"=== {dataset} | frozen activation probe ===", flush=True)
    print(summary.round(4).to_string(), flush=True)


def layer_drift(rows, clean, masked, labels, group, rep):
    labels = np.asarray(labels)
    for name, reference in clean.items():
        if name.startswith("fft_bins"):
            retained = (reference[:, :, None] == masked[name][:, None, :]).any(-1)
            shift = 1.0 - retained.mean(-1)
            measure = "fraction_of_clean_fft_bins_replaced"
        else:
            axes = tuple(range(1, reference.ndim))
            baseline = np.sqrt(np.mean(reference ** 2, axis=axes))
            shift = np.sqrt(np.mean((masked[name] - reference) ** 2, axis=axes)) / np.maximum(baseline, 1e-6)
            measure = "relative_rms_shift"
        for label in np.unique(labels):
            selected = shift[labels == label]
            rows.append(dict(group=group, stage=name, measure=measure, label=int(label),
                             replicate=rep, n=len(selected), value=float(selected.mean())))


def save_scores(rows, metadata, score, split, condition, rep):
    for row, value in zip(metadata, score, strict=True):
        rows.append(dict(sample_id=row.sample_id, state=row.state, label=row.label,
                         split=split, condition=condition, replicate=rep, score=float(value)))


def evaluate(dataset, rows, drift, out, sign=None):
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    group_cols = ["split", "condition", "replicate"]
    if scores.duplicated(group_cols + ["sample_id"]).any() or not np.isfinite(scores.score).all():
        raise ValueError("Invalid scores")
    val = scores[scores.split.eq("val")]
    test = scores[scores.split.eq("test")]
    if sign is None:
        sign = 1
        if dataset == "Z24":
            clean_val = val[val.condition.eq("clean")]
            sign = -1 if roc_auc_score(clean_val.label, clean_val.score) < .5 else 1
    scores["oriented_score"] = scores.score * sign
    scores.to_csv(out / "scores.csv.gz", index=False)
    expected = {0: 6, 1: 126} if dataset == "Z24" else {0: 80, 1: 640}
    low_fpr = 0.0 if dataset == "Z24" else .05
    clean = test.loc[test.condition.eq("clean")].set_index("sample_id").score * sign
    metrics, states = [], []
    for (condition, rep), trial in test.groupby(["condition", "replicate"]):
        if trial.label.value_counts().to_dict() != expected:
            raise ValueError(f"Unexpected test cohort for {dataset}/{condition}/{rep}")
        cal = val.loc[val.condition.eq(condition) & val.replicate.eq(rep) & val.label.eq(0), "score"] * sign
        threshold = threshold_from_healthy(cal.to_numpy(), .90)
        y, oriented = trial.label.to_numpy(), trial.score.to_numpy() * sign
        aligned = clean.loc[trial.sample_id].to_numpy()
        clean_correct = aligned[y == 1, None] > aligned[y == 0][None, :]
        masked_wrong = oriented[y == 1, None] <= oriented[y == 0][None, :]
        metrics.append(dict(
            condition=condition, replicate=rep, score_sign=sign, threshold=threshold,
            **publication_metrics(y, oriented, threshold),
            fp=int(((oriented >= threshold) & (y == 0)).sum()),
            recall_low_fpr=recall_at_fpr(y, oriented, low_fpr),
            rank_reversal=float((clean_correct & masked_wrong).sum() / max(1, clean_correct.sum())),
        ))
        healthy = oriented[y == 0]
        low_cutoff = np.sort(healthy)[::-1][int(np.floor(low_fpr * len(healthy)))]
        for state, damage in trial[trial.label.eq(1)].groupby("state"):
            d = damage.score.to_numpy() * sign
            states.append(dict(state=state, condition=condition, replicate=rep,
                               auc=roc_auc_score(np.r_[np.zeros(len(healthy)), np.ones(len(d))], np.r_[healthy, d]),
                               recall_q90=(d >= threshold).mean(), recall_low_fpr=(d > low_cutoff).mean()))
    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics_by_mask.csv", index=False)
    summary = table.groupby("condition", as_index=False)[METRICS].mean()
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(states).to_csv(out / "damage_states_by_mask.csv", index=False)
    logs = pd.DataFrame(drift)
    logs.to_csv(out / "layer_by_mask.csv", index=False)
    logs["weighted"] = logs.value * logs.n
    agg = logs.groupby(["stage", "measure", "label"], as_index=False)[["n", "weighted"]].sum()
    agg["value"] = agg.weighted / agg.n
    agg.drop(columns="weighted").to_csv(out / "layer_summary.csv", index=False)
    print(f"=== {dataset} · 1024 · sign={sign} ===", flush=True)
    print(summary.round(4).to_string(index=False), flush=True)
    print(agg[["stage", "measure", "label", "value"]].round(4).to_string(index=False), flush=True)
    return sign


def score_group(model, values, metadata, cases, rows, drift, group, reduce_record=False, probe_rows=None):
    role = metadata.iloc[0].role if "role" in metadata else metadata.iloc[0].split
    split = "test" if role.startswith("test") else "val"
    labels = metadata.label.to_numpy()
    for condition, rep, kept in cases:
        score, stages = model.score(values, kept, capture=split == "test")
        if reduce_record:
            save_scores(rows, metadata.itertuples(), [np.median(score)], split, condition, rep)
        else:
            save_scores(rows, metadata.itertuples(), score, split, condition, rep)
        if split == "test":
            if probe_rows is not None:
                for name, values_by_window in probe_layer_scores(stages, model.probe_reference).items():
                    values_by_sample = [np.median(values_by_window)] if reduce_record else values_by_window
                    probe_rows.extend(dict(stage=name, condition=condition, replicate=rep, sample_id=uid,
                                           label=int(label), score=float(probe))
                                      for uid, label, probe in zip(metadata.sample_id, labels, values_by_sample, strict=True))
            if condition == "clean":
                clean = stages
            else:
                window_labels = np.full(len(values), labels[0]) if reduce_record else labels
                layer_drift(drift, clean, stages, window_labels, group, rep)
        del stages


def finalize_dataset(dataset, results, out):
    ablation = len(results) > 1
    sign = None
    for method, (rows, drift) in results.items():
        if ablation:
            print(f"=== {dataset} | {method} ===", flush=True)
        sign = evaluate(dataset, rows, drift, out / method if ablation else out, sign=sign)


def run_hellbridge(args, out, device, methods):
    out.mkdir(exist_ok=True)
    split = out / "split"
    subprocess.run([
        sys.executable, "-m", "clmm_timesnet.data.prepare_hellbridge", "--h5", str(args.hell_h5),
        "--output-dir", str(split), "--profile", "svendsen2022", "--regime", "P2_NM_Z",
        "--seed", "20261113", "--strict-paper-counts",
    ], check=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, val, test = development_split(manifest)
    raw = load_hellbridge(args.hell_h5, manifest)
    rng = np.random.default_rng(20261119)
    cases = [("clean", 0, np.ones(40, dtype=bool))] + [
        (CONDITION, rep, sensor_mask(40, RATE, rng.permutation(40))) for rep in MASK_REPS
    ]
    results = {method: ([], []) for method in methods}
    probe_rows = [] if args.layer_probe else None
    for method, (rows, drift) in results.items():
        print(f"HellBridge training {method}", flush=True)
        model = Detector(raw[fit], raw[val], SEED, args, device, method)
        for role, ids in (("val", val), ("test", test)):
            metadata = manifest.loc[ids].assign(role=role)
            score_group(model, model.scaler.transform(raw[ids]), metadata, cases,
                        rows, drift, "P2_NM_Z", probe_rows=probe_rows)
            print(f"HellBridge scored {method} {role}", flush=True)
        del model
    finalize_dataset("HellBridge", results, out)
    if probe_rows is not None:
        evaluate_layer_probes("HellBridge", probe_rows, out)


def run_z24(args, out, device, methods):
    out.mkdir(exist_ok=True)
    manifest = load_manifest(args.z24_root).assign(sample_id=lambda f: f.record_id)
    results = {method: ([], []) for method in methods}
    probe_rows = [] if args.layer_probe else None
    limit = args.z24_windows
    for index, ((setup, excitation), channels) in enumerate(sorted(stable_sensor_map(manifest).items())):
        key = f"{setup:02d}/{excitation.upper()}"
        print(f"Z24 {key} group={index + 1}/18 input=[N,{len(channels)},1024]", flush=True)
        group = manifest[manifest.setup.eq(setup) & manifest.excitation.eq(excitation)]
        fit, ref = split_training_windows([
            load_z24(Path(row.path), channels) for row in group[group.split.eq("train")].itertuples()
        ])
        fit, ref = subset(fit, limit), subset(ref, limit)
        # Build once so every branch sees identical windows and masks.
        cohorts = []
        full = np.ones(len(channels), dtype=bool)
        for idx in group.index[group.split.isin(["val", "test"])]:
            metadata = group.loc[[idx]]
            row = metadata.iloc[0]
            values = subset(load_z24(Path(row.path), channels), limit)
            cases = [("clean", 0, full)]
            if row.split == "test" or row.label == 0:
                for rep in MASK_REPS:
                    order = stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                    cases.append((CONDITION, rep, sensor_mask(len(channels), RATE, order)))
            cohorts.append((metadata, values, cases))
        for method, (rows, drift) in results.items():
            print(f"  training {method}", flush=True)
            model = Detector(fit, ref, SEED + 1009 * index, args, device, method)
            for metadata, values, cases in cohorts:
                score_group(model, model.scaler.transform(values), metadata, cases,
                            rows, drift, key, reduce_record=True, probe_rows=probe_rows)
            del model
    finalize_dataset("Z24", results, out)
    if probe_rows is not None:
        evaluate_layer_probes("Z24", probe_rows, out)


def paired_ablation(out):
    rows = []
    for bridge in ("Z24", "HellBridge"):
        by_method = {
            method: pd.read_csv(out / bridge / method / "metrics_by_mask.csv")
            .set_index(["condition", "replicate"])
            for method in METHODS
        }
        fields = ["auc", "recall_low_fpr", "recall", "fpr", "ba", "rank_reversal"]
        for after, before, contrast in [
            ("sensor_dropout", "native", "dropout_minus_native"),
            ("sensor_dropout_mask", "sensor_dropout", "mask_minus_dropout"),
        ]:
            if not by_method[after].index.equals(by_method[before].index):
                raise ValueError(f"Cannot pair different condition/mask IDs on {bridge}")
            delta = (by_method[after][fields] - by_method[before][fields]).add_prefix("delta_")
            rows.append(delta.reset_index().assign(dataset=bridge, contrast=contrast))
    pd.concat(rows, ignore_index=True).to_csv(out / "paired_deltas.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--z24-windows", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--embedding-ablation", action="store_true")
    mode.add_argument("--layer-probe", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.z24_windows) < 1:
        parser.error("epochs, batch size, and window count must be positive")
    methods = METHODS if args.embedding_ablation else METHODS[:1]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (out / "protocol.json").write_text(json.dumps(dict(
        models=list(methods), fft="native batch-adaptive FFT", seed=SEED, device=str(device),
        window_points=LENGTH, detrend="linear", resampling=False, epochs=args.epochs,
        batch_size=args.batch_size, z24_windows=args.z24_windows,
        mask_reps=list(MASK_REPS), train_sensor_loss={
            method: ([0] if method == "native" else list(TRAIN_RATES)) for method in methods
        }, test_sensor_loss=[0, RATE],
        dropout_loss="MSE on observed channels only, shared by dropout branches",
        layer_probe=("Per-stage mean/log-std distance to clean held-out healthy reference"
                     if args.layer_probe else None),
        reference="healthy clean only; Z24 native clean validation orientation shared by all branches; HellBridge upper",
    ), indent=2), encoding="utf-8")
    run_hellbridge(args, out / "HellBridge", device, methods)
    run_z24(args, out / "Z24", device, methods)
    combined = pd.concat([
        pd.read_csv((out / bridge / method if args.embedding_ablation else out / bridge) / "summary.csv")
        .assign(dataset=bridge, method=method)
        for bridge in ("Z24", "HellBridge") for method in methods
    ], ignore_index=True)
    if args.embedding_ablation:
        paired_ablation(out)
    combined.to_csv(out / "summary.csv", index=False)
    print("TIMESNET_NATIVE_1024_PASS", flush=True)


if __name__ == "__main__":
    main()
