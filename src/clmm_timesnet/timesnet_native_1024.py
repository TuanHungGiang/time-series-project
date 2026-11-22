"""TimesNet reconstruction at native 1024-point resolution, with layer logs."""

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

from .data.hellbridge import Normalizer, development_split, read_al40
from .data.masking import sensor_mask, stable_rng
from .data.z24 import load_manifest, read_stable_vertical, split_training_windows, stable_sensor_map
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .models.backbone import fit_timesnet, reconstruct, seed_everything

LENGTH = 1024
MASK_REPS = range(3)
RATE = 0.50
SEED = 20270905
CONDITION = "sensor_0.50"
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


class Detector:
    def __init__(self, fit, reference, seed, args, device):
        self.device = device
        self.batch_size = args.batch_size
        self.scaler = Normalizer.fit(fit)
        self.full = np.ones(fit.shape[1], dtype=bool)
        seed_everything(seed)
        self.model = fit_timesnet(self.scaler.transform(fit), args.epochs, self.batch_size, device)
        self.reference_center = 0.0
        self.reference_scale = 1.0
        ref, _ = self.score(self.scaler.transform(reference), self.full)
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
            handles.append(self.model.enc_embedding.register_forward_hook(save("embedding", lambda _, a, y: y)))
            for i, block in enumerate(self.model.model, 1):
                handles.append(block.register_forward_hook(save(f"timesblock_{i}", lambda _, a, y: y)))
                handles.append(block.register_forward_hook(save(f"fft_bins_{i}", bins)))
            handles.append(self.model.projection.register_forward_hook(save("projection", lambda _, a, y: y)))

        errors = []
        try:
            for chunk in x.split(self.batch_size):
                visible = torch.where(mask.view(1, -1, 1), chunk, 0.0)
                prediction = reconstruct(self.model, visible)
                errors.append((prediction[:, mask] - chunk[:, mask]).square().mean((1, 2)).cpu().numpy())
        finally:
            for handle in handles:
                handle.remove()
        scores = np.log1p(np.concatenate(errors).astype(np.float64))
        return (scores - self.reference_center) / self.reference_scale, {
            name: np.concatenate(parts) for name, parts in logs.items()
        }


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


def evaluate(dataset, rows, drift, out):
    scores = pd.DataFrame(rows)
    group_cols = ["split", "condition", "replicate"]
    if scores.duplicated(group_cols + ["sample_id"]).any() or not np.isfinite(scores.score).all():
        raise ValueError("Invalid scores")
    val = scores[scores.split.eq("val")]
    test = scores[scores.split.eq("test")]
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


def run_hellbridge(args, out, device):
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
    model = Detector(raw[fit], raw[val], SEED, args, device)
    values = {role: model.scaler.transform(raw[ids]) for role, ids in [("val", val), ("test", test)]}
    del raw
    rng = np.random.default_rng(20261119)
    cases = [("clean", 0, model.full)] + [
        (CONDITION, rep, sensor_mask(40, RATE, rng.permutation(40))) for rep in MASK_REPS
    ]
    rows, drift = [], []
    for condition, rep, kept in cases:
        for role, ids in [("val", val), ("test", test)]:
            metadata = manifest.loc[ids]
            score, stages = model.score(values[role], kept, capture=role == "test")
            save_scores(rows, metadata.itertuples(), score, role, condition, rep)
            if role == "test":
                if condition == "clean":
                    clean = stages
                else:
                    layer_drift(drift, clean, stages, metadata.label, "P2_NM_Z", rep)
        print(f"HellBridge {condition} mask={rep}", flush=True)
    evaluate("HellBridge", rows, drift, out)


def run_z24(args, out, device):
    out.mkdir(exist_ok=True)
    manifest = load_manifest(args.z24_root).assign(sample_id=lambda f: f.record_id)
    rows, drift = [], []
    for index, ((setup, excitation), channels) in enumerate(sorted(stable_sensor_map(manifest).items())):
        key = f"{setup:02d}/{excitation.upper()}"
        print(f"Z24 {key} group={index + 1}/18 input=[N,{len(channels)},1024]", flush=True)
        group = manifest[manifest.setup.eq(setup) & manifest.excitation.eq(excitation)]
        fit, ref = split_training_windows([
            load_z24(Path(row.path), channels) for row in group[group.split.eq("train")].itertuples()
        ])
        limit = args.z24_windows
        model = Detector(subset(fit, limit), subset(ref, limit), SEED + 1009 * index, args, device)
        for row in group[group.split.isin(["val", "test"])].itertuples():
            values = model.scaler.transform(subset(load_z24(Path(row.path), channels), limit))
            score, clean = model.score(values, model.full, capture=row.split == "test")
            save_scores(rows, [row], [np.median(score)], row.split, "clean", 0)
            if row.split == "val" and row.label == 1:
                continue
            for rep in MASK_REPS:
                order = stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                kept = sensor_mask(len(channels), RATE, order)
                score, stages = model.score(values, kept, capture=row.split == "test")
                save_scores(rows, [row], [np.median(score)], row.split, CONDITION, rep)
                if row.split == "test":
                    layer_drift(drift, clean, stages, np.full(len(values), row.label), key, rep)
    evaluate("Z24", rows, drift, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--z24-windows", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.z24_windows) < 1:
        parser.error("epochs, batch size, and window count must be positive")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (out / "protocol.json").write_text(json.dumps(dict(
        model="original TimesNet, native batch-adaptive FFT", seed=SEED, device=str(device),
        window_points=LENGTH, detrend="linear", resampling=False, epochs=args.epochs,
        batch_size=args.batch_size, z24_windows=args.z24_windows,
        mask_reps=list(MASK_REPS), train_sensor_loss=0, test_sensor_loss=[0, RATE],
        reference="healthy clean only; score orientation selected on Z24 clean validation and upper on HellBridge",
    ), indent=2), encoding="utf-8")
    run_hellbridge(args, out / "HellBridge", device)
    run_z24(args, out / "Z24", device)
    combined = pd.concat([
        pd.read_csv(out / bridge / "summary.csv").assign(dataset=bridge)
        for bridge in ("Z24", "HellBridge")
    ], ignore_index=True)
    combined.to_csv(out / "summary.csv", index=False)
    print("TIMESNET_NATIVE_1024_PASS", flush=True)


if __name__ == "__main__":
    main()
