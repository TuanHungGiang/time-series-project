"""One-seed clean USAD diagnosis with shared feature and scoring rules."""

import argparse
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import detrend, resample_poly
import torch

from run_clmm_shared_configuration import BATCH, LENGTH, spectral_features
from run_clmm_multi_detector import recall_at_fpr


def representations(windows):
    return {
        "spectrum128": spectral_features(windows),
        "waveform512": resample_poly(
            detrend(windows, axis=-1, type="linear"), 1, 2, axis=-1,
        ).astype(np.float32),
    }


def tail_parameters(scores):
    q05, median, q95 = np.quantile(scores, [0.05, 0.5, 0.95])
    floor = max(0.10 * (q95 - q05), 1e-8)
    return median, max(median - q05, floor), max(q95 - median, floor), max(1.4826 * np.median(np.abs(scores - median)), 0.25 * np.std(scores), 1e-4)


def score_transform(scores, parameters):
    center, lower, upper, symmetric = parameters
    shift = scores - center
    return np.abs(shift) / symmetric, np.maximum(shift / upper, -shift / lower)


def train_compact_usad(fit, epochs, batch_size, device, model_class):
    """Preserve the two USAD objectives and fix the hidden widths."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    flat = np.ascontiguousarray(fit.reshape(len(fit), -1))
    width = flat.shape[1]

    class CompactUSAD(model_class):
        def __init__(self):
            nn.Module.__init__(self)
            self.encoder = nn.Sequential(
                nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 256),
                nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
            )
            self.decoder1 = self.make_decoder()
            self.decoder2 = self.make_decoder()

        @staticmethod
        def make_decoder():
            return nn.Sequential(
                nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 512),
                nn.ReLU(), nn.Linear(512, width), nn.Sigmoid(),
            )

    model = CompactUSAD().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(flat)), batch_size=batch_size, shuffle=True)
    first = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=1e-3)
    second = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=1e-3)
    for epoch in range(1, epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            model.zero_grad(set_to_none=True)
            loss1, _ = model.training_step(batch, epoch)
            loss1.backward()
            first.step()
            model.zero_grad(set_to_none=True)
            _, loss2 = model.training_step(batch, epoch)
            loss2.backward()
            second.step()
            losses.append(float((loss1 + loss2).detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch={epoch}/{epochs} usad_loss={np.mean(losses):.6f}", flush=True)
    return model.eval().requires_grad_(False)


def fit_usad_scores(features, seed, device, hb, fit_batch=BATCH):
    scaler = hb.FitMinMax(features)
    fit = scaler.transform(features)
    hb.seed_everything(seed)
    source = train_compact_usad(fit, 30, fit_batch, device, hb.UsadModel)

    @torch.no_grad()
    def residual(windows):
        parts = []
        for batch in torch.from_numpy(windows).split(BATCH):
            x = batch.to(device).flatten(1)
            first = source.decoder1(source.encoder(x))
            cycle = source.decoder2(source.encoder(first))
            squared = 0.5 * ((x - first).square() + (x - cycle).square())
            parts.append(squared.reshape_as(batch).cpu().numpy())
        return np.concatenate(parts).astype(np.float64)

    training = residual(fit)
    coordinate_scale = np.median(training, axis=0)
    # A floor specific to each channel prevents near-zero coordinates from
    # dominating the weighted reconstruction error.
    floor = np.maximum(np.quantile(coordinate_scale, 0.25, axis=1, keepdims=True), 1e-6)
    coordinate_scale = np.maximum(coordinate_scale, floor)

    def reductions(errors):
        return {
            "native": np.log1p(errors.mean((1, 2))),
            "weighted": np.log1p((errors / coordinate_scale).mean((1, 2))),
        }

    anchors = {name: tail_parameters(scores) for name, scores in reductions(training).items()}

    def score(features):
        results = {}
        for name, values in reductions(residual(scaler.transform(features))).items():
            symmetric, asymmetric = score_transform(values, anchors[name])
            results[f"{name}_symmetric"] = symmetric
            results[f"{name}_asymmetric"] = asymmetric
            # Upper and lower scores are diagnostic, with no test-derived threshold.
            results[f"{name}_signed_diagnostic"] = (values - anchors[name][0]) / anchors[name][3]
        return results

    return score


def report(dataset, rows, hb, seed, out):
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame.score).all():
        raise ValueError(f"{dataset}: nonfinite scores")
    frame.to_csv(out / f"{dataset}_scores.csv.gz", index=False, compression="gzip")
    result, diagnostic = [], []
    for (representation, rule), group in frame.groupby(["input", "rule"]):
        test = group[group.split.eq("test")]
        labels, scores = test.label.to_numpy(), test.score.to_numpy()
        row = {"dataset": dataset, "seed": seed, "input": representation, "rule": rule}
        if rule.endswith("_diagnostic"):
            from sklearn.metrics import roc_auc_score
            diagnostic.append({
                **row,
                "auc_upper": roc_auc_score(labels, scores),
                "auc_lower": roc_auc_score(labels, -scores),
            })
            continue
        cal = group.loc[group.split.eq("val") & group.label.eq(0), "score"].to_numpy()
        if len(cal) != (20 if dataset == "HellBridge" else 6):
            raise ValueError(f"{dataset}: incorrect healthy calibration count")
        threshold = hb.threshold_from_healthy(cal, 0.90)
        result.append({
            **row, "threshold": threshold,
            **hb.publication_metrics(labels, scores, threshold),
            "recall_at_fpr_05": recall_at_fpr(labels, scores, 0.05),
            "recall_at_fpr_10": recall_at_fpr(labels, scores, 0.10),
        })
    if len(result) != 8 or len(diagnostic) != 4:
        raise ValueError("Incomplete USAD clean ablation")
    return result, diagnostic


def add_scores(rows, values, representation, ids, labels, role):
    for rule, scores in values.items():
        rows.extend({
            "sample_id": uid, "label": int(label), "split": role,
            "input": representation, "rule": rule, "score": float(score),
        } for uid, label, score in zip(ids, labels, scores, strict=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hell-h5", required=True, type=Path)
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20270901)
    args = parser.parse_args()
    import run_hellbridge_source_baselines as hb
    import run_z24_source_baselines as z24
    from run_hellbridge_tsad_baselines import read_al40

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USAD clean seed={args.seed} device={device}", flush=True)

    split = out / "hellbridge_split"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("prepare_hellbridge_paper_split.py")),
        "--h5", str(args.hell_h5), "--output-dir", str(split),
        "--profile", "svendsen2022", "--regime", "P2_NM_Z", "--seed", "20261113",
        "--strict-paper-counts",
    ], check=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(
        ["record_id", "start"]
    ).reset_index(drop=True)
    fit, cal, test = hb.development_split(manifest)
    raw = np.empty((len(manifest), 40, LENGTH), dtype=np.float32)
    with h5py.File(args.hell_h5, "r") as h5:
        for record, group in manifest.groupby("record_id", sort=False):
            signal = read_al40(h5[record])
            for row in group.itertuples():
                raw[row.Index] = signal[:, int(row.start):int(row.stop)]
    rows = []
    # Limit peak FFT/resampling memory by transforming one record at a time.
    features = {"spectrum128": np.empty((len(raw), 40, 128), np.float32),
                "waveform512": np.empty((len(raw), 40, 512), np.float32)}
    for indices in manifest.groupby("record_id").indices.values():
        for name, values in representations(raw[indices]).items():
            features[name][indices] = values
    del raw
    for name, values in features.items():
        print(f"HellBridge {name}", flush=True)
        score = fit_usad_scores(values[fit], args.seed, device, hb)
        for indices, role in ((cal, "val"), (test, "test")):
            add_scores(rows, score(values[indices]), name,
                       manifest.loc[indices, "sample_id"], manifest.loc[indices, "label"], role)
        del score
    hell_results, hell_diag = report("HellBridge", rows, hb, args.seed, out)
    pd.DataFrame(hell_results).to_csv(out / "HellBridge_summary.csv", index=False)
    del rows, features

    manifest = z24.load_manifest(args.z24_root)
    rows = []
    for index, ((setup, excitation), channels) in enumerate(sorted(z24.stable_sensor_map(manifest).items())):
        group = manifest[(manifest.setup == setup) & (manifest.excitation == excitation)]

        def load(row):
            raw = z24.read_stable_vertical(Path(row.path), channels)
            usable = raw.shape[-1] // LENGTH * LENGTH
            return representations(raw[:, :usable].reshape(len(channels), -1, LENGTH).transpose(1, 0, 2))

        fit_rows = group[group.split.eq("train") & group.label.eq(0)]
        fit_records = [load(row) for row in fit_rows.itertuples()]
        query_rows = list(group[group.split.eq("test") | (group.split.eq("val") & group.label.eq(0))].itertuples())
        queries = [load(row) for row in query_rows]
        for name in ("spectrum128", "waveform512"):
            print(f"Z24 setup={setup} excitation={excitation} {name}", flush=True)
            fit, _ = z24.split_training_windows([rec[name] for rec in fit_records])
            score = fit_usad_scores(fit, args.seed + index * 1009, device, hb)
            for row, query in zip(query_rows, queries, strict=True):
                values = {k: [float(np.median(v))] for k, v in score(query[name]).items()}
                add_scores(rows, values, name, [row.record_id], [row.label], row.split)
            del score
            if device.type == "cuda":
                torch.cuda.empty_cache()
    z24_results, z24_diag = report("Z24", rows, hb, args.seed, out)
    summary = pd.DataFrame(hell_results + z24_results)
    diagnostics = pd.DataFrame(hell_diag + z24_diag)
    summary.to_csv(out / "summary.csv", index=False)
    diagnostics.to_csv(out / "tail_diagnostics_only.csv", index=False)
    fields = ["dataset", "input", "rule", "auc", "ba", "recall", "fpr", "recall_at_fpr_05"]
    print(summary[fields].round(4).to_string(index=False))
    print("\n=== SIGNED SCORE DIAGNOSTIC ONLY ===")
    print(diagnostics.round(4).to_string(index=False))
    print("USAD_CLEAN_TWO_BRIDGES_PASS", flush=True)


if __name__ == "__main__":
    main()
