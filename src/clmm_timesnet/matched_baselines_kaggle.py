"""Run the manuscript baselines on the native CLMM input, split, and loss masks."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data.hellbridge import Normalizer
from .hellbridge_spectral_kaggle import calibration_report
from .mask_pair_block import evaluate, native_score_direction, run_hellbridge, run_z24
from .models.backbone import seed_everything
from .targeted_checkpoints import Scorer, fit_scorer
from .timesnet_native_1024 import Detector
from .whitening_multiseed_kaggle import CONDITIONS, DATASETS, METRICS, SEEDS
from .whitening_two_bridges_kaggle import INPUT, LENGTH, MASKS, Z24_INPUT


NATIVE = "timesnet_native"
MODELS = ("pca_spe", "lstm_ae", NATIVE, "tranad", "moderntcn", "scatterad", "cnn_ae_1d")
EPOCHS, BATCH_SIZE, Z24_WINDOWS = 5, 8, 8
GROUP = ["dataset", "method", "condition", "policy"]


def native_scores(scorer, data):
    values = np.asarray(scorer(data), dtype=np.float64)
    if values.shape != (len(data),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected a finite nonnegative native score for every window")
    return np.log1p(values)


def fit_tranad(source, values, epochs, batch_size, device):
    normalizer = source.FitMinMax(values)
    fit = normalizer.transform(values)
    model = source.TranAD(*fit.shape[1:]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(fit)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for sample in batch.split(1):
                sample = sample.to(device)
                loss = (sample - source.model_forward("tranad", model, sample)).square().mean()
                (loss / len(batch)).backward()
                batch_loss += loss.detach() / len(batch)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(batch_loss.item())
        if os.environ.get("CLMM_LOG_EACH_EPOCH") == "1" or epoch in (1, epochs):
            print(f"  epoch={epoch}/{epochs} tranad_loss={np.mean(losses):.6f}", flush=True)
    return Scorer(source, "tranad", model.eval(), 1, device, (normalizer.low, normalizer.high))


class BaselineBlock:
    def __init__(self, fit, reference, seed, args, device, group, source):
        self.columns = args.active_models
        self.scaler = Normalizer.fit(fit)
        self.models = {}
        print(f"{group} | {fit.shape=} {reference.shape=} {seed=}", flush=True)
        normalized_fit, normalized_ref = self.scaler.transform(fit), self.scaler.transform(reference)
        for model in self.columns:
            print(f"  fitting {model}", flush=True)
            if model == NATIVE:
                self.detector = Detector(fit, reference, seed, args, device)
                continue
            seed_everything(seed)
            if model == "tranad":
                scorer = fit_tranad(source, normalized_fit, args.epochs, args.batch_size, device)
            else:
                scorer = fit_scorer(source, model, normalized_fit, args.epochs, args.batch_size, device)
            ref = native_scores(scorer, normalized_ref)
            center = np.median(ref)
            scale = max(1.4826 * np.median(np.abs(ref - center)), 0.25 * ref.std(), 1e-4)
            self.models[model] = (scorer, center, scale)

    def prepare(self, raw):
        return (self.scaler.transform(raw),)

    def scores(self, data, kept):
        masked = data.copy() if not kept.all() else data
        if not kept.all():
            masked[:, ~kept, :] = 0
        columns = []
        for model in self.columns:
            if model == NATIVE:
                score, _ = self.detector.score(data, kept)
            else:
                scorer, center, scale = self.models[model]
                score = (native_scores(scorer, masked) - center) / scale
            columns.append(score)
        return np.column_stack(columns)


def run_dataset(dataset, seed, args, device, source, out):
    args.seed = seed
    args.active_models = MODELS if seed == SEEDS[0] else tuple(m for m in MODELS if m != "pca_spe")

    def make_block(fit, ref, group_seed, options, accelerator, group):
        return BaselineBlock(fit, ref, group_seed, options, accelerator, group, source)

    print(f"\n{dataset} / {seed} / {args.active_models}", flush=True)
    signs = dict.fromkeys(args.active_models, 1)
    if dataset == "Z24":
        rows, masks, fits = run_z24(args, device, make_block)
        scores = pd.DataFrame(rows)
        for model in args.active_models:
            signs[model] = native_score_direction(dataset, scores, baseline_method=model)
        # evaluate applies the native TimesNet sign from the unmodified column.
        for row in rows:
            for model in args.active_models:
                if model != NATIVE:
                    row[model] *= signs[model]
        metrics = evaluate(dataset, rows, masks, fits, out, methods=args.active_models)
        metrics["policy"] = metrics.policy.map({"clean_q90": "clean_frozen", "masked_q90": "mask_matched"})
        metrics = metrics.rename(columns={"recall_low_fpr": "diagnostic_recall"})
    else:
        rows, masks, fits = run_hellbridge(args, device, out, make_block)
        if not np.isfinite(pd.DataFrame(rows)[list(args.active_models)]).all().all():
            raise ValueError("Invalid HellBridge scores")
        out.mkdir(parents=True, exist_ok=True)
        scores_path = out / "scores.csv.gz"
        pd.DataFrame(rows).to_csv(scores_path, index=False)
        pd.DataFrame(masks).to_csv(out / "masks.csv", index=False)
        (out / "fit_groups.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
        metrics, by_state = calibration_report(scores_path, methods=args.active_models)
        by_state.to_csv(out / "alarms_by_damage_state.csv", index=False)
        metrics = metrics.loc[metrics.alpha.eq(0.05)].assign(dataset=dataset).copy()
        metrics["policy"] = metrics.policy.map({
            "clean_calibration": "clean_frozen", "same_mask_calibration": "mask_matched",
        })
        metrics = metrics.rename(columns={"recall_at_fpr_05": "diagnostic_recall"})
    return metrics.assign(seed=seed, score_sign=metrics.method.map(signs))


def summarize(results, out):
    scores = pd.concat(results, ignore_index=True)
    expected = {
        (dataset, model, condition, policy, seed, rep)
        for dataset in DATASETS for model in MODELS
        for condition in CONDITIONS for policy in ("clean_frozen", "mask_matched")
        for seed in (SEEDS[:1] if model == "pca_spe" else SEEDS[:3])
        for rep in ((0,) if condition == "clean" else MASKS)
    }
    keys = [*GROUP, "seed", "replicate"]
    actual = set(scores[keys].itertuples(index=False, name=None))
    if actual != expected or len(actual) != len(scores) or not np.isfinite(scores[list(METRICS)]).all().all():
        raise ValueError("Incomplete or invalid baseline metrics")
    scores.to_csv(out / "per_mask_metrics.csv", index=False)
    per_seed = scores.groupby([*GROUP, "seed"], as_index=False)[list(METRICS)].mean()
    per_seed.to_csv(out / "per_seed_metrics.csv", index=False)
    summary = per_seed.groupby(GROUP)[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["runs"] = per_seed.groupby(GROUP).seed.size()
    summary.reset_index().to_csv(out / "summary_mean_sd.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, default=Z24_INPUT)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.z24_root / "metadata/records_manifest.csv", args.hell_h5,
                 args.baseline_dir / "run_hellbridge_source_baselines.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator on Kaggle.")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    source = importlib.import_module("run_hellbridge_source_baselines")
    args.epochs, args.batch_size, args.z24_windows = EPOCHS, BATCH_SIZE, Z24_WINDOWS
    args.layer_probe, args.replicates = False, MASKS
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        version=2, models=MODELS, seeds=SEEDS[:3], pca_seeds=SEEDS[:1],
        window_points=LENGTH, sample_rate_hz=100, masks=MASKS,
        epochs=EPOCHS, batch_size=BATCH_SIZE, z24_window_cap=Z24_WINDOWS,
        tranad_microbatch=1,
        z24_root=str(args.z24_root), hell_h5=str(args.hell_h5), baseline_dir=str(args.baseline_dir),
        inputs="native CLMM split, vertical sensors, detrend, healthy fit z-score, zero-fill",
        model_scoring="native scores; fit-only additional min-max for TranAD; TimesNet observed MSE",
        score_calibration="log1p, healthy reference median/scale; Z24 per-model clean validation sign",
        threshold_calibration="Z24 healthy Q90 and HellBridge healthy rank at nominal 5% FPR",
        variance="mean across masks per seed, then mean and sample SD across seeds",
    )
    protocol_path = out / "protocol.json"
    text = json.dumps(protocol, indent=2)
    if protocol_path.is_file() and protocol_path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing output has a different protocol: {out}")
    protocol_path.write_text(text, encoding="utf-8")
    results = []
    for seed in SEEDS[:3]:
        for dataset in DATASETS:
            group = out / dataset / str(seed)
            path = group / "complete_metrics.csv"
            if path.is_file():
                print(f"RESUME {dataset}/{seed}", flush=True)
                table = pd.read_csv(path, float_precision="round_trip")
            else:
                table = run_dataset(dataset, seed, args, device, source, group)
                temporary = path.with_suffix(".tmp")
                table.to_csv(temporary, index=False)
                temporary.replace(path)
            results.append(table)
    summarize(results, out)
    print(f"MATCHED_BASELINES_COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
