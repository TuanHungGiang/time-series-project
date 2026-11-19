"""Run source baselines on clean HellBridge and nested random sensor outages."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_hellbridge_source_baselines import (
    MODELS, Normalizer, development_split, load_windows, publication_metrics,
    seed_everything, threshold_from_healthy,
)
from run_z24_missingness_baselines import fit_scorer
from run_z24_sensor50_baselines_v1 import DEFAULT_MODELS, DEFAULT_SEEDS


def scenarios(channels, rates=(0.10, 0.30, 0.50)):
    rng = np.random.default_rng(20261119)
    orders = [rng.permutation(channels) for _ in range(5)]
    yield "clean", 0, np.array([], dtype=int)
    for fraction in rates:
        for replicate, order in enumerate(orders):
            yield f"sensor_{fraction:.2f}", replicate, order[:round(channels * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--paper-split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--paper-figure", action="store_true",
                        help="Use 10/20/30 percent sensor loss and one deterministic PCA fit")
    args = parser.parse_args()
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        parser.error("Exactly ten distinct seeds are required")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(args.paper_split_dir / "sample_manifest.csv")
    manifest = manifest.sort_values(["record_id", "start"]).reset_index(drop=True)
    fit_index, cal_index, test_index = development_split(manifest)
    raw = load_windows(args.h5, manifest, 128)
    values = Normalizer.fit(raw[fit_index]).transform(raw)
    fit, calibration, test = values[fit_index], values[cal_index], values[test_index]
    test_rows = manifest.loc[test_index, ["sample_id", "record_id", "state", "label"]].reset_index(drop=True)
    labels = test_rows.label.to_numpy()
    rates = (0.10, 0.20, 0.30) if args.paper_figure else (0.10, 0.30, 0.50)
    cases = list(scenarios(test.shape[1], rates))
    if sum(labels == 0) != 80 or sum(labels == 1) != 640:
        raise ValueError("Expected 80 healthy and 640 damage test windows")
    (out / "protocol.json").write_text(json.dumps({
        "dataset": "HellBridge", "models": args.models, "seeds": args.seeds,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "preprocessing": "AL40-z detrended, resampled to 128, train-only channel normalization",
        "fit_healthy_windows": len(fit), "calibration_healthy_windows": len(calibration),
        "loss": "whole sensor channels set to normalized healthy mean (zero)",
        "threshold": "clean healthy validation q90, fixed for all conditions",
        "paper_figure": args.paper_figure,
        "pca_seed_policy": "one deterministic fit" if args.paper_figure else "repeat deterministic fit",
        "patterns": [{"condition": c, "replicate": r, "removed_1_based": (idx+1).tolist()}
                     for c, r, idx in cases],
    }, indent=2), encoding="utf-8")

    metric_files = []
    for model in args.models:
        model_seeds = args.seeds[:1] if args.paper_figure and model == "pca_spe" else args.seeds
        for seed in model_seeds:
            run = out / "runs" / f"{model}__{seed}"
            run.mkdir(parents=True, exist_ok=True)
            metric_path = run / "metrics.csv"
            metric_files.append(metric_path)
            if metric_path.is_file() and (run / "scores.csv.gz").is_file():
                print(f"SKIP completed {model=} {seed=}", flush=True)
                continue
            print(f"{model=} {seed=} {device=}", flush=True)
            seed_everything(seed)
            score = fit_scorer(model, fit, args.epochs, args.batch_size, device)
            val = score(calibration)
            if val.shape != (len(calibration),) or not np.isfinite(val).all():
                raise ValueError(f"Invalid calibration scores for {model}/{seed}")
            threshold = threshold_from_healthy(val, 0.90)
            rows, exports = [], []
            for condition, replicate, dropped in cases:
                masked = test.copy()
                masked[:, dropped, :] = 0.0
                scores = score(masked)
                if scores.shape != labels.shape or not np.isfinite(scores).all():
                    raise ValueError(f"Invalid scores for {model}/{seed}/{condition}/{replicate}")
                key = dict(model=model, seed=seed, condition=condition, replicate=replicate, threshold=threshold)
                rows.append({**key, **publication_metrics(labels, scores, threshold)})
                exports.append(test_rows.assign(**key, score=scores))
            pd.concat(exports, ignore_index=True).to_csv(run / "scores.csv.gz", index=False)
            result = pd.DataFrame(rows)
            temporary = run / "metrics.tmp"
            result.to_csv(temporary, index=False)
            temporary.replace(metric_path)
            print(result.groupby("condition")[["auc", "ba", "recall", "fpr"]].mean().round(4), flush=True)
            del score

    pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True).to_csv(
        out / "all_metrics.csv", index=False
    )
    print("HELLBRIDGE_RANDOM_SENSOR_BASELINES_PASS", flush=True)


if __name__ == "__main__":
    main()
