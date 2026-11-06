"""Kaggle ablation: original modal input vs regime cointegration residuals."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def run(*cmd):
    print("\n+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--models", default="target_source,procrustes")
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--patience", type=int, default=50)
    p.add_argument("--rank", type=int, default=3); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--only-cointegration", action="store_true")
    args = p.parse_args(); root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}: root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    selected = [x.strip() for x in args.models.split(",") if x.strip()]
    scripts = {"target_source": "run_z24_target_source_ocsvm.py",
               "procrustes": "run_z24_procrustes_ae.py"}
    unknown = set(selected) - set(scripts)
    if unknown: raise ValueError(f"Unknown models: {sorted(unknown)}")
    variants = ["regime_cointegration"] if args.only_cointegration else ["modal", "regime_cointegration"]
    rows = []
    for model in selected:
        for preprocessing in variants:
            output = Path("/kaggle/working") / f"z24_{model}_{preprocessing}"
            run(sys.executable, "-u", PROJECT / scripts[model], "--manifest", manifest,
                "--data-root", root, "--output", output, "--epochs", args.epochs,
                "--patience", args.patience, "--batch-size", args.batch_size,
                "--preprocessing", preprocessing, "--cointegration-rank", args.rank)
            metrics = pd.read_csv(output / "metrics.csv"); metrics.insert(0, "baseline", model)
            metrics.insert(1, "preprocessing_variant", preprocessing); rows.append(metrics)
    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary.to_csv("/kaggle/working/z24_regime_cointegration_summary.csv", index=False)
    print("\nCOINTEGRATION ABLATION\n" + summary.to_string(index=False), flush=True)
    print("summary=/kaggle/working/z24_regime_cointegration_summary.csv", flush=True)


if __name__ == "__main__": main()

