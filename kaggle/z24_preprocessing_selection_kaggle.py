"""Kaggle launcher for leak-controlled Z24 preprocessing selection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--variants", default=(
        "mean,attention,mean_attention_max,mean_attention_avgtopk,"
        "mean_attention_trimmed_topk"
    ))
    parser.add_argument("--cpc-epochs", type=int, default=50)
    parser.add_argument("--compact-epochs", type=int, default=30)
    parser.add_argument("--downstream-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk-fraction", type=float, default=.25)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}:
        root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    command = [
        sys.executable, "-u", PROJECT / "run_z24_preprocessing_selection.py",
        "--manifest", manifest, "--data-root", root,
        "--output", "/kaggle/working/z24_preprocessing_selection",
        "--seeds", args.seeds, "--variants", args.variants,
        "--cpc-epochs", args.cpc_epochs,
        "--compact-epochs", args.compact_epochs,
        "--downstream-epochs", args.downstream_epochs,
        "--downstream-patience", args.patience,
        "--batch-size", args.batch_size,
        "--topk-fraction", args.topk_fraction,
    ]
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__":
    main()

