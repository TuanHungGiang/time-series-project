"""Kaggle launcher for the concise final modal-centroid selection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models", default="ocsvm,pca,iforest,ae")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--configurations", default=(
        "centroid_locked,centroid_modes4,centroid_window4096,"
        "centroid_separation070,centroid_modes4_window4096,"
        "centroid_modes4_window4096_separation070"
    ))
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}:
        root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    command = [
        sys.executable, "-u", PROJECT / "run_z24_centroid_final_selection.py",
        "--manifest", manifest, "--data-root", root,
        "--output", "/kaggle/working/z24_centroid_final",
        "--models", args.models, "--seeds", args.seeds,
        "--epochs", args.epochs, "--patience", args.patience,
        "--batch-size", args.batch_size,
        "--configurations", args.configurations,
    ]
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__": main()

