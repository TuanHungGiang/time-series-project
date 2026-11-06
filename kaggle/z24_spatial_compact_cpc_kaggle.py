"""Kaggle launcher for strict spatial Compact-CPC ablations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cpc-epochs", type=int, default=50)
    parser.add_argument("--compact-epochs", type=int, default=30)
    parser.add_argument("--paired-epochs", type=int, default=30)
    parser.add_argument("--topk-fraction", type=float, default=.25)
    parser.add_argument("--downstream-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}:
        root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    command = [sys.executable, "-u", PROJECT / "run_z24_spatial_compact_cpc.py",
               "--manifest", manifest, "--data-root", root,
               "--output", "/kaggle/working/z24_spatial_compact_cpc",
               "--cpc-epochs", args.cpc_epochs,
               "--compact-epochs", args.compact_epochs,
               "--paired-epochs", args.paired_epochs,
               "--topk-fraction", args.topk_fraction,
               "--downstream-epochs", args.downstream_epochs,
               "--downstream-patience", args.patience,
               "--batch-size", args.batch_size]
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__":
    main()

