"""Kaggle launcher for the leak-free CM-HRR multi-model benchmark."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--models", default="ocsvm,pca,iforest,ae")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}: root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    command = [sys.executable, "-u", str(PROJECT / "run_z24_cm_hrr_benchmark.py"),
               "--manifest", str(manifest), "--data-root", str(root),
               "--output", "/kaggle/working/z24_cm_hrr_benchmark",
               "--models", args.models, "--epochs", str(args.epochs),
               "--patience", str(args.patience)]
    print("+", " ".join(command), flush=True); subprocess.run(command, check=True)


if __name__ == "__main__": main()

