"""Kaggle launcher for locked modal-peak plus grouped-OOF EVT calibration."""
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
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--record-quantile", type=float, default=.95)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}:
        root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    command = [
        sys.executable, "-u", PROJECT / "run_z24_modal_evt_calibration.py",
        "--manifest", manifest, "--data-root", root,
        "--output", "/kaggle/working/z24_modal_evt",
        "--models", args.models, "--seeds", args.seeds,
        "--alpha", args.alpha,
        "--bootstrap-samples", args.bootstrap_samples,
        "--record-quantile", args.record_quantile,
        "--epochs", args.epochs, "--patience", args.patience,
        "--batch-size", args.batch_size,
    ]
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__":
    main()

