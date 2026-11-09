"""Run the corrected selection benchmark and validation-only lock audit."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def run(command: list[object]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--models", default="ocsvm,pca,iforest,ae")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--configurations", default=(
        "locked,modes_4,modes_8,radius_030,radius_060,"
        "window_4096,separation_030,separation_070"
    ))
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}:
        root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    shared = [
        "--manifest", manifest, "--data-root", root,
        "--models", args.models, "--seeds", args.seeds,
        "--epochs", args.epochs, "--patience", args.patience,
        "--batch-size", args.batch_size,
        "--pca-components", args.pca_components,
    ]
    if not args.skip_selection:
        run([
            sys.executable, "-u",
            PROJECT / "run_z24_deterministic_preprocessing_selection.py",
            *shared,
            "--output", "/kaggle/working/z24_preprocessing_lock_selection",
        ])
    if not args.skip_sensitivity:
        run([
            sys.executable, "-u",
            PROJECT / "run_z24_modal_peak_sensitivity.py",
            *shared, "--configurations", args.configurations,
            "--output", "/kaggle/working/z24_preprocessing_lock_sensitivity",
        ])


if __name__ == "__main__":
    main()

