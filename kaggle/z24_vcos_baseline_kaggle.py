"""Kaggle launcher for the supervised Z24 VCOS baseline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input/datasets/datnguyentien204/z24-dataset-my/kaggle_z24_split"),
    )
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_vcos_baseline"))
    parser.add_argument("--delta-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    generated = Path("/kaggle/working/z24_supervised_split_audit/records_manifest_supervised_grouped.csv")
    original = args.data_root / "metadata" / "records_manifest.csv"
    manifest = generated if generated.exists() else original
    command = [
        sys.executable,
        "-u",
        str(project / "run_z24_vcos_baseline.py"),
        "--manifest",
        str(manifest),
        "--data-root",
        str(args.data_root),
        "--output",
        str(args.output),
        "--delta-grid",
        args.delta_grid,
        "--max-features",
        str(args.max_features),
    ]
    if args.rebuild_features:
        command.append("--rebuild-features")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    print("\nIMPORTANT OUTPUTS", flush=True)
    for name in (
        "VCOS_BASELINE_REPORT.md",
        "locked_test_metrics.csv",
        "selected_validation_configuration.csv",
        "selected_features.csv",
        "locked_test_confusion_matrices.png",
    ):
        print(args.output / name, flush=True)


if __name__ == "__main__":
    main()

