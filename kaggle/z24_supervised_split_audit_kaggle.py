"""Kaggle entry point for the Z24 supervised split audit."""

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
    parser.add_argument(
        "--output", type=Path, default=Path("/kaggle/working/z24_supervised_split_audit")
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    manifest = args.data_root / "metadata" / "records_manifest.csv"
    command = [
        sys.executable,
        "-u",
        str(project / "run_z24_supervised_split_audit.py"),
        "--manifest",
        str(manifest),
        "--data-root",
        str(args.data_root),
        "--output",
        str(args.output),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    print("\nIMPORTANT OUTPUTS", flush=True)
    for name in (
        "SPLIT_AUDIT_REPORT.md",
        "split_audit.json",
        "records_manifest_supervised_grouped.csv",
        "legacy_vs_supervised_split_counts.png",
        "proposed_setup_group_matrix.png",
    ):
        print(args.output / name, flush=True)


if __name__ == "__main__":
    main()

