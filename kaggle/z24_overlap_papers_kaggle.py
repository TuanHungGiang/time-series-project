"""Kaggle launcher for the two MSSP healthy-domain alignment methods."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def run(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--procrustes-epochs", type=int, default=600)
    p.add_argument("--mapping-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--skip-procrustes", action="store_true")
    p.add_argument("--skip-target-source", action="store_true")
    args = p.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}:
        root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    shared = ["--manifest", manifest, "--data-root", root, "--patience", args.patience]
    if not args.skip_procrustes:
        run(sys.executable, "-u", PROJECT / "run_z24_procrustes_ae.py", *shared,
            "--epochs", args.procrustes_epochs,
            "--output", "/kaggle/working/z24_procrustes_ae")
    if not args.skip_target_source:
        run(sys.executable, "-u", PROJECT / "run_z24_target_source_ocsvm.py", *shared,
            "--epochs", args.mapping_epochs,
            "--output", "/kaggle/working/z24_target_source_ocsvm")


if __name__ == "__main__":
    main()
