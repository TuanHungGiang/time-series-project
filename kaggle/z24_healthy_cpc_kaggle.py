"""Kaggle launcher for the healthy-only CPC preprocessing ablation."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--backbones", default="target_source,procrustes")
    p.add_argument("--cpc-epochs", type=int, default=50); p.add_argument("--downstream-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20); p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(); root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}: root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    command = [sys.executable, "-u", PROJECT / "run_z24_healthy_cpc_benchmark.py",
               "--manifest", manifest, "--data-root", root,
               "--output", "/kaggle/working/z24_healthy_cpc", "--backbones", args.backbones,
               "--cpc-epochs", args.cpc_epochs, "--downstream-epochs", args.downstream_epochs,
               "--downstream-patience", args.patience, "--batch-size", args.batch_size]
    print("+", " ".join(map(str, command)), flush=True); subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__": main()

