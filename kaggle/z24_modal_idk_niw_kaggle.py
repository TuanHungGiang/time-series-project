"""Kaggle launcher for the Z24 Modal-IDK-NIW proposed-method experiment."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args(); root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}: root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    command = [sys.executable, "-u", PROJECT / "run_z24_modal_idk_niw.py",
               "--manifest", manifest, "--data-root", root,
               "--output", "/kaggle/working/z24_modal_idk_niw",
               "--seed", args.seed, "--alpha", args.alpha]
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


if __name__ == "__main__": main()

