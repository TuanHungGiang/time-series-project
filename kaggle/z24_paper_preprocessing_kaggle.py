"""Run the three paper preprocessing experiments on the Kaggle Z24 split."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def run(*command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--skip-mdfa", action="store_true")
    p.add_argument("--skip-timevertex", action="store_true")
    p.add_argument("--skip-edgeconv", action="store_true")
    args = p.parse_args()
    root = args.data_root.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}: root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    shared = ["--manifest", manifest, "--data-root", root, "--epochs", args.epochs,
              "--patience", args.patience, "--batch-size", args.batch_size]
    if not args.skip_mdfa:
        run(sys.executable, "-u", PROJECT / "run_z24_mdfa_paper.py", *shared,
            "--output", "/kaggle/working/z24_mdfa_paper")
    if not args.skip_timevertex:
        run(sys.executable, "-u", PROJECT / "run_z24_timevertex_paper.py", *shared,
            "--output", "/kaggle/working/z24_timevertex_paper")
    if not args.skip_edgeconv:
        edgeconv = ["--manifest", manifest, "--data-root", root, "--epochs", args.epochs,
                    "--patience", args.patience]
        run(sys.executable, "-u", PROJECT / "run_z24_graph_transformer_tailp.py", *edgeconv,
            "--output", "/kaggle/working/z24_edgeconvformer_paper")


if __name__ == "__main__": main()

