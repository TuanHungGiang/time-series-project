"""Kaggle bootstrap for the strict Z24 ScatterAD + BDC experiment."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCATTER = PROJECT / "baselines_repo" / "ScatterAD"
COMMIT = "7572fa954468e46c598badf1c9cfaf6c1835c4cb"


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True); subprocess.run(cmd, check=True)


def root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name in {"train", "val", "test", "metadata", "_excluded"}: path = path.parent
    if not (path / "metadata" / "records_manifest.csv").is_file():
        raise FileNotFoundError(path / "metadata" / "records_manifest.csv")
    return path


def patch_source() -> None:
    source = SCATTER / "model" / "ScatterAD.py"; text = source.read_text(encoding="utf-8")
    updated, count = re.subn(r"self\.center\.data\.norm\(dim=1(?:,\s*keepdim=True)?\)",
                             "self.center.data.norm()", text)
    if count == 0 and "self.center.data.norm()" not in text: raise RuntimeError("Patch line not found")
    source.write_text(updated, encoding="utf-8"); print(f"compatibility_patch_count={count}")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1); p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--alert-quantile", type=float, default=.95)
    p.add_argument("--alarm-alpha", type=float, default=.01)
    args = p.parse_args(); data = root(args.data_root)
    if not SCATTER.exists():
        SCATTER.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/jk-sounds/ScatterAD.git", str(SCATTER))
    run("git", "-C", str(SCATTER), "checkout", COMMIT); patch_source()
    run(sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.7.0")
    run(sys.executable, "-u", str(PROJECT / "run_z24_scatterad_bdc.py"),
        "--manifest", str(data / "metadata" / "records_manifest.csv"),
        "--data-root", str(data), "--output", "/kaggle/working/z24_scatterad_bdc",
        "--epochs", str(args.epochs), "--cv-folds", str(args.cv_folds),
        "--batch-size", str(args.batch_size), "--alert-quantile", str(args.alert_quantile),
        "--alarm-alpha", str(args.alarm_alpha))


if __name__ == "__main__": main()

