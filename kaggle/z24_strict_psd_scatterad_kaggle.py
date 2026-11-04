"""Kaggle bootstrap for strict S01-only log-PSD ScatterAD."""
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


def find_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if root.name in {"train", "val", "test", "metadata", "_excluded"}: root = root.parent
        if not (root / "metadata" / "records_manifest.csv").is_file():
            raise FileNotFoundError(root / "metadata" / "records_manifest.csv")
        return root
    matches = sorted(Path("/kaggle/input").rglob("metadata/records_manifest.csv"))
    if len(matches) != 1: raise RuntimeError(f"Expected one records_manifest.csv, found {len(matches)}")
    return matches[0].parents[1]


def patch_official_source() -> None:
    source = SCATTER / "model" / "ScatterAD.py"
    text = source.read_text(encoding="utf-8")
    updated, count = re.subn(r"self\.center\.data\.norm\(dim=1(?:,\s*keepdim=True)?\)",
                             "self.center.data.norm()", text)
    if count == 0 and "self.center.data.norm()" not in text:
        raise RuntimeError("ScatterAD compatibility line not found")
    source.write_text(updated, encoding="utf-8")
    print(f"compatibility_patch_count={count}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args(); root = find_root(args.data_root)
    if not SCATTER.exists():
        SCATTER.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/jk-sounds/ScatterAD.git", str(SCATTER))
    run("git", "-C", str(SCATTER), "checkout", COMMIT); patch_official_source()
    run(sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.7.0")
    output = Path("/kaggle/working/z24_strict_psd_scatterad")
    run(sys.executable, "-u", str(PROJECT / "run_z24_strict_psd_scatterad.py"),
        "--manifest", str(root / "metadata" / "records_manifest.csv"),
        "--data-root", str(root), "--output", str(output),
        "--epochs", str(args.epochs), "--cv-folds", str(args.cv_folds),
        "--batch-size", str(args.batch_size))


if __name__ == "__main__":
    main()

