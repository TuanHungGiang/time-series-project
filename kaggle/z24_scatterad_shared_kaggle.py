"""Clone and run official ScatterAD 2025 on the portable Z24 Kaggle split."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCATTER = PROJECT / "baselines_repo" / "ScatterAD"
PINNED_COMMIT = "7572fa954468e46c598badf1c9cfaf6c1835c4cb"


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name in {"train", "val", "test", "metadata", "_excluded"}: path = path.parent
    manifest = path / "metadata" / "records_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    return path


def patch_author_source() -> None:
    source = SCATTER / "model" / "ScatterAD.py"
    text = source.read_text(encoding="utf-8")
    updated, count = re.subn(r"self\.center\.data\.norm\(dim=1(?:,\s*keepdim=True)?\)",
                             "self.center.data.norm()", text)
    if count == 0 and "self.center.data.norm()" not in text:
        raise RuntimeError("Expected ScatterAD center-normalization line was not found")
    source.write_text(updated, encoding="utf-8")
    print(f"compatibility_patch_count={count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    root = dataset_root(args.data_root)
    manifest = root / "metadata" / "records_manifest.csv"
    if not SCATTER.exists():
        SCATTER.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "https://github.com/jk-sounds/ScatterAD.git", str(SCATTER))
    run("git", "-C", str(SCATTER), "checkout", PINNED_COMMIT)
    patch_author_source()
    run(sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.7.0")
    output = Path("/kaggle/working/z24_scatterad_shared")
    run(
        sys.executable, "-u", str(PROJECT / "run_z24_all_records_scatterad.py"),
        "--manifest", str(manifest), "--data-root", str(root),
        "--output", str(output), "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    )
    print(f"outputs={output}", flush=True)


if __name__ == "__main__":
    main()

