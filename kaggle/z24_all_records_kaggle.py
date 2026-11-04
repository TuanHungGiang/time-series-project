"""Kaggle bootstrap for the leakage-aware Z24 all-record CATCH baseline.

Notebook usage (Internet + GPU enabled):

    !git clone https://github.com/TuanHungGiang/time-series-project.git
    %cd time-series-project
    !python kaggle/z24_all_records_kaggle.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/z24_all_records_catch")
CATCH = PROJECT / "baselines_repo" / "CATCH"


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def normalize_data_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name in {"train", "val", "test", "_excluded", "metadata"}:
        path = path.parent
    return path


def find_dataset(explicit_root: Path | None = None) -> tuple[Path, Path]:
    if explicit_root is not None:
        root = normalize_data_root(explicit_root)
        manifest = root / "metadata" / "records_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        if not (root / "train" / "healthy").is_dir() or not (root / "test" / "damage").is_dir():
            raise FileNotFoundError(f"Invalid Z24 split layout below: {root}")
        return root, manifest

    matches = sorted(KAGGLE_INPUT.rglob("metadata/records_manifest.csv"))
    valid = []
    for manifest in matches:
        root = manifest.parents[1]
        if (root / "train" / "healthy").is_dir() and (root / "test" / "damage").is_dir():
            valid.append((root, manifest))
    if len(valid) != 1:
        found = "\n".join(str(item[1]) for item in valid) or "none"
        raise RuntimeError(
            "Expected exactly one Z24 all-record dataset manifest under /kaggle/input; "
            f"found {len(valid)}:\n{found}"
        )
    return valid[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ["Z24_DATA_ROOT"]) if "Z24_DATA_ROOT" in os.environ else None,
        help="Path to kaggle_z24_split or one of its train/val/test directories",
    )
    args = parser.parse_args()
    data_root, manifest = find_dataset(args.data_root)
    print(f"dataset_root={data_root}", flush=True)
    print(f"manifest={manifest}", flush=True)

    if not CATCH.exists():
        CATCH.parent.mkdir(parents=True, exist_ok=True)
        run(
            "git", "clone", "--depth", "1",
            "https://github.com/decisionintelligence/CATCH.git", str(CATCH),
        )

    # Kaggle already supplies torch/numpy/pandas/scipy/sklearn/matplotlib. This
    # lightweight dependency is required by the author CATCH transformer stack.
    run(sys.executable, "-m", "pip", "install", "-q", "reformer-pytorch==1.4.4")
    WORK.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable, "-u", str(PROJECT / "run_z24_all_records_official_catch.py"),
        "--manifest", str(manifest),
        "--data-root", str(data_root),
        "--output", str(WORK),
    )
    print(f"outputs={WORK}", flush=True)


if __name__ == "__main__":
    main()

