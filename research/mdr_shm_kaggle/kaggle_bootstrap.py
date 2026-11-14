"""Paste or run this file as the first Kaggle notebook cell."""
from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/TuanHungGiang/time-series-project.git"
BRANCH = "codex/mdr-shm-kaggle"
ROOT = Path("/kaggle/working/time-series-project")

if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(ROOT)], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(ROOT / "research/mdr_shm_kaggle/requirements.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, str(ROOT / "research/mdr_shm_kaggle/run_z24_b2.py"),
     "--input-root", "/kaggle/input",
     "--output-dir", "/kaggle/working/mdr_shm_b2"],
    check=True,
)
