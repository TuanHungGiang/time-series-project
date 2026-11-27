# HellBridge healthy-only TSAD baselines on Kaggle

This benchmark runs PCA, LSTM-AE, USAD, GDN, TranAD, TimesNet,
ModernTCN, and ScatterAD under one data protocol. The seven neural models are
compact architecture adapters rather than byte-identical executions of seven
official repositories. The output `protocol.json` records that fact.

## Cell 1 — clone from a fresh Kaggle session

```python
from pathlib import Path
import shutil, subprocess

REPO = Path("/kaggle/working/time-series-project")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    "git", "clone", "--depth", "1", "--branch", "codex/mdr-shm-kaggle",
    "https://github.com/TuanHungGiang/time-series-project.git", str(REPO),
], check=True)
print("repo:", REPO)
```

## Cell 2 — create/recreate the paper split

```python
from pathlib import Path
import subprocess, sys

H5 = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")
SPLIT = Path("/kaggle/working/hellbridge_split_svendsen2022")
PREP = REPO / "research/mdr_shm_kaggle/prepare_hellbridge_paper_split.py"

subprocess.run([
    sys.executable, "-u", str(PREP), "--h5", str(H5),
    "--output-dir", str(SPLIT), "--profile", "svendsen2022",
    "--regime", "P2_NM_Z", "--seed", "20261113", "--strict-paper-counts",
], check=True)
assert (SPLIT / "sample_manifest.csv").exists()
assert (SPLIT / "protocol.json").exists()
print("split ready:", SPLIT)
```

## Cell 3 — quick smoke run

```python
OUT = Path("/kaggle/working/hellbridge_tsad_quick")
RUNNER = REPO / "research/mdr_shm_kaggle/run_hellbridge_tsad_baselines.py"
subprocess.run([
    sys.executable, "-u", str(RUNNER), "--h5", str(H5),
    "--paper-split-dir", str(SPLIT), "--output-dir", str(OUT),
    "--mode", "quick",
], check=True)
```

## Cell 4 — full paper run

Run this only after Cell 3 passes. It uses 30 epochs and three seeds.

```python
OUT = Path("/kaggle/working/hellbridge_tsad_full")
subprocess.run([
    sys.executable, "-u", str(RUNNER), "--h5", str(H5),
    "--paper-split-dir", str(SPLIT), "--output-dir", str(OUT),
    "--mode", "full",
], check=True)
```

## Cell 5 — inspect and download

```python
from IPython.display import Image, display
import pandas as pd, shutil

display(pd.read_csv(OUT / "aggregate.csv").sort_values("auroc_mean", ascending=False))
display(pd.read_csv(OUT / "metrics.csv"))
display(Image(filename=str(OUT / "baseline_comparison.png")))
archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
print("download:", archive)
```

Do not rank models from the quick run. Use the full three-seed run and report
AUROC, balanced accuracy, recall, specificity, and their seed variation. No
test label is used for training or threshold selection, and no point adjustment
is applied.

