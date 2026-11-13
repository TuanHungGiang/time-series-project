# Z24 setup-conditioned source baselines on Kaggle

## Cell 1 — clean clone and dependencies

```python
from pathlib import Path
import shutil, subprocess, sys

REPO = Path("/kaggle/working/time-series-project-z24-baselines")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    "git", "clone", "--depth", "1", "--branch", "codex/mdr-shm-kaggle",
    "https://github.com/TuanHungGiang/time-series-project.git", str(REPO),
], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyod", "torch-geometric"], check=True)
```

## Cell 2 — paths and protocol check

```python
Z24 = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
RUNNER = REPO / "research/mdr_shm_kaggle/run_z24_source_baselines.py"
assert (Z24 / "metadata/records_manifest.csv").exists()
assert RUNNER.exists()
print("Z24:", Z24)
```

## Cell 3 — smoke test

This performs 18 setup/excitation fits for every requested model, so first use
one seed and one epoch.

```python
SMOKE = Path("/kaggle/working/z24_source_smoke")
subprocess.run([
    sys.executable, "-u", str(RUNNER), "--input-root", str(Z24),
    "--output-dir", str(SMOKE), "--epochs", "1", "--seeds", "20261113",
    "--models", "pca_spe", "lstm_ae", "usad", "gdn", "tranad",
    "timesnet", "moderntcn", "scatterad",
], check=True)
```

## Cell 4 — inspect smoke outputs

```python
import pandas as pd
from IPython.display import display
display(pd.read_csv(SMOKE / "metrics.csv"))
display(pd.read_csv(SMOKE / "aggregate.csv"))
```

## Cell 5 — full ten-seed run

The runner resumes at the setup/excitation level. Use a new output directory
for the paper run.

```python
OUT = Path("/kaggle/working/z24_source_10seeds")
SEEDS = [str(seed) for seed in range(20261113, 20261123)]
subprocess.run([
    sys.executable, "-u", str(RUNNER), "--input-root", str(Z24),
    "--output-dir", str(OUT), "--epochs", "30", "--seeds", *SEEDS,
    "--models", "pca_spe", "pca_pyod", "lstm_ae", "usad", "gdn",
    "tranad", "timesnet", "moderntcn", "scatterad",
], check=True)
```

## Cell 6 — table, figure, and archive

```python
from IPython.display import Image, display
display(pd.read_csv(OUT / "aggregate.csv").sort_values("auc_mean", ascending=False))
display(pd.read_csv(OUT / "metrics.csv"))
display(Image(filename=str(OUT / "baseline_comparison.png")))
archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
print("download:", archive)
```

Metrics are computed only after window scores are aggregated to one score per
complete MAT record. Damage is the positive class. The locked test contains 6
healthy and 126 unseen-damage records; validation damage is never used.

