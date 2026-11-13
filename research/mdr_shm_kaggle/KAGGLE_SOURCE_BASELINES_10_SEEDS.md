# Source-based HellBridge baselines — 10 seeds

This run uses the supplied source implementations or documented batch-safe
ports. It never uses test labels/scores to train, calibrate a threshold, tune a
hyperparameter, or point-adjust predictions.

## Cell 1 — clean clone and dependencies

```python
from pathlib import Path
import shutil, subprocess, sys

REPO = Path("/kaggle/working/time-series-project-source-baselines")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    "git", "clone", "--depth", "1", "--branch", "codex/mdr-shm-kaggle",
    "https://github.com/TuanHungGiang/time-series-project.git", str(REPO),
], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "pyod", "torch-geometric",
], check=True)
print("repo:", REPO)
```

## Cell 2 — recreate the paper split after every Kaggle reset

```python
H5 = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")
SPLIT = Path("/kaggle/working/hellbridge_split_svendsen2022")
PREP = REPO / "research/mdr_shm_kaggle/prepare_hellbridge_paper_split.py"

if SPLIT.exists():
    shutil.rmtree(SPLIT)
subprocess.run([
    sys.executable, "-u", str(PREP), "--h5", str(H5),
    "--output-dir", str(SPLIT), "--profile", "svendsen2022",
    "--regime", "P2_NM_Z", "--seed", "20261113", "--strict-paper-counts",
], check=True)
assert (SPLIT / "sample_manifest.csv").exists()
print("split ready:", SPLIT)
```

## Cell 3 — smoke test (one seed, two epochs)

```python
RUNNER = REPO / "research/mdr_shm_kaggle/run_hellbridge_source_baselines.py"
SMOKE = Path("/kaggle/working/hellbridge_source_smoke")
subprocess.run([
    sys.executable, "-u", str(RUNNER), "--h5", str(H5),
    "--paper-split-dir", str(SPLIT), "--output-dir", str(SMOKE),
    "--epochs", "2", "--seeds", "20261113",
], check=True)
```

## Cell 4 — full 10-seed run

The runner writes every `(model, seed)` immediately and skips completed runs
when rerun with the same output directory.

```python
OUT = Path("/kaggle/working/hellbridge_source_10seeds")
seeds = [str(seed) for seed in range(20261113, 20261123)]
subprocess.run([
    sys.executable, "-u", str(RUNNER), "--h5", str(H5),
    "--paper-split-dir", str(SPLIT), "--output-dir", str(OUT),
    "--epochs", "30", "--seeds", *seeds,
    "--models", "pca_spe", "pca_pyod", "lstm_ae", "usad", "gdn",
    "tranad", "timesnet", "moderntcn", "scatterad",
], check=True)
```

If a Kaggle session times out, save the output directory as a Kaggle dataset,
restore it to the same path in the next session, and rerun Cell 4. Completed
runs are not repeated.

## Cell 5 — results and archive

```python
import pandas as pd
from IPython.display import Image, display

display(pd.read_csv(OUT / "aggregate.csv").sort_values("auc_mean", ascending=False))
display(pd.read_csv(OUT / "metrics.csv"))
display(Image(filename=str(OUT / "baseline_comparison.png")))
archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
print("download:", archive)
```

`metrics.csv` reports, for every model and seed: ROC-AUC (`auc`), average
precision (`ap`), balanced accuracy (`ba`), recall, false-positive rate
(`fpr = 1 - specificity`), precision, F1, and specificity. `aggregate.csv`
reports mean and standard deviation over the completed seeds.

