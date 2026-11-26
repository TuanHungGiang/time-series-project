from pathlib import Path
import os
import shutil
import subprocess
import sys

import pandas as pd
from IPython.display import FileLink, Image, display

SEED_INDICES = list(globals().get("SEED_INDICES", [0]))
DATASETS = list(globals().get("DATASETS", ["Z24", "HellBridge"]))
MAIN_COMMIT = globals().get("MAIN_COMMIT")
SEEDS = (20270901, 20270913, 20270929, 20271011, 20271027, 20271109, 20271123, 20271207, 20271219, 20280103)

WORK = Path("/kaggle/working")
ROOT = WORK / "cnn_ae_1d_targeted"
REPO = ROOT / "source_main"
ARCHIVE = ROOT / "source_baselines"
OUT, WEIGHTS = ROOT / "reports", ROOT / "weights"
Z24 = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
HELL = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")
GIT_URL = "https://github.com/TuanHungGiang/time-series-project.git"

if not SEED_INDICES or any(index not in range(len(SEEDS)) for index in SEED_INDICES):
    raise ValueError("Set SEED_INDICES to one or more indices from 0 to 9.")
for dataset, path in (("Z24", Z24 / "metadata/records_manifest.csv"), ("HellBridge", HELL)):
    if dataset in DATASETS and not path.is_file():
        raise FileNotFoundError(path)
OUT.mkdir(parents=True, exist_ok=True)

def checkout(path, branch, commit):
    if not (path / ".git").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, GIT_URL, str(path)], check=True)
    git = ["git", "-C", str(path)]
    if commit and subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != commit:
        subprocess.run([*git, "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run([*git, "switch", "--detach", "FETCH_HEAD"], check=True)
    current = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if commit and current != commit:
        raise RuntimeError(f"Wrong commit for {branch}: {current}")
    print(f"{branch} = {current}", flush=True)

checkout(REPO, "main", MAIN_COMMIT)
checkout(ARCHIVE, "codex/mdr-shm-kaggle", "9f6e24b785e51462c5bfc93e2a06d8ca0df1e2df")

env = os.environ.copy()
env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
env.setdefault("OMP_NUM_THREADS", "2")
env.setdefault("OPENBLAS_NUM_THREADS", "2")
command = [
    sys.executable, "-u", "-m", "clmm_timesnet.targeted_baselines",
    "--z24-root", str(Z24), "--hell-h5", str(HELL),
    "--baseline-dir", str(ARCHIVE / "research/mdr_shm_kaggle"),
    "--output-dir", str(OUT), "--weights-dir", str(WEIGHTS),
    "--epochs", "30", "--batch-size", "16", "--models", "cnn_ae_1d",
    "--datasets", *DATASETS, "--seeds", *(str(SEEDS[index]) for index in SEED_INDICES),
]
with (OUT / "training.log").open("a", encoding="utf-8") as log, subprocess.Popen(
    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
table = pd.read_csv(OUT / "summary_mean_sd.csv", dtype={"key": str})
nearby = []
for dataset in DATASETS:
    subset = table.loc[table.dataset.eq(dataset)]
    clean = subset.loc[subset.design.eq("clean")]
    nearby.append(clean.assign(requested_loss_percent=0))
    cumulative = subset.loc[subset.design.eq("cumulative")]
    for percent in (10, 30, 50):
        best = min(cumulative.removed_percent.unique(), key=lambda x: (abs(x - percent), x))
        nearby.append(cumulative.loc[cumulative.removed_percent.eq(best)].assign(requested_loss_percent=percent))
    display(Image(filename=str(OUT / dataset / "cumulative_sensor_loss.png")))
cols = ["dataset", "model", "requested_loss_percent", "removed_percent", "runs"]
for metric in ("auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_low_fpr"):
    cols.extend([f"{metric}_mean", f"{metric}_std"])
focus = pd.concat(nearby, ignore_index=True)[cols]
focus.to_csv(OUT / "clean_and_nearest_10_30_50.csv", index=False)
display(focus[["dataset", "model", "removed_percent", "auc_mean", "auc_std", "ba_mean", "recall_mean", "fpr_mean", "runs"]].round(4))
zip_path = shutil.make_archive(str(WORK / "cnn_ae_1d_targeted_report"), "zip", root_dir=OUT)
display(FileLink(os.path.relpath(zip_path)))
print("Weights:", WEIGHTS)
