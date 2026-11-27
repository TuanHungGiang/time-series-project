from pathlib import Path
import os
import shutil
import subprocess
import sys

from IPython.display import FileLink, Image, display
import pandas as pd
import torch

WORK = Path("/kaggle/working")
ROOT = WORK / "clmm_matched_baselines_1seed_val"
ROOT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "reports"
URL = "https://github.com/TuanHungGiang/time-series-project.git"
Z24 = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
HELL = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")

for path in (Z24 / "metadata/records_manifest.csv", HELL):
    if not path.is_file():
        raise FileNotFoundError(path)
if not torch.cuda.is_available():
    raise RuntimeError("Kaggle → Settings → Accelerator → GPU")


def checkout(name, commit):
    path = ROOT / name
    if not (path / ".git").is_dir():
        subprocess.run(["git", "init", str(path)], check=True)
    git = ["git", "-C", str(path)]
    if subprocess.run([*git, "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
        subprocess.run([*git, "fetch", "--depth", "1", URL, commit], check=True)
    subprocess.run([*git, "switch", "--detach", commit], check=True)
    return path


repo = checkout("clmm", "176285e14f4c5825c0a7d3f930469385f62fff01")
archive = checkout("baseline_source", "09d81a4466668aa995d76446ae14c1d8e0c216bd")
try:
    from torch_geometric.nn import GATConv
except ImportError:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet", "--retries", "1",
        "torch-geometric==2.7.0",
    ], check=True)

env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo / "src"), env.get("PYTHONPATH"))))
env["OMP_NUM_THREADS"] = "2"
env["OPENBLAS_NUM_THREADS"] = "2"
env["CLMM_LOG_EACH_EPOCH"] = "1"
command = [
    sys.executable, "-u", "-m", "clmm_timesnet.matched_baselines_1seed_kaggle",
    "--z24-root", str(Z24), "--hell-h5", str(HELL),
    "--baseline-dir", str(archive / "research/mdr_shm_kaggle"),
    "--output-dir", str(OUT),
]
OUT.mkdir(parents=True, exist_ok=True)
log_path = OUT / "training.log"
with log_path.open("a", encoding="utf-8") as log, subprocess.Popen(
    command, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)

losses = pd.read_csv(OUT / "epoch_loss_mean_by_bridge.csv")
pca = losses.loc[losses["model"].eq("pca_spe"),
                 ["dataset", "groups", "train_loss", "val_loss"]]
print("PCA-SPE | Post-fit training and held-out loss")
display(pca.round(6))
for figure in sorted(OUT.glob("train_val_loss_*.png")):
    display(Image(filename=str(figure)))

summary = pd.read_csv(OUT / "summary_1seed.csv")
columns = ["dataset", "method", "condition", "auc", "ap", "ba", "recall", "fpr", "precision", "f1"]
for bridge, table in summary.loc[summary["policy"].eq("mask_matched")].groupby("dataset"):
    print(bridge)
    display(table[columns].round(4).reset_index(drop=True))
bundle = shutil.make_archive(str(ROOT / "clmm_matched_baselines_1seed_val"), "zip", OUT)
display(FileLink(os.path.relpath(bundle, Path.cwd())))
