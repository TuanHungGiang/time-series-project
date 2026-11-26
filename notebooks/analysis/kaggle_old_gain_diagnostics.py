from pathlib import Path
import os
import shutil
import subprocess
import sys

import pandas as pd
from IPython.display import FileLink, display

SEED = int(globals().get("SEED", 20270901))
DATASETS = list(globals().get("DATASETS", ["Z24", "HellBridge"]))
MAIN_COMMIT = globals().get("MAIN_COMMIT")
REUSE_OLD_CLMM = bool(globals().get("REUSE_OLD_CLMM", True))
WORK = Path("/kaggle/working")
ROOT = WORK / f"clmm_old_gain_diagnostics_{SEED}"
REPO, ARCHIVE = ROOT / "source_main", ROOT / "source_baselines"
OUT, WEIGHTS = ROOT / "reports", ROOT / "weights"
Z24 = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
HELL = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")
URL = "https://github.com/TuanHungGiang/time-series-project.git"

for dataset, path in (("Z24", Z24 / "metadata/records_manifest.csv"), ("HellBridge", HELL)):
    if dataset in DATASETS and not path.is_file():
        raise FileNotFoundError(path)
OUT.mkdir(parents=True, exist_ok=True)


def checkout(path, branch, commit):
    if not (path / ".git").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, URL, str(path)], check=True)
    git = ["git", "-C", str(path)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if commit and head != commit:
        subprocess.run([*git, "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run([*git, "switch", "--detach", "FETCH_HEAD"], check=True)
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if commit and head != commit:
        raise RuntimeError(f"Unexpected {branch} revision: {head}")
    print(f"{branch} = {head}", flush=True)


checkout(REPO, "main", MAIN_COMMIT)
checkout(ARCHIVE, "codex/mdr-shm-kaggle", "9f6e24b785e51462c5bfc93e2a06d8ca0df1e2df")
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
env.setdefault("OMP_NUM_THREADS", "2")
env.setdefault("OPENBLAS_NUM_THREADS", "2")
command = [
    sys.executable, "-u", "-m", "clmm_timesnet.old_gain_diagnostics",
    "--z24-root", str(Z24), "--hell-h5", str(HELL), "--datasets", *DATASETS,
    "--baseline-dir", str(ARCHIVE / "research/mdr_shm_kaggle"),
    "--output", str(OUT), "--weights-dir", str(WEIGHTS), "--seed", str(SEED),
]
if REUSE_OLD_CLMM:
    paths = [WORK / "clmm_gain_head_targeted" / "weights", WORK / "clmm_centered_gain_targeted" / "weights"]
    if globals().get("PREVIOUS_CLMM_WEIGHTS"):
        paths.insert(0, Path(PREVIOUS_CLMM_WEIGHTS))
    prior = next((p for p in paths if any((p / b / str(SEED)).is_dir() for b in DATASETS)), None)
    command.extend(("--baseline-runs", str(WORK)))
    if prior is not None:
        command.extend(("--previous-clmm-weights", str(prior)))

with (OUT / "training_and_layer_diagnostics.log").open("a", encoding="utf-8") as log, subprocess.Popen(
    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
metrics = pd.read_csv(OUT / "old_clmm_metrics.csv")
display(metrics[["dataset", "requested_loss", "actual_loss", "auc", "ba", "recall", "fpr",
                 "recall_low_fpr"]].round(4))
probes = pd.read_csv(OUT / "layer_probes.csv")
display(probes.pivot(index=["dataset", "stage"], columns="requested_loss", values="auc").round(4))
archive = shutil.make_archive(str(WORK / f"clmm_old_gain_diagnostics_{SEED}_report"), "zip", root_dir=OUT)
display(FileLink(os.path.relpath(archive)))
print("Weights:", WEIGHTS)
