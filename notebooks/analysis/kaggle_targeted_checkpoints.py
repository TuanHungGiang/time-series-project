"""Copy into one Kaggle code cell, or run as a script in a Kaggle session."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd
from IPython.display import FileLink, Image, display

SEED_INDEX = int(globals().get("SEED_INDEX", 0))  # 0–9, one shard per session
MODE = globals().get("MODE", "benchmark")  # "merge" combines saved reports
MAIN_COMMIT = globals().get("MAIN_COMMIT")
ARCHIVE_COMMIT = "9f6e24b785e51462c5bfc93e2a06d8ca0df1e2df"
SEEDS = (20270901, 20270913, 20270929, 20271011, 20271027, 20271109, 20271123, 20271207, 20271219, 20280103)
MODELS = ["lstm_ae", "usad", "tranad", "timesnet", "moderntcn", "scatterad"]
DATASETS = ["Z24", "HellBridge"]
URL = "https://github.com/TuanHungGiang/time-series-project.git"
WORK = Path("/kaggle/working")
SOURCE = WORK / "targeted_checkpoint_source"
MAIN = SOURCE / "main"
ARCHIVE = SOURCE / "archive"
Z24 = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
HELL = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")


def git_for(path, branch, expected):
    if not (path / ".git").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, URL, str(path)], check=True)
    git = ["git", "-C", str(path)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if expected and head != expected:
        if not re.fullmatch(r"[0-9a-f]{40}", expected):
            raise ValueError(f"Invalid Git SHA for {branch}")
        subprocess.run([*git, "fetch", "--depth", "1", "origin", expected], check=True)
        subprocess.run([*git, "switch", "--detach", "FETCH_HEAD"], check=True)
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        if head != expected:
            raise ValueError(f"Unexpected revision for {branch}: {head}")
    return head


SOURCE.mkdir(parents=True, exist_ok=True)
commits = {"main": git_for(MAIN, "main", MAIN_COMMIT)}
runner = MAIN / "src" / "clmm_timesnet" / "targeted_checkpoints.py"
if not runner.is_file():
    raise FileNotFoundError(f"Checkpoint code missing at {runner}. Clone the current main in a fresh source directory.")
env = os.environ.copy()
env["PYTHONPATH"] = str(MAIN / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
env.setdefault("OMP_NUM_THREADS", "2")
env.setdefault("OPENBLAS_NUM_THREADS", "2")

if MODE == "benchmark":
    if not 0 <= SEED_INDEX < len(SEEDS):
        raise ValueError("Use SEED_INDEX in 0..9")
    for path in (Z24 / "metadata" / "records_manifest.csv", HELL):
        if not path.is_file():
            raise FileNotFoundError(path)
    commits["codex/mdr-shm-kaggle"] = git_for(ARCHIVE, "codex/mdr-shm-kaggle", ARCHIVE_COMMIT)
    try:
        from torch_geometric.nn import GATConv
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.7.0"], check=True)

    seed = SEEDS[SEED_INDEX]
    shard = WORK / f"targeted_checkpoint_seed{SEED_INDEX:02d}"
    reports, weights = shard / "reports", shard / "weights"
    reports.mkdir(parents=True, exist_ok=True)
    provenance = reports / "git_commits.json"
    if provenance.exists() and json.loads(provenance.read_text(encoding="utf-8")) != commits:
        raise ValueError("This shard was started with different source commits")
    provenance.write_text(json.dumps(commits, indent=2), encoding="utf-8")
    models = ["pca_spe", *MODELS] if SEED_INDEX == 0 else MODELS
    free_gib = shutil.disk_usage(shard).free / 1024**3
    print(f"seed {SEED_INDEX + 1}/10: {seed} | free space {free_gib:.1f} GiB", flush=True)
    print(f"Checkpoints: {weights}", flush=True)
    print(commits, flush=True)
    command = [
        sys.executable, "-u", "-m", "clmm_timesnet.targeted_baselines",
        "--z24-root", str(Z24), "--hell-h5", str(HELL),
        "--baseline-dir", str(ARCHIVE / "research" / "mdr_shm_kaggle"),
        "--output-dir", str(reports), "--weights-dir", str(weights),
        "--epochs", "30", "--batch-size", "16", "--seeds", str(seed),
        "--datasets", *DATASETS, "--models", *models,
    ]
    log_path = reports / "training.log"
    with log_path.open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=MAIN, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
    active, history = None, []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        fit = re.fullmatch(r"(Z24|HellBridge) (\w+) seed=(\d+) group=(\S+)", line)
        event = re.fullmatch(r"\s*epoch=(\d+)/(\d+) (\w+)=(\S+)", line)
        if fit:
            active = fit.groups()
        if active and event:
            history.append((*active, *event.groups()))
    pd.DataFrame(history, columns=["dataset", "model", "seed", "group", "epoch", "epochs", "loss_name", "loss"]
                 ).to_csv(reports / "training_history.csv", index=False)
    archive_name = f"targeted_checkpoint_reports_seed{SEED_INDEX:02d}"
elif MODE == "merge":
    # Extract the ten report ZIPs in ten separate directories, then supply their locations.
    source_reports = globals().get("REPORT_DIRS")
    if not source_reports:
        source_reports = sorted(path / "reports" for path in WORK.glob("targeted_checkpoint_seed*")
                                if (path / "reports" / "protocol.json").is_file())
    if not source_reports:
        raise FileNotFoundError("Set REPORT_DIRS to the ten directories containing the saved protocol.json files")
    reports = WORK / "targeted_checkpoint_10seed_summary"
    subprocess.run(
        [sys.executable, "-m", "clmm_timesnet.targeted_artifacts", "--output-dir", str(reports),
         "--require-all", "--reports", *map(str, source_reports)],
        cwd=MAIN, env=env, check=True,
    )
    archive_name = "targeted_checkpoint_10seed_summary"
else:
    raise ValueError("Use MODE='benchmark' or MODE='merge'")

summary = pd.read_csv(reports / "summary_mean_sd.csv", dtype={"key": str})
display(summary.loc[summary.design.eq("clean"), ["dataset", "model", "auc_mean", "auc_std", "recall_mean", "runs"]])
for dataset in DATASETS:
    for name in ("input_pipeline", "timesnet_architecture", "waveform_examples",
                 "cumulative_sensor_loss", "single_sensor_impact", "roc_pr_clean_and_midloss"):
        figure = reports / dataset / f"{name}.png"
        if figure.is_file():
            display(Image(filename=str(figure)))
zip_path = shutil.make_archive(str(WORK / archive_name), "zip", root_dir=reports)
display(FileLink(os.path.relpath(zip_path)))
