from pathlib import Path
from zipfile import ZipFile, BadZipFile
import os
import shutil
import subprocess
import sys

from IPython.display import FileLink, Image, display

RUN_WAVEFORMS = globals().get("RUN_WAVEFORMS", False)
EXTRA_REPORTS = [Path(p) for p in globals().get("EXTRA_REPORTS", [])]
WORK = Path("/kaggle/working")
ROOT = WORK / "clmm_paper_figures"
REPO, OUT = ROOT / "source", ROOT / "figures"
URL = "https://github.com/TuanHungGiang/time-series-project.git"
COMMIT = "75f84f4609b89b77ad8e3f8061aa4cdbea2329fa"
Z24 = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
HELL = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")

ROOT.mkdir(parents=True, exist_ok=True)
if not (REPO / ".git").is_dir():
    subprocess.run(["git", "init", str(REPO)], check=True)
git = ["git", "-C", str(REPO)]
if subprocess.run([*git, "cat-file", "-e", f"{COMMIT}^{{commit}}"], capture_output=True).returncode:
    subprocess.run([*git, "fetch", "--depth", "1", URL, COMMIT], check=True)
subprocess.run([*git, "switch", "--detach", COMMIT], check=True)

members = {
    "Z24/scores.csv.gz", "Z24/protocol.json", "Z24/metrics_by_mask.csv",
    "HellBridge/scores.csv.gz", "HellBridge/protocol.json", "HellBridge/calibration_by_mask.csv",
}


def is_report(path):
    if path.is_dir():
        return all((path / member).is_file() for member in members)
    try:
        with ZipFile(path) as archive:
            return members.issubset(archive.namelist())
    except (BadZipFile, OSError):
        return False


candidates = {*WORK.glob("*/seed_*"), *WORK.glob("*_report.zip"), *WORK.glob("*/*_report.zip"),
              *Path("/kaggle/input").rglob("*_report.zip")}
for path in EXTRA_REPORTS:
    if not is_report(path):
        raise ValueError(f"Expected a full CLMM seed report: {path}")
reports = sorted(path for path in candidates if is_report(path))
reports = list(dict.fromkeys([*EXTRA_REPORTS, *reports]))
if reports:
    print("Full score reports found:", len(reports))
    for path in reports:
        print(path)
else:
    print("No full CLMM score reports found. The split diagram will still be generated.")

env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO / "src"), env.get("PYTHONPATH"))))
env["OMP_NUM_THREADS"] = "2"
env["OPENBLAS_NUM_THREADS"] = "2"
subprocess.run([
    sys.executable, "-u", "-m", "clmm_timesnet.clmm_paper_score_figures",
    "--output-dir", str(OUT), "--reports", *(str(path) for path in reports),
], cwd=WORK, env=env, check=True)

figures = ["two_bridge_split.png"]
if reports:
    figures.extend(["damage_state_recall.png", "healthy_damage_scores_50.png"])
if RUN_WAVEFORMS:
    subprocess.run([
        sys.executable, "-u", "-m", "clmm_timesnet.clmm_paper_waveforms",
        "--z24-root", str(Z24), "--hell-h5", str(HELL), "--output-dir", str(OUT),
    ], cwd=WORK, env=env, check=True)
    figures.append("observed_base_corrected_waveforms.png")
for figure in figures:
    display(Image(filename=str(OUT / figure)))
bundle = shutil.make_archive(str(ROOT / "clmm_paper_figures"), "zip", OUT)
display(FileLink(os.path.relpath(bundle, Path.cwd())))
