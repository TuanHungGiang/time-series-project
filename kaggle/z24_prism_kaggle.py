"""Kaggle entry point: clone this repo + PRISM, download the Z24 data, train PRISM on both splits.

Kaggle settings needed: GPU on, Internet on.
If the GitHub repo is private, add a Kaggle secret named GITHUB_TOKEN (read access to the repo).
Results (json, confusion matrices, checkpoints) end up in /kaggle/working/z24-prism/results.
"""
import os
import subprocess
import sys

GH_REPO = "TuanHungGiang/z24-prism"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/z24-prism"


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


token = ""
try:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass  # public repo, or no secret configured

url = f"https://{token + '@' if token else ''}github.com/{GH_REPO}.git"
os.chdir(WORK)
if not os.path.isdir(REPO_DIR):
    sh(f"git clone --depth 1 {url} {REPO_DIR}")
os.chdir(REPO_DIR)

if not os.path.isdir("PRISM"):
    sh("git clone --depth 1 https://github.com/fedezuc/PRISM.git PRISM")

sh(f"{sys.executable} -m pip install -q huggingface_hub scikit-learn")
from huggingface_hub import snapshot_download  # noqa: E402

snapshot_download("thanglexuan/Z24-dataset-processed", repo_type="dataset",
                  local_dir="Z24-dataset-processed", allow_patterns=["inputs.npy", "labels.npy"])

import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

for split in ("segment", "setup"):
    sh(f"{sys.executable} -u run_prism_z24.py --split {split}")

sh("ls -la results")
