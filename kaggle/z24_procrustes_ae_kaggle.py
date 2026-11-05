"""Kaggle launcher for the Z24 modal AE + Procrustes experiment."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
def run(*cmd): print("+", " ".join(map(str, cmd)), flush=True); subprocess.run(list(map(str, cmd)), check=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=600); p.add_argument("--patience",type=int,default=75)
    a=p.parse_args(); root=a.data_root.expanduser().resolve()
    if root.name in {"train","val","test","metadata","_excluded"}: root=root.parent
    manifest=root/"metadata"/"records_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    run(sys.executable,"-u",PROJECT/"run_z24_procrustes_ae.py","--manifest",manifest,
        "--data-root",root,"--output","/kaggle/working/z24_procrustes_ae",
        "--epochs",a.epochs,"--patience",a.patience)
if __name__=="__main__": main()
