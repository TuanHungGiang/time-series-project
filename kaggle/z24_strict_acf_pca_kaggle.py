"""Kaggle launcher for the restored strict ACF/PCA baseline."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,required=True);a=p.parse_args()
    root=a.data_root.expanduser().resolve()
    if root.name in {"train","val","test","metadata","_excluded"}:root=root.parent
    cmd=[sys.executable,"-u",str(PROJECT/"run_z24_strict_acf_pca.py"),"--manifest",
         str(root/"metadata"/"records_manifest.csv"),"--data-root",str(root),
         "--output","/kaggle/working/z24_strict_acf_pca"]
    print("+"," ".join(cmd),flush=True);subprocess.run(cmd,check=True)
if __name__=="__main__":main()

