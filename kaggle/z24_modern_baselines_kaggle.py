"""Run the modern Z24 baseline suite in one Kaggle notebook command.

This launcher establishes the original-paper/adaptation baselines first.  It
does not claim that CM-HRR has been integrated into these models yet.  Each
runner keeps its own published preprocessing and writes to a separate folder.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
KNOWN = {"scatterad", "target_source", "procrustes", "graph_tailp", "mdfa"}


def run(*command) -> None:
    print("\n+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def normalized_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name in {"train", "val", "test", "metadata", "_excluded"}: root = root.parent
    manifest = root / "metadata" / "records_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    return root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--models", default="scatterad,target_source,procrustes,graph_tailp,mdfa",
                   help="Comma-separated subset of: " + ",".join(sorted(KNOWN)))
    p.add_argument("--scatter-epochs", type=int, default=1)
    p.add_argument("--scatter-folds", type=int, default=3)
    p.add_argument("--mapping-epochs", type=int, default=300)
    p.add_argument("--procrustes-epochs", type=int, default=300)
    p.add_argument("--graph-epochs", type=int, default=20)
    p.add_argument("--mdfa-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args(); root = normalized_root(args.data_root)
    selected = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    unknown = set(selected) - KNOWN
    if unknown: raise ValueError(f"Unknown models: {sorted(unknown)}")
    failures = []

    commands = {
        "scatterad": [sys.executable, "-u", PROJECT / "kaggle" / "z24_strict_psd_scatterad_kaggle.py",
                      "--data-root", root, "--epochs", args.scatter_epochs,
                      "--cv-folds", args.scatter_folds, "--batch-size", 128],
        "target_source": [sys.executable, "-u", PROJECT / "run_z24_target_source_ocsvm.py",
                          "--manifest", root / "metadata" / "records_manifest.csv", "--data-root", root,
                          "--output", WORK / "z24_target_source_ocsvm", "--epochs", args.mapping_epochs,
                          "--patience", max(args.patience, 20), "--batch-size", min(args.batch_size, 64)],
        "procrustes": [sys.executable, "-u", PROJECT / "run_z24_procrustes_ae.py",
                       "--manifest", root / "metadata" / "records_manifest.csv", "--data-root", root,
                       "--output", WORK / "z24_procrustes_ae", "--epochs", args.procrustes_epochs,
                       "--patience", max(args.patience, 20), "--batch-size", min(args.batch_size, 64)],
        "graph_tailp": [sys.executable, "-u", PROJECT / "run_z24_graph_transformer_tailp.py",
                        "--manifest", root / "metadata" / "records_manifest.csv", "--data-root", root,
                        "--output", WORK / "z24_graph_transformer_tailp", "--epochs", args.graph_epochs,
                        "--patience", args.patience],
        "mdfa": [sys.executable, "-u", PROJECT / "run_z24_mdfa_paper.py",
                 "--manifest", root / "metadata" / "records_manifest.csv", "--data-root", root,
                 "--output", WORK / "z24_mdfa_paper", "--epochs", args.mdfa_epochs,
                 "--patience", args.patience, "--batch-size", args.batch_size],
    }
    output_dirs = {
        "scatterad": WORK / "z24_strict_psd_scatterad",
        "target_source": WORK / "z24_target_source_ocsvm",
        "procrustes": WORK / "z24_procrustes_ae",
        "graph_tailp": WORK / "z24_graph_transformer_tailp",
        "mdfa": WORK / "z24_mdfa_paper",
    }
    for name in selected:
        try: run(*commands[name])
        except subprocess.CalledProcessError as exc:
            failures.append({"baseline": name, "returncode": exc.returncode})
            if not args.continue_on_error: raise

    rows = []
    for name in selected:
        metrics = output_dirs[name] / "metrics.csv"
        if metrics.is_file():
            frame = pd.read_csv(metrics); frame.insert(0, "baseline", name); rows.append(frame)
    if rows:
        summary = pd.concat(rows, ignore_index=True, sort=False)
        summary.to_csv(WORK / "z24_modern_baselines_summary.csv", index=False)
        print("\nCOMBINED BASELINE SUMMARY\n" + summary.to_string(index=False), flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(WORK / "z24_modern_baselines_failures.csv", index=False)
        print("\nFAILURES\n" + pd.DataFrame(failures).to_string(index=False), flush=True)
    print(f"\nSelected baselines: {selected}", flush=True)
    print("Combined summary: /kaggle/working/z24_modern_baselines_summary.csv", flush=True)


if __name__ == "__main__": main()

