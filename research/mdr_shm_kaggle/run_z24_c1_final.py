from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and analyse the locked Z24 C1 protocol.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261110, 20261111, 20261112, 20261113, 20261114])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--matched-specificity", type=float, default=0.875)
    parser.add_argument("--force", action="store_true", help="Re-run completed seeds.")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = args.output_dir / "runs"
    for seed in args.seeds:
        seed_dir = runs_root / f"seed_{seed}"
        required = (seed_dir / "run.json", seed_dir / "validation_predictions.csv", seed_dir / "test_predictions.csv")
        if not args.force and all(path.exists() for path in required):
            print(f"SKIP completed seed={seed} directory={seed_dir}", flush=True)
            continue
        run([
            sys.executable, "-u", str(script_dir / "run_z24_supervised_consistency_c1.py"),
            "--input-root", str(args.input_root), "--output-dir", str(seed_dir),
            "--mode", args.mode, "--seed", str(seed),
        ])
    run([
        sys.executable, "-u", str(script_dir / "analyze_z24_supervised_c1_final.py"),
        "--runs-root", str(runs_root), "--output-dir", str(args.output_dir / "analysis"),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--matched-specificity", str(args.matched_specificity),
    ])
    print(f"Z24_C1_FINAL_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

