from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def execute(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261110, 20261111, 20261112, 20261113, 20261114])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--specificity-target", type=float, default=0.875)
    parser.add_argument("--run-occlusion", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    runs = args.output_dir / "runs"
    for seed in args.seeds:
        output = runs / f"seed_{seed}"
        required = (output / "run.json", output / "test_predictions.csv")
        if not args.force and all(path.exists() for path in required):
            if not args.run_occlusion or (output / "occlusion_predictions.csv").exists():
                print(f"SKIP completed seed={seed}", flush=True)
                continue
        command = [
            sys.executable, "-u", str(scripts / "run_z24_excitation_spectral_c2.py"),
            "--input-root", str(args.input_root), "--output-dir", str(output),
            "--mode", args.mode, "--seed", str(seed),
            "--specificity-target", str(args.specificity_target),
        ]
        if args.run_occlusion:
            command.append("--run-occlusion")
        execute(command)
    execute([
        sys.executable, "-u", str(scripts / "analyze_z24_c2_final.py"),
        "--runs-root", str(runs), "--output-dir", str(args.output_dir / "analysis"),
        "--bootstrap-samples", str(args.bootstrap_samples),
    ])
    print(f"Z24_C2_FINAL_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

