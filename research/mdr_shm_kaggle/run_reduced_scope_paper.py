from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_stage(name: str, marker: Path, command: list[str], force: bool) -> dict:
    if marker.exists() and not force:
        print(f"[SKIP] {name}: {marker}", flush=True)
        return {"stage": name, "status": "reused", "marker": str(marker)}
    print(f"[RUN ] {name}", flush=True)
    print("       " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not marker.exists():
        raise RuntimeError(f"{name} completed without expected marker: {marker}")
    return {"stage": name, "status": "completed", "marker": str(marker)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the reduced-scope reliability-aware SHM paper artifacts."
    )
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/reduced_scope_paper"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.z24_root.exists():
        raise FileNotFoundError(args.z24_root)
    if not args.hell_h5.exists():
        raise FileNotFoundError(args.hell_h5)
    args.output_root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    python = sys.executable

    quick = args.mode == "quick"
    bootstrap = 300 if quick else 2000
    v31_bootstrap = 300 if quick else 1000
    random_repeats = 50 if quick else 200
    detector_epochs = 12 if quick else 60
    detector_seeds = "0" if quick else "0,1,2"

    v31 = args.output_root / "z24_v31"
    reliability = args.output_root / "z24_reliability_paper"
    detectors = args.output_root / "z24_detectors"
    detector_paper = args.output_root / "z24_detector_paper"
    final_stats = args.output_root / "z24_final_statistics"
    hell = args.output_root / "hellbridge_supplement"

    stages = []
    stages.append(run_stage(
        "Z24 selective reliability V3.1",
        v31 / "record_scores.csv",
        [
            python, str(code / "run_z24_model_a_v31.py"),
            "--input-root", str(args.z24_root),
            "--output-dir", str(v31),
            "--mode", args.mode,
            "--bootstrap-samples", str(v31_bootstrap),
        ],
        args.force,
    ))
    stages.append(run_stage(
        "Z24 matched-coverage reliability analysis",
        reliability / "table5_matched_aurc.csv",
        [
            python, str(code / "analyze_z24_v31_paper.py"),
            "--input-dir", str(v31),
            "--output-dir", str(reliability),
            "--random-repeats", str(random_repeats),
        ],
        args.force,
    ))
    stages.append(run_stage(
        "Z24 classical and deep detector baselines",
        detectors / "record_scores.csv",
        [
            python, str(code / "run_z24_detector_baselines.py"),
            "--input-root", str(args.z24_root),
            "--output-dir", str(detectors),
            "--mode", args.mode,
            "--epochs", str(detector_epochs),
            "--seeds", detector_seeds,
        ],
        args.force,
    ))
    stages.append(run_stage(
        "Z24 detector tables and figures",
        detector_paper / "table2_clean_comparison.csv",
        [
            python, str(code / "analyze_z24_detector_baselines.py"),
            "--input-dir", str(detectors),
            "--output-dir", str(detector_paper),
        ],
        args.force,
    ))
    stages.append(run_stage(
        "Z24 paired statistics and evidence synthesis",
        final_stats / "table3_final_framework_evidence.csv",
        [
            python, str(code / "analyze_z24_final_statistics.py"),
            "--detector-dir", str(detectors),
            "--reliability-dir", str(reliability),
            "--output-dir", str(final_stats),
            "--bootstrap-samples", str(bootstrap),
            "--coverage", "0.70",
        ],
        args.force,
    ))
    stages.append(run_stage(
        "HellBridge supplementary protocol replication",
        hell / "metrics.csv",
        [
            python, str(code / "run_hellbridge_b2_external.py"),
            "--h5", str(args.hell_h5),
            "--output-dir", str(hell),
            "--mode", args.mode,
            "--window-seconds", "60",
            "--bootstrap-samples", str(bootstrap),
        ],
        args.force,
    ))

    artifact = {
        "scope": "healthy-only selective SHM under structured sensor corruption",
        "mode": args.mode,
        "primary_dataset": "Z24",
        "supplementary_dataset": "HellBridge",
        "cross_bridge_transfer_claimed": False,
        "damage_labels_used_for_fit_or_calibration": False,
        "proposed_detector": "frozen B2 modal/spectral detector",
        "proposed_component": "V3.1 integrity-aware selective reliability layer",
        "negative_ablations": ["multiview V4", "cross-spectral V5"],
        "stages": stages,
        "directories": {
            "z24_v31": str(v31),
            "z24_reliability_paper": str(reliability),
            "z24_detectors": str(detectors),
            "z24_detector_paper": str(detector_paper),
            "z24_final_statistics": str(final_stats),
            "hellbridge_supplement": str(hell),
        },
    }
    (args.output_root / "ARTIFACT_INDEX.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    (args.output_root / "README_RESULTS.md").write_text(
        "# Reduced-scope paper artifacts\n\n"
        "Primary claim: a healthy-calibrated reliability layer is evaluated at matched coverage "
        "under structured sensor corruption without changing the frozen damage detector.\n\n"
        "HellBridge is an independent protocol replication and limitation study; it is not "
        "presented as zero-shot cross-bridge transfer.\n\n"
        "Use `z24_final_statistics/table3_final_framework_evidence.csv` for the main synthesis, "
        "`z24_reliability_paper/table2_matched_coverage.csv` and "
        "`table5_matched_aurc.csv` for the selective claim, and "
        "`hellbridge_supplement/metrics.csv` only as supplementary evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2), flush=True)
    print(f"REDUCED_SCOPE_PAPER_PASS output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()

