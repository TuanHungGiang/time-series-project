"""Extend the original HellBridge CLMM gain with a mask-gated spectral term."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .hellbridge_spectral_kaggle import BANDS_HZ, INPUT, LENGTH, MASKS, METHODS, OUTPUT, SEED, SPECTRAL_WEIGHT
from .hellbridge_spectral_kaggle import write_calibration_reports


DESTINATION = OUTPUT.with_name("clmm_hellbridge_gain_spectral_1024_seed20270905")
ADAPTER_SPECTRAL = "adapter_spectral_gated"
COMBINED = "clmm_gain_spectral"
RATE_NAMES = ("sensor_0.10", "sensor_0.30", "sensor_0.50")


def add_spectral_block(adapter, legacy, spectral, missing):
    bonus = np.where(missing, spectral - adapter, 0.0)
    return adapter + bonus, legacy + bonus


def validate_source(source):
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    expected = dict(seed=SEED, dataset="HellBridge P2_NM_Z", native_points=LENGTH,
                    spectral_weight=SPECTRAL_WEIGHT, nested_replicates=list(MASKS),
                    bands_hz=[list(band) for band in BANDS_HZ], methods=list(METHODS))
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise ValueError(f"Incompatible spectral score source: {source}")
    scores = pd.read_csv(source / "scores.csv.gz", float_precision="round_trip")
    keys = ["condition", "replicate"]
    counts = scores.groupby(keys).size().to_dict()
    expected_masks = {("clean", 0), *((rate, mask) for rate in RATE_NAMES for mask in MASKS)}
    if (set(counts) != expected_masks or set(counts.values()) != {740}
            or scores.duplicated([*keys, "split", "sample_id"]).any()
            or not np.isfinite(scores[list(METHODS)].to_numpy()).all()):
        raise ValueError(f"Incomplete or invalid spectral scores in {source}")
    return protocol, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--source-dir", type=Path, default=OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    cli = parser.parse_args()
    source, out = cli.source_dir, cli.output_dir
    if source.resolve() == out.resolve():
        raise ValueError("Choose separate source and output directories")
    required = ("protocol.json", "scores.csv.gz", "masks.csv", "fit_groups.json")
    if any(not (source / name).is_file() for name in required):
        subprocess.run([sys.executable, "-u", "-m", "clmm_timesnet.hellbridge_spectral_kaggle",
                        "--hell-h5", str(cli.hell_h5), "--output-dir", str(source)], check=True)
    else:
        print(f"Reusing completed spectral scores from {source}", flush=True)

    protocol, scores = validate_source(source)
    scores[ADAPTER_SPECTRAL], scores[COMBINED] = add_spectral_block(
        scores.adapter_mse.to_numpy(), scores.clmm_gain_old.to_numpy(),
        scores.clmm_spectral.to_numpy(), scores.condition.ne("clean").to_numpy(),
    )
    out.mkdir(parents=True, exist_ok=True)
    for name in ("masks.csv", "fit_groups.json"):
        shutil.copy2(source / name, out / name)
    protocol.update(score_source=str(source), methods=[*METHODS, ADAPTER_SPECTRAL, COMBINED],
                    spectral_gate="bonus zero on clean, original spectral bonus on all masked observations")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    path = out / "scores.csv.gz"
    scores.to_csv(path, index=False)
    write_calibration_reports(path, out, (*METHODS, ADAPTER_SPECTRAL, COMBINED), candidate=COMBINED)
    bundle = shutil.make_archive(str(out) + "_report", "zip", root_dir=out)
    print(f"HELLBRIDGE_GAIN_SPECTRAL_PASS reports={out} zip={bundle}", flush=True)


if __name__ == "__main__":
    main()
