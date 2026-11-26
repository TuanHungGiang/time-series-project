"""Run and resume the paired CLMM whitening experiment on both bridges."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch

from .whitening_two_bridges_kaggle import CANDIDATE, INPUT, LENGTH, MASKS, METHODS, Z24_INPUT


SEEDS = (20270905, 20270913, 20270929, 20271011, 20271027,
         20271109, 20271123, 20271207, 20271219, 20280103)
OUTPUT = Path("/kaggle/working/clmm_whitening_10seed")
CONDITIONS = ("clean", "sensor_0.10", "sensor_0.30", "sensor_0.50")
DATASETS = ("Z24", "HellBridge")
POLICIES = ("clean_frozen", "mask_matched")
METRICS = ("auc", "ap", "ba", "recall", "fpr", "precision", "f1", "diagnostic_recall")
KEY = ["dataset", "condition", "policy", "method"]
CHECKPOINT_PROTOCOL = "clmm-whitening-native-1024-mask-300-304-v1"
N_BOOTSTRAPS = 5000


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def validate_seed(frame, seed):
    required = {"seed", "replicate", "threshold", *KEY, *METRICS}
    if not required.issubset(frame.columns) or set(frame.seed) != {seed}:
        raise ValueError(f"Invalid schema or seed for {seed}")
    expected = {
        (bridge, condition, policy, method, mask)
        for bridge in DATASETS for condition in CONDITIONS for policy in POLICIES
        for method in METHODS for mask in ((0,) if condition == "clean" else MASKS)
    }
    index = frame[[*KEY, "replicate"]]
    observed = set(index.itertuples(index=False, name=None))
    if index.duplicated().any() or observed != expected:
        raise ValueError(f"Incomplete, unexpected or duplicate mask pairs for seed {seed}")
    if not np.isfinite(frame[[*METRICS, "threshold"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite scores for seed {seed}")
    return frame


def collect_seed(out, seed):
    hell = out / "HellBridge"
    z24 = out / "Z24"
    for source, window_key, mask_key in (
        (hell / "protocol.json", "native_points", "nested_replicates"),
        (z24 / "protocol.json", "points_per_window", "masks"),
    ):
        metadata = json.loads(source.read_text(encoding="utf-8"))
        if (metadata["seed"] != seed or metadata[window_key] != LENGTH
                or tuple(metadata[mask_key]) != MASKS):
            raise ValueError(f"Different seed, window length or mask IDs in {source}")
    z24_rows = pd.read_csv(z24 / "metrics_by_mask.csv").rename(columns={"recall_low_fpr": "diagnostic_recall"})
    z24_rows["policy"] = z24_rows.policy.map({"clean_q90": "clean_frozen", "masked_q90": "mask_matched"})
    hb_rows = pd.read_csv(hell / "calibration_by_mask.csv")
    hb_rows = hb_rows.loc[hb_rows.alpha.eq(.05)].rename(columns={"recall_at_fpr_05": "diagnostic_recall"})
    hb_rows["policy"] = hb_rows.policy.map({"clean_calibration": "clean_frozen", "same_mask_calibration": "mask_matched"})
    hb_rows["dataset"] = "HellBridge"
    rows = pd.concat([z24_rows, hb_rows], ignore_index=True)
    columns = [*KEY, "replicate", "threshold", *METRICS]
    paired = rows.loc[rows.method.isin(METHODS), columns].assign(seed=seed)
    return validate_seed(paired, seed)


def checkpoint_path(directory, seed):
    return directory / f"seed_{seed}.csv"


def load_checkpoints(directory):
    return {
        seed: validate_seed(pd.read_csv(path), seed)
        for seed in SEEDS if (path := checkpoint_path(directory, seed)).is_file()
    }


def restore_checkpoints(archive, directory, protocol):
    with ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("protocol.json"))
        if manifest != protocol:
            raise ValueError(f"Different experiment settings or dataset versions in {archive}")
        for seed in SEEDS:
            path = checkpoint_path(directory, seed)
            if path.name not in bundle.namelist():
                continue
            restored = validate_seed(pd.read_csv(io.BytesIO(bundle.read(path.name))), seed)
            if path.is_file():
                local = validate_seed(pd.read_csv(path), seed)
                try:
                    pd.testing.assert_frame_equal(
                        local.sort_values([*KEY, "replicate"]).reset_index(drop=True),
                        restored.sort_values([*KEY, "replicate"]).reset_index(drop=True),
                    )
                except AssertionError as error:
                    raise ValueError(f"Conflicting checkpoint for seed {seed} in {archive}") from error
            else:
                write_csv(restored, path)


def save_checkpoint_archive(out, directory, protocol):
    path = out / "whitening_seed_checkpoints.zip"
    temp = path.with_suffix(".tmp")
    with ZipFile(temp, mode="w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("protocol.json", json.dumps(protocol, indent=2))
        for seed in SEEDS:
            checkpoint = checkpoint_path(directory, seed)
            if checkpoint.is_file():
                bundle.write(checkpoint, arcname=checkpoint.name)
    temp.replace(path)


def bootstrap_paired(deltas):
    rng = np.random.default_rng(20270905)
    summaries = []
    for keys, part in deltas.groupby([*KEY[:3], "baseline"]):
        metrics = [f"delta_{m}" for m in METRICS]
        values = part.sort_values("seed")[metrics].to_numpy()
        n = len(values)
        interval = np.full((2, len(metrics)), np.nan)
        if n > 1:
            sample_ids = rng.integers(n, size=(N_BOOTSTRAPS, n))
            interval = np.quantile(values[sample_ids].mean(axis=1), [.025, .975], axis=0)
        info = dict(zip([*KEY[:3], "baseline"], keys, strict=True))
        for index, metric in enumerate(METRICS):
            summaries.append(dict(**info, metric=metric, seeds=n,
                                  mean_delta=values[:, index].mean(),
                                  ci_lower=interval[0, index], ci_upper=interval[1, index]))
    return pd.DataFrame(summaries)


def report(out, runs, historical):
    if not runs:
        return
    masks = pd.concat([runs[seed] for seed in sorted(runs)], ignore_index=True)
    write_csv(masks, out / "per_mask_metrics.csv")
    by_seed = masks.groupby([*KEY, "seed"], as_index=False)[list(METRICS)].mean()
    write_csv(by_seed, out / "per_seed_metrics.csv")
    groups = by_seed.groupby(KEY)
    stats = groups[list(METRICS)].agg(["mean", "std"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    stats["seeds"] = groups.size()
    summary = stats.reset_index()
    write_csv(summary, out / "summary_mean_sd.csv")

    idx = [*KEY[:3], "seed"]
    candidate = by_seed.loc[by_seed.method.eq(CANDIDATE)].set_index(idx)[list(METRICS)]
    differences = []
    for comparator in METHODS:
        if comparator == CANDIDATE:
            continue
        baseline = by_seed.loc[by_seed.method.eq(comparator)].set_index(idx)[list(METRICS)]
        if set(candidate.index) != set(baseline.index):
            raise ValueError(f"Different same-seed pairs for {comparator}")
        differences.append((candidate - baseline).add_prefix("delta_").reset_index().assign(baseline=comparator))
    deltas = pd.concat(differences, ignore_index=True)
    write_csv(deltas, out / "paired_seed_deltas.csv")
    write_csv(bootstrap_paired(deltas), out / "paired_ci.csv")

    current = summary.assign(protocol="native_1024_paired_seed_sweep", design="nested_random", sd_over="seeds",
                             masks=np.where(summary.condition.eq("clean"), 1, len(MASKS)))
    write_csv(pd.concat([current, historical], ignore_index=True, sort=False), out / "baseline_comparison.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, default=Z24_INPUT)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--restore-zip", type=Path, action="append", default=[])
    cli = parser.parse_args()
    if len(set(cli.seeds)) != len(cli.seeds) or any(seed not in SEEDS for seed in cli.seeds):
        parser.error(f"Use distinct seeds from {SEEDS}")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator on Kaggle")
    catalog = cli.z24_root / "metadata" / "records_manifest.csv"
    for path in (catalog, cli.hell_h5, cli.baseline_reference, *cli.restore_zip):
        if not path.is_file():
            raise FileNotFoundError(path)
    out = cli.output_dir.resolve()
    directory = out / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"GPU={torch.cuda.get_device_name(0)} | torch={torch.__version__} | CUDA={torch.version.cuda}", flush=True)
    print("Fingerprinting the two dataset inputs for checkpoint portability...", flush=True)
    protocol = dict(
        name=CHECKPOINT_PROTOCOL, seeds=list(SEEDS), window_points=LENGTH, masks=list(MASKS),
        z24_manifest_sha256=file_sha256(catalog), hellbridge_h5_sha256=file_sha256(cli.hell_h5),
        backbone_epochs=5, adapter_epochs=8, batch_size=8, shrinkage=.5, gain_weight=.25,
        diagnostic_fpr_limit={"Z24": 0., "HellBridge": .05},
        aggregation="mean over masks, then mean and sample SD over seeds",
        confidence_intervals="paired percentile bootstrap over training seeds on a fixed test split and masks",
    )
    path = out / "protocol.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else protocol
    if stored != protocol:
        raise ValueError(f"Existing output directory uses different inputs or experiment settings: {out}")
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    historical = pd.read_csv(cli.baseline_reference)
    for archive in cli.restore_zip:
        restore_checkpoints(archive, directory, protocol)
    runs = load_checkpoints(directory)
    if runs:
        report(out, runs, historical)
        save_checkpoint_archive(out, directory, protocol)

    for seed in cli.seeds:
        if seed in runs:
            print(f"Reused completed seed {seed}", flush=True)
            continue
        target = out / f"seed_{seed}"
        print(f"\n=== SEED {seed} | {len(runs)}/{len(SEEDS)} completed ===", flush=True)
        if not (target / "paired_summary.csv").is_file():
            subprocess.run([
                sys.executable, "-u", "-m", "clmm_timesnet.whitening_two_bridges_kaggle",
                "--z24-root", str(cli.z24_root.resolve()), "--hell-h5", str(cli.hell_h5.resolve()),
                "--baseline-reference", str(cli.baseline_reference.resolve()),
                "--output-dir", str(target), "--seed", str(seed),
            ], check=True)
        runs[seed] = collect_seed(target, seed)
        write_csv(runs[seed], checkpoint_path(directory, seed))
        save_checkpoint_archive(out, directory, protocol)
        report(out, runs, historical)
        print(f"CHECKPOINT_READY seed={seed} total={len(runs)}/{len(SEEDS)} path={out / 'whitening_seed_checkpoints.zip'}", flush=True)

    state = "COMPLETE" if set(runs) == set(SEEDS) else "PARTIAL"
    print(f"WHITENING_MULTISEED_{state} seeds={sorted(runs)} reports={out}", flush=True)


if __name__ == "__main__":
    main()
