from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


FS = 100.0
NAME_PATTERN = re.compile(r"MVS_(P[12])_(UDS|DS\d+)_(SM|NM)_([YZ])_(\d+)")
PROFILES = {
    "svendsen2022": {
        "window_samples": 1024,
        "healthy_train_per_record": 40,
        "healthy_test_per_record": 40,
        "damage_test_per_record": 80,
        "ar_order": 5,
        "reference": "Svendsen et al. 2022 data-based SHM",
    },
    "delpozo2025": {
        "window_samples": 1070,
        "healthy_train_per_record": 20,
        "healthy_test_per_record": 20,
        "damage_test_per_record": 80,
        "ar_order": 50,
        "reference": "del Pozo et al. 2025 uncertainty-aware MSD",
    },
}


def parse_name(name: str) -> dict[str, object]:
    match = NAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(name)
    position, state, excitation, shaker_axis, repeat = match.groups()
    return {
        "record_id": name,
        "position": position,
        "state": state,
        "excitation": excitation,
        "shaker_axis": shaker_axis,
        "repeat": int(repeat),
        "regime": f"{position}_{excitation}_{shaker_axis}",
        "label": int(state != "UDS"),
    }


def channel_inventory(group: h5py.Group) -> dict[str, int]:
    acceleration = group["acceleration"]
    ag = [f"AG{i:02d}" for i in range(1, 19)]
    al = [f"AL{i:02d}" for i in range(1, 41)]
    counts = {
        "al_z_channels": sum(name in acceleration and "z" in acceleration[name] for name in al),
        "ag_y_channels": sum(name in acceleration and "y" in acceleration[name] for name in ag),
        "ag_z_channels": sum(name in acceleration and "z" in acceleration[name] for name in ag),
        "as_y_channels": int("AS" in acceleration and "y" in acceleration["AS"]),
        "as_z_channels": int("AS" in acceleration and "z" in acceleration["AS"]),
    }
    return {key: int(value) for key, value in counts.items()}


def common_length(group: h5py.Group) -> int:
    acceleration = group["acceleration"]
    lengths = []
    for sensor in acceleration.values():
        if not isinstance(sensor, h5py.Group):
            continue
        for dataset in sensor.values():
            if isinstance(dataset, h5py.Dataset) and dataset.ndim == 1:
                lengths.append(int(dataset.shape[0]))
    if not lengths:
        raise RuntimeError(f"No one-dimensional acceleration channels in {group.name}")
    return min(lengths)


def inventory(h5_path: Path, regime: str) -> pd.DataFrame:
    rows = []
    with h5py.File(h5_path, "r") as source:
        for name in sorted(source.keys()):
            if not NAME_PATTERN.fullmatch(name):
                continue
            item = parse_name(name)
            if item["regime"] != regime:
                continue
            group = source[name]
            samples = common_length(group)
            row = {**item, "samples": samples, "seconds": samples / FS, **channel_inventory(group)}
            for profile_name, profile in PROFILES.items():
                row[f"windows_{profile_name}"] = samples // int(profile["window_samples"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No records found for regime={regime}")
    expected = {"UDS", *[f"DS{i}" for i in range(1, 9)]}
    missing = expected - set(frame.state)
    if missing:
        raise RuntimeError(f"Missing states in {regime}: {sorted(missing)}")
    return frame


def allocate(
    record: pd.Series,
    profile_name: str,
    seed: int,
    strict: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    profile = PROFILES[profile_name]
    window = int(profile["window_samples"])
    available = int(record.samples) // window
    rng = np.random.default_rng(seed)
    order = rng.permutation(available)
    if record.state == "UDS":
        requested_train = int(profile["healthy_train_per_record"])
        requested_test = int(profile["healthy_test_per_record"])
        requested = requested_train + requested_test
        if strict and available < requested:
            raise RuntimeError(
                f"{record.record_id}: need {requested} non-overlapping windows for {profile_name}, "
                f"found {available}"
            )
        # Preserve both baseline campaigns in train and test when the public record
        # has fewer windows than the count reported in the original experiment.
        used = min(available, requested)
        train_count = min(requested_train, used // 2)
        test_count = min(requested_test, used - train_count)
        assignments = [
            *("train_healthy" for _ in range(train_count)),
            *("test_healthy" for _ in range(test_count)),
        ]
    else:
        requested = int(profile["damage_test_per_record"])
        if strict and available < requested:
            raise RuntimeError(
                f"{record.record_id}: need {requested} non-overlapping damage windows for "
                f"{profile_name}, found {available}"
            )
        assignments = ["test_damage"] * min(available, requested)

    rows = []
    for sample_number, (window_index, role) in enumerate(zip(order, assignments)):
        start = int(window_index) * window
        rows.append({
            "sample_id": f"{record.record_id}__w{int(window_index):03d}",
            "record_id": record.record_id,
            "state": record.state,
            "label": int(record.label),
            "regime": record.regime,
            "repeat": int(record.repeat),
            "window_index": int(window_index),
            "start": start,
            "stop": start + window,
            "role": role,
            "profile": profile_name,
        })
    audit = {
        "record_id": record.record_id,
        "state": record.state,
        "available_windows": available,
        "selected_windows": len(rows),
        "train_healthy": sum(row["role"] == "train_healthy" for row in rows),
        "test_healthy": sum(row["role"] == "test_healthy" for row in rows),
        "test_damage": sum(row["role"] == "test_damage" for row in rows),
        "paper_count_exact": len(rows) == (
            int(profile["healthy_train_per_record"] + profile["healthy_test_per_record"])
            if record.state == "UDS"
            else int(profile["damage_test_per_record"])
        ),
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="svendsen2022")
    parser.add_argument("--regime", default="P2_NM_Z")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--strict-paper-counts", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = inventory(args.h5, args.regime)
    records.to_csv(args.output_dir / "record_inventory.csv", index=False)
    print("=== RECORD INVENTORY ===")
    print(records.to_string(index=False))
    if args.audit_only:
        print(f"HELLBRIDGE_PAPER_SPLIT_AUDIT output={args.output_dir}")
        return

    manifest_rows, audit_rows = [], []
    for record_index, record in records.iterrows():
        rows, audit = allocate(
            record,
            args.profile,
            args.seed + 1009 * record_index,
            args.strict_paper_counts,
        )
        manifest_rows.extend(rows)
        audit_rows.append(audit)
    manifest = pd.DataFrame(manifest_rows).sort_values(["role", "state", "record_id", "window_index"])
    audit = pd.DataFrame(audit_rows)
    if manifest.sample_id.duplicated().any():
        raise RuntimeError("Leakage: duplicate sample_id in split")
    overlap = set(manifest.loc[manifest.role == "train_healthy", "sample_id"]) & set(
        manifest.loc[manifest.role != "train_healthy", "sample_id"]
    )
    if overlap:
        raise RuntimeError(f"Leakage between train and test: {sorted(overlap)[:3]}")
    if manifest.loc[manifest.role == "train_healthy", "label"].ne(0).any():
        raise RuntimeError("Leakage: damaged sample in healthy training split")

    manifest.to_csv(args.output_dir / "sample_manifest.csv", index=False)
    audit.to_csv(args.output_dir / "split_audit.csv", index=False)
    summary = (
        manifest.groupby(["role", "state"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
    )
    summary.to_csv(args.output_dir / "split_summary.csv", index=False)
    profile = PROFILES[args.profile]
    exact = bool(audit.paper_count_exact.all())
    protocol = {
        "dataset": "Hell Bridge Test Arena public HDF5",
        "profile": args.profile,
        "reference": profile["reference"],
        "regime": args.regime,
        "regime_mapping_note": (
            "P2_NM_Z is the predeclared mapping to the midspan/white-noise/vertical "
            "benchmark configuration; verify against dataset metadata before publication"
        ),
        "sample_rate_hz": FS,
        "window_samples": profile["window_samples"],
        "window_seconds": profile["window_samples"] / FS,
        "ar_order": profile["ar_order"],
        "selection": "seeded non-overlapping window allocation within each recording",
        "healthy_baselines_represented_in_both_train_and_test": True,
        "strict_paper_counts": args.strict_paper_counts,
        "paper_counts_exact": exact,
        "damage_labels_used_to_define_test_states_only": True,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("=== SPLIT AUDIT ===")
    print(audit.to_string(index=False))
    print("=== SPLIT SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"paper_counts_exact={exact}")
    print(f"HELLBRIDGE_PAPER_SPLIT_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

