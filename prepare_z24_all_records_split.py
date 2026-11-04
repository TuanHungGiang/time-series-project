"""Create a leakage-aware record-level split for all valid Z24 MAT records.

The script inventories all 306 files (17 states x 2 excitation types x 9
setups), marks the 18 state-02 installation records as excluded, and assigns
all remaining 288 records to train/validation/test without copying raw data.

Training contains healthy records only. Damage states are kept whole and are
assigned exclusively to validation or test. Healthy files are also assigned as
whole records, so no source MAT file appears in more than one split.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / ".packages"), str(ROOT / "baseline_packages"), str(ROOT)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat, whosmat

from run_z24_external import SCENARIOS


HEALTHY_STATES = {"01", "08"}
EXCLUDED_STATES = {"02"}
VAL_DAMAGE_STATES = {"04", "05", "06", "10", "14", "16", "17"}
TEST_DAMAGE_STATES = {"03", "07", "09", "11", "12", "13", "15"}

# Whole-record healthy split. The rotation between states prevents a split from
# being synonymous with one structural state while retaining both AVT and FVT.
# Train receives 24 records, validation 6, and test 6.
HEALTHY_ASSIGNMENT = {
    ("01", "fvt"): {"train": range(1, 7), "val": (7, 8), "test": (9,)},
    ("01", "avt"): {"train": range(1, 7), "val": (7,), "test": (8, 9)},
    ("08", "fvt"): {"train": range(4, 10), "val": (1, 2), "test": (3,)},
    ("08", "avt"): {"train": range(4, 10), "val": (1,), "test": (2, 3)},
}


def discover_records(root: Path) -> list[tuple[str, str, int, Path]]:
    records = []
    pattern = re.compile(r"^(\d{2})setup(\d{2})\.mat$", re.IGNORECASE)
    for path in root.rglob("*.mat"):
        match = pattern.match(path.name)
        excitation = path.parent.name.lower()
        state = path.parent.parent.name
        if match is None or excitation not in {"avt", "fvt"} or not state.isdigit():
            continue
        if match.group(1) != state:
            raise ValueError(f"Filename/state mismatch: {path}")
        records.append((state, excitation, int(match.group(2)), path.resolve()))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    return records


def mat_shape(path: Path) -> tuple[int, int]:
    variables = {name: shape for name, shape, _ in whosmat(path)}
    if "data" not in variables or len(variables["data"]) != 2:
        raise ValueError(f"Missing 2-D data matrix: {path}")
    return int(variables["data"][0]), int(variables["data"][1])


def sensor_labels(path: Path) -> tuple[list[str], list[str]]:
    mat = loadmat(path, variable_names=["labelshulp"])
    labels = [str(value).strip() for value in mat["labelshulp"].tolist()]
    vertical = [
        label for label in labels
        if label.endswith("V") and label[0].isdigit() and not label.startswith(("R", "DP"))
    ]
    return labels, vertical


def assign_split(state: str, excitation: str, setup: int) -> tuple[str, int, str]:
    if state in EXCLUDED_STATES:
        return "excluded", -1, "state 02 is an installation/intervention transition"
    if state in HEALTHY_STATES:
        assignment = HEALTHY_ASSIGNMENT[(state, excitation)]
        matches = [split for split, setups in assignment.items() if setup in setups]
        if len(matches) != 1:
            raise ValueError(f"Invalid healthy assignment: {state}/{excitation}/setup{setup:02d}")
        return matches[0], 0, "healthy"
    if state in VAL_DAMAGE_STATES:
        return "val", 1, "damage state held wholly in validation"
    if state in TEST_DAMAGE_STATES:
        return "test", 1, "damage state held wholly in test"
    raise ValueError(f"State {state} has no assignment")


def validate_manifest(frame: pd.DataFrame) -> dict:
    if len(frame) != 306:
        raise ValueError(f"Expected 306 records, found {len(frame)}")
    if frame.path.duplicated().any():
        raise ValueError("Duplicate source paths in manifest")
    per_cell = frame.groupby(["state", "excitation"]).size()
    if not (per_cell == 9).all():
        raise ValueError(f"Expected nine setups per state/excitation:\n{per_cell[per_cell != 9]}")

    used = frame[frame.split != "excluded"]
    train = used[used.split == "train"]
    if len(used) != 288 or len(frame[frame.split == "excluded"]) != 18:
        raise ValueError("Expected 288 used and 18 excluded records")
    if set(train.label) != {0} or set(train.state) != HEALTHY_STATES:
        raise ValueError("Training must contain both healthy states and no damage")
    expected = {
        ("train", 0): 24,
        ("val", 0): 6, ("val", 1): 126,
        ("test", 0): 6, ("test", 1): 126,
    }
    actual = Counter(zip(used.split, used.label))
    if actual != Counter(expected):
        raise ValueError(f"Unexpected split/label counts: {actual}")

    damage_membership = used[used.label == 1].groupby("state").split.nunique()
    if not (damage_membership == 1).all():
        raise ValueError("A damage state crosses validation/test")
    if set(used[(used.label == 1) & (used.split == "val")].state) != VAL_DAMAGE_STATES:
        raise ValueError("Validation damage states differ from locked protocol")
    if set(used[(used.label == 1) & (used.split == "test")].state) != TEST_DAMAGE_STATES:
        raise ValueError("Test damage states differ from locked protocol")

    train_sensors = set().union(*(set(value.split("|")) for value in train.vertical_labels))
    train_sensors.discard("")
    coverage = {}
    for split in ("train", "val", "test"):
        subset = used[used.split == split]
        sensors = set().union(*(set(value.split("|")) for value in subset.vertical_labels))
        sensors.discard("")
        coverage[split] = {
            "unique_vertical_sensor_ids": len(sensors),
            "not_seen_in_train": sorted(sensors - train_sensors),
        }
    return coverage


def plot_counts(frame: pd.DataFrame, output: Path) -> None:
    used = frame[frame.split != "excluded"]
    counts = used.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(
        ["train", "val", "test"]
    )
    counts = counts.rename(columns={0: "Healthy", 1: "Damage"})
    ax = counts.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#2ca02c", "#d62728"])
    ax.set(title="Z24 all-record split (state 02 excluded)", xlabel="Split", ylabel="Whole MAT records")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        labels = [str(int(value)) if value > 0 else "" for value in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center",
                     color="white", fontweight="bold")
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "z24")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs_baselines" / "z24_all_records_split")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.root)
    rows = []
    for index, (state, excitation, setup, path) in enumerate(records, 1):
        samples, channels = mat_shape(path)
        labels, vertical = sensor_labels(path)
        split, label, reason = assign_split(state, excitation, setup)
        rows.append({
            "record_id": f"z24_s{state}_{excitation}_setup{setup:02d}",
            "state": state,
            "scenario": SCENARIOS[state],
            "excitation": excitation,
            "setup": setup,
            "split": split,
            "label": label,
            "samples": samples,
            "all_channels": channels,
            "vertical_channels": len(vertical),
            "vertical_labels": "|".join(vertical),
            "assignment_reason": reason,
            "path": str(path),
        })
        if index % 36 == 0 or index == len(records):
            print(f"inventoried={index}/{len(records)}", flush=True)

    frame = pd.DataFrame(rows)
    coverage = validate_manifest(frame)
    frame.to_csv(args.output / "all_records_manifest.csv", index=False)
    used = frame[frame.split != "excluded"].copy()
    used.to_csv(args.output / "used_288_records_manifest.csv", index=False)
    frame[frame.split == "excluded"].to_csv(args.output / "excluded_state02_records.csv", index=False)

    counts = frame.groupby(["split", "label"]).size().rename("records").reset_index()
    counts.to_csv(args.output / "split_counts.csv", index=False)
    summary = {
        "total_discovered_records": len(frame),
        "used_records": len(used),
        "excluded_records": int((frame.split == "excluded").sum()),
        "train_records": int((frame.split == "train").sum()),
        "validation_records": int((frame.split == "val").sum()),
        "test_records": int((frame.split == "test").sum()),
        "train_healthy": int(((frame.split == "train") & (frame.label == 0)).sum()),
        "train_damage": int(((frame.split == "train") & (frame.label == 1)).sum()),
        "validation_healthy": int(((frame.split == "val") & (frame.label == 0)).sum()),
        "validation_damage": int(((frame.split == "val") & (frame.label == 1)).sum()),
        "test_healthy": int(((frame.split == "test") & (frame.label == 0)).sum()),
        "test_damage": int(((frame.split == "test") & (frame.label == 1)).sum()),
        "healthy_states": sorted(HEALTHY_STATES),
        "excluded_states": sorted(EXCLUDED_STATES),
        "validation_damage_states": sorted(VAL_DAMAGE_STATES),
        "test_damage_states": sorted(TEST_DAMAGE_STATES),
        "sensor_coverage": coverage,
        "leakage_checks": {
            "source_record_overlap": False,
            "damage_state_overlap_between_val_test": False,
            "damage_in_training": False,
            "split_unit": "whole MAT record",
        },
        "important_note": (
            "Variable sensor layouts and 15/17 vertical-channel counts require a sensor-aware "
            "or masked loader; raw channel indices must not be treated as identical across setups."
        ),
    }
    (args.output / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_counts(frame, args.output / "split_counts.png")

    print("\nSPLIT SUMMARY", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("\nCOUNTS", flush=True)
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

