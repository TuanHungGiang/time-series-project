"""Audit the legacy Z24 split and build a leakage-resistant supervised split.

This program does not train a model and does not inspect test signals.  It
audits only manifest metadata, so it is safe to run before VCOS experiments.

The recommended protocol holds out complete roving-sensor setups:

* train: setups 1--5
* validation: setups 6--7
* locked test: setups 8--9

Every retained structural state is therefore represented in every partition,
while no setup appears in more than one partition.  State 02 remains excluded
because the exported Z24 manifest labels it as a transition state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "record_id",
    "state",
    "scenario",
    "excitation",
    "setup",
    "split",
    "label",
    "relative_path",
}


def parse_int_set(value: str) -> set[int]:
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("setup list cannot be empty")
    return result


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"state": str})
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    frame = frame.copy()
    frame["state"] = frame["state"].str.zfill(2)
    frame["setup"] = pd.to_numeric(frame["setup"], errors="raise").astype(int)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["excitation"] = frame["excitation"].astype(str).str.lower()
    return frame


def split_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    kept = frame[frame[column] != "excluded"]
    table = (
        kept.groupby([column, "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        .rename(columns={0: "healthy", 1: "damage"})
    )
    table["total"] = table.sum(axis=1)
    table["damage_fraction"] = np.divide(
        table["damage"], table["total"], out=np.zeros(len(table)), where=table["total"] > 0
    )
    return table


def overlap_map(frame: pd.DataFrame, split_column: str, group_column: str) -> dict[str, list]:
    kept = frame[frame[split_column] != "excluded"]
    by_split = {
        split: set(part[group_column].tolist())
        for split, part in kept.groupby(split_column, observed=True)
    }
    result: dict[str, list] = {}
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            result[f"{left}__{right}"] = sorted(by_split[left].intersection(by_split[right]))
    return result


def class_coverage(counts: pd.DataFrame) -> dict[str, bool]:
    return {
        split: bool(row.get("healthy", 0) > 0 and row.get("damage", 0) > 0)
        for split, row in counts.iterrows()
    }


def state_coverage(frame: pd.DataFrame, split_column: str) -> pd.DataFrame:
    kept = frame[frame[split_column] != "excluded"]
    return (
        kept.assign(present=1)
        .pivot_table(index="state", columns=split_column, values="present", aggfunc="max", fill_value=0)
        .astype(int)
        .sort_index()
    )


def assign_grouped_split(
    frame: pd.DataFrame,
    train_setups: set[int],
    val_setups: set[int],
    test_setups: set[int],
) -> pd.DataFrame:
    groups = {"train": train_setups, "val": val_setups, "test": test_setups}
    union = set().union(*groups.values())
    pairwise_overlap = (
        train_setups.intersection(val_setups)
        | train_setups.intersection(test_setups)
        | val_setups.intersection(test_setups)
    )
    if pairwise_overlap:
        raise ValueError(f"setup partitions overlap: {sorted(pairwise_overlap)}")

    available = set(frame.loc[frame["split"] != "excluded", "setup"].unique())
    if union != available:
        raise ValueError(
            f"setup partitions must cover available setups exactly; available={sorted(available)}, "
            f"provided={sorted(union)}"
        )

    result = frame.copy()
    result = result.rename(columns={"split": "legacy_split"})
    result["split"] = "excluded"
    for name, setups in groups.items():
        mask = result["setup"].isin(setups) & (result["label"] >= 0)
        result.loc[mask, "split"] = name
    result["source_relative_path"] = result["relative_path"]
    return result


def file_audit(frame: pd.DataFrame, data_root: Path | None) -> dict:
    if data_root is None:
        return {"checked": False, "missing_count": None, "missing_examples": []}
    missing = []
    for relative in frame["relative_path"]:
        path = data_root / str(relative)
        if not path.exists():
            missing.append(str(relative))
    return {
        "checked": True,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
    }


def plot_counts(legacy: pd.DataFrame, proposed: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, table, title in (
        (axes[0], legacy, "Legacy healthy-only split"),
        (axes[1], proposed, "Proposed setup-grouped supervised split"),
    ):
        order = [name for name in ("train", "val", "test") if name in table.index]
        values = table.loc[order, ["healthy", "damage"]]
        values.plot(kind="bar", stacked=True, color=["#2ca25f", "#de2d26"], ax=ax)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Complete measurement records")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(frameon=False)
        for patch in ax.patches:
            if patch.get_height() > 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_y() + patch.get_height() / 2,
                    f"{int(patch.get_height())}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    fig.savefig(output / "legacy_vs_supervised_split_counts.png", dpi=180)
    plt.close(fig)


def plot_assignment(frame: pd.DataFrame, output: Path) -> None:
    retained = frame[frame["split"] != "excluded"].copy()
    states = sorted(retained["state"].unique())
    setups = sorted(retained["setup"].unique())
    mapping = {"train": 0, "val": 1, "test": 2}
    matrix = np.full((len(states), len(setups)), np.nan)
    for i, state in enumerate(states):
        for j, setup in enumerate(setups):
            values = retained.loc[
                (retained["state"] == state) & (retained["setup"] == setup), "split"
            ].unique()
            if len(values) == 1:
                matrix[i, j] = mapping[values[0]]

    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#3182bd", "#fd8d3c", "#756bb1"]), vmin=0, vmax=2)
    ax.set_xticks(range(len(setups)), [f"setup {x}" for x in setups], rotation=35, ha="right")
    ax.set_yticks(range(len(states)), [f"S{x}" for x in states])
    ax.set_xlabel("Roving-sensor setup (held out as a complete group)")
    ax.set_ylabel("Structural state")
    ax.set_title("Proposed Z24 supervised split", weight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in ("#3182bd", "#fd8d3c", "#756bb1")]
    ax.legend(handles, ["train", "validation", "locked test"], loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(output / "proposed_setup_group_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.reset_index().copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("z24_supervised_split_audit"))
    parser.add_argument("--train-setups", type=parse_int_set, default=parse_int_set("1,2,3,4,5"))
    parser.add_argument("--val-setups", type=parse_int_set, default=parse_int_set("6,7"))
    parser.add_argument("--test-setups", type=parse_int_set, default=parse_int_set("8,9"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    legacy = load_manifest(args.manifest)
    if legacy["record_id"].duplicated().any():
        duplicates = legacy.loc[legacy["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"duplicate record_id values: {duplicates[:10]}")

    proposed = assign_grouped_split(
        legacy, args.train_setups, args.val_setups, args.test_setups
    )
    legacy_counts = split_counts(legacy, "split")
    proposed_counts = split_counts(proposed, "split")
    legacy_coverage = class_coverage(legacy_counts)
    proposed_coverage = class_coverage(proposed_counts)
    legacy_setup_overlap = overlap_map(legacy, "split", "setup")
    proposed_setup_overlap = overlap_map(proposed, "split", "setup")

    retained_states = set(legacy.loc[legacy["split"] != "excluded", "state"])
    proposed_states = state_coverage(proposed, "split")
    all_states_each_split = all(
        set(proposed.loc[proposed["split"] == split, "state"]) == retained_states
        for split in ("train", "val", "test")
    )
    files = file_audit(legacy, args.data_root)

    legacy_vcos_valid = all(legacy_coverage.get(split, False) for split in ("train", "val", "test"))
    proposed_vcos_valid = (
        all(proposed_coverage.get(split, False) for split in ("train", "val", "test"))
        and not any(proposed_setup_overlap.values())
        and all_states_each_split
    )

    report = {
        "manifest": str(args.manifest),
        "records_total": int(len(legacy)),
        "records_retained": int((legacy["split"] != "excluded").sum()),
        "legacy": {
            "counts": legacy_counts.reset_index().to_dict(orient="records"),
            "both_classes_in_each_split": legacy_coverage,
            "setup_overlap": legacy_setup_overlap,
            "valid_for_healthy_only_one_class": bool(
                legacy_counts.loc["train", "damage"] == 0
                and legacy_counts.loc["train", "healthy"] > 0
            ),
            "valid_for_supervised_vcos": legacy_vcos_valid,
        },
        "proposed": {
            "train_setups": sorted(args.train_setups),
            "validation_setups": sorted(args.val_setups),
            "test_setups": sorted(args.test_setups),
            "counts": proposed_counts.reset_index().to_dict(orient="records"),
            "both_classes_in_each_split": proposed_coverage,
            "setup_overlap": proposed_setup_overlap,
            "all_retained_states_in_each_split": all_states_each_split,
            "valid_for_supervised_vcos": proposed_vcos_valid,
        },
        "file_audit": files,
        "decision": (
            "Legacy split is suitable only for healthy-only one-class anomaly detection; "
            "it is invalid for supervised VCOS. Use the setup-grouped supervised split."
        ),
    }

    proposed.to_csv(args.output / "records_manifest_supervised_grouped.csv", index=False)
    proposed_states.to_csv(args.output / "proposed_state_coverage.csv")
    with (args.output / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    plot_counts(legacy_counts, proposed_counts, args.output)
    plot_assignment(proposed, args.output)

    report_md = f"""# Z24 split audit for supervised VCOS

## Decision

**The legacy split is not valid for supervised VCOS.** Its training partition
contains healthy records only, so a supervised fuzzy-rough selector cannot
construct a healthy-versus-damage decision relation. It remains valid only for
the former one-class anomaly-detection question.

The proposed split holds out complete sensor setups and contains both classes
in train, validation and locked test. No setup is shared across partitions.

## Legacy split

{markdown_table(legacy_counts)}

- Both classes in train/validation/test: `{legacy_coverage}`
- Setup overlap: `{legacy_setup_overlap}`
- Valid for supervised VCOS: `{legacy_vcos_valid}`

## Proposed setup-grouped split

{markdown_table(proposed_counts)}

- Train setups: `{sorted(args.train_setups)}`
- Validation setups: `{sorted(args.val_setups)}`
- Locked-test setups: `{sorted(args.test_setups)}`
- Both classes in train/validation/test: `{proposed_coverage}`
- Setup overlap: `{proposed_setup_overlap}`
- All retained states represented in every split: `{all_states_each_split}`
- Valid for supervised VCOS: `{proposed_vcos_valid}`

## Interpretation

The proposed protocol measures generalization to unseen sensor configurations,
not generalization to unseen damage types. Class imbalance remains substantial,
so balanced accuracy, macro-F1 and MCC must be primary metrics. Accuracy alone
is not an acceptable selection metric.
"""
    (args.output / "SPLIT_AUDIT_REPORT.md").write_text(report_md, encoding="utf-8")

    print("\nZ24 SUPERVISED SPLIT AUDIT")
    print("=" * 72)
    print("LEGACY COUNTS")
    print(legacy_counts.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"legacy_valid_for_supervised_vcos={legacy_vcos_valid}")
    print(f"legacy_setup_overlap={legacy_setup_overlap}")
    print("\nPROPOSED COUNTS")
    print(proposed_counts.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"proposed_valid_for_supervised_vcos={proposed_vcos_valid}")
    print(f"proposed_setup_overlap={proposed_setup_overlap}")
    print("\nDECISION: legacy split is NOT suitable for supervised VCOS.")
    print("Use records_manifest_supervised_grouped.csv for the next VCOS stage.")
    print(f"outputs={args.output.resolve()}")


if __name__ == "__main__":
    main()

