from __future__ import annotations

"""MDR-SHM V12: healthy-only mask-aware Z24 screening.

This entry point deliberately reuses the verified data loader and model code
from ``run_z24_maskaware_v6.py`` while locking a stricter experiment:

* model fitting uses healthy training windows only;
* synthetic training masks are random complete-sensor masks only;
* targeted sensor loss is an unseen stress test, never a training mask;
* targeted sensors are ranked using healthy-training RMS only;
* score direction is fixed to two-sided healthy deviation;
* the threshold is fitted from clean healthy validation records only;
* neither damage validation nor test labels select a model or threshold.

Run this one-seed screen before spending compute on a multi-seed experiment.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_maskaware_v6 as core  # noqa: E402
from run_hellbridge_source_baselines import publication_metrics  # noqa: E402


METHODS = core.METHODS
RANDOM_RATES = (0.10, 0.30, 0.50)
RANDOM_REPLICATES = 5
TARGETED_RATES = (0.30, 0.50)


def scenarios_v12() -> list[tuple[str, str, float, int]]:
    rows: list[tuple[str, str, float, int]] = [("clean", "clean", 0.0, 0)]
    for rate in RANDOM_RATES:
        for replicate in range(RANDOM_REPLICATES):
            rows.append((f"sensor_{rate:.2f}", "sensor", rate, replicate))
    for rate in TARGETED_RATES:
        rows.append((f"targeted_{rate:.2f}", "targeted", rate, 0))
    return rows


def random_only_training_mask(
    batch: int,
    channels: int,
    importance: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """25% clean views; otherwise random 10/30/50% complete-sensor loss.

    ``importance`` is intentionally unused here.  It is accepted only to keep
    the verified V6 training interface.  This prevents targeted test patterns
    from leaking into representation learning.
    """

    del importance
    available = torch.ones(batch, channels, device=device)
    use_mask = torch.rand(batch, device=device) >= 0.25
    rates = torch.as_tensor(RANDOM_RATES, device=device)
    selected_rates = rates[torch.randint(len(rates), (batch,), device=device)]
    orders = torch.rand(batch, channels, device=device).argsort(dim=1)
    for index in range(batch):
        if not bool(use_mask[index]):
            continue
        count = min(
            channels - 1,
            max(1, int(round(float(selected_rates[index]) * channels))),
        )
        available[index, orders[index, :count]] = 0.0
    return available


def healthy_quantile_threshold(scores: np.ndarray, specificity: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise RuntimeError(
            f"Need at least two clean healthy validation records, got {len(values)}"
        )
    try:
        return float(np.quantile(values, specificity, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, specificity, interpolation="higher"))


def finish_healthy_only(output: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate with a fixed two-sided score and healthy-only calibration."""

    metric_rows: list[dict] = []
    scored_frames: list[pd.DataFrame] = []
    specificity_target = float(core.V12_SPECIFICITY_TARGET)

    for method in METHODS:
        validation = frame[
            (frame.method == method)
            & (frame.split == "val")
            & (frame.label == 0)
            & (frame.condition == "clean")
        ].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()

        threshold = healthy_quantile_threshold(
            validation.score_two_sided.to_numpy(), specificity_target,
        )
        test["score"] = test.score_two_sided
        test["threshold"] = threshold
        test["score_tail"] = "two_sided_fixed"
        test["calibration_records"] = len(validation)
        test["specificity_target"] = specificity_target
        scored_frames.append(test)

        rows: list[dict] = []
        clean_row: dict | None = None
        for (condition, replicate), current in test.groupby(
            ["condition", "replicate"], sort=False,
        ):
            row = {
                "method": method,
                "condition": condition,
                "replicate": int(replicate),
                "score_tail": "two_sided_fixed",
                "threshold": threshold,
                "calibration_records": len(validation),
                "specificity_target": specificity_target,
                **publication_metrics(current.label, current.score, threshold),
                "coverage_mean": float(current.coverage.mean()),
                "parameters": int(current.parameters.iloc[0]),
            }
            rows.append(row)
            if condition == "clean":
                clean_row = row
        if clean_row is None:
            raise RuntimeError(f"Missing clean test cohort for {method}")
        for row in rows:
            row["delta_auc_from_clean"] = row["auc"] - clean_row["auc"]
            row["delta_ba_from_clean"] = row["ba"] - clean_row["ba"]
            row["delta_recall_from_clean"] = row["recall"] - clean_row["recall"]
            row["delta_fpr_from_clean"] = row["fpr"] - clean_row["fpr"]
        metric_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(scored_frames, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"),
        ba_std=("ba", "std"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"),
        f1_mean=("f1", "mean"),
        coverage_mean=("coverage_mean", "mean"),
        delta_auc_from_clean=("delta_auc_from_clean", "mean"),
        delta_ba_from_clean=("delta_ba_from_clean", "mean"),
        delta_recall_from_clean=("delta_recall_from_clean", "mean"),
        delta_fpr_from_clean=("delta_fpr_from_clean", "mean"),
        parameters=("parameters", "mean"),
        runs=("replicate", "size"),
    )

    baseline = summary[summary.method == "compact_raw"].set_index("condition")
    proposed = summary[
        summary.method == "maskaware_consistency"
    ].set_index("condition")
    effects = proposed[
        ["auc_mean", "ba_mean", "recall_mean", "fpr_mean"]
    ].subtract(
        baseline[["auc_mean", "ba_mean", "recall_mean", "fpr_mean"]]
    )
    effects.columns = [
        "delta_auc_proposed_minus_baseline",
        "delta_ba_proposed_minus_baseline",
        "delta_recall_proposed_minus_baseline",
        "delta_fpr_proposed_minus_baseline",
    ]
    effects = effects.reset_index()

    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)

    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(12, 5.5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("MDR-SHM V12: healthy-only random and targeted sensor loss")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v12_auc.png", dpi=220)
    plt.close(axis.figure)

    print("\n=== V12 SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== PROPOSED - BASELINE ===", flush=True)
    print(effects.to_string(index=False), flush=True)
    return summary


def screening_decision(summary: pd.DataFrame) -> dict:
    table = summary.set_index(["method", "condition"])
    conditions = [
        "sensor_0.10", "sensor_0.30", "sensor_0.50",
        "targeted_0.30", "targeted_0.50",
    ]
    effects: dict[str, dict[str, float]] = {}
    for condition in ["clean", *conditions]:
        proposed = table.loc[("maskaware_consistency", condition)]
        baseline = table.loc[("compact_raw", condition)]
        effects[condition] = {
            "delta_auc": float(proposed.auc_mean - baseline.auc_mean),
            "delta_ba": float(proposed.ba_mean - baseline.ba_mean),
            "delta_recall": float(proposed.recall_mean - baseline.recall_mean),
            "delta_fpr": float(proposed.fpr_mean - baseline.fpr_mean),
        }

    sensor50 = effects["sensor_0.50"]
    targeted50 = effects["targeted_0.50"]
    clean = effects["clean"]
    promote = bool(
        clean["delta_auc"] >= -0.01
        and sensor50["delta_auc"] >= 0.03
        and sensor50["delta_ba"] >= 0.03
        and sensor50["delta_recall"] >= -0.05
        and targeted50["delta_auc"] >= 0.0
    )
    return {
        "effects_proposed_minus_baseline": effects,
        "promotion_gate": {
            "clean_delta_auc_min": -0.01,
            "sensor_0.50_delta_auc_min": 0.03,
            "sensor_0.50_delta_ba_min": 0.03,
            "sensor_0.50_delta_recall_min": -0.05,
            "targeted_0.50_delta_auc_min": 0.0,
        },
        "promote_to_three_seeds": promote,
        "decision": (
            "PROMOTE V12 to a three-seed pilot."
            if promote
            else "STOP after one seed; at least one pre-registered gate failed."
        ),
    }


def update_protocol(output: Path) -> None:
    path = output / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol.update({
        "experiment": "MDR-SHM V12 healthy-only mask-aware representation",
        "fit": "healthy train windows only",
        "calibration": "clean healthy validation records only",
        "score": "fixed two-sided deviation; direction not selected with damage labels",
        "test_used_for_fit_calibration_or_selection": False,
        "training_masks": "25% clean; otherwise random complete-sensor loss at 10/30/50%",
        "targeted_training_masks": False,
        "targeted_test_rule": "largest healthy-training mean-square-amplitude sensors",
        "random_test": "10/30/50%, five deterministic mask replicates",
        "targeted_test": "30/50%, one deterministic healthy-ranked stress mask",
        "specificity_target": float(core.V12_SPECIFICITY_TARGET),
    })
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    # Read the extra V12-only option, then let the verified V6 parser handle
    # every existing data/model argument unchanged.
    specificity = 0.90
    if "--specificity-target" in sys.argv:
        index = sys.argv.index("--specificity-target")
        specificity = float(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    core.V12_SPECIFICITY_TARGET = specificity

    output: Path | None = None
    if "--output-dir" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output-dir") + 1])

    core.scenarios = scenarios_v12
    core.paired_mask = random_only_training_mask
    core.finish = finish_healthy_only
    core.decision = screening_decision
    core.main()

    if output is None:
        raise RuntimeError("--output-dir is required")
    update_protocol(output)
    print(f"MDR_V12_PASS output={output}", flush=True)


if __name__ == "__main__":
    main()

