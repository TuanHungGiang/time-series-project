from __future__ import annotations

"""MDR-SHM V13: paper-guided improvement of the strongest Z24 USAD baseline.

V12 accidentally evaluated a lower-tail USAD detector with a fixed two-sided
score.  Its clean AUROC of about 0.086 was therefore the complement of the
established lower-tail result (about 0.914), and its threshold predicted no
damage records.  V13 corrects that protocol error and changes the proposed
branch only where missing-sensor papers motivate a change:

1. mask-aware cross-sensor recovery;
2. clean/masked healthy latent consistency;
3. reliability scoring over observed sensors only.

The baseline remains the same Compact USAD.  Both methods use a pre-registered
lower-tail anomaly score and thresholds fitted from clean healthy validation
records only.  Targeted masks are never used for training.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_mdr_v12 as v12  # noqa: E402
from run_hellbridge_source_baselines import publication_metrics  # noqa: E402


METHODS = v12.METHODS


@torch.no_grad()
def observed_only_maskaware_score(model, values, available, batch_size, device):
    """USAD error pooled only over sensors that are actually observed.

    Recovery still supplies a complete input representation to the encoder,
    but an imputed channel is not treated as physical damage evidence.  The
    denominator makes scores comparable across different mask coverage.
    """

    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(available)),
        batch_size=batch_size,
    )
    scores = []
    model.eval()
    for batch, mask in loader:
        batch = batch.to(device)
        mask = mask.to(device)
        completed, _, _, first, _, cycle = model.parts(batch, mask)
        sensor_evidence, _ = v12.core.anomaly_components(completed, first, cycle)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        score = (sensor_evidence * mask).sum(dim=1) / denominator
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def finish_lower_tail(output: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Healthy-only calibration for the pre-registered lower-tail detector."""

    metric_rows: list[dict] = []
    scored_frames: list[pd.DataFrame] = []
    specificity_target = float(v12.core.V13_SPECIFICITY_TARGET)

    for method in METHODS:
        validation = frame[
            (frame.method == method)
            & (frame.split == "val")
            & (frame.label == 0)
            & (frame.condition == "clean")
        ].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()

        # Reconstruction error is unusually LOW for Z24 damage under this
        # preprocessing.  Negation turns it into a conventional higher-is-more-
        # anomalous score without consulting damage validation or test labels.
        threshold = v12.healthy_quantile_threshold(
            validation.score_lower.to_numpy(), specificity_target,
        )
        test["score"] = test.score_lower
        test["threshold"] = threshold
        test["score_tail"] = "lower_fixed"
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
                "score_tail": "lower_fixed",
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
    axis.set_title("MDR-SHM V13: lower-tail USAD with observed-only reliability")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v13_auc.png", dpi=220)
    plt.close(axis.figure)

    print("\n=== V13 SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== PROPOSED - BEST USAD BASELINE ===", flush=True)
    print(effects.to_string(index=False), flush=True)
    return summary


def update_protocol(output: Path) -> None:
    path = output / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol.update({
        "experiment": "MDR-SHM V13 paper-guided best-USAD improvement",
        "baseline": "Compact USAD with the established fixed lower-tail score",
        "proposed_blocks": [
            "mask-aware cross-sensor recovery",
            "healthy clean/masked latent consistency",
            "coverage-normalized observed-sensor-only scoring",
        ],
        "fit": "healthy train windows only",
        "calibration": "clean healthy validation records only",
        "score": "negative reconstruction error (fixed lower-tail orientation)",
        "damage_labels_used_for_score_direction_or_threshold": False,
        "test_used_for_fit_calibration_or_selection": False,
        "training_masks": "25% clean; otherwise random complete-sensor loss at 10/30/50%",
        "targeted_training_masks": False,
        "targeted_test_rule": "largest healthy-training mean-square-amplitude sensors",
        "specificity_target": float(v12.core.V13_SPECIFICITY_TARGET),
    })
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    specificity = 0.90
    if "--specificity-target" in sys.argv:
        index = sys.argv.index("--specificity-target")
        specificity = float(sys.argv[index + 1])
        del sys.argv[index:index + 2]
    v12.core.V13_SPECIFICITY_TARGET = specificity

    output: Path | None = None
    if "--output-dir" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output-dir") + 1])

    v12.core.scenarios = v12.scenarios_v12
    v12.core.paired_mask = v12.random_only_training_mask
    v12.core.score_maskaware = observed_only_maskaware_score
    v12.core.finish = finish_lower_tail
    v12.core.decision = v12.screening_decision
    v12.core.main()

    if output is None:
        raise RuntimeError("--output-dir is required")
    update_protocol(output)
    print(f"MDR_V13_PASS output={output}", flush=True)


if __name__ == "__main__":
    main()

