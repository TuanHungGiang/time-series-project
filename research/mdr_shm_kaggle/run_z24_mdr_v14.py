from __future__ import annotations

"""MDR-SHM V14: anchored recovery and criticality-guided fusion.

This is an incremental improvement of the strongest Compact-USAD baseline,
not a replacement detector.  The clean USAD backbone is fitted first and then
frozen.  A small recovery adapter learns from randomly masked healthy views.
At inference, an availability/criticality policy blends the raw baseline with
the recovered observed-sensor branch.  Clean inputs therefore reduce exactly
to the original baseline, while loss of high-criticality sensors conservatively
shifts the policy back toward the baseline.

Model fitting is healthy-only.  Operational thresholds use the labeled clean
validation split (never test data), so the resulting protocol must be reported
as unsupervised representation learning with supervised threshold calibration.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_mdr_v12 as v12  # noqa: E402
import run_z24_mdr_v13 as v13  # noqa: E402
from run_hellbridge_source_baselines import publication_metrics  # noqa: E402
from run_z24_source_baselines import threshold_from_labeled_validation  # noqa: E402


BASE_METHODS = ("compact_raw", "maskaware_consistency")
ALL_METHODS = (*BASE_METHODS, "criticality_fusion")
_ORIGINAL_FIT_COMPACT = v12.core.fit_compact_usad
_LAST_COMPACT = None


class AnchoredRecoveryStudent(nn.Module):
    """Recovery adapter attached to an unchanged pretrained USAD backbone."""

    def __init__(self, backbone: nn.Module, channels: int, recovery_hidden: int):
        super().__init__()
        self.recovery = v12.core.RecoveryAdapter(channels, recovery_hidden)
        self.usad = backbone
        for parameter in self.usad.parameters():
            parameter.requires_grad_(False)
        self.usad.eval()

    def parts(self, values, available):
        completed, predicted = self.recovery(values, available)
        latent, first, second, cycle = self.usad.parts(completed)
        return completed, predicted, latent, first, second, cycle

    def train(self, mode: bool = True):
        super().train(mode)
        # The adapter trains, but BatchNorm/dropout state of the established
        # baseline (if later added) must remain frozen.
        self.usad.eval()
        return self


def fit_compact_and_capture(*args, **kwargs):
    global _LAST_COMPACT
    _LAST_COMPACT = _ORIGINAL_FIT_COMPACT(*args, **kwargs)
    return _LAST_COMPACT


def anchored_recovery_loss(clean, predicted, available):
    missing = 1.0 - available[:, :, None]
    denominator = (missing.sum() * clean.shape[-1]).clamp_min(1.0)
    return ((clean - predicted).pow(2) * missing).sum() / denominator


def fit_anchored_adapter(
    fit,
    importance,
    epochs,
    batch_size,
    recovery_hidden,
    hidden1,
    hidden2,
    latent,
    recovery_weight,
    consistency_weight,
    device,
):
    del hidden1, hidden2, latent
    global _LAST_COMPACT
    if _LAST_COMPACT is None:
        raise RuntimeError("Compact USAD must be fitted before the V14 adapter")

    channels = fit.shape[1]
    model = AnchoredRecoveryStudent(
        _LAST_COMPACT, channels, recovery_hidden,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.recovery.parameters(), lr=5e-4, weight_decay=1e-5,
    )

    best_loss = float("inf")
    best_state = None
    stale = 0
    patience = 3
    for epoch in range(1, epochs + 1):
        logs = []
        model.train()
        for (clean,) in loader:
            clean = clean.to(device)
            available = v12.random_only_training_mask(
                len(clean), channels, importance, device,
            )
            observed = clean * available[:, :, None]
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                clean_latent, clean_first, _, _ = model.usad.parts(clean)
            completed, predicted, latent_masked, first_masked, _, _ = model.parts(
                observed, available,
            )
            recovery = anchored_recovery_loss(clean, predicted, available)
            latent_alignment = (
                1.0
                - (
                    F.normalize(latent_masked, dim=1)
                    * F.normalize(clean_latent, dim=1)
                ).sum(1)
            ).mean()
            output_alignment = (first_masked - clean_first).pow(2).mean()
            loss = (
                recovery_weight * recovery
                + consistency_weight * latent_alignment
                + 0.10 * output_alignment
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.recovery.parameters(), 2.0)
            optimizer.step()
            logs.append((
                float(loss.detach().cpu()),
                float(recovery.detach().cpu()),
                float(latent_alignment.detach().cpu()),
                float(output_alignment.detach().cpu()),
            ))

        epoch_values = np.asarray(logs, dtype=float).mean(axis=0)
        epoch_loss = float(epoch_values[0])
        if epoch_loss < best_loss - 1e-7:
            best_loss = epoch_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.recovery.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"  anchored epoch={epoch}/{epochs} loss={epoch_values[0]:.6f} "
            f"recovery={epoch_values[1]:.6f} "
            f"latent={epoch_values[2]:.6f} output={epoch_values[3]:.6f}",
            flush=True,
        )
        if stale >= patience:
            print(f"  anchored early-stop epoch={epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Anchored adapter produced no checkpoint")
    model.recovery.load_state_dict(best_state)
    print(f"  anchored selected_training_loss={best_loss:.6f}", flush=True)
    return model.eval()


def build_values_with_criticality(
    records, cache, condition, kind, rate, replicate, importance,
):
    arrays, masks, counts, metadata = [], [], [], []
    weight = np.asarray(importance, dtype=float)
    weight = weight / max(float(weight.sum()), 1e-12)
    for row in records.itertuples():
        values, available = v12.core.sensor_corrupt(
            cache[str(row.record_id)], kind, rate, str(row.record_id),
            replicate, importance,
        )
        arrays.append(values)
        masks.append(available)
        counts.append(len(values))
        missing_fraction = float(np.mean(1.0 - available))
        missing_criticality = float(
            np.mean(np.sum((1.0 - available) * weight[None, :], axis=1))
        )
        relative_criticality = (
            missing_criticality / max(missing_fraction, 1e-12)
            if missing_fraction > 0.0 else 0.0
        )
        metadata.append({
            "record_id": row.record_id,
            "split": row.split,
            "label": int(row.label),
            "state": row.state,
            "setup": row.setup,
            "excitation": row.excitation,
            "condition": condition,
            "replicate": replicate,
            "rate": rate,
            "missing_fraction": missing_fraction,
            "missing_criticality": missing_criticality,
            "relative_criticality": relative_criticality,
        })
    return np.concatenate(arrays), np.concatenate(masks), counts, metadata


def add_criticality_fusion(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["record_id", "split", "condition", "replicate"]
    baseline = frame[frame.method == "compact_raw"].copy()
    proposed = frame[frame.method == "maskaware_consistency"].copy()
    proposed_scores = proposed[
        keys + ["score_upper", "score_lower", "score_two_sided"]
    ].rename(columns={
        "score_upper": "proposed_upper",
        "score_lower": "proposed_lower",
        "score_two_sided": "proposed_two_sided",
    })
    fused = baseline.merge(
        proposed_scores, on=keys, how="inner", validate="one_to_one",
    )
    if len(fused) != len(baseline):
        raise RuntimeError("Fusion alignment dropped records")

    # Missingness severity activates recovery.  Above-average missing sensor
    # criticality suppresses it, preserving the stronger targeted-loss branch.
    severity = np.clip((fused.missing_fraction - 0.10) / 0.40, 0.0, 1.0)
    criticality_penalty = np.exp(
        -3.0 * np.maximum(fused.relative_criticality - 1.0, 0.0)
    )
    weight = severity * criticality_penalty
    fused["fusion_weight"] = weight
    fused["score_upper"] = (
        (1.0 - weight) * fused.score_upper
        + weight * fused.proposed_upper
    )
    fused["score_lower"] = (
        (1.0 - weight) * fused.score_lower
        + weight * fused.proposed_lower
    )
    fused["score_two_sided"] = (
        (1.0 - weight) * fused.score_two_sided
        + weight * fused.proposed_two_sided
    )
    fused["method"] = "criticality_fusion"
    fused["parameters"] = proposed.parameters.to_numpy()
    fused = fused.drop(columns=[
        "proposed_upper", "proposed_lower", "proposed_two_sided",
    ])
    return pd.concat([frame, fused], ignore_index=True)


def finish_v14(output: Path, frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_criticality_fusion(frame)
    metric_rows, scored_frames = [], []

    for method in ALL_METHODS:
        validation = frame[
            (frame.method == method)
            & (frame.split == "val")
            & (frame.condition == "clean")
        ].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()
        threshold, validation_metrics = threshold_from_labeled_validation(
            validation.label.to_numpy(), validation.score_lower.to_numpy(),
        )
        test["score"] = test.score_lower
        test["threshold"] = threshold
        test["score_tail"] = "lower_fixed"
        test["calibration"] = "labeled_clean_validation"
        scored_frames.append(test)

        rows, clean_row = [], None
        for (condition, replicate), current in test.groupby(
            ["condition", "replicate"], sort=False,
        ):
            row = {
                "method": method,
                "condition": condition,
                "replicate": int(replicate),
                "score_tail": "lower_fixed",
                "threshold": threshold,
                "validation_ba": validation_metrics["ba"],
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
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ba_mean=("ba", "mean"),
        ba_std=("ba", "std"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        coverage_mean=("coverage_mean", "mean"),
        delta_auc_from_clean=("delta_auc_from_clean", "mean"),
        delta_ba_from_clean=("delta_ba_from_clean", "mean"),
        delta_recall_from_clean=("delta_recall_from_clean", "mean"),
        delta_fpr_from_clean=("delta_fpr_from_clean", "mean"),
        parameters=("parameters", "mean"), runs=("replicate", "size"),
    )

    baseline = summary[summary.method == "compact_raw"].set_index("condition")
    effect_rows = []
    for method in ("maskaware_consistency", "criticality_fusion"):
        proposed = summary[summary.method == method].set_index("condition")
        effects = proposed[
            ["auc_mean", "ba_mean", "recall_mean", "fpr_mean"]
        ].subtract(
            baseline[["auc_mean", "ba_mean", "recall_mean", "fpr_mean"]]
        )
        effects.columns = ["delta_auc", "delta_ba", "delta_recall", "delta_fpr"]
        effects["method"] = method
        effect_rows.append(effects.reset_index())
    effects = pd.concat(effect_rows, ignore_index=True)

    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)

    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(13, 5.5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("MDR-SHM V14: anchored recovery and criticality fusion")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v14_auc.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== V14 SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== EFFECTS VS BEST USAD BASELINE ===", flush=True)
    print(effects.to_string(index=False), flush=True)
    return summary


def decision_v14(summary: pd.DataFrame) -> dict:
    table = summary.set_index(["method", "condition"])
    effects = {}
    for condition in (
        "clean", "sensor_0.10", "sensor_0.30", "sensor_0.50",
        "targeted_0.30", "targeted_0.50",
    ):
        proposed = table.loc[("criticality_fusion", condition)]
        baseline = table.loc[("compact_raw", condition)]
        effects[condition] = {
            "delta_auc": float(proposed.auc_mean - baseline.auc_mean),
            "delta_ba": float(proposed.ba_mean - baseline.ba_mean),
            "delta_recall": float(proposed.recall_mean - baseline.recall_mean),
            "delta_fpr": float(proposed.fpr_mean - baseline.fpr_mean),
        }
    random_conditions = ("sensor_0.10", "sensor_0.30", "sensor_0.50")
    mean_random_auc = float(np.mean([
        effects[c]["delta_auc"] for c in random_conditions
    ]))
    promote = bool(
        effects["clean"]["delta_auc"] >= -0.005
        and mean_random_auc > 0.01
        and effects["sensor_0.50"]["delta_auc"] >= 0.02
        and effects["sensor_0.50"]["delta_ba"] >= 0.0
        and effects["sensor_0.50"]["delta_recall"] >= -0.02
        and effects["targeted_0.50"]["delta_auc"] >= -0.005
    )
    return {
        "effects_criticality_fusion_minus_best_usad": effects,
        "mean_random_loss_delta_auc": mean_random_auc,
        "promote_to_three_seeds": promote,
        "decision": (
            "PROMOTE V14 criticality fusion to three seeds."
            if promote else
            "STOP after one seed; V14 did not jointly pass ranking and operating-point gates."
        ),
    }


def update_protocol(output: Path) -> None:
    path = output / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol.update({
        "experiment": "MDR-SHM V14 anchored recovery and criticality fusion",
        "methods": list(ALL_METHODS),
        "model_fit": "healthy train only",
        "backbone": "established Compact USAD, frozen during adapter training",
        "adapter_training": "random complete-sensor masks at 10/30/50%",
        "targeted_masks_used_for_training": False,
        "fusion_inputs": "mask coverage and healthy-train sensor criticality only",
        "operational_threshold": "fixed lower-tail threshold optimized on labeled clean validation",
        "threshold_calibration_uses_damage_validation_labels": True,
        "test_used_for_fit_calibration_or_fusion_selection": False,
        "claim_scope": "healthy-only representation; supervised validation calibration",
    })
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    # V14 uses validation labels for an operational threshold, so the V13-only
    # specificity option is accepted and ignored to keep notebook calls robust.
    if "--specificity-target" in sys.argv:
        index = sys.argv.index("--specificity-target")
        del sys.argv[index:index + 2]

    output = None
    if "--output-dir" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output-dir") + 1])

    v12.core.scenarios = v12.scenarios_v12
    v12.core.paired_mask = v12.random_only_training_mask
    v12.core.fit_compact_usad = fit_compact_and_capture
    v12.core.fit_maskaware = fit_anchored_adapter
    v12.core.build_values = build_values_with_criticality
    v12.core.score_maskaware = v13.observed_only_maskaware_score
    v12.core.finish = finish_v14
    v12.core.decision = decision_v14
    v12.core.main()

    if output is None:
        raise RuntimeError("--output-dir is required")
    update_protocol(output)
    print(f"MDR_V14_PASS output={output}", flush=True)


if __name__ == "__main__":
    main()

