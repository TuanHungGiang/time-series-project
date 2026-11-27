from __future__ import annotations

"""Teacher-free compact/recovery dual-path screening on Z24.

The proposed ``dual_gate`` combines a high-sensitivity compact USAD branch and
a low-false-alarm recovery branch.  Its gate is trained only to predict masked
healthy recovery error; neither damage labels nor test data train the gate.
The source USAD is retained solely as an independent baseline.
"""

import argparse
import gc
import json
import sys
import time
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

from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax,
    Normalizer,
    fit_usad,
    publication_metrics,
    score_usad,
    seed_everything,
)
from run_z24_mir_usad_v0 import scenarios, sensor_corrupt  # noqa: E402
from run_z24_recovery_kd_v3 import (  # noqa: E402
    CompactUSAD,
    RecoveryAdapter,
    anomaly_components,
    count_parameters,
    curriculum_mask,
    fit_compact_usad,
    score_compact,
    usad_losses,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores,
    load_manifest,
    read_stable_vertical,
    record_windows,
    robust_score_reference,
    split_training_windows,
    stable_sensor_map,
    threshold_from_labeled_validation,
)


METHODS = (
    "source_usad",
    "compact_usad",
    "recovery_student",
    "dual_gate",
)


class GatedRecoveryStudent(nn.Module):
    def __init__(
        self,
        channels: int,
        length: int,
        recovery_hidden: int,
        hidden1: int,
        hidden2: int,
        latent: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.recovery = RecoveryAdapter(channels, recovery_hidden)
        self.usad = CompactUSAD(
            channels * length, hidden1, hidden2, latent,
        )
        # availability + observed reconstruction residual + coverage +
        # criticality-weighted missing fraction
        self.uncertainty_head = nn.Sequential(
            nn.Linear(2 * channels + 2, max(16, 2 * channels)),
            nn.GELU(),
            nn.Linear(max(16, 2 * channels), 1),
            nn.Softplus(),
        )

    def recovery_features(
        self,
        values: torch.Tensor,
        predicted: torch.Tensor,
        available: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        observed_residual = values.sub(predicted).abs().mean(2) * available
        coverage = available.mean(1, keepdim=True)
        importance = importance / importance.sum().clamp_min(1e-8)
        missing_criticality = (
            (1.0 - available) * importance[None, :]
        ).sum(1, keepdim=True)
        return torch.cat(
            [available, observed_residual, coverage, missing_criticality],
            dim=1,
        )

    def parts(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
        importance: torch.Tensor,
    ):
        completed, predicted = self.recovery(values, available)
        latent, first, second, cycle = self.usad.parts(completed)
        uncertainty = self.uncertainty_head(
            self.recovery_features(values, predicted, available, importance)
        ).squeeze(1)
        return completed, predicted, uncertainty, latent, first, second, cycle


def masked_recovery_targets(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    missing = 1.0 - available[:, :, None]
    denominator = missing.sum((1, 2)) * truth.shape[-1]
    valid = denominator > 0
    squared = (truth.sub(predicted).pow(2) * missing).sum((1, 2))
    per_sample = squared / denominator.clamp_min(1.0)
    recovery_loss = per_sample[valid].mean() if valid.any() else squared.sum() * 0.0
    # A fixed physical scale only makes the regression target numerically mild.
    uncertainty_target = torch.log1p(per_sample / 1e-3)
    return recovery_loss, uncertainty_target, valid


def fit_gated_recovery(
    fit: np.ndarray,
    importance: np.ndarray,
    epochs: int,
    batch_size: int,
    recovery_hidden: int,
    hidden1: int,
    hidden2: int,
    latent: int,
    recovery_weight: float,
    uncertainty_weight: float,
    device: torch.device,
) -> GatedRecoveryStudent:
    channels, length = fit.shape[1:]
    model = GatedRecoveryStudent(
        channels, length, recovery_hidden, hidden1, hidden2, latent,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)),
        batch_size=batch_size,
        shuffle=True,
    )
    importance_tensor = torch.as_tensor(
        importance, dtype=torch.float32, device=device,
    )
    targeted_order = torch.as_tensor(
        np.argsort(importance)[::-1].copy(),
        dtype=torch.long,
        device=device,
    )
    shared = (
        list(model.recovery.parameters())
        + list(model.usad.encoder.parameters())
        + list(model.uncertainty_head.parameters())
    )
    parameters1 = shared + list(model.usad.decoder1.parameters())
    parameters2 = shared + list(model.usad.decoder2.parameters())
    optimizer1 = torch.optim.Adam(parameters1, lr=1e-3)
    optimizer2 = torch.optim.Adam(parameters2, lr=1e-3)

    for epoch in range(1, epochs + 1):
        logs = []
        model.train()
        for (clean,) in loader:
            clean = clean.to(device)
            available = curriculum_mask(
                len(clean), channels, targeted_order, epoch, epochs, device,
            )
            observed = clean * available[:, :, None]

            optimizer1.zero_grad(set_to_none=True)
            completed, predicted, uncertainty, _, first, second, cycle = model.parts(
                observed, available, importance_tensor,
            )
            usad1, _ = usad_losses(completed, first, second, cycle, epoch)
            recovery, target, valid = masked_recovery_targets(
                clean, predicted, available,
            )
            uncertainty_loss = (
                F.smooth_l1_loss(uncertainty[valid], target[valid])
                if valid.any() else uncertainty.sum() * 0.0
            )
            loss1 = (
                usad1
                + recovery_weight * recovery
                + uncertainty_weight * uncertainty_loss
            )
            loss1.backward()
            nn.utils.clip_grad_norm_(parameters1, 5.0)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            completed, predicted, uncertainty, _, first, second, cycle = model.parts(
                observed, available, importance_tensor,
            )
            _, usad2 = usad_losses(completed, first, second, cycle, epoch)
            recovery2, target2, valid2 = masked_recovery_targets(
                clean, predicted, available,
            )
            uncertainty_loss2 = (
                F.smooth_l1_loss(uncertainty[valid2], target2[valid2])
                if valid2.any() else uncertainty.sum() * 0.0
            )
            loss2 = (
                usad2
                + 0.5 * recovery_weight * recovery2
                + 0.5 * uncertainty_weight * uncertainty_loss2
            )
            loss2.backward()
            nn.utils.clip_grad_norm_(parameters2, 5.0)
            optimizer2.step()
            logs.append({
                "loss": float((loss1 + loss2).detach().cpu()),
                "recovery": float(recovery.detach().cpu()),
                "uncertainty": float(uncertainty_loss.detach().cpu()),
            })

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            log = pd.DataFrame(logs).mean()
            print(
                f"  gated-recovery epoch={epoch}/{epochs} "
                f"loss={log.loss:.6f} rec={log.recovery:.6f} "
                f"unc={log.uncertainty:.6f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def predict_recovery_outputs(
    model: GatedRecoveryStudent,
    values: np.ndarray,
    available: np.ndarray,
    importance: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(available)),
        batch_size=batch_size,
    )
    importance_tensor = torch.as_tensor(
        importance, dtype=torch.float32, device=device,
    )
    scores, uncertainty = [], []
    for batch, mask in loader:
        batch, mask = batch.to(device), mask.to(device)
        completed, _, predicted_uncertainty, _, first, _, cycle = model.parts(
            batch, mask, importance_tensor,
        )
        _, score = anomaly_components(completed, first, cycle)
        scores.append(score.cpu().numpy())
        uncertainty.append(predicted_uncertainty.cpu().numpy())
    return np.concatenate(scores), np.concatenate(uncertainty)


@torch.no_grad()
def fit_uncertainty_reference(
    model: GatedRecoveryStudent,
    fit: np.ndarray,
    importance: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Calibrate predicted uncertainty using only masked healthy fit windows."""
    values, masks = [], []
    channels = fit.shape[1]
    targeted = np.argsort(importance)[::-1]
    rng = np.random.default_rng(20261113)
    for replicate in range(4):
        mask = np.ones((len(fit), channels), dtype=np.float32)
        rate = 0.30 if replicate < 2 else 0.50
        count = min(channels - 1, max(1, int(round(rate * channels))))
        for index in range(len(fit)):
            selected = (
                rng.permutation(channels)[:count]
                if replicate % 2 == 0 else targeted[:count]
            )
            mask[index, selected] = 0.0
        values.append(fit * mask[:, :, None])
        masks.append(mask)
    _, predicted = predict_recovery_outputs(
        model,
        np.concatenate(values).astype(np.float32),
        np.concatenate(masks).astype(np.float32),
        importance,
        batch_size,
        device,
    )
    center = float(np.median(predicted))
    scale = float(max(
        1.4826 * np.median(np.abs(predicted - center)),
        0.25 * np.std(predicted),
        1e-6,
    ))
    return center, scale


def reliability_from_uncertainty(
    predicted: np.ndarray,
    available: np.ndarray,
    reference: tuple[float, float],
) -> np.ndarray:
    standardized = (predicted - reference[0]) / reference[1]
    reliability = 1.0 / (1.0 + np.exp(np.clip(standardized, -30, 30)))
    reliability[available.mean(1) >= 1.0 - 1e-8] = 1.0
    return reliability


def standardized_window_score(
    score: np.ndarray,
    reference: tuple[float, float],
) -> np.ndarray:
    values = np.log1p(np.maximum(np.asarray(score, float), 0.0))
    return (values - reference[0]) / reference[1]


def aggregate_standardized(
    values: np.ndarray,
    counts: list[int],
) -> tuple[list[float], list[float]]:
    upper, two_sided, start = [], [], 0
    for count in counts:
        current = values[start:start + count]
        upper.append(float(np.median(current)))
        two_sided.append(float(np.median(np.abs(current))))
        start += count
    if start != len(values):
        raise RuntimeError("Dual-score aggregation length mismatch")
    return upper, two_sided


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def build_values(records, cache, condition, kind, rate, replicate, importance):
    arrays, masks, counts, metadata = [], [], [], []
    for row in records.itertuples():
        values, available = sensor_corrupt(
            cache[str(row.record_id)], kind, rate, str(row.record_id),
            replicate, importance,
        )
        arrays.append(values)
        masks.append(available)
        counts.append(len(values))
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
        })
    return np.concatenate(arrays), np.concatenate(masks), counts, metadata


def record_frame(
    method,
    raw_score,
    counts,
    metadata,
    reference,
    available,
    parameters,
    reliability=None,
    already_standardized=False,
):
    if already_standardized:
        upper, two_sided = aggregate_standardized(raw_score, counts)
    else:
        upper, two_sided = aggregate_record_scores(raw_score, counts, reference)
    frame = pd.DataFrame(metadata)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two_sided
    frame["windows"] = counts
    frame["parameters"] = parameters
    window_coverage = available.mean(1)
    if reliability is None:
        reliability = np.ones(len(available), dtype=float)
    start, coverage_rows, reliability_rows = 0, [], []
    for count in counts:
        coverage_rows.append(float(window_coverage[start:start + count].mean()))
        reliability_rows.append(float(reliability[start:start + count].mean()))
        start += count
    frame["coverage"] = coverage_rows
    frame["recovery_reliability"] = reliability_rows
    return frame


def run_group(group, labels, seed, group_index, args, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_by_record = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_by_record)
    normalizer = Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scale = FitMinMax(fit)
    fit_scaled = scale.transform(fit).astype(np.float32)
    internal_scaled = scale.transform(internal).astype(np.float32)
    print(
        f"  fit={tuple(fit_scaled.shape)} internal={tuple(internal_scaled.shape)}",
        flush=True,
    )

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    seed_everything(seed + group_index * 1009)
    source = fit_usad(fit_scaled, args.epochs, args.batch_size, device)
    source_internal = score_usad(source, internal_scaled, args.batch_size, device)

    seed_everything(seed + group_index * 1009 + 97)
    compact = fit_compact_usad(
        fit_scaled, args.epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    compact_internal = score_compact(
        compact, internal_scaled, args.batch_size, device,
    )

    seed_everything(seed + group_index * 1009 + 194)
    recovery = fit_gated_recovery(
        fit_scaled, importance, args.epochs, args.batch_size,
        args.recovery_hidden, args.hidden1, args.hidden2, args.latent,
        args.recovery_weight, args.uncertainty_weight, device,
    )
    internal_mask = np.ones(internal_scaled.shape[:2], dtype=np.float32)
    recovery_internal, _ = predict_recovery_outputs(
        recovery, internal_scaled, internal_mask, importance,
        args.batch_size, device,
    )
    uncertainty_reference = fit_uncertainty_reference(
        recovery, fit_scaled, importance, args.batch_size, device,
    )
    references = {
        "source_usad": robust_score_reference(source_internal),
        "compact_usad": robust_score_reference(compact_internal),
        "recovery_student": robust_score_reference(recovery_internal),
    }
    parameter_counts = {
        "source_usad": count_parameters(source),
        "compact_usad": count_parameters(compact),
        "recovery_student": count_parameters(recovery),
        "dual_gate": count_parameters(compact) + count_parameters(recovery),
    }
    print(
        f"  parameters={parameter_counts} "
        f"uncertainty_reference={uncertainty_reference}",
        flush=True,
    )

    cohorts = [(validation, "clean", "clean", 0.0, 0)]
    cohorts.extend([
        (test, condition, kind, rate, replicate)
        for condition, kind, rate, replicate in scenarios()
    ])
    frames = []
    for records, condition, kind, rate, replicate in cohorts:
        values, available, counts, metadata = build_values(
            records, cache, condition, kind, rate, replicate, importance,
        )
        scaled = scale.transform(values).astype(np.float32)
        source_score = score_usad(source, scaled, args.batch_size, device)
        compact_score = score_compact(compact, scaled, args.batch_size, device)
        recovery_score, predicted_uncertainty = predict_recovery_outputs(
            recovery, scaled, available.astype(np.float32), importance,
            args.batch_size, device,
        )
        reliability = reliability_from_uncertainty(
            predicted_uncertainty, available, uncertainty_reference,
        )
        compact_z = standardized_window_score(
            compact_score, references["compact_usad"],
        )
        recovery_z = standardized_window_score(
            recovery_score, references["recovery_student"],
        )
        dual_score = reliability * recovery_z + (1.0 - reliability) * compact_z
        frames.extend([
            record_frame(
                "source_usad", source_score, counts, metadata,
                references["source_usad"], available,
                parameter_counts["source_usad"],
            ),
            record_frame(
                "compact_usad", compact_score, counts, metadata,
                references["compact_usad"], available,
                parameter_counts["compact_usad"],
            ),
            record_frame(
                "recovery_student", recovery_score, counts, metadata,
                references["recovery_student"], available,
                parameter_counts["recovery_student"], reliability,
            ),
            record_frame(
                "dual_gate", dual_score, counts, metadata, None, available,
                parameter_counts["dual_gate"], reliability,
                already_standardized=True,
            ),
        ])

    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del source, compact, recovery
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finish(output: Path, frame: pd.DataFrame) -> pd.DataFrame:
    metric_rows, scored = [], []
    for method in METHODS:
        validation = frame[(frame.method == method) & (frame.split == "val")].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                validation.label, validation[f"score_{tail}"],
            )
            candidates.append(((metric["ba"], metric["auc"], metric["f1"]), tail, threshold))
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        test["score"] = test[f"score_{tail}"]
        test["threshold"] = threshold
        test["score_tail"] = tail
        scored.append(test)
        rows, clean = [], None
        for (condition, replicate), current in test.groupby(["condition", "replicate"]):
            row = {
                "method": method,
                "condition": condition,
                "replicate": int(replicate),
                "score_tail": tail,
                "threshold": threshold,
                **publication_metrics(current.label, current.score, threshold),
                "coverage_mean": float(current.coverage.mean()),
                "reliability_mean": float(current.recovery_reliability.mean()),
                "parameters": int(current.parameters.iloc[0]),
            }
            rows.append(row)
            if condition == "clean":
                clean = row
        for row in rows:
            row["delta_auc"] = row["auc"] - clean["auc"]
            row["delta_ba"] = row["ba"] - clean["ba"]
        metric_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(scored, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ba_mean=("ba", "mean"),
        recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"),
        f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_ba_mean=("delta_ba", "mean"),
        coverage_mean=("coverage_mean", "mean"),
        reliability_mean=("reliability_mean", "mean"),
        parameters=("parameters", "mean"),
    )
    source_parameters = float(
        summary.loc[summary.method == "source_usad", "parameters"].mean()
    )
    summary["parameter_ratio_vs_source"] = summary.parameters / source_parameters
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(13, 5.5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 teacher-free compact/recovery dual path")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "dual_gate_auc.png", dpi=220)
    plt.close(axis.figure)
    print(summary.to_string(index=False), flush=True)
    return summary


def make_decision(summary: pd.DataFrame) -> dict:
    table = summary.set_index(["method", "condition"])
    conditions = ["clean", "sensor_0.30", "sensor_0.50", "targeted_0.50"]
    comparisons = {}
    for baseline in ("source_usad", "compact_usad", "recovery_student"):
        comparisons[baseline] = {
            condition: float(
                table.loc[("dual_gate", condition), "auc_mean"]
                - table.loc[(baseline, condition), "auc_mean"]
            )
            for condition in conditions
        }
    best_single = {
        condition: max(
            float(table.loc[("compact_usad", condition), "auc_mean"]),
            float(table.loc[("recovery_student", condition), "auc_mean"]),
        )
        for condition in conditions
    }
    dual = {
        condition: float(table.loc[("dual_gate", condition), "auc_mean"])
        for condition in conditions
    }
    promote = bool(
        np.mean([
            dual[c] - float(table.loc[("source_usad", c), "auc_mean"])
            for c in conditions[1:]
        ]) >= 0.01
        and dual["targeted_0.50"] >= float(
            table.loc[("source_usad", "targeted_0.50"), "auc_mean"]
        )
        and dual["clean"] >= float(
            table.loc[("source_usad", "clean"), "auc_mean"]
        ) - 0.02
    )
    return {
        "dual_gate_auc_gain": comparisons,
        "best_single_branch_auc": best_single,
        "dual_gate_auc": dual,
        "promote_to_multiseed": promote,
        "decision": (
            "PROMOTE teacher-free dual_gate to a three-seed pilot."
            if promote else
            "DO NOT PROMOTE dual_gate; retain the best single teacher-free branch."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--recovery-hidden", type=int, default=48)
    parser.add_argument("--recovery-weight", type=float, default=0.10)
    parser.add_argument("--uncertainty-weight", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={METHODS} epochs={args.epochs}",
        flush=True,
    )
    started = time.time()
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, *key)
        if path.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(group, labels, args.seed, group_index, args, device, path)
    paths = [group_file(args.output_dir, *key) for key in sorted(sensor_map)]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(METHODS) * (132 + 132 * len(scenarios()))
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/record/condition keys")
    summary = finish(args.output_dir, frame)
    decision = make_decision(summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 teacher-free dual-path recovery v4",
        "seed": args.seed,
        "epochs": args.epochs,
        "fit": "healthy train only",
        "selection": "clean labeled validation only",
        "test_used_for_selection": False,
        "teacher_or_distillation_used": False,
        "source_usad_role": "independent baseline only",
        "methods": list(METHODS),
        "gate_training": "healthy masked recovery-error regression only",
        "gate_inputs": [
            "availability mask",
            "observed recovery residual",
            "coverage",
            "healthy-train sensor criticality",
        ],
        "mask_curriculum": "clean/random to clean/random/targeted 30-50%",
        "loss_weights": {
            "recovery": args.recovery_weight,
            "uncertainty": args.uncertainty_weight,
        },
        "conditions": scenarios(),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(json.dumps(decision, indent=2), flush=True)
    print(f"DUAL_GATE_V4_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

