from __future__ import annotations

"""Healthy-only full-sensor teacher to masked compact student on Z24.

Screening methods
-----------------
teacher_usad
    The existing full-input USAD reference.
compact_usad
    A substantially smaller clean-only USAD student (compression control).
recovery_student
    Compact student preceded by a missing-only recovery adapter, trained with
    clean/random/targeted masks but without knowledge distillation.
recovery_kd
    The same recovery student with full-sensor teacher latent, sensor-evidence
    and robust anomaly-score distillation.

All learned models fit healthy training records only.  Labeled clean validation
selects a single score tail and threshold per method; those decisions remain
fixed for clean, random complete-sensor loss and targeted sensor loss tests.
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
    "teacher_usad",
    "compact_usad",
    "recovery_student",
    "recovery_kd",
)


class CompactEncoder(nn.Module):
    def __init__(self, width: int, hidden1: int, hidden2: int, latent: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, latent),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CompactDecoder(nn.Module):
    def __init__(self, latent: int, hidden2: int, hidden1: int, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, width),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class CompactUSAD(nn.Module):
    def __init__(
        self,
        width: int,
        hidden1: int,
        hidden2: int,
        latent: int,
    ) -> None:
        super().__init__()
        self.encoder = CompactEncoder(width, hidden1, hidden2, latent)
        self.decoder1 = CompactDecoder(latent, hidden2, hidden1, width)
        self.decoder2 = CompactDecoder(latent, hidden2, hidden1, width)

    def parts(self, values: torch.Tensor):
        flat = values.flatten(1)
        latent = self.encoder(flat)
        first = self.decoder1(latent)
        second = self.decoder2(latent)
        cycle = self.decoder2(self.encoder(first))
        shape = values.shape
        return (
            latent,
            first.reshape(shape),
            second.reshape(shape),
            cycle.reshape(shape),
        )


class RecoveryAdapter(nn.Module):
    """Predict missing channels while copying every observed value exactly."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2 * channels, hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = available[:, :, None]
        mask_sequence = observed.expand_as(values)
        predicted = self.network(torch.cat([values * observed, mask_sequence], dim=1))
        completed = values * observed + predicted * (1.0 - observed)
        return completed, predicted


class RecoveryStudent(nn.Module):
    def __init__(
        self,
        channels: int,
        length: int,
        recovery_hidden: int,
        hidden1: int,
        hidden2: int,
        latent: int,
        teacher_latent: int,
    ) -> None:
        super().__init__()
        self.recovery = RecoveryAdapter(channels, recovery_hidden)
        self.usad = CompactUSAD(
            channels * length, hidden1, hidden2, latent,
        )
        self.latent_projection = nn.Linear(latent, teacher_latent)

    def parts(self, values: torch.Tensor, available: torch.Tensor):
        completed, predicted = self.recovery(values, available)
        latent, first, second, cycle = self.usad.parts(completed)
        return completed, predicted, latent, first, second, cycle


def model_parts(model, values: torch.Tensor):
    """Return latent and reconstructions for either source or compact USAD."""
    if isinstance(model, CompactUSAD):
        return model.parts(values)
    flat = values.flatten(1)
    latent = model.encoder(flat)
    first = model.decoder1(latent).reshape_as(values)
    second = model.decoder2(latent).reshape_as(values)
    cycle = model.decoder2(model.encoder(first.flatten(1))).reshape_as(values)
    return latent, first, second, cycle


def anomaly_components(
    truth: torch.Tensor,
    first: torch.Tensor,
    cycle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_error = 0.5 * truth.sub(first).pow(2)
    point_error += 0.5 * truth.sub(cycle).pow(2)
    sensor_evidence = point_error.mean(2)
    score = sensor_evidence.mean(1)
    return sensor_evidence, score


def usad_losses(
    truth: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    cycle: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fraction = 1.0 / epoch
    loss1 = fraction * truth.sub(first).pow(2).mean()
    loss1 += (1.0 - fraction) * truth.sub(cycle).pow(2).mean()
    loss2 = fraction * truth.sub(second).pow(2).mean()
    loss2 -= (1.0 - fraction) * truth.sub(cycle).pow(2).mean()
    return loss1, loss2


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def fit_compact_usad(
    fit: np.ndarray,
    epochs: int,
    batch_size: int,
    hidden1: int,
    hidden2: int,
    latent: int,
    device: torch.device,
) -> CompactUSAD:
    width = int(np.prod(fit.shape[1:]))
    model = CompactUSAD(width, hidden1, hidden2, latent).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)),
        batch_size=batch_size,
        shuffle=True,
    )
    parameters1 = list(model.encoder.parameters()) + list(model.decoder1.parameters())
    parameters2 = list(model.encoder.parameters()) + list(model.decoder2.parameters())
    optimizer1 = torch.optim.Adam(parameters1, lr=1e-3)
    optimizer2 = torch.optim.Adam(parameters2, lr=1e-3)

    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for (clean,) in loader:
            clean = clean.to(device)
            optimizer1.zero_grad(set_to_none=True)
            _, first, second, cycle = model.parts(clean)
            loss1, _ = usad_losses(clean, first, second, cycle, epoch)
            loss1.backward()
            nn.utils.clip_grad_norm_(parameters1, 5.0)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            _, first, second, cycle = model.parts(clean)
            _, loss2 = usad_losses(clean, first, second, cycle, epoch)
            loss2.backward()
            nn.utils.clip_grad_norm_(parameters2, 5.0)
            optimizer2.step()
            losses.append(float((loss1 + loss2).detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"  compact epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    return model.eval()


def curriculum_mask(
    batch: int,
    channels: int,
    targeted_order: torch.Tensor,
    epoch: int,
    epochs: int,
    device: torch.device,
) -> torch.Tensor:
    """Progress from random mild loss to balanced random/targeted loss."""
    progress = (epoch - 1) / max(epochs - 1, 1)
    clean_probability = 0.70 - 0.30 * progress
    targeted_probability = 0.30 * progress
    random_boundary = clean_probability + (1.0 - clean_probability - targeted_probability)
    available = torch.ones(batch, channels, device=device)
    mode = torch.rand(batch, device=device)
    random_order = torch.rand(batch, channels, device=device).argsort(dim=1)
    for index in range(batch):
        if mode[index] < clean_probability:
            continue
        if progress < 0.25:
            rate = 0.30
        else:
            rate = 0.30 if torch.rand((), device=device) < 0.5 else 0.50
        count = min(channels - 1, max(1, int(round(rate * channels))))
        selected = (
            random_order[index, :count]
            if mode[index] < random_boundary
            else targeted_order[:count]
        )
        available[index, selected] = 0.0
    return available


@torch.no_grad()
def teacher_reference(
    teacher,
    fit: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    loader = DataLoader(TensorDataset(torch.from_numpy(fit)), batch_size=batch_size)
    evidences, scores = [], []
    for (batch,) in loader:
        batch = batch.to(device)
        _, first, _, cycle = model_parts(teacher, batch)
        evidence, score = anomaly_components(batch, first, cycle)
        evidences.append(evidence.cpu().numpy())
        scores.append(score.cpu().numpy())
    evidence = np.concatenate(evidences)
    score = np.concatenate(scores)
    evidence_scale = np.maximum(np.median(evidence, axis=0), 1e-6)
    score_scale = max(float(np.median(score)), 1e-6)
    return {"evidence_scale": evidence_scale, "score_scale": score_scale}


def missing_recovery_loss(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    missing = 1.0 - available[:, :, None]
    denominator = (missing.sum() * truth.shape[-1]).clamp_min(1.0)
    return (truth.sub(predicted).pow(2) * missing).sum() / denominator


def fit_recovery_student(
    fit: np.ndarray,
    importance: np.ndarray,
    teacher,
    teacher_stats: dict,
    epochs: int,
    batch_size: int,
    recovery_hidden: int,
    hidden1: int,
    hidden2: int,
    latent: int,
    recovery_weight: float,
    latent_weight: float,
    evidence_weight: float,
    score_weight: float,
    use_kd: bool,
    device: torch.device,
) -> RecoveryStudent:
    channels, length = fit.shape[1:]
    teacher_latent = teacher.encoder.linear3.out_features
    model = RecoveryStudent(
        channels,
        length,
        recovery_hidden,
        hidden1,
        hidden2,
        latent,
        teacher_latent,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)),
        batch_size=batch_size,
        shuffle=True,
    )
    targeted_order = torch.as_tensor(
        np.argsort(importance)[::-1].copy(),
        dtype=torch.long,
        device=device,
    )
    evidence_scale = torch.as_tensor(
        teacher_stats["evidence_scale"], dtype=torch.float32, device=device,
    )
    score_scale = torch.tensor(
        teacher_stats["score_scale"], dtype=torch.float32, device=device,
    )
    shared = (
        list(model.recovery.parameters())
        + list(model.usad.encoder.parameters())
        + list(model.latent_projection.parameters())
    )
    parameters1 = shared + list(model.usad.decoder1.parameters())
    parameters2 = shared + list(model.usad.decoder2.parameters())
    optimizer1 = torch.optim.Adam(parameters1, lr=1e-3)
    optimizer2 = torch.optim.Adam(parameters2, lr=1e-3)

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    for epoch in range(1, epochs + 1):
        rows = []
        model.train()
        for (clean,) in loader:
            clean = clean.to(device)
            available = curriculum_mask(
                len(clean), channels, targeted_order, epoch, epochs, device,
            )
            observed = clean * available[:, :, None]
            with torch.no_grad():
                teacher_latent_value, teacher_first, _, teacher_cycle = model_parts(
                    teacher, clean,
                )
                teacher_evidence, teacher_score = anomaly_components(
                    clean, teacher_first, teacher_cycle,
                )

            optimizer1.zero_grad(set_to_none=True)
            completed, predicted, student_latent, first, second, cycle = model.parts(
                observed, available,
            )
            usad1, _ = usad_losses(completed, first, second, cycle, epoch)
            recovery = missing_recovery_loss(clean, predicted, available)
            latent_term = kd_parts_tensor(
                model, student_latent, teacher_latent_value,
            )
            evidence_term = evidence_loss_tensor(
                completed, first, cycle, teacher_evidence, evidence_scale,
            )
            score_term = score_loss_tensor(
                completed, first, cycle, teacher_score, score_scale,
            )
            kd_parts = {
                "latent": float(latent_term.detach().cpu()),
                "evidence": float(evidence_term.detach().cpu()),
                "score": float(score_term.detach().cpu()),
            }
            if use_kd:
                kd_weighted = (
                    latent_weight * latent_term
                    + evidence_weight * evidence_term
                    + score_weight * score_term
                )
            else:
                kd_weighted = torch.zeros((), device=device)
            loss1 = usad1 + recovery_weight * recovery + kd_weighted
            loss1.backward()
            nn.utils.clip_grad_norm_(parameters1, 5.0)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            completed, predicted, student_latent, first, second, cycle = model.parts(
                observed, available,
            )
            _, usad2 = usad_losses(completed, first, second, cycle, epoch)
            recovery2 = missing_recovery_loss(clean, predicted, available)
            if use_kd:
                kd_weighted2 = 0.5 * (
                    latent_weight * kd_parts_tensor(model, student_latent, teacher_latent_value)
                    + evidence_weight * evidence_loss_tensor(
                        completed, first, cycle, teacher_evidence, evidence_scale,
                    )
                    + score_weight * score_loss_tensor(
                        completed, first, cycle, teacher_score, score_scale,
                    )
                )
            else:
                kd_weighted2 = torch.zeros((), device=device)
            loss2 = usad2 + 0.5 * recovery_weight * recovery2 + kd_weighted2
            loss2.backward()
            nn.utils.clip_grad_norm_(parameters2, 5.0)
            optimizer2.step()
            rows.append({
                "loss": float((loss1 + loss2).detach().cpu()),
                "recovery": float(recovery.detach().cpu()),
                **kd_parts,
            })

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            log = pd.DataFrame(rows).mean()
            label = "recovery-kd" if use_kd else "recovery"
            print(
                f"  {label} epoch={epoch}/{epochs} loss={log.loss:.6f} "
                f"rec={log.recovery:.6f} latent={log.latent:.6f} "
                f"evidence={log.evidence:.6f} score={log.score:.6f}",
                flush=True,
            )
    return model.eval()


def kd_parts_tensor(
    model: RecoveryStudent,
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(
        model.latent_projection(student_latent), teacher_latent, dim=1,
    ).mean()


def evidence_loss_tensor(
    completed: torch.Tensor,
    first: torch.Tensor,
    cycle: torch.Tensor,
    teacher_evidence: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    student_evidence, _ = anomaly_components(completed, first, cycle)
    return F.smooth_l1_loss(
        torch.log1p(student_evidence / scale[None, :]),
        torch.log1p(teacher_evidence / scale[None, :]),
    )


def score_loss_tensor(
    completed: torch.Tensor,
    first: torch.Tensor,
    cycle: torch.Tensor,
    teacher_score: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    _, student_score = anomaly_components(completed, first, cycle)
    return F.smooth_l1_loss(
        torch.log1p(student_score / scale),
        torch.log1p(teacher_score / scale),
    )


@torch.no_grad()
def score_compact(
    model: CompactUSAD,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        _, first, _, cycle = model.parts(batch)
        _, score = anomaly_components(batch, first, cycle)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def score_recovery(
    model: RecoveryStudent,
    values: np.ndarray,
    available: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(available)),
        batch_size=batch_size,
    )
    scores = []
    for batch, mask in loader:
        batch, mask = batch.to(device), mask.to(device)
        completed, _, _, first, _, cycle = model.parts(batch, mask)
        _, score = anomaly_components(completed, first, cycle)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)


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


def make_record_frame(method, scores, counts, metadata, reference, available, parameters):
    upper, two_sided = aggregate_record_scores(scores, counts, reference)
    frame = pd.DataFrame(metadata)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two_sided
    frame["windows"] = counts
    frame["parameters"] = parameters
    window_coverage = available.mean(1)
    start, coverage = 0, []
    for count in counts:
        coverage.append(float(window_coverage[start:start + count].mean()))
        start += count
    frame["coverage"] = coverage
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
    teacher = fit_usad(fit_scaled, args.epochs, args.batch_size, device)
    teacher.eval()
    teacher_stats = teacher_reference(
        teacher, fit_scaled, args.batch_size, device,
    )
    teacher_internal = score_usad(
        teacher, internal_scaled, args.batch_size, device,
    )
    teacher_parameters = count_parameters(teacher)
    print(f"  teacher parameters={teacher_parameters:,}", flush=True)

    seed_everything(seed + group_index * 1009 + 97)
    compact = fit_compact_usad(
        fit_scaled, args.epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    compact_internal = score_compact(
        compact, internal_scaled, args.batch_size, device,
    )

    seed_everything(seed + group_index * 1009 + 194)
    recovery = fit_recovery_student(
        fit_scaled, importance, teacher, teacher_stats,
        args.epochs, args.batch_size, args.recovery_hidden,
        args.hidden1, args.hidden2, args.latent,
        args.recovery_weight, args.latent_weight,
        args.evidence_weight, args.score_weight, False, device,
    )
    all_available = np.ones(internal_scaled.shape[:2], dtype=np.float32)
    recovery_internal = score_recovery(
        recovery, internal_scaled, all_available, args.batch_size, device,
    )

    seed_everything(seed + group_index * 1009 + 291)
    recovery_kd = fit_recovery_student(
        fit_scaled, importance, teacher, teacher_stats,
        args.epochs, args.batch_size, args.recovery_hidden,
        args.hidden1, args.hidden2, args.latent,
        args.recovery_weight, args.latent_weight,
        args.evidence_weight, args.score_weight, True, device,
    )
    recovery_kd_internal = score_recovery(
        recovery_kd, internal_scaled, all_available, args.batch_size, device,
    )
    parameter_counts = {
        "teacher_usad": teacher_parameters,
        "compact_usad": count_parameters(compact),
        "recovery_student": count_parameters(recovery),
        "recovery_kd": count_parameters(recovery_kd),
    }
    print(f"  parameter_counts={parameter_counts}", flush=True)
    references = {
        "teacher_usad": robust_score_reference(teacher_internal),
        "compact_usad": robust_score_reference(compact_internal),
        "recovery_student": robust_score_reference(recovery_internal),
        "recovery_kd": robust_score_reference(recovery_kd_internal),
    }

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
        method_scores = {
            "teacher_usad": score_usad(
                teacher, scaled, args.batch_size, device,
            ),
            "compact_usad": score_compact(
                compact, scaled, args.batch_size, device,
            ),
            "recovery_student": score_recovery(
                recovery, scaled, available.astype(np.float32),
                args.batch_size, device,
            ),
            "recovery_kd": score_recovery(
                recovery_kd, scaled, available.astype(np.float32),
                args.batch_size, device,
            ),
        }
        for method, scores in method_scores.items():
            frames.append(make_record_frame(
                method, scores, counts, metadata, references[method],
                available, parameter_counts[method],
            ))
    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del teacher, compact, recovery, recovery_kd
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
        parameters=("parameters", "mean"),
    )
    teacher_parameters = float(
        summary.loc[summary.method == "teacher_usad", "parameters"].mean()
    )
    summary["parameter_ratio_vs_teacher"] = summary.parameters / teacher_parameters
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(13, 5.5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 full-sensor teacher to masked compact student")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "recovery_kd_auc.png", dpi=220)
    plt.close(axis.figure)
    print(summary.to_string(index=False), flush=True)
    return summary


def decision(summary: pd.DataFrame) -> dict:
    table = summary.set_index(["method", "condition"])
    corruptions = ["sensor_0.30", "sensor_0.50", "targeted_0.50"]
    proposed = "recovery_kd"
    gains = {
        condition: float(
            table.loc[(proposed, condition), "auc_mean"]
            - table.loc[("teacher_usad", condition), "auc_mean"]
        )
        for condition in ["clean", *corruptions]
    }
    kd_effect = {
        condition: float(
            table.loc[("recovery_kd", condition), "auc_mean"]
            - table.loc[("recovery_student", condition), "auc_mean"]
        )
        for condition in ["clean", *corruptions]
    }
    promote = bool(
        np.mean([gains[c] for c in corruptions]) >= 0.01
        and gains["targeted_0.50"] >= 0.0
        and gains["clean"] >= -0.02
    )
    return {
        "proposed_vs_teacher_auc_gain": gains,
        "incremental_kd_auc_effect": kd_effect,
        "promote_to_multiseed": promote,
        "decision": (
            "PROMOTE recovery_kd to a longer multi-seed ablation."
            if promote
            else "STOP one-seed screening; inspect recovery and KD effects before adding complexity."
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
    parser.add_argument("--latent-weight", type=float, default=0.05)
    parser.add_argument("--evidence-weight", type=float, default=0.10)
    parser.add_argument("--score-weight", type=float, default=0.05)
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
    screening = decision(summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 recovery plus healthy-only distillation v3",
        "seed": args.seed,
        "epochs": args.epochs,
        "fit": "healthy train only",
        "selection": "clean labeled validation only",
        "test_used_for_selection": False,
        "methods": list(METHODS),
        "teacher": "full-sensor source USAD, frozen during student fit",
        "student": {
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "latent": args.latent,
            "recovery_hidden": args.recovery_hidden,
        },
        "distillation": [
            "cosine latent projection",
            "log-scaled sensor evidence SmoothL1",
            "log-scaled anomaly score SmoothL1",
        ],
        "loss_weights": {
            "recovery": args.recovery_weight,
            "latent": args.latent_weight,
            "evidence": args.evidence_weight,
            "score": args.score_weight,
        },
        "mask_curriculum": "clean/random to clean/random/targeted 30-50%",
        "conditions": scenarios(),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(json.dumps(screening, indent=2), flush=True)
    print(f"RECOVERY_KD_V3_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

