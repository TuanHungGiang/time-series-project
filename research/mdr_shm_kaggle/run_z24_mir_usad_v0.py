from __future__ import annotations

"""One-seed Z24 prototype: source USAD versus sensor-loss-aware MIR-USAD-v0.

This is an ablation-oriented screening runner, not a publication run. Both
methods use the same healthy train records, labeled clean validation protocol,
test records, nested sensor masks, epochs and record-level aggregation.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax, Normalizer, fit_usad, publication_metrics, score_usad,
    seed_everything,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)
from third_party.usad import UsadModel  # noqa: E402


METHODS = ("usad", "mir_usad_v0")


def stable_rng(record_id: str, kind: str, replicate: int) -> np.random.Generator:
    token = f"z24-mir-usad-v0|{record_id}|{kind}|{replicate}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(seed)


def scenarios() -> list[tuple[str, str, float, int]]:
    rows = [("clean", "clean", 0.0, 0)]
    for rate in (0.30, 0.50):
        for replicate in range(3):
            rows.append((f"sensor_{rate:.2f}", "sensor", rate, replicate))
    rows.append(("targeted_0.50", "targeted", 0.50, 0))
    return rows


def sensor_corrupt(
    values: np.ndarray,
    kind: str,
    rate: float,
    record_id: str,
    replicate: int,
    importance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = values.copy()
    windows, channels, _ = output.shape
    available = np.ones((windows, channels), dtype=np.float32)
    if kind == "clean":
        return output, available
    count = min(channels - 1, max(1, int(round(rate * channels))))
    if kind == "sensor":
        selected = stable_rng(record_id, kind, replicate).permutation(channels)[:count]
    elif kind == "targeted":
        selected = np.argsort(importance)[-count:]
    else:
        raise ValueError(kind)
    output[:, selected, :] = 0.0
    available[:, selected] = 0.0
    return output, available


class MIRUSAD(nn.Module):
    """USAD backbone plus a compact explicit sensor-mask embedding."""

    def __init__(self, channels: int, length: int):
        super().__init__()
        self.channels = channels
        self.length = length
        width = channels * length
        self.usad = UsadModel(width, max(5, width // 8))
        hidden = max(16, 4 * channels)
        self.mask_projector = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(),
            nn.Linear(hidden, width), nn.Tanh(),
        )
        self.mask_scale = nn.Parameter(torch.tensor(0.10))

    @property
    def encoder(self):
        return self.usad.encoder

    @property
    def decoder1(self):
        return self.usad.decoder1

    @property
    def decoder2(self):
        return self.usad.decoder2

    def encode(self, values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        flat = values.flatten(1)
        missing = 1.0 - available
        mask_embedding = self.mask_projector(missing)
        return self.encoder(flat + self.mask_scale * mask_embedding)

    def parts(self, values: torch.Tensor, available: torch.Tensor):
        latent = self.encode(values, available)
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


def random_training_mask(batch: int, channels: int, device: torch.device) -> torch.Tensor:
    rates = torch.tensor([0.10, 0.30, 0.50], device=device)
    chosen = rates[torch.randint(0, len(rates), (batch,), device=device)]
    random_order = torch.rand(batch, channels, device=device).argsort(dim=1)
    mask = torch.ones(batch, channels, device=device)
    for index in range(batch):
        count = min(channels - 1, max(1, int(round(float(chosen[index]) * channels))))
        mask[index, random_order[index, :count]] = 0.0
    return mask


def masked_reconstruction_loss(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    available: torch.Tensor,
    recovery_weight: float,
) -> torch.Tensor:
    error = (truth - prediction).pow(2)
    observed = available[:, :, None]
    missing = 1.0 - observed
    observed_loss = (error * observed).sum() / observed.sum().clamp_min(1.0) / truth.shape[-1]
    recovery_loss = (error * missing).sum() / missing.sum().clamp_min(1.0) / truth.shape[-1]
    return observed_loss + recovery_weight * recovery_loss


def fit_mir_usad(
    fit: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    recovery_weight: float,
    consistency_weight: float,
) -> MIRUSAD:
    channels, length = fit.shape[1:]
    model = MIRUSAD(channels, length).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)), batch_size=batch_size, shuffle=True,
    )
    parameters1 = (
        list(model.encoder.parameters()) + list(model.decoder1.parameters())
        + list(model.mask_projector.parameters()) + [model.mask_scale]
    )
    parameters2 = (
        list(model.encoder.parameters()) + list(model.decoder2.parameters())
        + list(model.mask_projector.parameters()) + [model.mask_scale]
    )
    optimizer1 = torch.optim.Adam(parameters1, lr=1e-3)
    optimizer2 = torch.optim.Adam(parameters2, lr=1e-3)
    fill = torch.from_numpy(
        np.median(fit, axis=0, keepdims=True).astype(np.float32)
    ).to(device)

    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for (clean,) in loader:
            clean = clean.to(device)
            available = random_training_mask(len(clean), channels, device)
            observed = available[:, :, None]
            masked = clean * observed + fill * (1.0 - observed)
            all_available = torch.ones_like(available)

            optimizer1.zero_grad(set_to_none=True)
            latent, first, _, cycle = model.parts(masked, available)
            with torch.no_grad():
                clean_latent = model.encode(clean, all_available)
            reconstruction1 = masked_reconstruction_loss(
                clean, first, available, recovery_weight,
            )
            cycle1 = masked_reconstruction_loss(
                clean, cycle, available, recovery_weight,
            )
            consistency = (latent - clean_latent).pow(2).mean()
            loss1 = (
                reconstruction1 / epoch
                + (1.0 - 1.0 / epoch) * cycle1
                + consistency_weight * consistency
            )
            loss1.backward()
            nn.utils.clip_grad_norm_(parameters1, 5.0)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            _, _, second, cycle = model.parts(masked, available)
            reconstruction2 = masked_reconstruction_loss(
                clean, second, available, recovery_weight,
            )
            cycle2 = masked_reconstruction_loss(
                clean, cycle, available, recovery_weight,
            )
            loss2 = reconstruction2 / epoch - (1.0 - 1.0 / epoch) * cycle2
            loss2.backward()
            nn.utils.clip_grad_norm_(parameters2, 5.0)
            optimizer2.step()
            losses.append(float((loss1 + loss2).detach().cpu()))

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"  MIR epoch={epoch}/{epochs} loss={np.mean(losses):.6f} "
                f"mask_scale={float(model.mask_scale.detach().cpu()):.4f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def mir_components(
    model: MIRUSAD,
    values: np.ndarray,
    available: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(available)),
        batch_size=batch_size,
    )
    sensor_errors, disagreements = [], []
    for batch, mask in loader:
        batch, mask = batch.to(device), mask.to(device)
        _, first, second, cycle = model.parts(batch, mask)
        error = 0.5 * (batch - first).pow(2) + 0.5 * (batch - cycle).pow(2)
        sensor_errors.append(error.mean(2).cpu().numpy())
        disagreements.append((first - second).pow(2).mean(2).cpu().numpy())
    return np.concatenate(sensor_errors), np.concatenate(disagreements)


def mir_sensor_weights(sensor_errors: np.ndarray) -> np.ndarray:
    median = np.median(sensor_errors, axis=0)
    mad = np.median(np.abs(sensor_errors - median), axis=0)
    weights = 1.0 / np.maximum(median + mad, 1e-6)
    lower, upper = np.quantile(weights, [0.10, 0.90])
    weights = np.clip(weights, lower, upper)
    return weights / np.mean(weights)


def fuse_mir(
    sensor_errors: np.ndarray,
    disagreement: np.ndarray,
    available: np.ndarray,
    weights: np.ndarray,
    disagreement_reference: float,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_mask = available * weights[None, :]
    coverage = weighted_mask.sum(1) / weights.sum()
    score = (sensor_errors * weighted_mask).sum(1) / weighted_mask.sum(1).clip(1e-6)
    missing = 1.0 - available
    missing_disagreement = (disagreement * missing).sum(1) / missing.sum(1).clip(1.0)
    reliability = coverage * np.exp(
        -missing_disagreement / max(disagreement_reference, 1e-6)
    )
    reliability[missing.sum(1) == 0] = 1.0
    return score, reliability


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def run_group(
    group: pd.DataFrame,
    labels: list[str],
    seed: int,
    group_index: int,
    args,
    device: torch.device,
    output: Path,
) -> None:
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
    fit_scaled = scale.transform(fit)
    internal_scaled = scale.transform(internal)

    seed_everything(seed + group_index * 1009)
    baseline = fit_usad(fit_scaled, args.epochs, args.batch_size, device)
    baseline_internal = score_usad(
        baseline, internal_scaled, args.batch_size, device,
    )

    seed_everything(seed + group_index * 1009)
    proposed = fit_mir_usad(
        fit_scaled, args.epochs, args.batch_size, device,
        args.recovery_weight, args.consistency_weight,
    )
    internal_mask = np.ones(internal.shape[:2], dtype=np.float32)
    internal_errors, internal_disagreement = mir_components(
        proposed, internal_scaled, internal_mask, args.batch_size, device,
    )
    weights = mir_sensor_weights(internal_errors)
    disagreement_reference = float(
        np.median(internal_disagreement) + np.median(
            np.abs(internal_disagreement - np.median(internal_disagreement))
        )
    )
    proposed_internal, _ = fuse_mir(
        internal_errors, internal_disagreement, internal_mask, weights,
        disagreement_reference,
    )
    references = {
        "usad": robust_score_reference(baseline_internal),
        "mir_usad_v0": robust_score_reference(proposed_internal),
    }

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw)

    frames = []
    for split, records, scenario_list in (
        ("val", validation, [("clean", "clean", 0.0, 0)]),
        ("test", test, scenarios()),
    ):
        for condition, kind, rate, replicate in scenario_list:
            values, masks, counts, metadata = [], [], [], []
            for row in records.itertuples():
                current, available = sensor_corrupt(
                    cache[str(row.record_id)], kind, rate, str(row.record_id),
                    replicate, importance,
                )
                values.append(current)
                masks.append(available)
                counts.append(len(current))
                metadata.append({
                    "record_id": row.record_id, "split": split,
                    "label": int(row.label), "state": row.state,
                    "setup": row.setup, "excitation": row.excitation,
                    "condition": condition, "replicate": replicate,
                    "rate": rate,
                })
            values_array = np.concatenate(values)
            mask_array = np.concatenate(masks)
            scaled = scale.transform(values_array)

            baseline_scores = score_usad(
                baseline, scaled, args.batch_size, device,
            )
            proposed_errors, proposed_disagreement = mir_components(
                proposed, scaled, mask_array, args.batch_size, device,
            )
            proposed_scores, reliability = fuse_mir(
                proposed_errors, proposed_disagreement, mask_array, weights,
                disagreement_reference,
            )

            for method, window_scores in (
                ("usad", baseline_scores), ("mir_usad_v0", proposed_scores),
            ):
                upper, two_sided = aggregate_record_scores(
                    window_scores, counts, references[method],
                )
                frame = pd.DataFrame(metadata)
                frame["method"] = method
                frame["score_upper"] = upper
                frame["score_lower"] = -np.asarray(upper)
                frame["score_two_sided"] = two_sided
                frame["windows"] = counts
                if method == "mir_usad_v0":
                    start = 0
                    record_reliability = []
                    for count in counts:
                        record_reliability.append(float(np.mean(reliability[start:start + count])))
                        start += count
                    frame["reliability"] = record_reliability
                else:
                    frame["reliability"] = 1.0
                frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def finish(output: Path, frame: pd.DataFrame, reliability_gate: float) -> None:
    all_metrics = []
    scored_frames = []
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
        scored_frames.append(test)
        clean = None
        method_rows = []
        for (condition, replicate), current in test.groupby(["condition", "replicate"]):
            row = {
                "method": method, "condition": condition,
                "replicate": int(replicate), "score_tail": tail,
                "threshold": threshold,
                **publication_metrics(current.label, current.score, threshold),
                "reliability_mean": float(current.reliability.mean()),
                "acceptance_rate": float((current.reliability >= reliability_gate).mean()),
            }
            method_rows.append(row)
            if condition == "clean":
                clean = row
        for row in method_rows:
            row["delta_auc"] = row["auc"] - clean["auc"]
            row["delta_ba"] = row["ba"] - clean["ba"]
        all_metrics.extend(method_rows)
    metrics = pd.DataFrame(all_metrics)
    scores = pd.concat(scored_frames, ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        fpr_mean=("fpr", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        reliability_mean=("reliability_mean", "mean"),
        acceptance_rate=("acceptance_rate", "mean"),
    )
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(11, 5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 one-seed sensor-loss screening")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "mir_usad_v0_comparison.png", dpi=220)
    plt.close(axis.figure)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--recovery-weight", type=float, default=0.50)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--reliability-gate", type=float, default=0.45)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    started = time.time()
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, key[0], key[1])
        if path.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(group, labels, args.seed, group_index, args, device, path)
    paths = [group_file(args.output_dir, *key) for key in sorted(sensor_map)]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = 2 * (132 + 132 * len(scenarios()))
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} record rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/record/condition keys")
    finish(args.output_dir, frame, args.reliability_gate)
    protocol = {
        "prototype": "MIR-USAD-v0", "seed": args.seed, "epochs": args.epochs,
        "fit": "healthy train only", "selection": "clean labeled validation",
        "conditions": scenarios(), "nested_random_sensor_masks": True,
        "targeted_rule": "highest healthy-train mean-square-amplitude sensors",
        "blocks": [
            "explicit low-dimensional sensor-mask embedding",
            "observed plus masked-sensor recovery loss",
            "clean/masked latent consistency",
            "sensor-wise coverage-normalized robust fusion",
            "retained-information reliability",
        ],
        "recovery_weight": args.recovery_weight,
        "consistency_weight": args.consistency_weight,
        "reliability_gate": args.reliability_gate,
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"MIR_USAD_V0_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

