from __future__ import annotations

"""One-seed HeMIS-USAD screening for complete sensor loss on Z24.

The runner deliberately tests only the next two architectural hypotheses:

1. sensor-set mean aggregation (``hemis_mean``); and
2. HeMIS mean/variance aggregation (``hemis_meanvar``).

Both models use a shared sensor-wise temporal autoencoder with learned sensor
identity embeddings.  During healthy-only fitting, half of the batches are
clean, one quarter use random complete-sensor dropout, and one quarter drop
the healthy-train sensors with the largest mean-square response.  USAD is
trained on the resulting fixed-width set representation.  The raw-input USAD
baseline, records, masks, epochs and validation/test protocol are unchanged.

This is a one-seed block screening runner, not a publication experiment.
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
from third_party.usad import UsadModel  # noqa: E402


METHODS = ("usad", "hemis_mean", "hemis_meanvar")


class SensorSetUSAD(nn.Module):
    """Sensor-wise temporal encoder plus a permutation-safe HeMIS abstraction."""

    def __init__(
        self,
        channels: int,
        length: int,
        embedding: int,
        use_variance: bool,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.length = length
        self.embedding = embedding
        self.use_variance = use_variance

        # The temporal weights are shared across sensors.  A learned identity
        # vector preserves the fact that sensor positions are not exchangeable.
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, embedding),
        )
        self.sensor_identity = nn.Embedding(channels, embedding)
        self.sensor_decoder = nn.Sequential(
            nn.Linear(embedding, 128),
            nn.GELU(),
            nn.Linear(128, length),
            nn.Sigmoid(),
        )

        fused_width = embedding * (2 if use_variance else 1)
        self.usad = UsadModel(fused_width, max(3, fused_width // 4))

    @property
    def usad_encoder(self):
        return self.usad.encoder

    @property
    def decoder1(self):
        return self.usad.decoder1

    @property
    def decoder2(self):
        return self.usad.decoder2

    def sensor_features(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, length = values.shape
        flattened = values.reshape(batch * channels, 1, length)
        features = self.temporal_encoder(flattened).reshape(
            batch, channels, self.embedding,
        )
        sensor_ids = torch.arange(channels, device=values.device)
        features = torch.sigmoid(
            features + self.sensor_identity(sensor_ids)[None, :, :]
        )
        reconstruction = self.sensor_decoder(features).reshape(
            batch, channels, length,
        )
        return features, reconstruction

    def aggregate(self, features: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        observed = available[:, :, None]
        denominator = observed.sum(1).clamp_min(1.0)
        mean = (features * observed).sum(1) / denominator
        if not self.use_variance:
            return mean
        variance = (
            (features - mean[:, None, :]).pow(2) * observed
        ).sum(1) / denominator
        return torch.cat([mean, variance], dim=1)

    def parts(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        features, sensor_reconstruction = self.sensor_features(values)
        fused = self.aggregate(features, available)
        latent = self.usad_encoder(fused)
        first = self.decoder1(latent)
        second = self.decoder2(latent)
        cycle = self.decoder2(self.usad_encoder(first))
        return fused, sensor_reconstruction, first, second, cycle


def balanced_training_mask(
    batch: int,
    channels: int,
    targeted_order: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """50% clean, 25% random loss, 25% targeted loss; rates 30%/50%."""
    available = torch.ones(batch, channels, device=device)
    mode = torch.rand(batch, device=device)
    rates = torch.where(
        torch.rand(batch, device=device) < 0.5,
        torch.full((batch,), 0.30, device=device),
        torch.full((batch,), 0.50, device=device),
    )
    random_order = torch.rand(batch, channels, device=device).argsort(dim=1)
    for index in range(batch):
        if mode[index] < 0.50:
            continue
        count = min(
            channels - 1,
            max(1, int(round(float(rates[index]) * channels))),
        )
        if mode[index] < 0.75:
            selected = random_order[index, :count]
        else:
            selected = targeted_order[:count]
        available[index, selected] = 0.0
    return available


def fit_sensor_set_usad(
    fit: np.ndarray,
    importance: np.ndarray,
    epochs: int,
    batch_size: int,
    embedding: int,
    use_variance: bool,
    sensor_reconstruction_weight: float,
    device: torch.device,
) -> SensorSetUSAD:
    channels, length = fit.shape[1:]
    model = SensorSetUSAD(
        channels, length, embedding, use_variance,
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
    shared = (
        list(model.temporal_encoder.parameters())
        + list(model.sensor_identity.parameters())
        + list(model.sensor_decoder.parameters())
        + list(model.usad_encoder.parameters())
    )
    parameters1 = shared + list(model.decoder1.parameters())
    parameters2 = shared + list(model.decoder2.parameters())
    optimizer1 = torch.optim.Adam(parameters1, lr=1e-3)
    optimizer2 = torch.optim.Adam(parameters2, lr=1e-3)

    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for (clean,) in loader:
            clean = clean.to(device)
            available = balanced_training_mask(
                len(clean), channels, targeted_order, device,
            )

            optimizer1.zero_grad(set_to_none=True)
            fused, sensor_reconstruction, first, _, cycle = model.parts(
                clean, available,
            )
            sensor_loss = (clean - sensor_reconstruction).pow(2).mean()
            loss1 = (
                fused.sub(first).pow(2).mean() / epoch
                + (1.0 - 1.0 / epoch) * fused.sub(cycle).pow(2).mean()
                + sensor_reconstruction_weight * sensor_loss
            )
            loss1.backward()
            nn.utils.clip_grad_norm_(parameters1, 5.0)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            fused, sensor_reconstruction, _, second, cycle = model.parts(
                clean, available,
            )
            sensor_loss = (clean - sensor_reconstruction).pow(2).mean()
            loss2 = (
                fused.sub(second).pow(2).mean() / epoch
                - (1.0 - 1.0 / epoch) * fused.sub(cycle).pow(2).mean()
                + 0.5 * sensor_reconstruction_weight * sensor_loss
            )
            loss2.backward()
            nn.utils.clip_grad_norm_(parameters2, 5.0)
            optimizer2.step()
            losses.append(float((loss1 + loss2).detach().cpu()))

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            label = "meanvar" if use_variance else "mean"
            print(
                f"  HeMIS-{label} epoch={epoch}/{epochs} "
                f"loss={np.mean(losses):.6f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def score_sensor_set_usad(
    model: SensorSetUSAD,
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
        fused, _, first, _, cycle = model.parts(batch, mask)
        score = 0.5 * fused.sub(first).pow(2).mean(1)
        score += 0.5 * fused.sub(cycle).pow(2).mean(1)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def build_values(
    records: pd.DataFrame,
    cache: dict[str, np.ndarray],
    condition: str,
    kind: str,
    rate: float,
    replicate: int,
    importance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]]:
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


def make_record_frame(
    method: str,
    window_scores: np.ndarray,
    counts: list[int],
    metadata: list[dict],
    reference: dict,
    availability: np.ndarray,
) -> pd.DataFrame:
    upper, two_sided = aggregate_record_scores(
        window_scores, counts, reference,
    )
    frame = pd.DataFrame(metadata)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two_sided
    frame["windows"] = counts
    start, coverage = 0, []
    window_coverage = availability.mean(1)
    for count in counts:
        coverage.append(float(window_coverage[start:start + count].mean()))
        start += count
    frame["coverage"] = coverage
    return frame


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
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
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
        f"  fit_shape={tuple(fit_scaled.shape)} "
        f"internal_shape={tuple(internal_scaled.shape)}",
        flush=True,
    )

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    cohorts = [(validation, "clean", "clean", 0.0, 0)]
    cohorts.extend([
        (test, condition, kind, rate, replicate)
        for condition, kind, rate, replicate in scenarios()
    ])
    frames = []

    for method_index, method in enumerate(METHODS):
        seed_everything(seed + group_index * 1009 + method_index * 97)
        if method == "usad":
            model = fit_usad(
                fit_scaled, args.epochs, args.batch_size, device,
            )
            internal_scores = score_usad(
                model, internal_scaled, args.batch_size, device,
            )

            def scorer(values, available):
                return score_usad(
                    model,
                    scale.transform(values).astype(np.float32),
                    args.batch_size,
                    device,
                )
        else:
            use_variance = method == "hemis_meanvar"
            model = fit_sensor_set_usad(
                fit_scaled,
                importance,
                args.epochs,
                args.batch_size,
                args.embedding,
                use_variance,
                args.sensor_reconstruction_weight,
                device,
            )
            internal_mask = np.ones(
                internal_scaled.shape[:2], dtype=np.float32,
            )
            internal_scores = score_sensor_set_usad(
                model,
                internal_scaled,
                internal_mask,
                args.batch_size,
                device,
            )

            def scorer(values, available):
                return score_sensor_set_usad(
                    model,
                    scale.transform(values).astype(np.float32),
                    available.astype(np.float32),
                    args.batch_size,
                    device,
                )

        reference = robust_score_reference(internal_scores)
        print(f"  scoring method={method}", flush=True)
        for records, condition, kind, rate, replicate in cohorts:
            values, available, counts, metadata = build_values(
                records, cache, condition, kind, rate, replicate, importance,
            )
            scores = scorer(values, available)
            frames.append(make_record_frame(
                method,
                scores,
                counts,
                metadata,
                reference,
                available,
            ))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def finish(output: Path, frame: pd.DataFrame) -> pd.DataFrame:
    metric_rows, scored = [], []
    for method in METHODS:
        validation = frame[
            (frame.method == method) & (frame.split == "val")
        ].copy()
        test = frame[
            (frame.method == method) & (frame.split == "test")
        ].copy()
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                validation.label,
                validation[f"score_{tail}"],
            )
            candidates.append(
                ((metric["ba"], metric["auc"], metric["f1"]), tail, threshold)
            )
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        test["score"] = test[f"score_{tail}"]
        test["threshold"] = threshold
        test["score_tail"] = tail
        scored.append(test)

        rows, clean = [], None
        for (condition, replicate), current in test.groupby(
            ["condition", "replicate"]
        ):
            row = {
                "method": method,
                "condition": condition,
                "replicate": int(replicate),
                "score_tail": tail,
                "threshold": threshold,
                **publication_metrics(
                    current.label, current.score, threshold,
                ),
                "coverage_mean": float(current.coverage.mean()),
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
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    pivot = summary.pivot(
        index="condition", columns="method", values="auc_mean",
    )
    axis = pivot.plot.bar(
        figsize=(12, 5.5), ylim=(0, 1.02), rot=15, grid=True,
    )
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 complete-sensor-loss: HeMIS-USAD screening")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "hemis_usad_auc.png", dpi=220)
    plt.close(axis.figure)
    print(summary.to_string(index=False), flush=True)
    return summary


def screening_decision(summary: pd.DataFrame) -> dict:
    table = summary.set_index(["method", "condition"])
    corruptions = ["sensor_0.30", "sensor_0.50", "targeted_0.50"]
    rows = []
    for method in ("hemis_mean", "hemis_meanvar"):
        clean_gain = float(
            table.loc[(method, "clean"), "auc_mean"]
            - table.loc[("usad", "clean"), "auc_mean"]
        )
        corrupted_gain = float(np.mean([
            table.loc[(method, condition), "auc_mean"]
            - table.loc[("usad", condition), "auc_mean"]
            for condition in corruptions
        ]))
        targeted_gain = float(
            table.loc[(method, "targeted_0.50"), "auc_mean"]
            - table.loc[("usad", "targeted_0.50"), "auc_mean"]
        )
        rows.append({
            "method": method,
            "clean_auc_gain": clean_gain,
            "mean_corrupted_auc_gain": corrupted_gain,
            "targeted_auc_gain": targeted_gain,
        })
    gain_table = pd.DataFrame(rows)
    best = gain_table.sort_values(
        ["mean_corrupted_auc_gain", "targeted_auc_gain"],
        ascending=False,
    ).iloc[0]
    promote = bool(
        best.mean_corrupted_auc_gain >= 0.01
        and best.targeted_auc_gain >= 0.0
        and best.clean_auc_gain >= -0.01
    )
    return {
        "gains": rows,
        "best_proposed": str(best.method),
        "promote_to_recovery_ablation": promote,
        "decision": (
            "PROMOTE: add a decoupled recovery head to the best HeMIS block."
            if promote
            else "STOP: do not add recovery/teacher yet; the sensor-set representation did not beat USAD."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=16)
    parser.add_argument(
        "--sensor-reconstruction-weight", type=float, default=0.10,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={METHODS} "
        f"epochs={args.epochs}",
        flush=True,
    )
    started = time.time()

    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, *key)
        if path.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[
            (used.setup == key[0]) & (used.excitation == key[1])
        ]
        run_group(
            group,
            labels,
            args.seed,
            group_index,
            args,
            device,
            path,
        )

    paths = [
        group_file(args.output_dir, *key) for key in sorted(sensor_map)
    ]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(METHODS) * (132 + 132 * len(scenarios()))
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/record/condition keys")

    summary = finish(args.output_dir, frame)
    decision = screening_decision(summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 HeMIS-USAD v2 block screening",
        "seed": args.seed,
        "epochs": args.epochs,
        "fit": "healthy train only",
        "selection": "clean labeled validation only",
        "test_used_for_selection": False,
        "methods": list(METHODS),
        "sensor_set_training": {
            "clean_probability": 0.50,
            "random_loss_probability": 0.25,
            "targeted_loss_probability": 0.25,
            "loss_rates": [0.30, 0.50],
            "targeted_rule": "highest healthy-train mean-square-amplitude sensors",
        },
        "hemis_mean": "mean of observed sensor embeddings",
        "hemis_meanvar": "mean and variance of observed sensor embeddings",
        "sensor_reconstruction_weight": args.sensor_reconstruction_weight,
        "conditions": scenarios(),
        "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(json.dumps(decision, indent=2), flush=True)
    print(f"HEMIS_USAD_V2_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

