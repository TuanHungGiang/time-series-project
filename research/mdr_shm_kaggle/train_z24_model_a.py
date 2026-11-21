from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_z24_b2 import (
    Model as B2Model,
    build_file_index,
    excitation,
    features as b2_features,
    find_manifest,
    learn_anchors,
    load_vertical,
    metric_dict,
    normalized_sensor_spectra,
    resolve_source,
    stable_rng,
)


HERE = Path(__file__).resolve().parent


def prepare_frame(root: Path) -> tuple[pd.DataFrame, Path]:
    manifest = find_manifest(root)
    frame = pd.read_csv(manifest)
    frame["label"] = frame["label"].astype(int)
    if "role" not in frame.columns:
        frame["role"] = frame["split"]
    frame["role"] = frame["role"].astype(str).str.lower().replace({"val": "calibration"})
    frame.loc[frame["label"] == 1, "role"] = "test"
    frame = frame[frame["role"].isin(["train", "calibration", "test"])].copy()
    if frame.loc[frame.role.isin(["train", "calibration"]), "label"].ne(0).any():
        raise RuntimeError("Leakage: damage found in train/calibration")
    dataset_root = manifest.parent.parent
    file_index = build_file_index(dataset_root)
    frame["resolved_path"] = [
        str(resolve_source(row, file_index, dataset_root)) for _, row in frame.iterrows()
    ]
    frame["excitation_resolved"] = [excitation(row) for _, row in frame.iterrows()]
    counts = frame.groupby(["role", "label"]).size().to_dict()
    expected = {
        ("train", 0): 24,
        ("calibration", 0): 6,
        ("test", 0): 6,
        ("test", 1): 252,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected frozen split: {counts}; expected {expected}")
    return frame, manifest


def choose_bins(frequencies: np.ndarray, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, len(frequencies) - 1, count).round().astype(int))


@dataclass
class RecordFeatures:
    record_id: str
    label: int
    role: str
    excitation: str
    path: str
    frequencies: np.ndarray
    sensor_psd: np.ndarray
    tokens: np.ndarray


@dataclass
class TokenScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "TokenScaler":
        values = np.concatenate(arrays, axis=0)
        center = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        scale = np.maximum(scale, 0.05)
        return cls(center.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.center) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.center


def extract_record(row: pd.Series, bin_count: int) -> RecordFeatures:
    signals = load_vertical(Path(row.resolved_path))
    frequencies, sensor_psd = normalized_sensor_spectra(signals)
    indices = choose_bins(frequencies, bin_count)
    tokens = np.log10(sensor_psd[:, indices] + 1e-16).astype(np.float32)
    return RecordFeatures(
        record_id=str(row.record_id),
        label=int(row.label),
        role=str(row.role),
        excitation=str(row.excitation_resolved),
        path=str(row.resolved_path),
        frequencies=frequencies,
        sensor_psd=sensor_psd.astype(np.float32),
        tokens=tokens,
    )


class MaskAwareDeepSet(nn.Module):
    def __init__(self, bins: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(bins + 1, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent),
            nn.GELU(),
        )
        self.attention = nn.Sequential(nn.Linear(latent, 32), nn.Tanh(), nn.Linear(32, 1))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, bins),
        )

    def forward(
        self, tokens: torch.Tensor, availability: torch.Tensor, quality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(torch.cat([tokens, quality.unsqueeze(-1)], dim=-1))
        logits = self.attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~availability, -1e4)
        weights = torch.softmax(logits, dim=1)
        latent = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.decoder(latent), latent


def pad_batch(
    records: list[np.ndarray],
    qualities: list[np.ndarray],
    availabilities: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_sensors = max(item.shape[0] for item in records)
    bins = records[0].shape[1]
    x = np.zeros((len(records), max_sensors, bins), dtype=np.float32)
    q = np.zeros((len(records), max_sensors), dtype=np.float32)
    a = np.zeros((len(records), max_sensors), dtype=bool)
    for i, (tokens, quality, available) in enumerate(
        zip(records, qualities, availabilities)
    ):
        count = tokens.shape[0]
        x[i, :count] = tokens
        q[i, :count] = quality
        a[i, :count] = available
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(a).to(device),
        torch.from_numpy(q).to(device),
    )


def augment_tokens(
    tokens: np.ndarray, rng: np.random.Generator, config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = tokens.copy()
    sensors, bins = output.shape
    availability = np.ones(sensors, dtype=bool)
    quality = np.ones(sensors, dtype=np.float32)

    rate = float(rng.uniform(0.0, config["sensor_dropout_max"]))
    drop_count = min(sensors - 1, int(round(sensors * rate)))
    if drop_count:
        availability[rng.choice(sensors, drop_count, replace=False)] = False

    noise_std = float(rng.uniform(0.0, config["spectral_noise_std_max"]))
    output += rng.normal(0.0, noise_std, output.shape).astype(np.float32)

    affected_count = max(1, int(round(0.30 * sensors)))
    affected = rng.choice(sensors, affected_count, replace=False)
    fraction = float(rng.uniform(0.05, config["frequency_block_mask_max"]))
    length = max(1, int(round(bins * fraction)))
    coordinates = np.arange(bins)
    for sensor in affected:
        start = int(rng.integers(0, bins - length + 1))
        keep = np.ones(bins, dtype=bool)
        keep[start : start + length] = False
        output[sensor, ~keep] = np.interp(
            coordinates[~keep], coordinates[keep], output[sensor, keep]
        )
        quality[sensor] = 1.0 - fraction

    clip = float(rng.uniform(2.0, 5.0))
    output = np.clip(output, -clip, clip)
    return output, availability, quality


def deterministic_validation_loss(
    model: nn.Module,
    records: list[RecordFeatures],
    scaler: TokenScaler,
    device: torch.device,
    config: dict,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for record in records:
            clean = scaler.transform(record.tokens)
            target = np.median(clean, axis=0)
            for replicate in range(4):
                rng = stable_rng(record.record_id, "model_a_validation", replicate)
                corrupted, availability, quality = augment_tokens(clean, rng, config)
                x, a, q = pad_batch([corrupted], [quality], [availability], device)
                prediction, _ = model(x, a, q)
                losses.append(float(torch.mean((prediction[0] - torch.from_numpy(target).to(device)) ** 2)))
    return float(np.mean(losses))


def fit_model(
    train: list[RecordFeatures],
    calibration: list[RecordFeatures],
    scaler: TokenScaler,
    config: dict,
    device: torch.device,
    group_seed: int,
) -> tuple[MaskAwareDeepSet, list[dict]]:
    torch.manual_seed(group_seed)
    model = MaskAwareDeepSet(
        config["spectral_bins"], config["hidden_dim"], config["latent_dim"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-4
    )
    rng = np.random.default_rng(group_seed)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        batch_tokens, batch_quality, batch_availability, targets = [], [], [], []
        clean_tokens, clean_quality, clean_availability = [], [], []
        for _ in range(config["batch_size"]):
            record = train[int(rng.integers(0, len(train)))]
            clean = scaler.transform(record.tokens)
            corrupted, availability, quality = augment_tokens(clean, rng, config)
            batch_tokens.append(corrupted)
            batch_quality.append(quality)
            batch_availability.append(availability)
            targets.append(np.median(clean, axis=0))
            clean_tokens.append(clean)
            clean_quality.append(np.ones(clean.shape[0], dtype=np.float32))
            clean_availability.append(np.ones(clean.shape[0], dtype=bool))

        x, a, q = pad_batch(batch_tokens, batch_quality, batch_availability, device)
        xc, ac, qc = pad_batch(
            clean_tokens, clean_quality, clean_availability, device
        )
        target = torch.from_numpy(np.stack(targets).astype(np.float32)).to(device)
        pred_corrupt, latent_corrupt = model(x, a, q)
        pred_clean, latent_clean = model(xc, ac, qc)
        weights = config["loss_weights"]
        loss_clean = torch.mean((pred_clean - target) ** 2)
        loss_corrupt = torch.mean((pred_corrupt - target) ** 2)
        loss_prediction = torch.mean((pred_corrupt - pred_clean.detach()) ** 2)
        loss_latent = torch.mean((latent_corrupt - latent_clean.detach()) ** 2)
        loss = (
            weights["clean_reconstruction"] * loss_clean
            + weights["corrupted_reconstruction"] * loss_corrupt
            + weights["prediction_consistency"] * loss_prediction
            + weights["latent_consistency"] * loss_latent
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            validation = deterministic_validation_loss(
                model, calibration, scaler, device, config
            )
            history.append(
                {"epoch": epoch, "train_loss": float(loss.detach()), "validation_loss": validation}
            )
            print(
                f"epoch={epoch:03d} train={float(loss.detach()):.6f} "
                f"healthy_cal={validation:.6f}",
                flush=True,
            )
            if validation < best_loss - 1e-6:
                best_loss = validation
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 5
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history


@dataclass
class ComponentCalibration:
    center: np.ndarray
    scale: np.ndarray
    threshold: float
    reliability_threshold: float


def reconstruct_full_spectrum(
    prediction_std: np.ndarray,
    scaler: TokenScaler,
    frequencies: np.ndarray,
) -> np.ndarray:
    log_sampled = scaler.inverse(prediction_std)
    indices = choose_bins(frequencies, len(log_sampled))
    log_full = np.interp(np.arange(len(frequencies)), indices, log_sampled)
    spectrum = 10.0 ** log_full
    spectrum /= max(float(np.trapezoid(spectrum, frequencies)), 1e-30)
    return spectrum


def forward_record(
    model: nn.Module,
    record: RecordFeatures,
    scaler: TokenScaler,
    device: torch.device,
    availability: np.ndarray | None = None,
    quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    clean = scaler.transform(record.tokens)
    if availability is None:
        availability = np.ones(clean.shape[0], dtype=bool)
    if quality is None:
        quality = np.ones(clean.shape[0], dtype=np.float32)
    x, a, q = pad_batch([clean], [quality], [availability], device)
    model.eval()
    with torch.no_grad():
        prediction, latent = model(x, a, q)
    return prediction[0].cpu().numpy(), latent[0].cpu().numpy()


def raw_components(
    model: nn.Module,
    record: RecordFeatures,
    scaler: TokenScaler,
    b2: B2Model,
    device: torch.device,
    availability: np.ndarray | None = None,
    quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prediction, latent = forward_record(
        model, record, scaler, device, availability, quality
    )
    full_spectrum = reconstruct_full_spectrum(
        prediction, scaler, record.frequencies
    )
    b2_score = b2.raw_score(
        b2_features(full_spectrum, record.frequencies, b2.anchors)
    )
    observed = scaler.transform(record.tokens)
    if availability is None:
        availability = np.ones(observed.shape[0], dtype=bool)
    observed_median = np.median(observed[availability], axis=0)
    residual = float(np.mean((observed_median - prediction) ** 2))
    return np.asarray([b2_score, residual], dtype=float), latent


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    return center, np.maximum(scale, 1e-4)


def composite_score(components: np.ndarray, calibration: ComponentCalibration) -> float:
    standardized = (components - calibration.center) / calibration.scale
    return float(np.mean(np.maximum(standardized, 0.0)) / calibration.threshold)


def mc_uncertainty(
    model: nn.Module,
    record: RecordFeatures,
    scaler: TokenScaler,
    device: torch.device,
    availability: np.ndarray,
    quality: np.ndarray,
    repeats: int = 8,
) -> float:
    predictions = []
    present = np.flatnonzero(availability)
    for replicate in range(repeats):
        keep = availability.copy()
        count = min(len(present) - 1, max(0, int(round(0.10 * len(present)))))
        if count:
            rng = stable_rng(record.record_id, "model_a_mc", replicate)
            keep[rng.choice(present, count, replace=False)] = False
        prediction, _ = forward_record(model, record, scaler, device, keep, quality)
        predictions.append(prediction)
    disagreement = float(np.mean(np.var(np.stack(predictions), axis=0)))
    coverage_penalty = max(0.0, 0.5 - float(availability.mean())) ** 2
    return disagreement + coverage_penalty


def corrupt_record(
    record: RecordFeatures,
    condition: str,
    replicate: int,
    b2: B2Model,
) -> tuple[RecordFeatures, np.ndarray, np.ndarray]:
    signals = load_vertical(Path(record.path))
    sensors, samples = signals.shape
    availability = np.ones(sensors, dtype=bool)
    quality = np.ones(sensors, dtype=np.float32)
    rng = stable_rng(record.record_id, condition, replicate)
    rate = float(condition.split("_")[-1]) if condition != "clean" else 0.0
    count = min(sensors - 1, max(1, int(round(0.30 * sensors))))
    affected = rng.choice(sensors, count, replace=False)

    if condition.startswith("missing"):
        length = int(round(samples * rate))
        coordinates = np.arange(samples)
        for sensor in affected:
            start = int(rng.integers(0, samples - length + 1))
            keep = np.ones(samples, dtype=bool)
            keep[start : start + length] = False
            signals[sensor, ~keep] = np.interp(
                coordinates[~keep], coordinates[keep], signals[sensor, keep]
            )
            quality[sensor] = 1.0 - rate
    elif condition.startswith("noise"):
        snr = rate
        for sensor in affected:
            centered = signals[sensor] - signals[sensor].mean()
            power = float(np.mean(centered ** 2))
            sigma = np.sqrt(max(power, 1e-30) / 10 ** (snr / 10))
            signals[sensor] += rng.normal(0.0, sigma, samples).astype(np.float32)
    elif condition.startswith("dropout"):
        drop_count = min(sensors - 1, int(round(sensors * rate)))
        availability[rng.choice(sensors, drop_count, replace=False)] = False
    elif condition.startswith("targeted"):
        freq, psd = normalized_sensor_spectra(signals)
        evidence = [
            b2.raw_score(b2_features(item, freq, b2.anchors))
            for item in psd
        ]
        drop_count = min(sensors - 1, int(round(sensors * rate)))
        availability[np.argsort(evidence)[-drop_count:]] = False
    elif condition != "clean":
        raise ValueError(condition)

    frequencies, sensor_psd = normalized_sensor_spectra(signals)
    indices = choose_bins(frequencies, record.tokens.shape[1])
    changed = RecordFeatures(
        record.record_id,
        record.label,
        record.role,
        record.excitation,
        record.path,
        frequencies,
        sensor_psd.astype(np.float32),
        np.log10(sensor_psd[:, indices] + 1e-16).astype(np.float32),
    )
    return changed, availability, quality


def selective_curve(group: pd.DataFrame) -> list[dict]:
    output = []
    ordered = group.sort_values("reliability")
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(2, int(np.ceil(len(ordered) * coverage)))
        kept = ordered.iloc[:count]
        if kept.label.nunique() < 2:
            continue
        output.append(
            {
                "coverage": coverage,
                "records": len(kept),
                **metric_dict(kept.label.tolist(), kept.score.tolist()),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/model_a_v1"))
    parser.add_argument("--config", type=Path, default=HERE / "model_a_config.json")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.epochs:
        config["epochs"] = args.epochs
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    print(f"device={device} mode={args.mode}")
    frame, manifest = prepare_frame(args.input_root)

    rows_by_group = {
        group: frame[frame.excitation_resolved == group] for group in ("avt", "fvt")
    }
    artifacts = {}
    all_scores = []
    all_history = []

    for group_index, group in enumerate(("avt", "fvt")):
        rows = rows_by_group[group]
        print(f"extracting={group}")
        records = [
            extract_record(row, config["spectral_bins"]) for _, row in rows.iterrows()
        ]
        train = [r for r in records if r.role == "train"]
        calibration = [r for r in records if r.role == "calibration"]
        test = [r for r in records if r.role == "test"]
        scaler = TokenScaler.fit([r.tokens for r in train])

        train_spectra = [np.median(r.sensor_psd, axis=0) for r in train]
        frequencies = train[0].frequencies
        anchors = learn_anchors(train_spectra, frequencies)
        x_b2 = np.stack([
            b2_features(spectrum, frequencies, anchors)
            for spectrum in train_spectra
        ])
        b2 = B2Model.fit(anchors, x_b2)

        model, history = fit_model(
            train,
            calibration,
            scaler,
            config,
            device,
            config["seed"] + group_index,
        )
        for item in history:
            all_history.append({"excitation": group, **item})

        train_components, train_latents = [], []
        for record in train:
            components, latent = raw_components(
                model, record, scaler, b2, device
            )
            train_components.append(components)
            train_latents.append(latent)
        component_center, component_scale = robust_center_scale(
            np.stack(train_components)
        )

        provisional = ComponentCalibration(
            component_center, component_scale, 1.0, 1.0
        )
        calibration_raw = []
        calibration_reliability = []
        for record in calibration:
            components, _ = raw_components(
                model, record, scaler, b2, device
            )
            standardized = (components - component_center) / component_scale
            calibration_raw.append(float(np.mean(np.maximum(standardized, 0.0))))
            availability = np.ones(record.tokens.shape[0], dtype=bool)
            quality = np.ones(record.tokens.shape[0], dtype=np.float32)
            calibration_reliability.append(
                mc_uncertainty(
                    model, record, scaler, device, availability, quality
                )
            )
        score_threshold = max(calibration_raw)
        reliability_threshold = max(calibration_reliability)
        calibrated = ComponentCalibration(
            component_center,
            component_scale,
            max(score_threshold, 1e-6),
            max(reliability_threshold, 1e-12),
        )

        conditions = (
            ["clean", "missing_0.30", "noise_5", "dropout_0.30", "targeted_0.30"]
            if args.mode == "quick"
            else [
                "clean",
                "missing_0.10", "missing_0.30", "missing_0.50",
                "noise_20", "noise_10", "noise_5",
                "dropout_0.10", "dropout_0.30", "dropout_0.50",
                "targeted_0.10", "targeted_0.30", "targeted_0.50",
            ]
        )
        for record_index, record in enumerate(test, 1):
            for condition in conditions:
                repeats = 1 if condition == "clean" or args.mode == "quick" else 3
                for replicate in range(repeats):
                    changed, availability, quality = corrupt_record(
                        record, condition, replicate, b2
                    )
                    components, _ = raw_components(
                        model,
                        changed,
                        scaler,
                        b2,
                        device,
                        availability,
                        quality,
                    )
                    score = composite_score(components, calibrated)
                    reliability = mc_uncertainty(
                        model,
                        changed,
                        scaler,
                        device,
                        availability,
                        quality,
                    ) / calibrated.reliability_threshold
                    all_scores.append(
                        {
                            "record_id": record.record_id,
                            "label": record.label,
                            "excitation": group,
                            "condition": condition,
                            "replicate": replicate,
                            "score": score,
                            "reliability": reliability,
                            "accepted": reliability <= 1.0,
                            "sensor_coverage": float(availability.mean()),
                            "observed_quality": float(quality.mean()),
                            "component_b2_reconstructed": float(components[0]),
                            "component_reconstruction_residual": float(components[1]),
                        }
                    )
            if record_index % 36 == 0:
                print(f"{group}_test={record_index}/{len(test)}", flush=True)

        artifacts[group] = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "scaler_center": scaler.center,
            "scaler_scale": scaler.scale,
            "b2": b2.as_dict(),
            "component_center": calibrated.center,
            "component_scale": calibrated.scale,
            "score_threshold": calibrated.threshold,
            "reliability_threshold": calibrated.reliability_threshold,
        }

    scores = pd.DataFrame(all_scores)
    summaries = []
    risk_coverage = {}
    for (condition, replicate), group in scores.groupby(["condition", "replicate"]):
        summaries.append(
            {
                "condition": condition,
                "replicate": int(replicate),
                **metric_dict(group.label.tolist(), group.score.tolist()),
                "acceptance_rate": float(group.accepted.mean()),
            }
        )
        risk_coverage[f"{condition}_rep{replicate}"] = selective_curve(group)
    summary = pd.DataFrame(summaries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    summary.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(all_history).to_csv(args.output_dir / "training_history.csv", index=False)
    (args.output_dir / "risk_coverage.json").write_text(
        json.dumps(risk_coverage, indent=2)
    )
    torch.save(
        {
            "protocol": "model-a-v1",
            "manifest": str(manifest),
            "config": config,
            "groups": artifacts,
        },
        args.output_dir / "model_a.pt",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps({"mode": args.mode, "device": str(device), **config}, indent=2)
    )
    print(summary.to_string(index=False))
    print(f"MODEL_A_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()
