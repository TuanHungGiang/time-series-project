from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_z24_b2 import features, load_vertical, normalized_sensor_spectra
from run_z24_model_a_v3 import b2_score, fit_b2, prepare_frame
from run_z24_model_a_v31 import corrupt_v31


CLASSICAL_MODELS = ("pca_spe", "ocsvm", "isolation_forest")
DEEP_MODELS = ("cnn_ae", "masked_cnn_ae", "deep_svdd")
ALL_MODELS = ("b2",) + CLASSICAL_MODELS + DEEP_MODELS
SPECTRAL_BINS = 256


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def conditions_for(mode: str) -> list[str]:
    if mode == "quick":
        return ["clean", "missing_0.30", "noise_5", "dropout_0.30", "targeted_0.30"]
    return [
        "clean",
        "missing_0.10",
        "missing_0.30",
        "missing_0.50",
        "noise_20",
        "noise_10",
        "noise_5",
        "dropout_0.10",
        "dropout_0.30",
        "dropout_0.50",
        "targeted_0.10",
        "targeted_0.30",
        "targeted_0.50",
    ]


def robust_spectrum_features(model, frequencies: np.ndarray, psd: np.ndarray, availability: np.ndarray) -> np.ndarray:
    spectrum = np.median(psd[availability], axis=0)
    return features(spectrum, frequencies, model.anchors)


@dataclass
class SpectralNormalizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "SpectralNormalizer":
        matrix = np.concatenate(arrays, axis=0)
        center = np.median(matrix, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(matrix - center), axis=0), 0.05)
        return cls(center.astype(np.float32), scale.astype(np.float32))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = (matrix - self.center) / self.scale
        return np.clip(values, -8.0, 8.0).astype(np.float32)


def record_log_spectrum(psd: np.ndarray, availability: np.ndarray) -> np.ndarray:
    values = np.log10(np.maximum(np.median(psd[availability], axis=0), 1e-16))
    old = np.linspace(0.0, 1.0, values.shape[0])
    new = np.linspace(0.0, 1.0, SPECTRAL_BINS)
    return np.interp(new, old, values).astype(np.float32)


def augmented_record_spectra(psd: np.ndarray, record_id: str, count: int = 12) -> np.ndarray:
    sensors = psd.shape[0]
    token = hashlib.sha256(str(record_id).encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(token, "little"))
    output = [record_log_spectrum(psd, np.ones(sensors, dtype=bool))]
    keep_count = max(2, int(round(0.70 * sensors)))
    for _ in range(count - 1):
        availability = np.zeros(sensors, dtype=bool)
        availability[rng.choice(sensors, keep_count, replace=False)] = True
        output.append(record_log_spectrum(psd, availability))
    return np.stack(output)


class SpectralAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.GELU(),
        )
        self.to_latent = nn.Linear(64 * 32, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 32)
        self.deconvolution = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.convolution(x[:, None, :]).flatten(1)
        return self.to_latent(hidden)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.from_latent(z).reshape(-1, 64, 32)
        return self.deconvolution(hidden)[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def masked_input(x: torch.Tensor) -> torch.Tensor:
    output = x.clone()
    batch, width = output.shape
    random_mask = torch.rand_like(output) < 0.10
    output[random_mask] = 0.0
    for index in range(batch):
        band = int(torch.randint(max(2, width // 32), max(3, width // 8), (1,), device=x.device))
        start = int(torch.randint(0, width - band + 1, (1,), device=x.device))
        output[index, start : start + band] = 0.0
    return output


def train_autoencoder(
    matrix: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    denoising: bool,
) -> SpectralAutoencoder:
    set_seed(seed)
    model = SpectralAutoencoder(latent_dim).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(matrix)),
        batch_size=min(batch_size, len(matrix)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            target = batch.to(device)
            model_input = masked_input(target) if denoising else target
            loss = torch.mean((model(model_input) - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def ae_sensor_scores(model: SpectralAutoencoder, matrix: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        values = torch.from_numpy(matrix).to(device)
        error = torch.mean((model(values) - values) ** 2, dim=1)
    return error.cpu().numpy()


@dataclass
class DeepSVDDModel:
    network: SpectralAutoencoder
    center: torch.Tensor


def train_deep_svdd(
    matrix: np.ndarray,
    device: torch.device,
    seed: int,
    pretrain_epochs: int,
    svdd_epochs: int,
    batch_size: int,
    latent_dim: int,
) -> DeepSVDDModel:
    network = train_autoencoder(
        matrix, device, seed, pretrain_epochs, batch_size, latent_dim, False
    )
    values = torch.from_numpy(matrix).to(device)
    with torch.no_grad():
        center = network.encode(values).mean(dim=0)
        center[torch.abs(center) < 0.05] = 0.05
    loader = DataLoader(
        TensorDataset(torch.from_numpy(matrix)),
        batch_size=min(batch_size, len(matrix)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 17),
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=1e-5)
    network.train()
    for _ in range(svdd_epochs):
        for (batch,) in loader:
            latent = network.encode(batch.to(device))
            loss = torch.mean(torch.sum((latent - center) ** 2, dim=1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    network.eval()
    return DeepSVDDModel(network, center)


def svdd_sensor_scores(model: DeepSVDDModel, matrix: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        latent = model.network.encode(torch.from_numpy(matrix).to(device))
        scores = torch.sum((latent - model.center) ** 2, dim=1)
    return scores.cpu().numpy()


@dataclass
class ClassicalModel:
    scaler: RobustScaler
    estimator: object
    name: str

    def score(self, features_matrix: np.ndarray) -> np.ndarray:
        values = self.scaler.transform(features_matrix)
        if self.name == "pca_spe":
            reconstructed = self.estimator.inverse_transform(self.estimator.transform(values))
            return np.mean((values - reconstructed) ** 2, axis=1)
        if self.name == "ocsvm":
            return -self.estimator.decision_function(values)
        return -self.estimator.score_samples(values)


def fit_classical(name: str, matrix: np.ndarray, seed: int) -> ClassicalModel:
    scaler = RobustScaler().fit(matrix)
    values = scaler.transform(matrix)
    if name == "pca_spe":
        estimator = PCA(n_components=min(6, len(matrix) - 1, matrix.shape[1]), random_state=seed).fit(values)
    elif name == "ocsvm":
        estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10).fit(values)
    elif name == "isolation_forest":
        estimator = IsolationForest(n_estimators=300, max_samples="auto", contamination="auto", random_state=seed).fit(values)
    else:
        raise ValueError(name)
    return ClassicalModel(scaler, estimator, name)


@dataclass
class Calibration:
    threshold: float
    scale: float

    @classmethod
    def fit(cls, scores: list[float]) -> "Calibration":
        values = np.asarray(scores, dtype=float)
        threshold = float(np.max(values))
        scale = float(max(1.4826 * np.median(np.abs(values - np.median(values))), np.std(values), 1e-6))
        return cls(threshold, scale)

    def normalize(self, score: float) -> float:
        return float(1.0 + (score - self.threshold) / self.scale)


def metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = scores > 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def record_deep_raw(name: str, model, normalizer: SpectralNormalizer, psd: np.ndarray, availability: np.ndarray, device: torch.device) -> float:
    matrix = normalizer.transform(record_log_spectrum(psd, availability)[None, :])
    if name == "deep_svdd":
        values = svdd_sensor_scores(model, matrix, device)
    else:
        values = ae_sensor_scores(model, matrix, device)
    return float(values[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/z24_detector_baselines"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    args = parser.parse_args()

    requested = tuple(item.strip() for item in args.models.split(",") if item.strip())
    unknown = sorted(set(requested) - set(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choices={ALL_MODELS}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.mode == "quick":
        seeds = seeds[:1]
        args.epochs = min(args.epochs, 12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} models={requested} seeds={seeds}", flush=True)

    frame, manifest = prepare_frame(args.input_root)
    b2_models = {}
    feature_models: dict[tuple[str, str], ClassicalModel] = {}
    deep_models: dict[tuple[str, str, int], object] = {}
    normalizers = {}
    calibrations: dict[tuple[str, str, int], Calibration] = {}
    training_rows = []

    for excitation in ("avt", "fvt"):
        rows = frame[frame.excitation_resolved == excitation]
        b2 = fit_b2(rows)
        b2_models[excitation] = b2
        train_features, calibration_features = [], []
        train_spectra, calibration_cache = [], []
        for _, row in rows[rows.role.isin(["train", "calibration"])].iterrows():
            frequencies, psd = normalized_sensor_spectra(load_vertical(Path(row.resolved_path)))
            availability = np.ones(psd.shape[0], dtype=bool)
            vector = robust_spectrum_features(b2, frequencies, psd, availability)
            if row.role == "train":
                train_features.append(vector)
                train_spectra.append(augmented_record_spectra(psd, str(row.record_id)))
            else:
                calibration_features.append(vector)
                calibration_cache.append((frequencies, psd, availability))
        train_features_matrix = np.stack(train_features)
        normalizer = SpectralNormalizer.fit(train_spectra)
        normalizers[excitation] = normalizer
        deep_train = normalizer.transform(np.concatenate(train_spectra, axis=0))

        if "b2" in requested:
            calibrations[(excitation, "b2", 0)] = Calibration(1.0, 1.0)
        for name in CLASSICAL_MODELS:
            if name not in requested:
                continue
            fitted = fit_classical(name, train_features_matrix, 0)
            feature_models[(excitation, name)] = fitted
            raw = fitted.score(np.stack(calibration_features)).tolist()
            calibrations[(excitation, name, 0)] = Calibration.fit(raw)

        for seed in seeds:
            for name in DEEP_MODELS:
                if name not in requested:
                    continue
                start = time.time()
                if name == "deep_svdd":
                    fitted = train_deep_svdd(
                        deep_train, device, seed, max(4, args.epochs // 3), args.epochs,
                        args.batch_size, args.latent_dim,
                    )
                else:
                    fitted = train_autoencoder(
                        deep_train, device, seed, args.epochs, args.batch_size,
                        args.latent_dim, name == "masked_cnn_ae",
                    )
                deep_models[(excitation, name, seed)] = fitted
                raw = [
                    record_deep_raw(name, fitted, normalizer, psd, availability, device)
                    for _, psd, availability in calibration_cache
                ]
                calibrations[(excitation, name, seed)] = Calibration.fit(raw)
                training_rows.append({
                    "excitation": excitation,
                    "model": name,
                    "seed": seed,
                    "training_seconds": time.time() - start,
                    "train_augmented_spectra": len(deep_train),
                    "parameters": sum(p.numel() for p in (fitted.network if name == "deep_svdd" else fitted).parameters()),
                })
                print(f"trained {excitation}/{name}/seed{seed}", flush=True)

    output = []
    conditions = conditions_for(args.mode)
    test = frame[frame.role == "test"]
    for record_index, (_, row) in enumerate(test.iterrows(), 1):
        signals = load_vertical(Path(row.resolved_path))
        excitation = row.excitation_resolved
        b2 = b2_models[excitation]
        for condition in conditions:
            repeats = 1 if condition == "clean" or args.mode == "quick" else 3
            if args.mode == "quick" and condition.startswith("targeted"):
                repeats = 3
            for replicate in range(repeats):
                changed, availability, _, signature = corrupt_v31(
                    signals, condition, str(row.record_id), replicate, b2
                )
                frequencies, psd = normalized_sensor_spectra(changed)
                feature_vector = robust_spectrum_features(b2, frequencies, psd, availability)
                for name in requested:
                    model_seeds = seeds if name in DEEP_MODELS else [0]
                    for seed in model_seeds:
                        if name == "b2":
                            score = b2_score(b2, frequencies, psd, availability)
                        elif name in CLASSICAL_MODELS:
                            raw = float(feature_models[(excitation, name)].score(feature_vector[None, :])[0])
                            score = calibrations[(excitation, name, 0)].normalize(raw)
                        else:
                            fitted = deep_models[(excitation, name, seed)]
                            raw = record_deep_raw(name, fitted, normalizers[excitation], psd, availability, device)
                            score = calibrations[(excitation, name, seed)].normalize(raw)
                        output.append({
                            "record_id": row.record_id,
                            "label": int(row.label),
                            "excitation": excitation,
                            "condition": condition,
                            "replicate": replicate,
                            "model": name,
                            "seed": seed,
                            "score": score,
                            "corrupted_sensors": signature,
                        })
        if record_index % 24 == 0 or record_index == len(test):
            print(f"test={record_index}/{len(test)}", flush=True)

    scores = pd.DataFrame(output)
    metrics = []
    for keys, group in scores.groupby(["model", "seed", "condition", "replicate"], sort=True):
        model_name, seed, condition, replicate = keys
        metrics.append({
            "model": model_name,
            "seed": int(seed),
            "condition": condition,
            "replicate": int(replicate),
            "records": len(group),
            "healthy_records": int((group.label == 0).sum()),
            "damage_records": int((group.label == 1).sum()),
            **metric_dict(group.label.to_numpy(), group.score.to_numpy()),
        })
    metrics_frame = pd.DataFrame(metrics)
    summary = metrics_frame.groupby(["model", "condition"]).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        recall_mean=("recall", "mean"), specificity_mean=("specificity", "mean"),
        f1_mean=("f1", "mean"), runs=("auroc", "size"),
    ).reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(training_rows).to_csv(args.output_dir / "training_summary.csv", index=False)
    artifact = {
        "protocol": "z24-healthy-only-detector-baselines-v1",
        "manifest": str(manifest),
        "mode": args.mode,
        "models": list(requested),
        "seeds": seeds,
        "epochs": args.epochs,
        "device": str(device),
        "selection_uses_damage": False,
        "window_leakage": False,
        "spectral_bins": SPECTRAL_BINS,
        "deep_input": "median sensor log-spectrum with train-only sensor-subset augmentation",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(artifact, indent=2))
    print(summary.to_string(index=False))
    print(f"DETECTOR_BASELINES_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

