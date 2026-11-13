from __future__ import annotations

"""Leakage-controlled healthy-only TSAD benchmark for the HellBridge split.

The neural networks below are compact, common-protocol architecture adapters.
They are intentionally kept in one file so a fresh Kaggle session can run the
benchmark without cloning seven mutually incompatible research environments.
They are not claimed to be byte-for-byte executions of the authors' repos; the
protocol.json output records that distinction.
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, resample
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_NAMES = (
    "pca", "lstm_ae", "usad", "gdn", "tranad", "timesnet",
    "moderntcn", "scatterad",
)
MODEL_PROVENANCE = {
    "pca": "sklearn exact PCA/SPE baseline",
    "lstm_ae": "compact architecture adapter of Malhotra et al. LSTM-EncDec-AD",
    "usad": "compact architecture adapter preserving the two-decoder USAD loss",
    "gdn": "compact graph-attention/deviation architecture adapter",
    "tranad": "compact transformer self-conditioning architecture adapter",
    "timesnet": "compact multi-period temporal-convolution architecture adapter",
    "moderntcn": "compact large-kernel depthwise TCN architecture adapter",
    "scatterad": "compact multiscale scattering plus graph architecture adapter",
}
OFFICIAL_SOURCES = {
    "lstm_ae": "https://arxiv.org/abs/1607.00148",
    "usad": "https://github.com/manigalati/usad",
    "gdn": "https://github.com/d-ailin/GDN",
    "tranad": "https://github.com/imperial-qore/TranAD",
    "timesnet": "https://github.com/thuml/TimesNet",
    "moderntcn": "https://github.com/luodhhh/ModernTCN",
    "scatterad": "https://github.com/jk-sounds/ScatterAD",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_al40(group: h5py.Group) -> np.ndarray:
    acceleration = group["acceleration"]
    values = [np.asarray(acceleration[f"AL{i:02d}"]["z"], np.float32) for i in range(1, 41)]
    length = min(map(len, values))
    return np.stack([value[:length] for value in values])


def load_windows(h5_path: Path, manifest: pd.DataFrame, length: int) -> np.ndarray:
    output = np.empty((len(manifest), 40, length), np.float32)
    with h5py.File(h5_path, "r") as source:
        total = manifest.record_id.nunique()
        for position, (record_id, rows) in enumerate(manifest.groupby("record_id", sort=False), 1):
            recording = read_al40(source[record_id])
            for row in rows.itertuples():
                window = recording[:, int(row.start):int(row.stop)]
                window = detrend(window, axis=1, type="linear")
                output[row.Index] = resample(window, length, axis=1).astype(np.float32)
            print(f"load record={position}/{total} id={record_id} windows={len(rows)}", flush=True)
    return output


def development_split(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = manifest[manifest.role.eq("train_healthy")].sort_values(["record_id", "start"])
    if len(train) != 80 or train.label.ne(0).any() or train.record_id.nunique() != 2:
        raise ValueError("Expected 80 healthy development windows from two recordings")
    fit, calibration = [], []
    for _, group in train.groupby("record_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) != 40:
            raise ValueError("Expected 40 healthy windows per development recording")
        # Interleaved blocked allocation avoids putting calibration in only one time region.
        calibration.extend(indices.reshape(5, 8)[:, -2:].ravel())
        fit.extend(indices.reshape(5, 8)[:, :-2].ravel())
    test = manifest.index[manifest.role.isin(["test_healthy", "test_damage"])].to_numpy()
    if len(fit) != 60 or len(calibration) != 20 or len(test) != 720:
        raise RuntimeError("Unexpected fit/calibration/test sizes")
    return np.asarray(fit), np.asarray(calibration), test


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        mean = values.mean(axis=(0, 2), keepdims=True)
        scale = values.std(axis=(0, 2), keepdims=True)
        return cls(mean.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)


class LSTMAE(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(channels, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        _, (hidden, cell) = self.encoder(sequence)
        repeated = hidden[-1].unsqueeze(1).expand(-1, sequence.size(1), -1)
        decoded, _ = self.decoder(repeated, (hidden, cell))
        return self.output(decoded).transpose(1, 2)


class USAD(nn.Module):
    def __init__(self, channels: int, length: int, latent: int = 32):
        super().__init__()
        dimension = channels * length
        hidden = min(256, max(64, dimension // 8))
        self.shape = (channels, length)
        self.encoder = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU())
        decoder = lambda: nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dimension))
        self.decoder1, self.decoder2 = decoder(), decoder()

    def components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.flatten(1)
        latent = self.encoder(flat)
        first = self.decoder1(latent)
        second = self.decoder2(latent)
        cycle = self.decoder2(self.encoder(first))
        return first.view_as(x), second.view_as(x), cycle.view_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.components(x)[0]


class GDNAdapter(nn.Module):
    def __init__(self, channels: int, length: int, hidden: int = 32):
        super().__init__()
        self.temporal = nn.Conv1d(channels, channels * hidden, 5, padding=2, groups=channels)
        self.embedding = nn.Parameter(torch.randn(channels, hidden) * 0.1)
        self.project = nn.Conv1d(channels * hidden, channels, 1, groups=channels)
        self.length = length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        features = torch.relu(self.temporal(x)).view(batch, channels, -1, length)
        attention = torch.softmax(self.embedding @ self.embedding.T / self.embedding.size(1) ** 0.5, dim=1)
        mixed = torch.einsum("ij,bjht->biht", attention, features)
        return self.project(mixed.reshape(batch, -1, length))


class TranADAdapter(nn.Module):
    def __init__(self, channels: int, length: int, width: int = 64):
        super().__init__()
        self.input = nn.Linear(channels * 2, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.output = nn.Linear(width, channels)
        self.position = nn.Parameter(torch.randn(1, length, width) * 0.02)

    def pass_once(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        sequence = torch.cat([x, condition], dim=1).transpose(1, 2)
        hidden = self.input(sequence) + self.position[:, :sequence.size(1)]
        return self.output(self.encoder(hidden)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.pass_once(x, torch.zeros_like(x))
        return self.pass_once(x, (x - first).pow(2).detach())


class TimesBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
            for kernel in (3, 5, 9, 17)
        ])
        self.mix = nn.Conv1d(channels * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(torch.cat([torch.relu(branch(x)) for branch in self.branches], dim=1))


class TimesNetAdapter(nn.Module):
    def __init__(self, channels: int, length: int):
        super().__init__()
        self.blocks = nn.Sequential(TimesBlock(channels), TimesBlock(channels), TimesBlock(channels))
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(x))


class ModernBlock(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise = nn.Sequential(nn.Conv1d(channels, channels * 2, 1), nn.GELU(), nn.Conv1d(channels * 2, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.norm(self.depthwise(x)))


class ModernTCNAdapter(nn.Module):
    def __init__(self, channels: int, length: int):
        super().__init__()
        self.network = nn.Sequential(ModernBlock(channels, 31), ModernBlock(channels, 15), ModernBlock(channels, 7))
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.network(x))


class ScatterADAdapter(nn.Module):
    def __init__(self, channels: int, length: int, hidden: int = 32):
        super().__init__()
        self.widths = (3, 7, 15, 31)
        for index, width in enumerate(self.widths):
            kernel = torch.zeros(channels, 1, width)
            kernel[:, 0, : width // 2] = -1.0 / max(width // 2, 1)
            kernel[:, 0, width // 2 + 1:] = 1.0 / max(width // 2, 1)
            self.register_buffer(f"wavelet_{index}", kernel)
        self.encoder = nn.Conv1d(channels * 4, channels * hidden, 1, groups=channels)
        self.embedding = nn.Parameter(torch.randn(channels, hidden) * 0.1)
        self.decoder = nn.Conv1d(channels * hidden, channels, 1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for index, width in enumerate(self.widths):
            kernel = getattr(self, f"wavelet_{index}")
            pieces.append(torch.abs(nn.functional.conv1d(x, kernel, padding=width // 2, groups=x.size(1))))
        scatter = torch.stack(pieces, dim=2)  # B,C,scale,T
        encoded = torch.relu(self.encoder(scatter.flatten(1, 2))).view(x.size(0), x.size(1), -1, x.size(2))
        graph = torch.softmax(self.embedding @ self.embedding.T / self.embedding.size(1) ** 0.5, dim=1)
        encoded = torch.einsum("ij,bjht->biht", graph, encoded)
        return self.decoder(encoded.flatten(1, 2))


def make_model(name: str, channels: int, length: int) -> nn.Module:
    return {
        "lstm_ae": lambda: LSTMAE(channels),
        "usad": lambda: USAD(channels, length),
        "gdn": lambda: GDNAdapter(channels, length),
        "tranad": lambda: TranADAdapter(channels, length),
        "timesnet": lambda: TimesNetAdapter(channels, length),
        "moderntcn": lambda: ModernTCNAdapter(channels, length),
        "scatterad": lambda: ScatterADAdapter(channels, length),
    }[name]()


def reconstruction_loss(model: nn.Module, x: torch.Tensor, epoch: int) -> torch.Tensor:
    if isinstance(model, USAD):
        first, second, cycle = model.components(x)
        weight = 1.0 / max(epoch, 1)
        loss1 = weight * (x - first).pow(2).mean() + (1 - weight) * (x - cycle).pow(2).mean()
        loss2 = weight * (x - second).pow(2).mean() - (1 - weight) * (x - cycle).pow(2).mean()
        return loss1 + loss2
    return (x - model(x)).pow(2).mean()


def fit_torch(name: str, train: np.ndarray, epochs: int, batch_size: int, device: torch.device) -> nn.Module:
    model = make_model(name, train.shape[1], train.shape[2]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(train)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model, batch, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch={epoch:03d}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    return model.eval()


@torch.no_grad()
def score_torch(model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        if isinstance(model, USAD):
            first, _, cycle = model.components(batch)
            error = 0.5 * (batch - first).pow(2) + 0.5 * (batch - cycle).pow(2)
        else:
            error = (batch - model(batch)).pow(2)
        scores.extend(error.mean(dim=(1, 2)).cpu().numpy())
    return np.asarray(scores)


def threshold_from_healthy(scores: np.ndarray, specificity: float) -> float:
    try:
        return float(np.quantile(scores, specificity, method="higher"))
    except TypeError:
        return float(np.quantile(scores, specificity, interpolation="higher"))


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = scores >= threshold
    healthy, damage = labels == 0, labels == 1
    recall = float(prediction[damage].mean())
    specificity = float((~prediction[healthy]).mean())
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": float(f1_score(labels, prediction)),
    }


def run_one(name: str, fit: np.ndarray, calibration: np.ndarray, test: np.ndarray,
            epochs: int, batch_size: int, specificity: float, device: torch.device):
    started = time.time()
    if name == "pca":
        flat = fit.reshape(len(fit), -1)
        model = PCA(n_components=min(len(fit) - 1, flat.shape[1]), svd_solver="full").fit(flat)
        def score(values):
            values = values.reshape(len(values), -1)
            restored = model.inverse_transform(model.transform(values))
            return np.mean((values - restored) ** 2, axis=1)
    else:
        model = fit_torch(name, fit, epochs, batch_size, device)
        score = lambda values: score_torch(model, values, batch_size, device)
    calibration_scores = score(calibration)
    test_scores = score(test)
    threshold = threshold_from_healthy(calibration_scores, specificity)
    return model, calibration_scores, test_scores, threshold, time.time() - started


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    aggregate = summary.groupby("model", sort=False)[["auroc", "balanced_accuracy", "recall", "specificity"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    aggregate[["auroc", "balanced_accuracy"]].plot.bar(ax=axes[0], ylim=(0, 1.02), rot=35)
    aggregate[["recall", "specificity"]].plot.bar(ax=axes[1], ylim=(0, 1.02), rot=35)
    axes[0].set_title("Ranking metrics (no point adjustment)")
    axes[1].set_title("Healthy-calibrated operating point")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--paper-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--specificity-target", type=float, default=0.90)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or (5 if args.mode == "quick" else 30)
    seeds = args.seeds or ([20261113] if args.mode == "quick" else [20261113, 20261114, 20261115])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(args.paper_split_dir / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit_index, calibration_index, test_index = development_split(manifest)
    raw = load_windows(args.h5, manifest, args.length)
    normalizer = Normalizer.fit(raw[fit_index])
    values = normalizer.transform(raw)
    labels = manifest.loc[test_index, "label"].to_numpy(int)
    print(f"device={device} fit={len(fit_index)} cal={len(calibration_index)} test={len(test_index)}", flush=True)

    metric_rows, score_rows = [], []
    for seed in seeds:
        for name in args.models:
            print(f"\n=== model={name} seed={seed} ===", flush=True)
            seed_everything(seed)
            _, cal_scores, test_scores, threshold, seconds = run_one(
                name, values[fit_index], values[calibration_index], values[test_index],
                epochs, args.batch_size, args.specificity_target, device,
            )
            result = evaluate(labels, test_scores, threshold)
            metric_rows.append({
                "model": name, "seed": seed, **result, "threshold": threshold,
                "calibration_specificity": float(np.mean(cal_scores < threshold)),
                "seconds": seconds,
            })
            for row, score in zip(manifest.loc[test_index].itertuples(), test_scores):
                score_rows.append({
                    "model": name, "seed": seed, "sample_id": row.sample_id,
                    "record_id": row.record_id, "state": row.state, "label": row.label,
                    "score": float(score), "threshold": threshold,
                })
            print(json.dumps(metric_rows[-1], indent=2), flush=True)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.DataFrame(score_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    aggregate = metrics.groupby("model", sort=False).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        recall_mean=("recall", "mean"), specificity_mean=("specificity", "mean"),
        f1_mean=("f1", "mean"), runs=("seed", "size"),
    ).reset_index()
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    make_figure(metrics, args.output_dir / "baseline_comparison.png")
    protocol = {
        "dataset": "HellBridge", "task": "healthy-only unseen-damage detection",
        "split": {"fit_healthy": 60, "calibration_healthy": 20, "test_healthy": 80, "test_damage": 640},
        "threshold": f"healthy calibration quantile targeting specificity={args.specificity_target}",
        "test_labels_used_for_training_or_threshold": False,
        "point_adjustment": False, "input": f"AL40-z, detrend, resample to {args.length}",
        "mode": args.mode, "epochs": epochs, "seeds": seeds,
        "model_provenance": MODEL_PROVENANCE, "official_sources": OFFICIAL_SOURCES,
        "warning": "Architecture adapters are common-protocol reimplementations, not byte-identical official-repository runs.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("\n=== AGGREGATE ===")
    print(aggregate.to_string(index=False))
    print(f"BASELINE_SUITE_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

