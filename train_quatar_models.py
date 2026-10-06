"""Benchmark Mamba, 1D-CNN and BiLSTM on independent QUGS recordings.

The default runs two cross-dataset folds (A -> B and B -> A). In each fold the
test dataset is never used for training, normalisation, or checkpoint selection.
This avoids the leakage caused by splitting one recording into train and test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score)
from torch.utils.data import DataLoader, Dataset

from models_seq import MambaClassifier, RNNClassifier


ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_a_dir: str = ""
    data_b_dir: str = ""
    out_dir: str = str(ROOT / "results" / "quatar_independent_benchmark")
    cache_dir: str = str(ROOT / "results" / "quatar_cache")
    models: tuple[str, ...] = ("mamba", "cnn1d", "bilstm")
    epochs: int = 15
    patience: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 48
    seed: int = 42
    window_raw: int = 2048
    downsample: int = 2
    train_windows: int = 96
    guard_windows: int = 8
    val_windows: int = 24
    num_workers: int = 0
    direction: str = "both"
    normalization: str = "per_window"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_condition(path: Path, prefix: str) -> str | None:
    match = re.fullmatch(rf"zzz{prefix}(U|D\d+)", path.stem, re.IGNORECASE)
    if not match:
        return None
    condition = match.group(1).upper()
    return "AU" if condition == "U" else f"AD{int(condition[1:])}"


def class_sort_key(name: str) -> int:
    return -1 if name == "AU" else int(name[2:])


def discover_files(data_dir: Path, prefix: str) -> list[Path]:
    matched = [(parse_condition(path, prefix), path) for path in data_dir.glob("*") if path.is_file()]
    matched = [(name, path) for name, path in matched if name is not None]
    if not matched:
        raise FileNotFoundError(
            f"No zzz{prefix}U.TXT/zzz{prefix}D*.TXT files found in {data_dir}"
        )
    return [path for _, path in sorted(matched, key=lambda item: class_sort_key(item[0]))]


def find_dataset_dir(prefix: str) -> Path | None:
    roots = [ROOT, Path("/kaggle/input")]
    candidates: dict[Path, int] = {}
    pattern = re.compile(rf"zzz{prefix}(?:U|D\d+)\.TXT$", re.IGNORECASE)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.TXT"):
            if pattern.fullmatch(path.name):
                candidates[path.parent] = candidates.get(path.parent, 0) + 1
    return max(candidates, key=candidates.get) if candidates else None


def resolve_dataset_dir(configured: str, prefix: str) -> Path:
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset {prefix} directory does not exist: {path}")
        return path
    discovered = find_dataset_dir(prefix)
    if discovered is None:
        flag = "--data-a" if prefix == "A" else "--data-b"
        raise FileNotFoundError(
            f"Dataset {prefix} was not found. Add it under /kaggle/input or pass {flag} PATH. "
            "Independent evaluation requires both Dataset A and Dataset B."
        )
    return discovered


def cache_is_current(meta_path: Path, files: list[Path], cfg: Config,
                     prefix: str, class_names: list[str]) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    signature = [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                 for p in files]
    return (meta.get("source_signature") == signature
            and meta.get("window_raw") == cfg.window_raw
            and meta.get("downsample") == cfg.downsample
            and meta.get("dataset_prefix") == prefix
            and meta.get("class_names") == class_names)


def build_cache(cfg: Config, data_dir: Path, prefix: str,
                required_classes: list[str] | None = None
                ) -> tuple[Path, np.ndarray, np.ndarray, list[str], dict]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_files(data_dir, prefix)
    by_class = {parse_condition(path, prefix): path for path in discovered}
    class_names = (sorted(by_class, key=class_sort_key) if required_classes is None
                   else list(required_classes))
    missing = [name for name in class_names if name not in by_class]
    if missing:
        raise ValueError(f"Dataset {prefix} is missing classes required by Dataset A: {missing}")
    files = [by_class[name] for name in class_names]
    tag = prefix.lower()
    x_path = cache_dir / f"inputs_{tag}.npy"
    y_path = cache_dir / f"labels_{tag}.npy"
    w_path = cache_dir / f"window_ids_{tag}.npy"
    meta_path = cache_dir / f"metadata_{tag}.json"

    if (cache_is_current(meta_path, files, cfg, prefix, class_names)
            and all(p.exists() for p in (x_path, y_path, w_path))):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (x_path, np.load(y_path), np.load(w_path), meta["class_names"], meta)

    rows_per_file = None
    n_windows = None
    output = None
    labels, window_ids = [], []
    effective_length = cfg.window_raw // cfg.downsample
    class_to_id = {name: index for index, name in enumerate(class_names)}
    print(f"Building Dataset {prefix} cache from {len(files)} files ...", flush=True)

    for path in files:
        label = class_to_id[parse_condition(path, prefix)]
        started = time.time()
        values = np.loadtxt(path, delimiter="\t", skiprows=11,
                            usecols=range(1, 31), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 30:
            raise ValueError(f"Expected 30 channels in {path.name}, got {values.shape}")
        current_windows = values.shape[0] // cfg.window_raw
        if rows_per_file is None:
            rows_per_file = values.shape[0]
            n_windows = current_windows
            output = np.lib.format.open_memmap(
                x_path, mode="w+", dtype=np.float16,
                shape=(len(files) * n_windows, effective_length, 30),
            )
        if values.shape[0] != rows_per_file or current_windows != n_windows:
            raise ValueError("All recordings must have the same number of samples")

        start = label * n_windows
        stop = start + n_windows
        windows = values[:n_windows * cfg.window_raw].reshape(n_windows, cfg.window_raw, 30)
        output[start:stop] = windows[:, ::cfg.downsample, :].astype(np.float16)
        labels.extend([label] * n_windows)
        window_ids.extend(range(n_windows))
        print(f"  {path.name:12s} -> {n_windows} windows ({time.time() - started:.1f}s)", flush=True)

    output.flush()
    del output
    labels_array = np.asarray(labels, dtype=np.int64)
    windows_array = np.asarray(window_ids, dtype=np.int16)
    np.save(y_path, labels_array)
    np.save(w_path, windows_array)
    signature = [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                 for p in files]
    meta = {
        "dataset_prefix": prefix,
        "data_dir": str(data_dir),
        "source_signature": signature,
        "class_names": class_names,
        "window_raw": cfg.window_raw,
        "downsample": cfg.downsample,
        "window_length": effective_length,
        "raw_sampling_hz": 1024,
        "effective_sampling_hz": 1024 / cfg.downsample,
        "windows_per_file": n_windows,
        "shape": [len(files) * n_windows, effective_length, 30],
        "dtype": "float16",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return x_path, labels_array, windows_array, class_names, meta


def make_train_val_split(window_ids: np.ndarray, cfg: Config,
                         train_name: str, test_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    train_end = cfg.train_windows
    val_start = train_end + cfg.guard_windows
    val_end = val_start + cfg.val_windows
    max_window = int(window_ids.max()) + 1
    if val_end > max_window:
        raise ValueError(f"A split needs {val_end} windows per class, found {max_window}")
    train = np.flatnonzero(window_ids < train_end)
    val = np.flatnonzero((window_ids >= val_start) & (window_ids < val_end))
    manifest = {
        "strategy": f"{train_name} train/validation; independent {test_name} test",
        "train_window_ids": [0, train_end - 1],
        "guard_window_ids": [train_end, val_start - 1],
        "validation_window_ids": [val_start, val_end - 1],
        "unused_a_window_ids": [val_end, max_window - 1] if val_end < max_window else [],
        "test": f"all windows from {test_name}",
        "counts": {"train": len(train), "validation": len(val)},
    }
    return train, val, manifest


def channel_stats(x_path: Path, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path, mmap_mode="r")
    total = np.zeros(x.shape[2], dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = 0
    for start in range(0, len(train_idx), 64):
        batch = np.asarray(x[train_idx[start:start + 64]], dtype=np.float32)
        total += batch.sum(axis=(0, 1), dtype=np.float64)
        total_sq += np.square(batch, dtype=np.float32).sum(axis=(0, 1), dtype=np.float64)
        count += batch.shape[0] * batch.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, x_path: Path, labels: np.ndarray, indices: np.ndarray,
                 mean: np.ndarray, std: np.ndarray, normalization: str):
        self.x = np.load(x_path, mmap_mode="r")
        self.labels = labels
        self.indices = np.asarray(indices)
        self.mean = mean[None, :]
        self.std = std[None, :]
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        sample_id = int(self.indices[position])
        x = np.asarray(self.x[sample_id], dtype=np.float32)
        if self.normalization == "per_window":
            center = x.mean(axis=0, keepdims=True)
            scale = x.std(axis=0, keepdims=True) + 1e-6
            x = (x - center) / scale
        elif self.normalization == "train_global":
            x = (x - self.mean) / self.std
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        x = np.clip(x, -10.0, 10.0)
        return torch.from_numpy(x), int(self.labels[sample_id]), sample_id


class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_cls: int, hidden: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 15, stride=2, padding=7),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, 9, stride=2, padding=4),
            nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(hidden * 2, hidden * 2, 5, padding=2),
            nn.BatchNorm1d(hidden * 2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(hidden * 2, n_cls))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.transpose(1, 2)))


def build_model(name: str, n_cls: int, hidden: int) -> nn.Module:
    if name == "mamba":
        return MambaClassifier(30, n_cls, hidden=hidden, layers=1, d_state=12,
                               stem_stride=8, dropout=0.1)
    if name == "cnn1d":
        return CNN1D(30, n_cls, hidden=hidden)
    if name == "bilstm":
        return RNNClassifier(30, n_cls, kind="lstm", hidden=hidden, layers=1,
                             stem_stride=8, dropout=0.1)
    raise ValueError(f"Unknown model {name!r}; choose mamba, cnn1d or bilstm")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> dict:
    model.eval()
    losses, targets, predictions, probabilities, ids = 0.0, [], [], [], []
    for x, y, sample_ids in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        losses += criterion(logits, y).item() * len(y)
        targets.append(y.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ids.append(sample_ids.numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": losses / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "probability": np.concatenate(probabilities),
        "sample_ids": np.concatenate(ids),
    }


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize="true")
    size = max(9, len(class_names) * 0.42)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one(name: str, cfg: Config, loaders: dict[str, DataLoader], class_names: list[str],
              device: torch.device, run_dir: Path, fold_name: str) -> dict:
    set_seed(cfg.seed)
    model_dir = run_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(name, len(class_names), cfg.hidden).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_loss, best_epoch, stale = math.inf, 0, 0
    checkpoint = model_dir / "best.pt"
    started = time.time()

    print(f"\n{name.upper()} ({parameters:,} parameters)", flush=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, seen, correct = 0.0, 0, 0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        validation = evaluate(model, loaders["validation"], criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss / seen,
               "train_accuracy": correct / seen, "val_loss": validation["loss"],
               "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"]}
        history.append(row)
        print(f"  epoch {epoch:02d} | train {row['train_accuracy']:.3f} | "
              f"val {row['val_accuracy']:.3f} | val F1 {row['val_macro_f1']:.3f}", flush=True)

        if validation["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "class_names": class_names,
                        "config": asdict(cfg), "epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"  early stopping after epoch {epoch}", flush=True)
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    test = evaluate(model, loaders["test"], criterion, device)
    elapsed = time.time() - started
    recording_y = []
    recording_pred = []
    recording_probability = []
    for class_id in range(len(class_names)):
        positions = np.flatnonzero(test["y_true"] == class_id)
        if not len(positions):
            raise ValueError(f"Independent test set has no samples for {class_names[class_id]}")
        probability = test["probability"][positions].mean(axis=0, dtype=np.float64)
        probability /= probability.sum()
        recording_y.append(class_id)
        recording_pred.append(int(probability.argmax()))
        recording_probability.append(probability)
    recording_y = np.asarray(recording_y, dtype=np.int64)
    recording_pred = np.asarray(recording_pred, dtype=np.int64)
    recording_probability = np.asarray(recording_probability)
    recording_accuracy = accuracy_score(recording_y, recording_pred)
    recording_macro_f1 = f1_score(recording_y, recording_pred, average="macro", zero_division=0)
    report = classification_report(test["y_true"], test["y_pred"], labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    np.savez(model_dir / "test_predictions.npz",
             y_true=test["y_true"], y_pred=test["y_pred"], probability=test["probability"],
             sample_ids=test["sample_ids"], recording_y=recording_y,
             recording_pred=recording_pred, recording_probability=recording_probability)
    save_confusion(test["y_true"], test["y_pred"], class_names, model_dir / "confusion_matrix.png")
    (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metrics = {"fold": fold_name, "model": name, "parameters": parameters, "best_epoch": best_epoch,
               "training_seconds": elapsed, "test_loss": test["loss"],
               "test_accuracy": test["accuracy"],
               "test_balanced_accuracy": test["balanced_accuracy"],
               "test_macro_f1": test["macro_f1"],
               "recording_accuracy": recording_accuracy,
               "recording_macro_f1": recording_macro_f1,
               "per_class": report}
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  DATASET B window accuracy={test['accuracy']:.3f}, macro-F1={test['macro_f1']:.3f}",
          flush=True)
    print(f"  DATASET B recording accuracy={recording_accuracy:.3f}, "
          f"macro-F1={recording_macro_f1:.3f}", flush=True)
    return metrics


def save_summary(results: list[dict], run_dir: Path) -> None:
    fields = ["fold", "model", "parameters", "best_epoch", "training_seconds", "test_loss",
              "test_accuracy", "test_balanced_accuracy", "test_macro_f1",
              "recording_accuracy", "recording_macro_f1"]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in results])
    names = [f"{row['fold']}\n{row['model']}" for row in results]
    accuracy = [row["test_accuracy"] for row in results]
    macro_f1 = [row["test_macro_f1"] for row in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
    ax.bar(x - 0.18, accuracy, 0.36, label="Accuracy")
    ax.bar(x + 0.18, macro_f1, 0.36, label="Macro-F1")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("QUGS cross-dataset test results")
    ax.legend()
    ax.bar_label(ax.containers[0], fmt="%.3f")
    ax.bar_label(ax.containers[1], fmt="%.3f")
    fig.tight_layout()
    fig.savefig(run_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def run_fold(cfg: Config, fold_name: str, train_bundle: tuple, test_bundle: tuple,
             class_names: list[str], device: torch.device, run_dir: Path) -> list[dict]:
    x_train_path, labels_train, windows_train, train_meta = train_bundle
    x_test_path, labels_test, _, test_meta = test_bundle
    train_idx, val_idx, manifest = make_train_val_split(
        windows_train, cfg, train_meta["dataset_prefix"], test_meta["dataset_prefix"]
    )
    test_idx = np.arange(len(labels_test), dtype=np.int64)
    per_class_counts = {}
    for split_name, labels, indices in (
        ("train", labels_train, train_idx),
        ("validation", labels_train, val_idx),
        ("independent_test", labels_test, test_idx),
    ):
        counts = np.bincount(labels[indices], minlength=len(class_names))
        if not np.all(counts == counts[0]):
            raise AssertionError(f"Unbalanced {fold_name}/{split_name}: {counts.tolist()}")
        per_class_counts[split_name] = int(counts[0])

    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest.update({
        "fold": fold_name,
        "per_class_counts": per_class_counts,
        "classes": class_names,
        "window_seconds": cfg.window_raw / 1024,
        "normalization": cfg.normalization,
        "train_dataset": train_meta,
        "test_dataset": test_meta,
    })
    (fold_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mean, std = channel_stats(x_train_path, train_idx)
    np.savez(fold_dir / "normalization.npz", mean=mean, std=std,
             mode=np.asarray(cfg.normalization))
    datasets = {
        "train": WindowDataset(x_train_path, labels_train, train_idx, mean, std, cfg.normalization),
        "validation": WindowDataset(x_train_path, labels_train, val_idx, mean, std, cfg.normalization),
        "test": WindowDataset(x_test_path, labels_test, test_idx, mean, std, cfg.normalization),
    }
    loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=name == "train",
                         num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                         persistent_workers=cfg.num_workers > 0)
        for name, ds in datasets.items()
    }
    print(f"\n=== {fold_name}: Dataset {train_meta['dataset_prefix']} -> "
          f"Dataset {test_meta['dataset_prefix']} ===")
    print(f"Samples: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    return [train_one(name, cfg, loaders, class_names, device, fold_dir, fold_name)
            for name in cfg.models]


def save_model_aggregate(results: list[dict], run_dir: Path) -> None:
    fields = ["model", "folds", "window_accuracy_mean", "window_accuracy_std",
              "macro_f1_mean", "macro_f1_std", "recording_accuracy_mean",
              "recording_accuracy_std"]
    rows = []
    for model in sorted({row["model"] for row in results}):
        selected = [row for row in results if row["model"] == model]
        accuracy = np.asarray([row["test_accuracy"] for row in selected])
        macro_f1 = np.asarray([row["test_macro_f1"] for row in selected])
        recording = np.asarray([row["recording_accuracy"] for row in selected])
        rows.append({
            "model": model,
            "folds": len(selected),
            "window_accuracy_mean": accuracy.mean(),
            "window_accuracy_std": accuracy.std(ddof=0),
            "macro_f1_mean": macro_f1.mean(),
            "macro_f1_std": macro_f1.std(ddof=0),
            "recording_accuracy_mean": recording.mean(),
            "recording_accuracy_std": recording.std(ddof=0),
        })
    with (run_dir / "summary_by_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: Config | None = None) -> tuple[Path, list[dict]]:
    cfg = cfg or Config()
    if cfg.direction not in {"a_to_b", "b_to_a", "both"}:
        raise ValueError("direction must be a_to_b, b_to_a, or both")
    if cfg.normalization not in {"per_window", "train_global"}:
        raise ValueError("normalization must be per_window or train_global")
    set_seed(cfg.seed)
    data_a_dir = resolve_dataset_dir(cfg.data_a_dir, "A")
    data_b_dir = resolve_dataset_dir(cfg.data_b_dir, "B")
    run_dir = Path(cfg.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    x_a, y_a, w_a, class_names, meta_a = build_cache(cfg, data_a_dir, "A")
    x_b, y_b, w_b, _, meta_b = build_cache(cfg, data_b_dir, "B", required_classes=class_names)
    bundles = {
        "A": (x_a, y_a, w_a, meta_a),
        "B": (x_b, y_b, w_b, meta_b),
    }
    folds = []
    if cfg.direction in {"a_to_b", "both"}:
        folds.append(("a_to_b", bundles["A"], bundles["B"]))
    if cfg.direction in {"b_to_a", "both"}:
        folds.append(("b_to_a", bundles["B"], bundles["A"]))

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Dataset A: {data_a_dir}")
    print(f"Dataset B: {data_b_dir}")
    print(f"Normalization: {cfg.normalization}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    results = []
    for fold_name, train_bundle, test_bundle in folds:
        results.extend(run_fold(cfg, fold_name, train_bundle, test_bundle,
                                class_names, device, run_dir))
    save_summary(results, run_dir)
    save_model_aggregate(results, run_dir)
    print("\nCROSS-DATASET RESULTS")
    for row in results:
        print(f"  {row['fold']:6s} {row['model']:7s} window_accuracy={row['test_accuracy']:.3f} "
              f"recording_accuracy={row['recording_accuracy']:.3f} "
              f"macro-F1={row['test_macro_f1']:.3f} time={row['training_seconds']:.1f}s")
    print(f"Saved to: {run_dir}")
    return run_dir, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-a", default="",
                        help="Dataset A directory. Auto-detected under the repo or /kaggle/input when omitted.")
    parser.add_argument("--data-b", default="",
                        help="Dataset B directory. Auto-detected under the repo or /kaggle/input when omitted.")
    parser.add_argument("--out-dir", default="",
                        help="Output directory (default: /kaggle/working/qugs_results on Kaggle).")
    parser.add_argument("--cache-dir", default="",
                        help="Preprocessed cache directory (default: /kaggle/working/qugs_cache on Kaggle).")
    parser.add_argument("--models", nargs="+", choices=["mamba", "cnn1d", "bilstm"],
                        default=["mamba", "cnn1d", "bilstm"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--direction", choices=["a_to_b", "b_to_a", "both"], default="both",
                        help="Cross-dataset evaluation direction (default: both directions).")
    parser.add_argument("--normalization", choices=["per_window", "train_global"],
                        default="per_window",
                        help="Per-window sensor z-score removes acquisition gain fingerprints.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    on_kaggle = Path("/kaggle/working").is_dir()
    default_out = "/kaggle/working/qugs_results" if on_kaggle else str(ROOT / "results" / "quatar_independent_benchmark")
    default_cache = "/kaggle/working/qugs_cache" if on_kaggle else str(ROOT / "results" / "quatar_cache")
    run(Config(data_a_dir=args.data_a, data_b_dir=args.data_b,
               out_dir=args.out_dir or default_out, cache_dir=args.cache_dir or default_cache,
               models=tuple(args.models), epochs=args.epochs, patience=args.patience,
               batch_size=args.batch_size, hidden=args.hidden,
               learning_rate=args.learning_rate, seed=args.seed,
               num_workers=args.num_workers, direction=args.direction,
               normalization=args.normalization))
