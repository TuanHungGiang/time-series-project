"""Train Mamba on 10000-point Z24 windows.

The default temporal-holdout split uses three early windows for training, one for
validation, leaves a full window as a gap, and tests on the final window from every
(scenario, setup) recording. Use the grouped modes for stricter generalisation tests.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models_seq import build_model
from report import (compute_metrics, plot_class_accuracy, plot_class_setup_accuracy,
                    plot_confusion, plot_roc, plot_training_curves, plot_tsne)


@dataclass
class TrainConfig:
    data_dir: str = str(ROOT / "Z24-dataset-processed")
    out_dir: str = str(ROOT / "results")
    run_name: str = "mamba_temporal_holdout_seed42"
    epochs: int = 100
    patience: int = 12
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 64
    layers: int = 2
    stem_stride: int = 25
    clip_value: float = 10.0
    grad_clip: float = 4.0
    seed: int = 42
    num_workers: int = 0
    log_every: int = 10
    max_train_batches: int = 0
    skip_tsne: bool = False
    use_augmentation: bool = True
    split_mode: str = "temporal_holdout"


class RunLogger:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")

    def __call__(self, message=""):
        print(message, flush=True)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_10k_data(data_dir: Path) -> Path:
    """Rebuild (918, 27, 10000) once from the processed (1530, 27, 6000) source."""
    output_path = data_dir / "inputs_10000.npy"
    if output_path.exists():
        return output_path

    source = np.load(data_dir / "inputs.npy", mmap_mode="r")
    if source.shape != (1530, 27, 6000):
        raise ValueError(f"Expected source shape (1530, 27, 6000), got {source.shape}")
    source_5d = source.reshape(17, 9, 10, 27, 6000)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=source.dtype, shape=(918, 27, 10000)
    )
    for scenario in range(17):
        for setup in range(9):
            recording = source_5d[scenario, setup].transpose(1, 0, 2).reshape(27, 60000)
            first = (scenario * 9 + setup) * 6
            output[first:first + 6] = recording.reshape(27, 6, 10000).transpose(1, 0, 2)
    output.flush()
    del output
    return output_path


def metadata():
    y = np.repeat(np.arange(17, dtype=np.int64), 9 * 6)
    setup = np.broadcast_to(np.arange(9)[None, :, None], (17, 9, 6)).reshape(-1)
    window = np.broadcast_to(np.arange(6)[None, None, :], (17, 9, 6)).reshape(-1)
    group = np.repeat(np.arange(17 * 9), 6)
    return y, setup, window, group


def split_indices(labels: np.ndarray, setup: np.ndarray, window: np.ndarray, group: np.ndarray,
                  mode="temporal_holdout"):
    """Create the requested split while keeping class counts exactly balanced."""
    if mode == "temporal_holdout":
        train_idx = np.where(window <= 2)[0]
        val_idx = np.where(window == 3)[0]
        test_idx = np.where(window == 5)[0]
    elif mode == "within_recording":
        train_idx = np.where(window <= 3)[0]
        val_idx = np.where(window == 4)[0]
        test_idx = np.where(window == 5)[0]
    elif mode == "unseen_setup":
        train_idx = np.where(setup <= 5)[0]
        val_idx = np.where(setup == 6)[0]
        test_idx = np.where(setup >= 7)[0]
    elif mode == "balanced":
        split_code = np.full(len(labels), -1, dtype=np.int8)  # 0=train, 1=val, 2=test
        for class_id in range(17):
            val_setup = class_id % 9
            test_setups = {(class_id + 1) % 9, (class_id + 2) % 9}
            selected = labels == class_id
            split_code[selected] = 0
            split_code[selected & (setup == val_setup)] = 1
            split_code[selected & np.isin(setup, list(test_setups))] = 2
        train_idx = np.where(split_code == 0)[0]
        val_idx = np.where(split_code == 1)[0]
        test_idx = np.where(split_code == 2)[0]
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    if mode in ("temporal_holdout", "within_recording"):
        assert set(group[train_idx]) == set(group[val_idx]) == set(group[test_idx])
        expected_train = 27 if mode == "temporal_holdout" else 36
        assert np.all(np.bincount(labels[train_idx], minlength=17) == expected_train)
        assert np.all(np.bincount(labels[val_idx], minlength=17) == 9)
        assert np.all(np.bincount(labels[test_idx], minlength=17) == 9)
    else:
        assert set(group[train_idx]).isdisjoint(group[val_idx])
        assert set(group[train_idx]).isdisjoint(group[test_idx])
        assert set(group[val_idx]).isdisjoint(group[test_idx])
        assert np.all(np.bincount(labels[train_idx], minlength=17) == 36)
        assert np.all(np.bincount(labels[val_idx], minlength=17) == 6)
        assert np.all(np.bincount(labels[test_idx], minlength=17) == 12)
    return train_idx, val_idx, test_idx


def save_split_manifest(path, labels, setup, window, group, train_idx, val_idx, test_idx, mode):
    membership = np.full(len(labels), "", dtype="<U10")
    membership[train_idx] = "train"
    membership[val_idx] = "validation"
    membership[test_idx] = "test"
    rows = []
    for recording_id in np.unique(group):
        positions = np.where(group == recording_id)[0]
        rows.append({
            "recording_id": int(recording_id),
            "scenario": int(labels[positions[0]]),
            "setup": int(setup[positions[0]]),
            "train_windows": window[positions][membership[positions] == "train"].astype(int).tolist(),
            "validation_windows": window[positions][membership[positions] == "validation"].astype(int).tolist(),
            "test_windows": window[positions][membership[positions] == "test"].astype(int).tolist(),
            "unused_windows": window[positions][membership[positions] == ""].astype(int).tolist(),
        })
    path.write_text(json.dumps({"mode": mode, "recordings": rows}, indent=2), encoding="utf-8")


def robust_train_stats(x_path: Path, train_idx: np.ndarray):
    x = np.load(x_path, mmap_mode="r")
    center = np.empty((1, x.shape[1], 1), dtype=np.float32)
    scale = np.empty_like(center)
    for channel in range(x.shape[1]):
        channel_data = np.asarray(x[train_idx, channel, :])
        center[0, channel, 0] = np.median(channel_data)
        scale[0, channel, 0] = np.median(channel_data.std(axis=1)) + 1e-8
    return center, scale


class Z24Dataset(Dataset):
    def __init__(self, x_path, labels, indices, center, scale, clip_value):
        self.x = np.load(x_path, mmap_mode="r")
        self.labels = labels
        self.indices = np.asarray(indices)
        self.center = center[0]
        self.scale = scale[0]
        self.clip_value = clip_value

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        sample_id = int(self.indices[position])
        x = (np.array(self.x[sample_id], dtype=np.float32, copy=True) - self.center) / self.scale
        np.clip(x, -self.clip_value, self.clip_value, out=x)
        return torch.from_numpy(x), int(self.labels[sample_id]), sample_id


def augment_train_batch(x: torch.Tensor) -> torch.Tensor:
    """Hard-but-label-preserving augmentation. Shape remains (B, C, 10000)."""
    batch, channels, length = x.shape
    out = x.clone()
    selected = torch.rand(batch, device=x.device) >= 0.20  # retain 20% clean anchors
    if not selected.any():
        return out

    ids = selected.nonzero(as_tuple=True)[0]
    work = out[ids]
    n = len(ids)

    # Additive sensor noise after robust normalization.
    apply = torch.rand(n, 1, 1, device=x.device) < 0.70
    sigma = torch.empty(n, 1, 1, device=x.device).uniform_(0.01, 0.06)
    work = work + apply * sigma * torch.randn_like(work)

    # One gain value per sensor over the whole sequence.
    apply = torch.rand(n, 1, 1, device=x.device) < 0.70
    gain = torch.empty(n, channels, 1, device=x.device).uniform_(0.85, 1.15)
    work = work * torch.where(apply, gain, torch.ones_like(gain))

    # Smooth baseline drift; no resampling and no change in length.
    apply = torch.rand(n, 1, 1, device=x.device) < 0.30
    ramp = torch.linspace(-1.0, 1.0, length, device=x.device)[None, None, :]
    slope = torch.empty(n, channels, 1, device=x.device).uniform_(-0.04, 0.04)
    work = work + apply * slope * ramp

    for i in range(n):
        # Circular phase shift keeps all samples.
        if torch.rand((), device=x.device) < 0.50:
            shift = int(torch.randint(-length // 10, length // 10 + 1, (), device=x.device))
            work[i] = torch.roll(work[i], shift, dims=-1)

        # Short missing interval, shared by all sensors.
        if torch.rand((), device=x.device) < 0.35:
            width = int(torch.randint(max(1, length // 200), max(2, length // 20), (), device=x.device))
            start = int(torch.randint(0, length - width + 1, (), device=x.device))
            work[i, :, start:start + width] = 0.0

        # Up to three failed sensors; zero is the normalized median.
        if torch.rand((), device=x.device) < 0.35:
            n_drop = int(torch.randint(1, min(3, channels) + 1, (), device=x.device))
            dropped = torch.randperm(channels, device=x.device)[:n_drop]
            work[i, dropped] = 0.0

    out[ids] = work.clamp_(-10.0, 10.0)
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_embeddings=False):
    model.eval()
    total_loss = 0.0
    labels, predictions, probabilities, embeddings, sample_ids = [], [], [], [], []
    for x, y, ids in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z = model.embed(x.transpose(1, 2))
        logits = model.head(z)
        total_loss += criterion(logits, y).item() * len(y)
        labels.append(y.cpu())
        predictions.append(logits.argmax(1).cpu())
        probabilities.append(torch.softmax(logits.float(), 1).cpu())
        sample_ids.append(ids)
        if return_embeddings:
            embeddings.append(z.cpu())
    result = {
        "loss": total_loss / len(loader.dataset),
        "y": torch.cat(labels).numpy(),
        "pred": torch.cat(predictions).numpy(),
        "prob": torch.cat(probabilities).numpy(),
        "ids": torch.cat(sample_ids).numpy(),
    }
    result["accuracy"] = float((result["pred"] == result["y"]).mean())
    if return_embeddings:
        result["embeddings"] = torch.cat(embeddings).numpy()
    return result


def aggregate_recordings(window_result, group_ids, setup_ids):
    """Average six window probabilities/embeddings into one independent recording."""
    ids = window_result["ids"]
    sample_groups = group_ids[ids]
    records = {key: [] for key in ("y", "pred", "prob", "ids", "setup", "window_count")}
    if "embeddings" in window_result:
        records["embeddings"] = []

    for recording_id in np.unique(sample_groups):
        positions = np.where(sample_groups == recording_id)[0]
        labels = np.unique(window_result["y"][positions])
        setups = np.unique(setup_ids[ids[positions]])
        if len(labels) != 1 or len(setups) != 1:
            raise ValueError(f"Recording {recording_id} has inconsistent labels or setups")

        probability = window_result["prob"][positions].mean(axis=0, dtype=np.float64)
        probability /= probability.sum()
        records["y"].append(int(labels[0]))
        records["pred"].append(int(probability.argmax()))
        records["prob"].append(probability)
        records["ids"].append(int(recording_id))
        records["setup"].append(int(setups[0]))
        records["window_count"].append(len(positions))
        if "embeddings" in window_result:
            records["embeddings"].append(window_result["embeddings"][positions].mean(axis=0))

    for key in records:
        records[key] = np.asarray(records[key])
    true_probability = records["prob"][np.arange(len(records["y"])), records["y"]]
    records["loss"] = float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean())
    records["accuracy"] = float((records["pred"] == records["y"]).mean())
    return records


def save_tsne(model, loader, criterion, device, setup, group, run_dir, name, seed):
    window_result = evaluate(model, loader, criterion, device, return_embeddings=True)
    result = aggregate_recordings(window_result, group, setup)
    target_dir = run_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        target_dir / "window_embeddings.npz", embeddings=window_result["embeddings"],
        y=window_result["y"], setup=setup[window_result["ids"]], sample_ids=window_result["ids"],
    )
    np.savez(
        target_dir / "embeddings.npz",
        embeddings=result["embeddings"], y=result["y"], setup=result["setup"],
        recording_ids=result["ids"], window_count=result["window_count"],
    )
    plot_tsne(result["embeddings"], result["y"], result["setup"], target_dir, seed=seed)
    return result


def train(config: TrainConfig | None = None):
    cfg = config or TrainConfig()
    set_seed(cfg.seed)
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir / "trainlog.txt")
    t0_all = time.time()

    try:
        logger("Mamba Z24 training")
        logger(json.dumps(asdict(cfg), indent=2))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger(f"device={device} | gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")

        data_dir = Path(cfg.data_dir)
        x_path = ensure_10k_data(data_dir)
        x_shape = np.load(x_path, mmap_mode="r").shape
        y, setup, window, group = metadata()
        train_idx, val_idx, test_idx = split_indices(y, setup, window, group, cfg.split_mode)
        save_split_manifest(run_dir / "split_manifest.json", y, setup, window, group,
                            train_idx, val_idx, test_idx, cfg.split_mode)
        logger(f"data={x_shape} | windows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        logger(f"source recordings represented: train={len(np.unique(group[train_idx]))} "
               f"val={len(np.unique(group[val_idx]))} test={len(np.unique(group[test_idx]))}")
        if cfg.split_mode == "temporal_holdout":
            logger("split=temporal_holdout: windows 0-2 train | 3 validation | 4 gap/unused | 5 test; "
                   "all classes/setups occur in every split")
            logger("note: splits contain temporally separated windows from the same source recordings")
        elif cfg.split_mode == "within_recording":
            logger("split=within_recording: windows 0-3 train | 4 validation | 5 test; "
                   "all classes/setups occur in every split")
            logger("note: splits contain different windows from the same source recordings")
        elif cfg.split_mode == "balanced":
            logger("split=balanced: held-out setups rotate by scenario; every setup appears globally in train")
        else:
            logger("split=unseen_setup: train setups 0-5 | validation setup 6 | test setups 7-8")

        logger("computing robust normalization from train only ...")
        center, scale = robust_train_stats(x_path, train_idx)
        np.savez(run_dir / "normalization.npz", center=center, scale=scale)

        datasets = {
            "train": Z24Dataset(x_path, y, train_idx, center, scale, cfg.clip_value),
            "val": Z24Dataset(x_path, y, val_idx, center, scale, cfg.clip_value),
            "test": Z24Dataset(x_path, y, test_idx, center, scale, cfg.clip_value),
        }
        generator = torch.Generator().manual_seed(cfg.seed)
        common = dict(
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )
        loaders = {
            "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True,
                                generator=generator, **common),
            "val": DataLoader(datasets["val"], batch_size=max(cfg.batch_size, 8), shuffle=False, **common),
            "test": DataLoader(datasets["test"], batch_size=max(cfg.batch_size, 8), shuffle=False, **common),
        }

        model_cfg = SimpleNamespace(enc_in=27, num_class=17, seq_len=10000)
        model_args = SimpleNamespace(hidden=cfg.hidden, layers=cfg.layers, stem_stride=cfg.stem_stride)
        model = build_model("mamba", model_cfg, model_args).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger(f"model=mamba | params={n_params:,} | stem_stride={cfg.stem_stride}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RAdam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                   "val_window_loss": [], "val_window_acc": [], "lr": []}
        best_val_loss = math.inf
        best_epoch = 0
        bad_epochs = 0
        last_epoch = 0

        for epoch in range(1, cfg.epochs + 1):
            last_epoch = epoch
            epoch_start = time.time()
            lr = 0.5 * cfg.learning_rate * (1 + math.cos(math.pi * (epoch - 1) / cfg.epochs))
            for group_ in optimizer.param_groups:
                group_["lr"] = lr

            model.train()
            running_loss = 0.0
            correct = 0
            seen = 0
            batches_run = 0
            for batch_no, (x, target, _) in enumerate(loaders["train"], start=1):
                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if cfg.use_augmentation:
                    x = augment_train_batch(x)  # train only
                optimizer.zero_grad(set_to_none=True)
                logits = model(x.transpose(1, 2))
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                running_loss += loss.item() * len(target)
                correct += (logits.argmax(1) == target).sum().item()
                seen += len(target)
                batches_run += 1
                if batch_no % cfg.log_every == 0:
                    logger(f"epoch {epoch:03d} batch {batch_no:03d}/{len(loaders['train'])} "
                           f"loss={running_loss/seen:.4f} acc={correct/seen:.4f}")
                if cfg.max_train_batches and batch_no >= cfg.max_train_batches:
                    break

            train_loss = running_loss / seen
            train_acc = correct / seen
            val_window = evaluate(model, loaders["val"], criterion, device)
            val_result = aggregate_recordings(val_window, group, setup)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_result["loss"])
            history["val_acc"].append(val_result["accuracy"])
            history["val_window_loss"].append(val_window["loss"])
            history["val_window_acc"].append(val_window["accuracy"])
            history["lr"].append(lr)
            logger(f"epoch {epoch:03d}/{cfg.epochs} lr={lr:.3e} train_loss={train_loss:.4f} "
                   f"train_acc={train_acc:.4f} val_record_loss={val_result['loss']:.4f} "
                   f"val_record_acc={val_result['accuracy']:.4f} "
                   f"val_window_acc={val_window['accuracy']:.4f} time={time.time()-epoch_start:.1f}s")

            if epoch == 1 and not cfg.skip_tsne:
                logger("saving test embeddings and t-SNE after epoch 1 ...")
                save_tsne(model, loaders["test"], criterion, device, setup, group, run_dir,
                          "tsne_epoch_001", cfg.seed)

            torch.save(model.state_dict(), run_dir / "mamba_final.pt")
            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), run_dir / "mamba_best.pt")
                logger("  -> saved mamba_best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                    break

        if not cfg.skip_tsne:
            logger(f"saving test embeddings and t-SNE after final epoch {last_epoch} ...")
            save_tsne(model, loaders["test"], criterion, device, setup, group, run_dir,
                      "tsne_epoch_final", cfg.seed)

        # Grouped modes average windows from a recording. In within-recording mode each
        # validation/test recording contributes exactly one held-out window.
        model.load_state_dict(torch.load(run_dir / "mamba_best.pt", map_location=device, weights_only=True))
        val_window = evaluate(model, loaders["val"], criterion, device)
        test_window = evaluate(model, loaders["test"], criterion, device)
        val_result = aggregate_recordings(val_window, group, setup)
        test_result = aggregate_recordings(test_window, group, setup)
        val_metrics = compute_metrics(val_result["y"], val_result["pred"], val_result["prob"], 17)
        test_metrics = compute_metrics(test_result["y"], test_result["pred"], test_result["prob"], 17)
        val_window_metrics = compute_metrics(val_window["y"], val_window["pred"], val_window["prob"], 17)
        test_window_metrics = compute_metrics(test_window["y"], test_window["pred"], test_window["prob"], 17)
        metrics = {
            "model": "mamba",
            "difficulty": ({"temporal_holdout": "medium_temporal_holdout",
                            "within_recording": "easy_within_recording",
                            "balanced": "medium_balanced_group",
                            "unseen_setup": "hard_unseen_setup"}[cfg.split_mode]),
            "split_mode": cfg.split_mode,
            "primary_evaluation_unit": ("held_out_window_per_recording"
                                        if cfg.split_mode in ("temporal_holdout", "within_recording")
                                        else "recording"),
            "input_shape": [27, 10000],
            "augmentation": "train_only" if cfg.use_augmentation else "disabled",
            "parameters": n_params,
            "best_epoch": best_epoch,
            "final_epoch": last_epoch,
            "split_sizes_windows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
            "unused_windows": int(len(y) - len(train_idx) - len(val_idx) - len(test_idx)),
            "source_recordings_represented": {
                "train": len(np.unique(group[train_idx])),
                "validation": len(np.unique(group[val_idx])),
                "test": len(np.unique(group[test_idx])),
            },
            "validation": val_metrics,
            "test": test_metrics,
            "validation_window": val_window_metrics,
            "test_window": test_window_metrics,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        np.savez(run_dir / "predictions_test.npz", y_true=test_window["y"], y_pred=test_window["pred"],
                 probability=test_window["prob"], sample_ids=test_window["ids"],
                 recording_ids=group[test_window["ids"]], setup=setup[test_window["ids"]])
        np.savez(run_dir / "predictions_test_recording.npz", y_true=test_result["y"],
                 y_pred=test_result["pred"], probability=test_result["prob"],
                 recording_ids=test_result["ids"], setup=test_result["setup"],
                 window_count=test_result["window_count"])

        plot_training_curves(history, run_dir / "training_curves.png")
        plot_confusion(test_result["y"], test_result["pred"], 17,
                       run_dir / "confusion_matrix.png", title="Mamba test recordings -")
        plot_confusion(test_window["y"], test_window["pred"], 17,
                       run_dir / "confusion_matrix_window.png", title="Mamba test windows -")
        plot_roc(test_result["y"], test_result["prob"], 17, run_dir / "roc_curve.png")
        plot_roc(test_window["y"], test_window["prob"], 17, run_dir / "roc_curve_window.png")
        plot_class_accuracy(test_result["y"], test_result["pred"], 17,
                            run_dir / "accuracy_per_class.png")
        plot_class_setup_accuracy(test_result["y"], test_result["pred"], test_result["setup"], 17,
                                  run_dir / "accuracy_per_class_by_setup.png")

        logger("\nTEST PRIMARY METRICS (mamba_best.pt)")
        for key, value in test_metrics.items():
            if key != "per_class":
                logger(f"{key:24s}: {value}")
        logger("\nACCURACY BY CLASS")
        for class_id, values in test_metrics["per_class"].items():
            logger(f"class {int(class_id):02d}: {values['recall']:.4f} "
                   f"({round(values['recall'] * values['support'])}/{values['support']})")
        logger(f"\nwindow_accuracy (secondary): {test_window_metrics['accuracy']}")
        logger(f"elapsed_minutes: {(time.time()-t0_all)/60:.2f}")
        logger(f"artifacts: {run_dir}")
        return run_dir, metrics
    finally:
        logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--stem-stride", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (use 2-4 on Kaggle, 0 on Windows if needed)")
    parser.add_argument("--run-name", default="mamba_temporal_holdout_seed42")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable all training augmentation for an ablation run")
    parser.add_argument("--split-mode",
                        choices=("temporal_holdout", "within_recording", "balanced", "unseen_setup"),
                        default="temporal_holdout",
                        help="temporal_holdout is the middle-difficulty default; grouped modes are stricter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden=args.hidden,
        layers=args.layers,
        stem_stride=args.stem_stride,
        seed=args.seed,
        num_workers=args.num_workers,
        run_name=args.run_name,
        max_train_batches=args.max_train_batches,
        skip_tsne=args.skip_tsne,
        use_augmentation=not args.no_augment,
        split_mode=args.split_mode,
    ))
