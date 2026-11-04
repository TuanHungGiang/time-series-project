"""Shared-sensor ScatterAD (NeurIPS 2025) baseline for the Z24 record split.

The author architecture, objective and anomaly score are retained. The data
adapter applies the same model to univariate sensor windows pooled from all
healthy training records, avoiding invalid channel-index alignment across the
roving Z24 setups. Results must be described as a shared univariate-sensor
adapter, not an exact reproduction of the paper's fixed-dimensional benchmarks.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCATTER_ROOT = ROOT / "baselines_repo" / "ScatterAD"
sys.path.insert(0, str(SCATTER_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from model.ScatterAD import Model


SCENARIOS = {
    "01": "Undamaged reference", "02": "Pier-settlement system installation",
    "03": "Pier lowering 20 mm", "04": "Pier lowering 40 mm",
    "05": "Pier lowering 80 mm", "06": "Pier lowering 95 mm",
    "07": "Foundation tilt", "08": "New reference after lifting",
    "09": "Concrete spalling 12 m2", "10": "Concrete spalling 24 m2",
    "11": "Abutment landslide", "12": "Concrete hinge failure",
    "13": "Two anchor heads failure", "14": "Four anchor heads failure",
    "15": "Two tendons ruptured", "16": "Four tendons ruptured",
    "17": "Six tendons ruptured",
}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def vertical_data(path: Path) -> tuple[np.ndarray, list[str]]:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    labels = [str(value).strip() for value in mat["labelshulp"].tolist()]
    selected = [
        index for index, label in enumerate(labels)
        if label.endswith("V") and label[0].isdigit() and not label.startswith(("R", "DP"))
    ]
    names = [labels[index] for index in selected]
    values = np.asarray(mat["data"][:, selected], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"NaN/Inf in {path}")
    # Match the 50 Hz protocol used by the earlier official-code Z24 adapters.
    return resample_poly(values, 1, 2, axis=0).astype(np.float32), names


def resolve_manifest(manifest_path: Path, data_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, dtype={"state": str})
    frame["state"] = frame.state.str.zfill(2)
    frame = frame[frame.split != "excluded"].copy()
    if "relative_path" in frame:
        frame["path"] = frame.relative_path.map(lambda value: (data_root / value).resolve())
    elif "path" in frame:
        frame["path"] = frame.path.map(Path)
    else:
        raise ValueError("Manifest needs relative_path or path")
    missing = frame.loc[~frame.path.map(Path.exists), "path"]
    if len(missing):
        raise FileNotFoundError(f"Missing {len(missing)} MAT files; first={missing.iloc[0]}")
    return frame


def fit_sensor_scaler(train_rows: pd.DataFrame, fraction: float = 0.8):
    pools: dict[str, list[np.ndarray]] = {}
    for row in train_rows.itertuples():
        values, labels = vertical_data(row.path)
        stop = int(len(values) * fraction)
        for index, label in enumerate(labels):
            pools.setdefault(label, []).append(values[:stop, index])
    scaler = {}
    for label, pieces in pools.items():
        values = np.concatenate(pieces).astype(np.float64)
        center = float(np.median(values))
        scale = float(1.4826 * np.median(np.abs(values - center)))
        if scale < 1e-8: scale = float(np.std(values))
        scaler[label] = (center, max(scale, 1e-8))
    return scaler


def normalized_series(path: Path, scaler: dict[str, tuple[float, float]]) -> list[np.ndarray]:
    values, labels = vertical_data(path)
    unseen = set(labels) - set(scaler)
    if unseen:
        raise ValueError(f"Sensors absent from healthy train: {sorted(unseen)}")
    return [
        ((values[:, index] - scaler[label][0]) / scaler[label][1]).astype(np.float32)
        for index, label in enumerate(labels)
    ]


def windows(series: list[np.ndarray], window: int, stride: int,
            start_fraction: float = 0.0, end_fraction: float = 1.0) -> np.ndarray:
    chunks = []
    for values in series:
        start = int(len(values) * start_fraction)
        stop = int(len(values) * end_fraction)
        values = values[start:stop]
        count = 1 + (len(values) - window) // stride
        if count > 0:
            starts = np.arange(count) * stride
            chunks.append(np.stack([values[item:item + window] for item in starts]))
    if not chunks:
        return np.empty((0, window), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray): self.values = torch.from_numpy(values)
    def __len__(self): return len(self.values)
    def __getitem__(self, index): return self.values[index]


def single_edges(window: int) -> torch.Tensor:
    edges = []
    for index in range(window):
        if index: edges.append((index, index - 1))
        if index + 1 < window: edges.append((index, index + 1))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class GraphBatcher:
    def __init__(self, window: int, device: torch.device):
        self.window, self.device = window, device
        self.base = single_edges(window).to(device)
        self.cache: dict[int, torch.Tensor] = {}

    def convert(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device, non_blocking=True)
        size = len(batch)
        if size not in self.cache:
            offsets = torch.arange(size, device=self.device).view(-1, 1, 1) * self.window
            edges = self.base.view(1, 2, -1) + offsets
            self.cache[size] = edges.permute(1, 0, 2).reshape(2, -1)
        return batch.reshape(size * self.window, 1), self.cache[size]


def make_loader(values: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(WindowDataset(values), batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=False)


def evaluate_loss(model, loader, batcher) -> float:
    model.eval(); losses = []
    with torch.no_grad():
        for batch in loader:
            x, edge = batcher.convert(batch)
            h_pred, h_target = model.model(x, edge)
            losses.append(float(model.model.compute_loss(x, h_pred, h_target, edge).cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def score_window_array(model, values: np.ndarray, batch_size: int,
                       batcher: GraphBatcher) -> np.ndarray:
    model.eval(); result = []
    for batch in make_loader(values, batch_size, False):
        x, edge = batcher.convert(batch)
        point = model.model.get_anomaly_scores(x, edge)
        result.append(point.reshape(len(batch), batcher.window).mean(1).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0,
    )
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "specificity": float(tn / max(tn + fp, 1)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_scatterad_shared"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--window", type=int, default=110)
    parser.add_argument("--train-stride", type=int, default=110)
    parser.add_argument("--eval-stride", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = resolve_manifest(args.manifest, args.data_root)
    train_rows = frame[(frame.split == "train") & (frame.label == 0)]
    scaler = fit_sensor_scaler(train_rows)
    print(f"device={device} train_records={len(train_rows)} sensors={len(scaler)}", flush=True)

    train_parts, internal_val_parts = [], []
    for index, row in enumerate(train_rows.itertuples(), 1):
        series = normalized_series(row.path, scaler)
        train_parts.append(windows(series, args.window, args.train_stride, 0.0, 0.8))
        internal_val_parts.append(windows(series, args.window, args.eval_stride, 0.8, 1.0))
        print(f"loaded healthy train {index}/{len(train_rows)}", flush=True)
    train_windows = np.concatenate(train_parts)
    internal_val = np.concatenate(internal_val_parts)
    print(f"train_windows={len(train_windows)} internal_val_windows={len(internal_val)}", flush=True)

    model = Model(input_dim=1, hidden_dim=args.hidden, num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = make_loader(train_windows, args.batch_size, True)
    val_loader = make_loader(internal_val, args.batch_size, False)
    batcher = GraphBatcher(args.window, device)
    history, best_loss, best_state = [], float("inf"), None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_losses = []
        for step, batch in enumerate(train_loader, 1):
            x, edge = batcher.convert(batch)
            optimizer.zero_grad(set_to_none=True)
            h_pred, h_target = model.model(x, edge)
            loss = model.model.compute_loss(x, h_pred, h_target, edge)
            loss.backward(); optimizer.step(); model.model.update_target()
            epoch_losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(epoch_losses[-100:]):.6f}", flush=True)
        train_loss = float(np.mean(epoch_losses))
        val_loss = evaluate_loss(model, val_loader, batcher)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_healthy_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        print(f"epoch={epoch}/{args.epochs} train={train_loss:.6f} val_healthy={val_loss:.6f}", flush=True)
    model.load_state_dict(best_state)
    seconds = time.time() - started
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    torch.save({"model_state_dict": best_state, "scaler": scaler, "args": vars(args)},
               args.output / "scatterad_shared_best.pt")

    rows = []
    eval_rows = frame[frame.split.isin(["val", "test"])]
    for index, row in enumerate(eval_rows.itertuples(), 1):
        series = normalized_series(row.path, scaler)
        record_windows = windows(series, args.window, args.eval_stride)
        scores = score_window_array(model, record_windows, args.batch_size, batcher)
        rows.append({
            "record_id": row.record_id, "state": row.state, "scenario": SCENARIOS[row.state],
            "excitation": row.excitation, "setup": int(row.setup), "split": row.split,
            "label": int(row.label), "sensor_windows": len(scores),
            "score_mean": float(np.mean(scores)), "score_median": float(np.median(scores)),
            "score_q95": float(np.quantile(scores, 0.95)),
        })
        if index % 12 == 0 or index == len(eval_rows):
            print(f"scored records={index}/{len(eval_rows)}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.output / "record_scores.csv", index=False)
    val, test = scores[scores.split == "val"].copy(), scores[scores.split == "test"].copy()
    record_threshold = float(val.loc[val.label == 0, "score_median"].max())
    record_result = binary_metrics(test.label.to_numpy(), test.score_median.to_numpy(), record_threshold)

    def state_table(part: pd.DataFrame) -> pd.DataFrame:
        return part.groupby(["state", "scenario", "label"], as_index=False).agg(
            records=("record_id", "size"), score_median=("score_median", "mean"),
            score_mean=("score_mean", "mean"), score_q95=("score_q95", "mean"),
        )
    val_states, test_states = state_table(val), state_table(test)
    state_threshold = float(val_states.loc[val_states.label == 0, "score_median"].max())
    state_result = binary_metrics(test_states.label.to_numpy(), test_states.score_median.to_numpy(), state_threshold)
    test["alarm"] = test.score_median > record_threshold
    test_states["alarm"] = test_states.score_median > state_threshold
    test.to_csv(args.output / "test_record_scores.csv", index=False)
    test_states.to_csv(args.output / "test_state_scores.csv", index=False)

    result = {
        "method": "ScatterAD_NeurIPS2025_shared_univariate_sensor_adapter",
        "epochs": args.epochs, "training_seconds": seconds,
        **{f"record_{key}": value for key, value in record_result.items()},
        **{f"state_{key}": value for key, value in state_result.items()},
    }
    pd.DataFrame([result]).to_csv(args.output / "metrics.csv", index=False)
    metadata = {
        "paper": "ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection, NeurIPS 2025",
        "official_repository": "https://github.com/jk-sounds/ScatterAD",
        "official_commit": "7572fa954468e46c598badf1c9cfaf6c1835c4cb",
        "compatibility_patch": "center is 1-D: norm(dim=1) replaced by norm()",
        "adapter": "one shared input_dim=1 model over sensor windows; record median aggregation",
        "author_architecture_loss_score_modified": False,
        "important_scope": "shared univariate-sensor adapter, not exact fixed-dimensional benchmark reproduction",
        "damage_used_for_training_threshold_or_checkpoint": False,
        "batch_boundary_note": "Author loss treats concatenated graph nodes as a sequence; batched boundaries are below 1% of adjacency comparisons.",
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    hist = pd.DataFrame(history)
    axes[0, 0].plot(hist.epoch, hist.train_loss, "o-", label="Train healthy")
    axes[0, 0].plot(hist.epoch, hist.val_healthy_loss, "s--", label="Internal healthy val")
    axes[0, 0].set(title="ScatterAD loss", xlabel="Epoch", ylabel="Official loss")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.25)
    fpr, tpr, _ = roc_curve(test.label, test.score_median)
    axes[0, 1].plot(fpr, tpr, label=f"AUROC={record_result['auroc']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set(title="Test record ROC", xlabel="FPR", ylabel="TPR")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.25)
    precision, recall, _ = precision_recall_curve(test.label, test.score_median)
    axes[1, 0].plot(recall, precision, label=f"AUPRC={record_result['auprc']:.3f}")
    axes[1, 0].set(title="Test record precision-recall", xlabel="Recall", ylabel="Precision")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.25)
    predictions = (test_states.score_median > state_threshold).astype(int)
    ConfusionMatrixDisplay(
        confusion_matrix(test_states.label, predictions, labels=[0, 1]),
        display_labels=["Healthy", "Damage"],
    ).plot(ax=axes[1, 1], colorbar=False, cmap="Greens")
    axes[1, 1].set_title("Test state confusion matrix")
    fig.tight_layout(); fig.savefig(args.output / "scatterad_shared_results.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("\nRESULT", flush=True); print(pd.DataFrame([result]).to_string(index=False), flush=True)
    print("\nTEST STATES", flush=True); print(test_states.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

