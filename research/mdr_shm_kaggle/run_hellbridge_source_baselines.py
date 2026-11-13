from __future__ import annotations

"""Source-based 10-seed baselines under one HellBridge protocol.

Model architectures/losses come from the repositories vendored in
``third_party``. Dataset splitting, threshold calibration, scoring and metrics
are controlled here to prevent test-contamination and point adjustment.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_tsad_baselines import (  # noqa: E402
    Normalizer, development_split, evaluate, load_windows, seed_everything,
    threshold_from_healthy,
)
from third_party.lstm_ae import LSTMAE  # noqa: E402
from third_party.tranad import TranAD  # noqa: E402
from third_party.usad import UsadModel  # noqa: E402
from third_party.timesnet import Model as TimesNet  # noqa: E402
from third_party.moderntcn import Model as ModernTCN  # noqa: E402


MODELS = (
    "pca_spe", "pca_pyod", "lstm_ae", "usad", "gdn", "tranad",
    "timesnet", "moderntcn", "scatterad",
)
SOURCE_KIND = {
    "pca_spe": "scikit-learn PCA SPE/Q reconstruction baseline",
    "pca_pyod": "PyOD PCA 3.6.8 public API",
    "lstm_ae": "batch-safe port of supplied TSAD LSTM_AE",
    "usad": "supplied official EURECOM USAD source and two-optimizer loss",
    "gdn": "supplied official GDN source, PyG compatibility import patch",
    "tranad": "batch-safe port of supplied official TranAD two-phase architecture",
    "timesnet": "supplied THUML Time-Series-Library TimesNet source",
    "moderntcn": "supplied official ModernTCN anomaly-detection source",
    "scatterad": "supplied ScatterAD source with center-shape compatibility patch",
}


class FitMinMax:
    def __init__(self, fit: np.ndarray):
        self.low = fit.min(axis=(0, 2), keepdims=True)
        self.high = fit.max(axis=(0, 2), keepdims=True)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.low) / np.maximum(self.high - self.low, 1e-6), 0, 1).astype(np.float32)


def timesnet_config(channels: int, length: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="anomaly_detection", seq_len=length, label_len=0, pred_len=0,
        top_k=3, num_kernels=4, d_model=64, d_ff=64, e_layers=2,
        enc_in=channels, c_out=channels, embed="timeF", freq="h", dropout=0.1,
    )


def moderntcn_config(channels: int, length: int) -> SimpleNamespace:
    # Based on the official SWaT anomaly script; dimension is reduced from 128
    # to 64 because the HellBridge healthy fit cohort contains only 60 windows.
    return SimpleNamespace(
        task_name="anomaly_detection", stem_ratio=6, downsample_ratio=2,
        ffn_ratio=1, num_blocks=[3], large_size=[51], small_size=[5],
        dims=[64], dw_dims=[64], enc_in=channels, small_kernel_merged=False,
        dropout=0.1, head_dropout=0.0, use_multi_scale=False, revin=True,
        affine=True, subtract_last=False, freq="h", seq_len=length,
        individual=False, pred_len=0, kernel_size=25, patch_size=8,
        patch_stride=4, decomposition=False,
    )


def model_forward(name: str, model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    sequence = batch.transpose(1, 2)  # B,T,C
    if name == "lstm_ae":
        return model(sequence).transpose(1, 2)
    if name == "tranad":
        _, second = model(sequence)
        return second.transpose(1, 2)
    if name == "timesnet":
        return model(sequence, None, None, None).transpose(1, 2)
    if name == "moderntcn":
        return model(sequence, None, None, None).transpose(1, 2)
    raise KeyError(name)


def fit_reconstructor(name: str, fit: np.ndarray, epochs: int, batch_size: int,
                      device: torch.device) -> nn.Module:
    channels, length = fit.shape[1:]
    constructors = {
        "lstm_ae": lambda: LSTMAE(channels, length),
        "tranad": lambda: TranAD(channels, length),
        "timesnet": lambda: TimesNet(timesnet_config(channels, length)),
        "moderntcn": lambda: ModernTCN(moderntcn_config(channels, length)),
    }
    model = constructors[name]().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(fit)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model_forward(name, model, batch)
            loss = (batch - output).pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    return model.eval()


@torch.no_grad()
def score_reconstructor(name: str, model: nn.Module, values: np.ndarray,
                        batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    output = []
    for (batch,) in loader:
        batch = batch.to(device)
        error = (batch - model_forward(name, model, batch)).pow(2).mean((1, 2))
        output.extend(error.cpu().numpy())
    return np.asarray(output)


def fit_usad(fit: np.ndarray, epochs: int, batch_size: int, device: torch.device) -> UsadModel:
    flat = fit.reshape(len(fit), -1)
    model = UsadModel(flat.shape[1], max(5, flat.shape[1] // 8)).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(flat)), batch_size=batch_size, shuffle=True)
    optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=1e-3)
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=1e-3)
    for epoch in range(1, epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer1.zero_grad(set_to_none=True)
            loss1, _ = model.training_step(batch, epoch)
            loss1.backward()
            optimizer1.step()
            optimizer2.zero_grad(set_to_none=True)
            _, loss2 = model.training_step(batch, epoch)
            loss2.backward()
            optimizer2.step()
            losses.append(float((loss1 + loss2).detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch={epoch}/{epochs} usad_loss={np.mean(losses):.6f}", flush=True)
    return model.eval()


@torch.no_grad()
def score_usad(model: UsadModel, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    loader = DataLoader(TensorDataset(torch.from_numpy(flat)), batch_size=batch_size)
    scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        first = model.decoder1(model.encoder(batch))
        cycle = model.decoder2(model.encoder(first))
        scores.extend((0.5 * (batch - first).pow(2) + 0.5 * (batch - cycle).pow(2)).mean(1).cpu().numpy())
    return np.asarray(scores)


def complete_edge(channels: int) -> torch.Tensor:
    source, target = np.meshgrid(np.arange(channels), np.arange(channels), indexing="ij")
    keep = source.ravel() != target.ravel()
    return torch.as_tensor(np.stack([source.ravel()[keep], target.ravel()[keep]]), dtype=torch.long)


def fit_gdn(fit: np.ndarray, epochs: int, batch_size: int, device: torch.device):
    try:
        from third_party.gdn import GDN
    except ImportError as error:
        raise RuntimeError("GDN requires torch-geometric. Run the Kaggle install cell.") from error
    channels, length = fit.shape[1:]
    history = 16
    x = np.stack([fit[:, :, start:start + history] for start in range(length - history)], axis=2)
    y = np.stack([fit[:, :, start + history] for start in range(length - history)], axis=2)
    x, y = x.transpose(0, 2, 1, 3).reshape(-1, channels, history), y.transpose(0, 2, 1).reshape(-1, channels)
    edge = complete_edge(channels)
    model = GDN([edge], channels, dim=32, out_layer_inter_dim=64, input_dim=history,
                out_layer_num=1, topk=min(10, channels)).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(batch_x, edge) - batch_y).pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch={epoch}/{epochs} gdn_loss={np.mean(losses):.6f}", flush=True)
    return model.eval(), edge, history


@torch.no_grad()
def score_gdn(bundle, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model, edge, history = bundle
    channels, length = values.shape[1:]
    all_scores = []
    for values_batch in np.array_split(values, max(1, int(np.ceil(len(values) / batch_size)))):
        x = np.stack([values_batch[:, :, start:start + history] for start in range(length - history)], axis=2)
        y = np.stack([values_batch[:, :, start + history] for start in range(length - history)], axis=2)
        batch, steps = x.shape[0], x.shape[2]
        prediction = model(torch.from_numpy(x.transpose(0, 2, 1, 3).reshape(-1, channels, history)).to(device), edge)
        truth = torch.from_numpy(y.transpose(0, 2, 1).reshape(-1, channels)).to(device)
        error = (prediction - truth).pow(2).mean(1).reshape(batch, steps).mean(1)
        all_scores.extend(error.cpu().numpy())
    return np.asarray(all_scores)


def temporal_edge(length: int, radius: int = 2) -> torch.Tensor:
    pairs = [(i, j) for i in range(length) for j in range(max(0, i - radius), min(length, i + radius + 1)) if i != j]
    return torch.as_tensor(pairs, dtype=torch.long).T.contiguous()


def fit_scatterad(fit: np.ndarray, epochs: int, device: torch.device):
    try:
        from third_party.scatterad import Model
    except ImportError as error:
        raise RuntimeError("ScatterAD requires torch-geometric. Run the Kaggle install cell.") from error
    sequence = fit.transpose(0, 2, 1)
    edge = temporal_edge(sequence.shape[1]).to(device)
    model = Model(input_dim=fit.shape[1], hidden_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1, epochs + 1):
        order, losses = np.random.permutation(len(sequence)), []
        model.train()
        for index in order:
            x = torch.from_numpy(sequence[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, target = model.model(x, edge)
            loss = model.compute_loss(x, prediction, target, edge)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_target()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch={epoch}/{epochs} scatter_loss={np.mean(losses):.6f}", flush=True)
    return model.eval(), edge


@torch.no_grad()
def score_scatterad(bundle, values: np.ndarray, device: torch.device) -> np.ndarray:
    model, edge = bundle
    return np.asarray([
        float(model.get_anomaly_scores(torch.from_numpy(window.T).to(device), edge).mean().cpu())
        for window in values
    ])


def run_model(name: str, fit: np.ndarray, calibration: np.ndarray, test: np.ndarray,
              epochs: int, batch_size: int, device: torch.device):
    if name == "pca_spe":
        train = fit.reshape(len(fit), -1)
        model = PCA(n_components=0.95, svd_solver="full").fit(train)
        score = lambda x: np.mean((x.reshape(len(x), -1) - model.inverse_transform(model.transform(x.reshape(len(x), -1)))) ** 2, axis=1)
    elif name == "pca_pyod":
        try:
            from pyod.models.pca import PCA as PyODPCA
        except ImportError as error:
            raise RuntimeError("pca_pyod requires `pip install pyod`.") from error
        model = PyODPCA(n_components=None, n_selected_components=10, contamination=0.1, random_state=0).fit(fit.reshape(len(fit), -1))
        score = lambda x: model.decision_function(x.reshape(len(x), -1))
    elif name == "usad":
        scale = FitMinMax(fit); fit, calibration, test = map(scale.transform, (fit, calibration, test))
        model = fit_usad(fit, epochs, batch_size, device)
        score = lambda x: score_usad(model, x, batch_size, device)
    elif name == "gdn":
        model = fit_gdn(fit, epochs, batch_size, device)
        score = lambda x: score_gdn(model, x, batch_size, device)
    elif name == "scatterad":
        model = fit_scatterad(fit, epochs, device)
        score = lambda x: score_scatterad(model, x, device)
    else:
        if name == "tranad":
            scale = FitMinMax(fit); fit, calibration, test = map(scale.transform, (fit, calibration, test))
        model = fit_reconstructor(name, fit, epochs, batch_size, device)
        score = lambda x: score_reconstructor(name, model, x, batch_size, device)
    return score(calibration), score(test)


def refresh_outputs(output: Path, manifest: pd.DataFrame, test_index: np.ndarray) -> None:
    metric_files = sorted((output / "runs").glob("*/metrics.json"))
    score_files = sorted((output / "runs").glob("*/scores.csv"))
    if not metric_files:
        return
    metrics = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in metric_files])
    scores = pd.concat([pd.read_csv(path) for path in score_files], ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    aggregate = metrics.groupby("model", sort=False).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        specificity_mean=("specificity", "mean"), specificity_std=("specificity", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), runs=("seed", "size"),
    ).reset_index()
    aggregate.to_csv(output / "aggregate.csv", index=False)
    chart = aggregate.set_index("model")[["auroc_mean", "balanced_accuracy_mean", "recall_mean", "specificity_mean"]]
    axis = chart.plot.bar(figsize=(14, 6), ylim=(0, 1.03), rot=35, grid=True)
    axis.set_title("HellBridge healthy-only source baselines (completed runs)")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "baseline_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--paper-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20261113, 20261123)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--specificity-target", type=float, default=0.90)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(args.paper_split_dir / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit_index, calibration_index, test_index = development_split(manifest)
    raw = load_windows(args.h5, manifest, args.length)
    normalizer = Normalizer.fit(raw[fit_index])
    values = normalizer.transform(raw)
    labels = manifest.loc[test_index, "label"].to_numpy(int)
    print(f"device={device} models={args.models} seeds={args.seeds}", flush=True)

    protocol = {
        "dataset": "HellBridge", "task": "healthy-only unseen-damage detection",
        "fit_healthy": 60, "calibration_healthy": 20, "test_healthy": 80,
        "test_damage": 640, "seeds": args.seeds, "epochs": args.epochs,
        "threshold": f"healthy calibration quantile at specificity {args.specificity_target}",
        "test_labels_used_for_training_threshold_or_selection": False,
        "point_adjustment": False, "source_kind": SOURCE_KIND,
        "compatibility_patches": [
            "ModernTCN c_in trailing-comma tuple corrected",
            "ScatterAD center changed from [H] to [1,H] to match upstream dim=1 operations",
            "GDN imports adapted to current torch-geometric package",
            "TranAD and LSTM-AE made batch-safe without changing their core architecture",
        ],
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for model_name in args.models:
        for seed in args.seeds:
            run_dir = args.output_dir / "runs" / f"{model_name}__{seed}"
            metric_path = run_dir / "metrics.json"
            if metric_path.exists():
                print(f"SKIP completed model={model_name} seed={seed}", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== model={model_name} seed={seed} ===", flush=True)
            seed_everything(seed)
            started = time.time()
            calibration_scores, test_scores = run_model(
                model_name, values[fit_index].copy(), values[calibration_index].copy(),
                values[test_index].copy(), args.epochs, args.batch_size, device,
            )
            threshold = threshold_from_healthy(calibration_scores, args.specificity_target)
            metric = {
                "model": model_name, "seed": seed,
                **evaluate(labels, test_scores, threshold),
                "threshold": threshold,
                "calibration_specificity": float(np.mean(calibration_scores < threshold)),
                "seconds": time.time() - started,
            }
            score_frame = manifest.loc[test_index, ["sample_id", "record_id", "state", "label"]].copy()
            score_frame.insert(0, "seed", seed)
            score_frame.insert(0, "model", model_name)
            score_frame["score"] = test_scores
            score_frame["threshold"] = threshold
            score_frame.to_csv(run_dir / "scores.csv", index=False)
            metric_path.write_text(json.dumps(metric, indent=2), encoding="utf-8")
            print(json.dumps(metric, indent=2), flush=True)
            refresh_outputs(args.output_dir, manifest, test_index)

    refresh_outputs(args.output_dir, manifest, test_index)
    print(pd.read_csv(args.output_dir / "aggregate.csv").to_string(index=False))
    print(f"SOURCE_BASELINES_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

