"""Time-Vertex sensor-selection reproduction for the Z24 PDT records.

Paper: Niresi et al., "Time-Vertex machine learning for optimal sensor
placement in temporal graph signals: Applications in structural health
monitoring", Reliability Engineering & System Safety 270 (2026) 112153,
DOI 10.1016/j.ress.2025.112153.

The implementation follows the paper's three published algorithms:
statistical/spectral sensor descriptors, feature-space sensor clustering, and
selection of a central representative from every cluster.  A healthy-only
autoencoder then uses the selected sensors; its threshold is mean + 2 standard
deviations of healthy validation RMSE, as reported in the paper.

No TVML author repository was available at implementation time.  The "Code"
link on the first author's site resolves to EPFL-IMOS/PEGNN_SS, which belongs to
a different paper.  This file is therefore an equation/pseudocode reproduction,
not author code.  Z24 additionally requires one sensor graph per AVT/FVT setup
because its roving layouts are not a single fixed graph.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import detrend, resample_poly, welch
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_z24_strict_psd_scatterad as common


VAL_DAMAGE = {"04", "05", "06", "10", "14", "16", "17"}
TEST_DAMAGE = {"03", "07", "09", "11", "12", "13", "15"}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def named_vertical(path: Path) -> tuple[list[str], np.ndarray]:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    labels = [str(x).strip() for x in mat["labelshulp"].tolist()]
    use = [i for i, name in enumerate(labels)
           if name.endswith("V") and name[0].isdigit() and not name.startswith(("R", "DP"))]
    names = [labels[i] for i in use]
    values = resample_poly(np.asarray(mat["data"][:, use], np.float32), 1, 2, axis=0)
    if not np.isfinite(values).all(): raise ValueError(f"NaN/Inf in {path}")
    return names, detrend(values, axis=0, type="linear").astype(np.float32)


def entropy_histogram(x: np.ndarray, bins: int = 32) -> float:
    count, _ = np.histogram(x, bins=bins); p = count[count > 0] / max(count.sum(), 1)
    return float(-np.sum(p * np.log2(p)))


def sensor_descriptors(values: np.ndarray, fs: float, fmax: float) -> np.ndarray:
    """Published TVML-style statistical and spectral feature matrix (sensor x feature)."""
    descriptors = []
    for sensor in range(values.shape[1]):
        x = values[:, sensor]; rms = float(np.sqrt(np.mean(x * x)))
        f, p = welch(x, fs=fs, nperseg=min(2048, len(x)))
        keep = (f > 0) & (f <= fmax); f, p = f[keep], p[keep]
        prob = p / max(float(p.sum()), np.finfo(float).tiny)
        centroid = float(np.sum(f * prob)); spread = float(np.sqrt(np.sum((f - centroid) ** 2 * prob)))
        spectral_entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))
        dominant = float(f[np.argmax(p)])
        edges = np.linspace(0., fmax, 5)
        bands = [float(np.sum(p[(f >= edges[i]) & (f < edges[i + 1])]) / max(p.sum(), 1e-12))
                 for i in range(4)]
        descriptors.append([float(np.mean(x)), float(np.std(x)), rms, float(np.max(np.abs(x))),
                            float(np.max(np.abs(x)) / max(rms, 1e-12)), float(skew(x)),
                            float(kurtosis(x, fisher=False)), entropy_histogram(x), dominant,
                            centroid, spread, spectral_entropy, *bands])
    return np.nan_to_num(np.asarray(descriptors, np.float64))


def select_timevertex_sensors(values: np.ndarray, fs: float, fmax: float,
                              count: int, seed: int) -> tuple[np.ndarray, dict]:
    """Cluster non-redundant sensors and choose maximum-centrality representatives."""
    features = sensor_descriptors(values, fs, fmax)
    center = np.median(features, axis=0)
    scale = 1.4826 * np.median(np.abs(features - center), axis=0); scale[scale < 1e-8] = 1.
    z = np.clip((features - center) / scale, -10, 10)
    clusters = min(count, len(z))
    labels = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(z)
    distance2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    nonzero = distance2[distance2 > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.
    adjacency = np.exp(-distance2 / max(2 * bandwidth, 1e-12)); np.fill_diagonal(adjacency, 0.)
    selected, cluster_rows = [], []
    for cluster in range(clusters):
        members = np.where(labels == cluster)[0]
        # Weighted-degree centrality is the stable graph-centrality representative
        # for the tiny (15--17 node) Z24 setup graphs.
        centrality = adjacency[np.ix_(members, members)].sum(axis=1)
        winner = int(members[np.argmax(centrality)])
        selected.append(winner)
        cluster_rows.append({"cluster": cluster, "members": members.tolist(), "selected": winner,
                             "centrality": float(np.max(centrality))})
    fallback_order = np.argsort(-adjacency.sum(axis=1)).astype(int).tolist()
    return np.asarray(sorted(selected), dtype=int), {"descriptors": features.tolist(),
                                                      "clusters": cluster_rows,
                                                      "fallback_order": fallback_order}


def record_windows(values: np.ndarray, selected: np.ndarray, fs: float, window: int,
                   stride: int, bins: int, fmin: float, fmax: float) -> np.ndarray:
    grid = np.linspace(fmin, fmax, bins); output = []
    for start in range(0, len(values) - window + 1, stride):
        nodes = []
        for sensor in selected:
            f, p = welch(values[start:start + window, sensor], fs=fs,
                         nperseg=min(1024, window), noverlap=min(512, window // 2))
            band = (f >= fmin) & (f <= fmax); shape = np.interp(grid, f[band], p[band])
            shape /= max(float(np.trapezoid(shape, grid)), np.finfo(float).tiny)
            nodes.append(np.log10(shape + 1e-12))
        output.append(np.stack(nodes).reshape(-1))
    if not output: raise ValueError("Record shorter than --window")
    return np.asarray(output, np.float32)


def fit_scaler(records: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(records); center = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - center), axis=0); scale[scale < 1e-7] = 1.
    return center.astype(np.float32), scale.astype(np.float32)


def transform(x, center, scale): return np.clip((x - center) / scale, -10, 10).astype(np.float32)


class Autoencoder(nn.Module):
    def __init__(self, dimension: int, hidden: int, latent: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.ELU(), nn.LayerNorm(hidden),
                                     nn.Linear(hidden, latent), nn.ELU(), nn.Linear(latent, hidden),
                                     nn.ELU(), nn.Linear(hidden, dimension))
    def forward(self, x): return self.network(x)


def loader(x, batch, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def errors(model, x, batch, device):
    model.eval(); output = []
    for (value,) in loader(x, batch, False):
        value = value.to(device); output.append(torch.mean((model(value) - value) ** 2, 1).cpu().numpy())
    return np.concatenate(output)


def train_model(train, validation, args, device):
    model = Autoencoder(train.shape[1], args.hidden, args.latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for (value,) in loader(train, args.batch_size, True):
            value = value.to(device); optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(value) - value) ** 2); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses)); val_loss = float(errors(model, validation, args.batch_size, device).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "healthy_validation_loss": val_loss})
        if val_loss < best - 1e-7: best, state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch}/{args.epochs} train={train_loss:.6f} healthy_val={val_loss:.6f}", flush=True)
        if stale >= args.patience: break
    if state is None: raise RuntimeError("No model checkpoint")
    model.load_state_dict(state); return model, pd.DataFrame(history)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_timevertex_paper"))
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256); p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent", type=int, default=32); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--fs", type=float, default=50.)
    p.add_argument("--window", type=int, default=4096); p.add_argument("--stride", type=int, default=2048)
    p.add_argument("--psd-bins", type=int, default=48); p.add_argument("--fmin", type=float, default=3.)
    p.add_argument("--fmax", type=float, default=20.); p.add_argument("--selected-sensors", type=int, default=4)
    p.add_argument("--record-quantile", type=float, default=.95); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = common.resolve_manifest(args.manifest, args.data_root)
    train_rows = frame[(frame.state == "01") & (frame.setup.astype(int) <= 6)].sort_values(["setup", "excitation"])
    cal_rows = frame[(frame.state == "01") & (frame.setup.astype(int) >= 7)].sort_values(["setup", "excitation"])
    validation_rows = frame[frame.state.isin(VAL_DAMAGE)].sort_values(["state", "setup", "excitation"])
    test_rows = frame[frame.state.isin({"08"} | TEST_DAMAGE)].sort_values(["state", "setup", "excitation"])
    # Learn a distinct sensor subset from the S01 healthy record of every setup/excitation.
    selections, audit, healthy_values = {}, {}, {}
    for row in frame[frame.state == "01"].itertuples():
        names, values = named_vertical(row.path); key = (row.excitation, int(row.setup))
        selected, details = select_timevertex_sensors(values, args.fs, args.fmax,
                                                       args.selected_sensors, args.seed + int(row.setup))
        selected_names = [names[i] for i in selected]
        fallback_names = [names[i] for i in details["fallback_order"]]
        selections[key] = {"selected_names": selected_names, "fallback_names": fallback_names}
        healthy_values[row.record_id] = values
        audit[f"{row.excitation}_setup{int(row.setup):02d}"] = {
            "sensor_names": names, "selected_indices": selected.tolist(),
            "selected_names": selected_names, "fallback_names": fallback_names, **details}
    print(f"device={device} learned_timevertex_graphs={len(selections)} train_healthy={len(train_rows)} "
          f"cal_healthy={len(cal_rows)} validation_damage={len(validation_rows)} locked_test={len(test_rows)}", flush=True)
    cache = {}
    all_rows = pd.concat([train_rows, cal_rows, validation_rows, test_rows]).drop_duplicates("record_id")
    for i, row in enumerate(all_rows.itertuples(), 1):
        names, values = named_vertical(row.path); key = (row.excitation, int(row.setup))
        selection = selections[key]; name_to_index = {name: i for i, name in enumerate(names)}
        chosen_names = [name for name in selection["selected_names"] if name in name_to_index]
        for name in selection["fallback_names"]:
            if len(chosen_names) >= args.selected_sensors: break
            if name in name_to_index and name not in chosen_names: chosen_names.append(name)
        if len(chosen_names) != args.selected_sensors:
            raise ValueError(f"Only {len(chosen_names)} selected/fallback sensors in {row.record_id}")
        selected = np.asarray([name_to_index[name] for name in chosen_names], dtype=int)
        audit_key = f"{row.excitation}_setup{int(row.setup):02d}"
        if chosen_names != selection["selected_names"]:
            audit[audit_key].setdefault("fallback_events", []).append(
                {"record_id": row.record_id, "available": names, "used": chosen_names})
        cache[row.record_id] = record_windows(values, selected, args.fs, args.window, args.stride,
                                              args.psd_bins, args.fmin, args.fmax)
        if i % 18 == 0 or i == len(all_rows): print(f"features={i}/{len(all_rows)}", flush=True)
    center, scale = fit_scaler([cache[x] for x in train_rows.record_id])
    scaled = {key: transform(value, center, scale) for key, value in cache.items()}
    train = np.concatenate([scaled[x] for x in train_rows.record_id])
    cal = np.concatenate([scaled[x] for x in cal_rows.record_id])
    started = time.time(); model, history = train_model(train, cal, args, device)
    history.to_csv(args.output / "training_history.csv", index=False)

    def score(record_id):
        return float(np.quantile(errors(model, scaled[record_id], args.batch_size, device), args.record_quantile))
    cal_scores = np.asarray([score(x) for x in cal_rows.record_id])
    # Exact threshold rule reported by the paper: healthy validation mean + 2 sigma.
    threshold = float(cal_scores.mean() + 2 * cal_scores.std(ddof=1))
    def evaluate(rows, split):
        output = []
        for row in rows.itertuples():
            value = score(row.record_id)
            output.append({"record_id": row.record_id, "state": row.state, "scenario": row.scenario,
                           "excitation": row.excitation, "setup": int(row.setup), "label": int(row.label),
                           "split": split, "score": value, "alarm": value > threshold})
        return pd.DataFrame(output)
    val_healthy = pd.DataFrame([{"record_id": row.record_id, "state": "01", "scenario": row.scenario,
                                 "excitation": row.excitation, "setup": int(row.setup), "label": 0,
                                 "split": "healthy_validation", "score": value, "alarm": value > threshold}
                                for row, value in zip(cal_rows.itertuples(), cal_scores)])
    validation = pd.concat([val_healthy, evaluate(validation_rows, "damage_validation")], ignore_index=True)
    test = evaluate(test_rows, "locked_test")
    validation.to_csv(args.output / "validation_record_scores.csv", index=False)
    test.to_csv(args.output / "locked_test_record_scores.csv", index=False)
    val_metrics = common.metrics(validation.label, validation.score, threshold)
    test_metrics = common.metrics(test.label, test.score, threshold)
    states = test.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("score", "median"))
    states["alarm"] = states.score > threshold; states.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    state_metrics = common.metrics(states.label, states.score, threshold)
    summary = {"method": "TVML_RESS2026_pseudocode_reproduction_Z24_setup_graphs",
               "healthy_train_records": len(train_rows), "healthy_calibration_records": len(cal_rows),
               "damage_validation_records": len(validation_rows), "damage_used_for_gradient_fit": False,
               "damage_used_for_sensor_selection": False, "damage_used_for_threshold": False,
               "damage_validation_opened_before_locked_test": True, "selected_sensors_per_setup": args.selected_sensors,
               "epochs_run": int(history.epoch.max()), "seconds": time.time() - started,
               **{f"validation_{k}": v for k, v in val_metrics.items()},
               **{f"record_{k}": v for k, v in test_metrics.items()},
               **{f"state_{k}": v for k, v in state_metrics.items()}}
    pd.DataFrame([summary]).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "sensor_selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "center": center, "scale": scale,
                "selections": selections, "threshold": threshold, "args": vars(args)}, args.output / "model.pt")
    metadata = {"paper": "Niresi et al., RESS 270 (2026) 112153",
                "doi": "10.1016/j.ress.2025.112153", "author_code_available": False,
                "repository_dead_end": "Author-site Code link resolved to PEGNN_SS for a different IEEE IoTJ paper",
                "reproduced": ["statistical/spectral descriptors", "sensor clustering",
                               "graph-central representative selection", "healthy-only AE",
                               "healthy validation mean + 2 sigma threshold"],
                "z24_adaptation": "one graph per excitation/setup because Z24 sensors are roving",
                "split": {"train": "S01 setups 1-6 healthy", "calibration": "S01 setups 7-9 healthy",
                          "validation": sorted(VAL_DAMAGE), "locked_test": ["08", *sorted(TEST_DAMAGE)]},
                "parameters": vars(args)}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(history.epoch, history.train_loss, label="train healthy")
    axes[0].plot(history.epoch, history.healthy_validation_loss, label="validation healthy")
    axes[0].legend(); axes[0].set_title("TVML-selected AE loss")
    fpr, tpr, _ = roc_curve(test.label, test.score); axes[1].plot(fpr, tpr, label=f"AUROC={test_metrics['auroc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray"); axes[1].legend(); axes[1].set_title("Locked-test ROC")
    ConfusionMatrixDisplay(confusion_matrix(test.label, test.alarm, labels=[0, 1]),
                           display_labels=["Healthy", "Damage"]).plot(ax=axes[2], cmap="Purples", colorbar=False)
    axes[2].set_title("Locked-test confusion matrix"); fig.tight_layout()
    fig.savefig(args.output / "timevertex_results.png", dpi=190); plt.close(fig)
    print("\nVALIDATION\n" + pd.DataFrame([{f"validation_{k}": v for k, v in val_metrics.items()}]).to_string(index=False))
    print("\nLOCKED TEST\n" + pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__": main()

