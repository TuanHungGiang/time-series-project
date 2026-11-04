"""Run official CATCH on the locked Z24 all-record split.

One author CATCH model is trained from scratch for each AVT/FVT x setup sensor
configuration. This preserves physical channel semantics while allowing every
valid Z24 record to participate in the record-level protocol. Author model,
loss, optimizer, scheduler and anomaly score are not modified.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT), str(ROOT / "baselines_repo" / "CATCH"),
    str(ROOT / "baseline_packages"), str(ROOT / ".packages"),
]

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

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from ts_benchmark.baselines.catch.CATCH import CATCH


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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


HYPERPARAMETERS = {
    "Mlr": 0.001, "auxi_lambda": 0.01, "batch_size": 128,
    "cf_dim": 16, "d_ff": 32, "d_model": 16, "dc_lambda": 0.05,
    "dropout": 0.3, "e_layers": 1, "head_dim": 32,
    "inference_patch_size": 96, "lr": 0.005, "n_heads": 4,
    "num_epochs": 1, "patch_size": 16, "patch_stride": 8,
    "score_lambda": 0.5, "seq_len": 192, "patience": 3,
}


def vertical_labels(mat: dict) -> list[str]:
    labels = [str(value).strip() for value in mat["labelshulp"].tolist()]
    return [
        label for label in labels
        if label.endswith("V") and label[0].isdigit() and not label.startswith(("R", "DP"))
    ]


def common_labels(paths: list[Path]) -> list[str]:
    ordered = vertical_labels(loadmat(paths[0], variable_names=["labelshulp"]))
    intersection = set(ordered)
    for path in paths[1:]:
        intersection &= set(vertical_labels(loadmat(path, variable_names=["labelshulp"])))
    labels = [label for label in ordered if label in intersection]
    if not labels:
        raise ValueError(f"No common vertical channels for {paths[0].parent}")
    return labels


def load_values(path: Path, labels: list[str]) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    all_labels = [str(value).strip() for value in mat["labelshulp"].tolist()]
    indices = [all_labels.index(label) for label in labels]
    values = np.asarray(mat["data"][:, indices], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"NaN/Inf in {path}")
    return resample_poly(values, 1, 2, axis=0).astype(np.float32)


def dataframe(values: np.ndarray, labels: list[str]) -> pd.DataFrame:
    index = pd.date_range("1998-08-01", periods=len(values), freq=pd.Timedelta(seconds=1 / 50))
    return pd.DataFrame(values, index=index, columns=labels)


def point_scores(model: CATCH, values: np.ndarray, labels: list[str]) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        scores, _ = model.detect_score(dataframe(values, labels))
    return np.asarray(scores, dtype=np.float64)


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if scale < 1e-12:
        scale = float(np.std(values))
    return center, max(scale, 1e-12)


def parse_training_log(text: str) -> list[dict]:
    pattern = re.compile(
        r"Epoch:\s*(\d+), Steps:\s*(\d+) \| Train Loss:\s*([0-9.eE+-]+) "
        r"Vali Loss:\s*([0-9.eE+-]+)"
    )
    return [
        {"epoch": int(epoch), "steps": int(steps), "train_loss": float(train),
         "validation_loss": float(validation)}
        for epoch, steps, train, validation in pattern.findall(text)
    ]


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0,
    )
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float((labels == predictions).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "specificity": float(tn / max(tn + fp, 1)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs_baselines" /
                        "z24_all_records_split" / "used_288_records_manifest.csv")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Dataset root used to resolve manifest relative_path values")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_baselines" /
                        "z24_all_records_catch_official")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    logs = args.output / "logs"
    checkpoints.mkdir(exist_ok=True); logs.mkdir(exist_ok=True)

    manifest = pd.read_csv(args.manifest, dtype={"state": str})
    manifest["state"] = manifest.state.str.zfill(2)
    if "relative_path" in manifest.columns:
        if args.data_root is None:
            raise ValueError("--data-root is required for a portable relative-path manifest")
        manifest = manifest[manifest.split != "excluded"].copy()
        manifest["path"] = manifest.relative_path.map(lambda value: (args.data_root / value).resolve())
    else:
        manifest["path"] = manifest.path.map(Path)
    record_rows, history_rows, config_rows = [], [], []
    groups = list(manifest.groupby(["excitation", "setup"], sort=True))

    for config_index, ((excitation, setup), group) in enumerate(groups, 1):
        name = f"{excitation}_setup{int(setup):02d}"
        paths = group.path.tolist()
        labels = common_labels(paths)
        train_group = group[(group.split == "train") & (group.label == 0)]
        if train_group.empty:
            raise ValueError(f"No healthy training record for {name}")

        train_parts, internal_val_parts, full_train_parts = [], [], []
        for row in train_group.itertuples():
            values = load_values(row.path, labels)
            split = int(len(values) * 0.8)
            train_parts.append(values[:split])
            internal_val_parts.append(values[split:])
            full_train_parts.append(values)
        train_values = np.concatenate(train_parts)
        internal_val = np.concatenate(internal_val_parts)
        full_train = np.concatenate(full_train_parts)

        seed_everything(4200 + config_index)
        model = CATCH(**HYPERPARAMETERS)
        buffer = io.StringIO()
        started = time.time()
        with contextlib.redirect_stdout(buffer):
            model.detect_fit(dataframe(train_values, labels), dataframe(internal_val, labels))
        elapsed = time.time() - started
        log = buffer.getvalue()
        (logs / f"{name}.txt").write_text(log, encoding="utf-8")
        history = parse_training_log(log)
        for item in history:
            history_rows.append({"configuration": name, **item})

        train_scores = point_scores(model, full_train, labels)
        center, scale = robust_location_scale(train_scores)
        for row in group[group.split.isin(["val", "test"])].itertuples():
            values = load_values(row.path, labels)
            normalized = (point_scores(model, values, labels) - center) / scale
            record_rows.append({
                "record_id": row.record_id, "state": row.state,
                "scenario": SCENARIOS[row.state], "excitation": excitation,
                "setup": int(setup), "split": row.split, "label": int(row.label),
                "common_channels": len(labels), "score_mean": float(np.mean(normalized)),
                "score_q95": float(np.quantile(normalized, 0.95)),
            })

        torch.save({
            "model_state_dict": model.early_stopping.check_point,
            "scaler_mean": model.scaler.mean_, "scaler_scale": model.scaler.scale_,
            "labels": labels, "hyperparameters": HYPERPARAMETERS,
            "point_score_center": center, "point_score_scale": scale,
            "training_records": train_group.record_id.tolist(),
        }, checkpoints / f"{name}.pt")
        config_rows.append({
            "configuration": name, "common_channels": len(labels),
            "training_records": len(train_group), "train_points": len(train_values),
            "internal_validation_points": len(internal_val), "training_seconds": elapsed,
        })
        print(f"config={config_index}/18 {name} channels={len(labels)} "
              f"train_records={len(train_group)} seconds={elapsed:.1f} history={history}", flush=True)
        del model, train_values, internal_val, full_train, full_train_parts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = pd.DataFrame(record_rows)
    scores.to_csv(args.output / "record_scores.csv", index=False)
    pd.DataFrame(history_rows).to_csv(args.output / "training_history.csv", index=False)
    pd.DataFrame(config_rows).to_csv(args.output / "configuration_metadata.csv", index=False)

    val = scores[scores.split == "val"].copy()
    test = scores[scores.split == "test"].copy()
    record_threshold = float(val.loc[val.label == 0, "score_mean"].max())
    record_result = metrics(test.label.to_numpy(), test.score_mean.to_numpy(), record_threshold)

    def aggregate_states(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby(["state", "scenario", "label"], as_index=False).agg(
            records=("record_id", "size"), score_mean=("score_mean", "mean"),
            score_median=("score_mean", "median"), score_q95=("score_q95", "mean"),
        )

    val_states = aggregate_states(val)
    test_states = aggregate_states(test)
    state_threshold = float(val_states.loc[val_states.label == 0, "score_mean"].max())
    state_result = metrics(test_states.label.to_numpy(), test_states.score_mean.to_numpy(), state_threshold)
    test["alarm"] = test.score_mean > record_threshold
    test_states["alarm"] = test_states.score_mean > state_threshold
    test.to_csv(args.output / "test_record_scores.csv", index=False)
    val_states.to_csv(args.output / "validation_state_scores.csv", index=False)
    test_states.to_csv(args.output / "test_state_scores.csv", index=False)

    result = {
        "method": "CATCH_ICLR2025_official_18_configuration_1epoch",
        "records_used": 288, "models": 18, "epochs_per_model": 1,
        **{f"record_{key}": value for key, value in record_result.items()},
        **{f"state_{key}": value for key, value in state_result.items()},
    }
    pd.DataFrame([result]).to_csv(args.output / "metrics.csv", index=False)
    metadata = {
        "official_repository": "https://github.com/decisionintelligence/CATCH.git",
        "official_commit": "3647c69be5eb56649b072596cf89098e689e20c3",
        "author_model_modified": False,
        "adapter": "one model per AVT/FVT x setup; common physical sensor-label intersection",
        "training": "random initialization; one epoch/config; healthy train records only",
        "internal_validation": "last 20% of each training record; never damage",
        "record_threshold": "maximum mean score among six healthy validation records",
        "state_threshold": "maximum state-mean score among two healthy validation states",
        "damage_labels_used_for_training_or_threshold": False,
        "boundary_note": "When a configuration has two training records, concatenation creates one boundary.",
        "license_warning": "Pinned CATCH repository has no LICENSE file; author source is not redistributed.",
        "hyperparameters": HYPERPARAMETERS,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    history = pd.DataFrame(history_rows).sort_values("configuration")
    x = np.arange(len(history))
    axes[0, 0].plot(x, history.train_loss, "o-", label="Train")
    axes[0, 0].plot(x, history.validation_loss, "s--", label="Internal healthy val")
    axes[0, 0].set(title="Official CATCH loss, one epoch/config", xlabel="Sensor configuration", ylabel="Loss")
    axes[0, 0].set_xticks(x, history.configuration, rotation=75, fontsize=7)
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.25)

    fpr, tpr, _ = roc_curve(test.label, test.score_mean)
    axes[0, 1].plot(fpr, tpr, label=f"AUROC={record_result['auroc']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set(title="Test record ROC", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.25)

    precision, recall, _ = precision_recall_curve(test.label, test.score_mean)
    axes[1, 0].plot(recall, precision, label=f"AUPRC={record_result['auprc']:.3f}")
    axes[1, 0].set(title="Test record precision-recall", xlabel="Recall", ylabel="Precision")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.25)

    predictions = (test_states.score_mean > state_threshold).astype(int)
    ConfusionMatrixDisplay(
        confusion_matrix(test_states.label, predictions, labels=[0, 1]),
        display_labels=["Healthy", "Damage"],
    ).plot(ax=axes[1, 1], colorbar=False, cmap="Oranges")
    axes[1, 1].set_title("Test state-level confusion matrix")
    fig.tight_layout()
    fig.savefig(args.output / "z24_all_records_catch_results.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    print("\nRESULT", flush=True)
    print(pd.DataFrame([result]).to_string(index=False), flush=True)
    print("\nTEST STATES", flush=True)
    print(test_states.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

