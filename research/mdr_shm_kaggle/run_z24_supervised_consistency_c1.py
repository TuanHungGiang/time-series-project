from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from run_z24_b2 import load_vertical, normalized_sensor_spectra
from run_z24_crossfit_fusion_b11 import fit_modal_raw
from run_z24_model_a_v31 import corrupt_v31
from run_z24_preprocessing_audit import infer_setup
from run_z24_shape_fusion_b1 import record_metadata, shape_feature
from run_z24_model_a_v3 import prepare_frame


TRAIN_CONDITIONS = ("dropout_0.30", "missing_0.30", "noise_5", "targeted_0.30")
TEST_CONDITIONS = (
    "clean", "dropout_0.30", "missing_0.30", "noise_5",
    "targeted_0.30", "targeted_0.50",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_supervised(root: Path) -> tuple[pd.DataFrame, Path]:
    frame, manifest = prepare_frame(root)
    frame["setup_resolved"] = [infer_setup(row) for _, row in frame.iterrows()]
    frame["supervised_split"] = np.select(
        [frame.setup_resolved <= 5, frame.setup_resolved <= 7],
        ["train", "validation"], default="test",
    )
    return frame.reset_index(drop=True), manifest


def feature_from_signals(
    signals: np.ndarray,
    excitation: str,
    availability: np.ndarray | None = None,
) -> np.ndarray:
    frequencies, psd = normalized_sensor_spectra(signals)
    if availability is None:
        availability = np.ones(psd.shape[0], dtype=bool)
    shape = shape_feature(frequencies, psd, availability)
    return np.concatenate([shape, [float(excitation == "fvt")]])


def build_views(
    frame: pd.DataFrame,
    modal_models: dict,
    mode: str,
) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(frame.itertuples(), start=1):
        signals = load_vertical(Path(row.resolved_path))
        excitation = row.excitation_resolved
        clean = feature_from_signals(signals, excitation)
        state, setup = record_metadata(str(row.record_id))
        base = {
            "record_id": row.record_id,
            "label": int(row.label),
            "excitation": excitation,
            "state": state,
            "setup": setup,
            "split": row.supervised_split,
        }
        rows.append({**base, "condition": "clean", "replicate": 0, "feature": clean})
        conditions = TRAIN_CONDITIONS if row.supervised_split != "test" else TEST_CONDITIONS[1:]
        repeats = 1 if mode == "quick" else (2 if row.supervised_split == "train" else 3)
        for condition in conditions:
            for replicate in range(repeats):
                changed, availability, _, signature = corrupt_v31(
                    signals, condition, str(row.record_id), replicate,
                    modal_models[excitation],
                )
                feature = feature_from_signals(changed, excitation, availability)
                rows.append({
                    **base, "condition": condition, "replicate": replicate,
                    "corrupted_sensors": signature, "feature": feature,
                })
        if index % 36 == 0:
            print(f"features={index}/{len(frame)}", flush=True)
    return pd.DataFrame(rows)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = scores >= threshold
    positive = labels == 1
    negative = ~positive
    recall = float(np.mean(prediction[positive])) if np.any(positive) else np.nan
    specificity = float(np.mean(~prediction[negative])) if np.any(negative) else np.nan
    return {
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2),
        "threshold": float(threshold),
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best = None
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, float(threshold))
        key = (metrics["balanced_accuracy"], metrics["specificity"], metrics["auroc"])
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return best[1], best[2]


def arrays(views: pd.DataFrame, split: str, conditions: tuple[str, ...] | None = None):
    subset = views[views.split == split]
    if conditions is not None:
        subset = subset[subset.condition.isin(conditions)]
    x = np.stack(subset.feature.to_numpy())
    y = subset.label.to_numpy(dtype=int)
    return subset, x, y


class ConsistencyLinear(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).squeeze(1)


def fit_consistency(
    clean_x: np.ndarray,
    clean_y: np.ndarray,
    paired_x: np.ndarray,
    paired_clean_x: np.ndarray,
    paired_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    consistency_weight: float,
) -> tuple[ConsistencyLinear, np.ndarray, np.ndarray, float, dict]:
    seed_all(seed)
    center = clean_x.mean(axis=0)
    scale = np.maximum(clean_x.std(axis=0), 1e-6)
    transform = lambda value: (value - center) / scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConsistencyLinear(clean_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    counts = np.bincount(clean_y, minlength=2)
    class_weights = np.asarray([
        len(clean_y) / (2 * max(counts[0], 1)),
        len(clean_y) / (2 * max(counts[1], 1)),
    ], dtype=np.float32)
    px = torch.tensor(transform(paired_x), dtype=torch.float32, device=device)
    pc = torch.tensor(transform(paired_clean_x), dtype=torch.float32, device=device)
    py = torch.tensor(paired_y, dtype=torch.float32, device=device)
    weights = torch.tensor(class_weights[paired_y], dtype=torch.float32, device=device)
    vx = torch.tensor(transform(validation_x), dtype=torch.float32, device=device)
    best = None
    for epoch in range(401):
        model.train()
        optimizer.zero_grad()
        clean_logits = model(pc)
        corrupt_logits = model(px)
        classification = (
            nn.functional.binary_cross_entropy_with_logits(
                clean_logits, py, weight=weights
            )
            + nn.functional.binary_cross_entropy_with_logits(
                corrupt_logits, py, weight=weights
            )
        ) / 2
        consistency = torch.mean((clean_logits - corrupt_logits) ** 2)
        loss = classification + consistency_weight * consistency
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                score = torch.sigmoid(model(vx)).cpu().numpy()
            threshold, metrics = choose_threshold(validation_y, score)
            key = (metrics["balanced_accuracy"], metrics["auroc"], -float(loss.item()))
            if best is None or key > best[0]:
                best = (key, copy.deepcopy(model.state_dict()), threshold, metrics, epoch)
    assert best is not None
    model.load_state_dict(best[1])
    return model, center, scale, best[2], {**best[3], "epoch": best[4]}


def predict_consistency(model, x: np.ndarray, center: np.ndarray, scale: np.ndarray):
    device = next(model.parameters()).device
    values = torch.tensor((x - center) / scale, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(values)).cpu().numpy()


def evaluate_model(name, views, predictor, threshold, split="test"):
    rows, predictions = [], []
    evaluation = views[views.split == split]
    for (condition, replicate), group in evaluation.groupby(["condition", "replicate"]):
        x = np.stack(group.feature.to_numpy())
        score = predictor(x)
        metrics = binary_metrics(group.label.to_numpy(), score, threshold)
        rows.append({"model": name, "condition": condition, "replicate": int(replicate), **metrics})
        for record, value in zip(group.itertuples(), score):
            predictions.append({
                "model": name, "record_id": record.record_id, "label": int(record.label),
                "condition": condition, "replicate": int(replicate),
                "excitation": record.excitation, "state": int(record.state),
                "setup": int(record.setup), "score": float(value),
                "prediction": int(value >= threshold),
                "threshold": float(threshold), "split": split,
            })
    return rows, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--consistency-weight", type=float, default=0.20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    frame, manifest = prepare_supervised(args.input_root)
    modal_models = {}
    for excitation in ("avt", "fvt"):
        healthy_train = frame[
            (frame.supervised_split == "train") & (frame.label == 0)
            & (frame.excitation_resolved == excitation)
        ]
        modal_models[excitation] = fit_modal_raw(healthy_train)
    views = build_views(frame, modal_models, args.mode)

    train_clean, x_clean, y_clean = arrays(views, "train", ("clean",))
    val_all, x_val, y_val = arrays(views, "validation")
    train_aug, x_aug, y_aug = arrays(views, "train")
    clean_map = {row.record_id: row.feature for row in train_clean.itertuples()}
    paired_clean = np.stack([clean_map[row.record_id] for row in train_aug.itertuples()])

    clean_logistic = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    ).fit(x_clean, y_clean)
    augmented_logistic = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    ).fit(x_aug, y_aug)
    candidates = {}
    for name, model in (("clean_logistic", clean_logistic), ("augmented_logistic", augmented_logistic)):
        score = model.predict_proba(x_val)[:, 1]
        threshold, metrics = choose_threshold(y_val, score)
        candidates[name] = {"model": model, "threshold": threshold, "validation": metrics}

    consistency, center, scale, threshold, val_metrics = fit_consistency(
        x_clean, y_clean, x_aug, paired_clean, y_aug, x_val, y_val,
        args.seed, args.consistency_weight,
    )
    candidates["consistency_linear"] = {
        "model": consistency, "threshold": threshold, "validation": val_metrics,
        "center": center, "scale": scale,
    }
    selected = max(
        candidates,
        key=lambda name: (
            candidates[name]["validation"]["balanced_accuracy"],
            candidates[name]["validation"]["auroc"],
        ),
    )
    metric_rows, prediction_rows, validation_prediction_rows = [], [], []
    for name, item in candidates.items():
        if name == "consistency_linear":
            predictor = lambda x, item=item: predict_consistency(
                item["model"], x, item["center"], item["scale"]
            )
        else:
            predictor = lambda x, item=item: item["model"].predict_proba(x)[:, 1]
        rows, predictions = evaluate_model(name, views, predictor, item["threshold"])
        metric_rows.extend(rows)
        prediction_rows.extend(predictions)
        _, val_predictions = evaluate_model(
            name, views, predictor, item["threshold"], split="validation"
        )
        validation_prediction_rows.extend(val_predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    validation_predictions = pd.DataFrame(validation_prediction_rows)
    metrics.insert(0, "seed", args.seed)
    predictions.insert(0, "seed", args.seed)
    validation_predictions.insert(0, "seed", args.seed)
    validation = pd.DataFrame([
        {"model": name, **item["validation"], "selected": name == selected}
        for name, item in candidates.items()
    ])
    validation.to_csv(args.output_dir / "validation_metrics.csv", index=False)
    metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    validation_predictions.to_csv(
        args.output_dir / "validation_predictions.csv", index=False
    )
    views.drop(columns="feature").to_csv(args.output_dir / "view_manifest.csv", index=False)
    for name in ("clean_logistic", "augmented_logistic"):
        joblib.dump(candidates[name]["model"], args.output_dir / f"{name}.joblib")
    torch.save({
        "state_dict": consistency.state_dict(), "center": center, "scale": scale,
        "threshold": candidates["consistency_linear"]["threshold"],
    }, args.output_dir / "consistency_linear.pt")
    artifact = {
        "protocol": "z24-supervised-setup-heldout-consistency-c1",
        "manifest": str(manifest), "seed": args.seed, "mode": args.mode,
        "train_setups": [1, 2, 3, 4, 5], "validation_setups": [6, 7],
        "test_setups": [8, 9], "selected_model": selected,
        "selection_uses_validation_damage_labels": True,
        "test_labels_used_for_selection": False,
        "consistency_weight": args.consistency_weight,
    }
    (args.output_dir / "run.json").write_text(json.dumps(artifact, indent=2))
    print("\n=== VALIDATION ===")
    print(validation.sort_values("balanced_accuracy", ascending=False).to_string(index=False))
    print("\n=== TEST ===")
    print(metrics.sort_values(["condition", "model", "replicate"]).to_string(index=False))
    print(f"SUPERVISED_CONSISTENCY_C1_PASS selected={selected} output={args.output_dir}")


if __name__ == "__main__":
    main()

