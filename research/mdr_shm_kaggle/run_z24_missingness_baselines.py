from __future__ import annotations

"""Missing-data stress test for the canonical Z24 source baselines.

Models and normalization are fitted on clean healthy training windows. Score
polarity and threshold are selected once on clean labeled validation records.
The frozen decision rule is then evaluated on clean and corrupted test records.
All models receive identical deterministic masks for a given record/condition.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    MODELS, FitMinMax, Normalizer, fit_gdn, fit_reconstructor, fit_scatterad,
    fit_usad, publication_metrics, score_gdn, score_reconstructor,
    score_scatterad, score_usad, seed_everything,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores,
    load_manifest,
    read_stable_vertical,
    record_windows,
    robust_score_reference,
    split_training_windows,
    stable_sensor_map,
    threshold_from_labeled_validation,
)


def stable_rng(record_id: str, condition: str, replicate: int) -> np.random.Generator:
    token = f"z24-missing-v1|{record_id}|{condition}|{replicate}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(seed)


def scenarios(mode: str) -> list[tuple[str, str, float, int]]:
    output = [("clean", "clean", 0.0, 0)]
    if mode == "quick":
        for kind in ("point", "block", "sensor", "targeted"):
            output.append((f"{kind}_0.30", kind, 0.30, 0))
        return output
    for rate in (0.10, 0.30, 0.50):
        for kind in ("point", "block", "sensor"):
            for replicate in range(5):
                output.append((f"{kind}_{rate:.2f}", kind, rate, replicate))
        output.append((f"targeted_{rate:.2f}", "targeted", rate, 0))
    return output


def corrupt(
    values: np.ndarray,
    kind: str,
    rate: float,
    record_id: str,
    condition: str,
    replicate: int,
    sensor_importance: np.ndarray,
) -> np.ndarray:
    """Apply missingness after train-only z-scoring; zero is healthy mean."""
    output = values.copy()
    if kind == "clean":
        return output
    rng = stable_rng(record_id, condition, replicate)
    windows, channels, length = output.shape
    if kind == "point":
        output[rng.random(output.shape) < rate] = 0.0
    elif kind == "block":
        block = max(1, int(round(rate * length)))
        for window in range(windows):
            for channel in range(channels):
                start = int(rng.integers(0, length - block + 1))
                output[window, channel, start:start + block] = 0.0
    elif kind in {"sensor", "targeted"}:
        count = min(channels - 1, max(1, int(round(rate * channels))))
        if kind == "sensor":
            selected = rng.choice(channels, size=count, replace=False)
        else:
            selected = np.argsort(sensor_importance)[-count:]
        output[:, selected, :] = 0.0
    else:
        raise ValueError(kind)
    return output


def group_path(run_dir: Path, setup: int, excitation: str) -> Path:
    return run_dir / "groups" / f"setup{setup:02d}_{excitation}.csv"


def fit_scorer(
    name: str,
    fit: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
):
    """Fit once and return a callable so conditions can be scored sequentially."""
    if name == "pca_spe":
        train = fit.reshape(len(fit), -1)
        model = PCA(n_components=0.95, svd_solver="full").fit(train)

        def score(values):
            flat = values.reshape(len(values), -1)
            reconstructed = model.inverse_transform(model.transform(flat))
            return np.mean((flat - reconstructed) ** 2, axis=1)

        return score
    if name == "pca_pyod":
        try:
            from pyod.models.pca import PCA as PyODPCA
        except ImportError as error:
            raise RuntimeError("pca_pyod requires `pip install pyod`.") from error
        model = PyODPCA(
            n_components=None, n_selected_components=10,
            contamination=0.1, random_state=0,
        ).fit(fit.reshape(len(fit), -1))
        return lambda values: model.decision_function(values.reshape(len(values), -1))
    if name == "usad":
        scale = FitMinMax(fit)
        model = fit_usad(scale.transform(fit), epochs, batch_size, device)
        return lambda values: score_usad(
            model, scale.transform(values), batch_size, device
        )
    if name == "gdn":
        model = fit_gdn(fit, epochs, batch_size, device)
        return lambda values: score_gdn(model, values, batch_size, device)
    if name == "scatterad":
        model = fit_scatterad(fit, epochs, device)
        return lambda values: score_scatterad(model, values, device)
    scale = FitMinMax(fit) if name == "tranad" else None
    training = scale.transform(fit) if scale is not None else fit
    model = fit_reconstructor(name, training, epochs, batch_size, device)

    def score(values):
        transformed = scale.transform(values) if scale is not None else values
        return score_reconstructor(name, model, transformed, batch_size, device)

    return score


def run_group(
    model_name: str,
    seed: int,
    group_index: int,
    group: pd.DataFrame,
    labels: list[str],
    scenario_rows: list[tuple[str, str, float, int]],
    raw_window: int,
    stride: int,
    downsample: int,
    epochs: int,
    batch_size: int,
    device,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    train = group[(group.split == "train") & (group.label == 0)].copy()
    validation = group[group.split == "val"].copy()
    test = group[group.split == "test"].copy()
    if len(train) not in (1, 2):
        raise RuntimeError(f"Expected one/two healthy train records, got {len(train)}")

    train_by_record = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), raw_window, stride,
            "raw_temporal", downsample, 128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_by_record)
    normalizer = Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    sensor_importance = np.mean(np.square(fit_raw), axis=(0, 2))

    clean_cache: dict[str, np.ndarray] = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), raw_window, stride,
            "raw_temporal", downsample, 128,
        )
        clean_cache[str(row.record_id)] = normalizer.transform(raw)

    # Fit exactly once. Score validation and each corruption cohort separately
    # so full mode never materializes all 49 cohorts in RAM simultaneously.
    seed_everything(seed + 1009 * group_index)
    scorer = fit_scorer(model_name, fit, epochs, batch_size, device)
    internal_scores = scorer(internal)
    reference = robust_score_reference(internal_scores)

    arrays, counts, metadata = [], [], []
    for row in validation.itertuples():
        current = clean_cache[str(row.record_id)]
        arrays.append(current)
        counts.append(len(current))
        metadata.append({
            "record_id": row.record_id, "split": "val", "label": int(row.label),
            "state": row.state, "setup": row.setup, "excitation": row.excitation,
            "condition": "clean", "kind": "clean", "rate": 0.0, "replicate": 0,
        })
    validation_values = np.concatenate(arrays)
    validation_scores = scorer(validation_values)
    upper, two_sided = aggregate_record_scores(validation_scores, counts, reference)
    validation_frame = pd.DataFrame(metadata)
    validation_frame["score_upper"] = upper
    validation_frame["score_lower"] = -np.asarray(upper, dtype=float)
    validation_frame["score_two_sided"] = two_sided
    validation_frame["windows"] = counts

    condition_frames = []
    for condition, kind, rate, replicate in scenario_rows:
        arrays, counts, metadata = [], [], []
        for row in test.itertuples():
            current = corrupt(
                clean_cache[str(row.record_id)], kind, rate, str(row.record_id),
                condition, replicate, sensor_importance,
            )
            arrays.append(current)
            counts.append(len(current))
            metadata.append({
                "record_id": row.record_id, "split": "test", "label": int(row.label),
                "state": row.state, "setup": row.setup, "excitation": row.excitation,
                "condition": condition, "kind": kind, "rate": rate,
                "replicate": replicate,
            })
        condition_values = np.concatenate(arrays)
        condition_scores = scorer(condition_values)
        upper, two_sided = aggregate_record_scores(condition_scores, counts, reference)
        condition_frame = pd.DataFrame(metadata)
        condition_frame["score_upper"] = upper
        condition_frame["score_lower"] = -np.asarray(upper, dtype=float)
        condition_frame["score_two_sided"] = two_sided
        condition_frame["windows"] = counts
        condition_frames.append(condition_frame)

    result = pd.concat([validation_frame, *condition_frames], ignore_index=True)
    result["stable_sensors"] = len(labels)
    result["model"] = model_name
    result["seed"] = seed
    result.to_csv(output, index=False)


def collect(run_dir: Path, sensor_map: dict, scenario_count: int) -> pd.DataFrame:
    paths = [group_path(run_dir, setup, excitation) for setup, excitation in sorted(sensor_map)]
    if not all(path.exists() for path in paths):
        raise RuntimeError("Incomplete setup/excitation groups")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = 132 + 132 * scenario_count
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} validation/test rows, got {len(frame)}")
    keys = ["record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate stress-test record keys")
    return frame


def finish(model_name: str, seed: int, run_dir: Path, frame: pd.DataFrame) -> None:
    validation = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    candidates = []
    for tail in ("upper", "lower", "two_sided"):
        threshold, metric = threshold_from_labeled_validation(
            validation.label, validation[f"score_{tail}"]
        )
        key = (metric["ba"], metric["auc"], metric["f1"], metric["specificity"])
        candidates.append((key, tail, threshold, metric))
    _, tail, threshold, validation_metric = max(candidates, key=lambda item: item[0])
    validation["score"] = validation[f"score_{tail}"]
    test["score"] = test[f"score_{tail}"]
    validation["threshold"] = threshold
    test["threshold"] = threshold

    rows = []
    for (condition, replicate), current in test.groupby(["condition", "replicate"], sort=False):
        rows.append({
            "model": model_name, "seed": seed, "condition": condition,
            "replicate": int(replicate), "score_tail": tail,
            "threshold_from_clean_validation": threshold,
            **publication_metrics(current.label, current.score, threshold),
            "records": len(current),
            "healthy_records": int((current.label == 0).sum()),
            "damage_records": int((current.label == 1).sum()),
        })
    metrics = pd.DataFrame(rows)
    clean = metrics[metrics.condition == "clean"].iloc[0]
    for name in ("auc", "ba", "recall", "specificity", "f1"):
        metrics[f"delta_{name}"] = metrics[name] - float(clean[name])
    metrics["validation_auc"] = validation_metric["auc"]
    metrics["validation_ba"] = validation_metric["ba"]
    metrics["validation_recall"] = validation_metric["recall"]
    metrics["validation_specificity"] = validation_metric["specificity"]
    validation.to_csv(run_dir / "validation_scores.csv", index=False)
    test.to_csv(run_dir / "scores.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "complete.json").write_text(
        json.dumps({"model": model_name, "seed": seed, "tail": tail,
                    "threshold": threshold}, indent=2), encoding="utf-8"
    )


def refresh(output: Path) -> None:
    metric_paths = sorted((output / "runs").glob("*/metrics.csv"))
    if not metric_paths:
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    scores = pd.concat([pd.read_csv(path.parent / "scores.csv") for path in metric_paths],
                       ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary = metrics.groupby(["model", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"), delta_ba_mean=("delta_ba", "mean"),
        runs=("seed", "size"),
    )
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="model", columns="condition", values="auc_mean")
    axis = pivot.plot.bar(figsize=(14, 6), ylim=(0, 1.02), rot=25, grid=True)
    axis.set_title("Z24 missing-data baseline stress test")
    axis.set_ylabel("Record-level AUROC")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "missingness_auc.png", dpi=220, bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(frame)
    scenario_rows = scenarios(args.mode)
    protocol = {
        "dataset": "Z24", "experiment": "missing-data baseline stress test",
        "fit": "clean healthy train only",
        "selection": "clean labeled validation only",
        "test_conditions": [list(value) for value in scenario_rows],
        "imputation": "zero after train-only channel z-score (healthy mean)",
        "mask_identity": "SHA256(record_id, condition, replicate), shared by all models",
        "targeted_sensor_rule": "highest mean-square amplitude on clean healthy train only",
        "threshold_recalibrated_under_corruption": False,
        "models": args.models, "seeds": args.seeds, "epochs": args.epochs,
        "input": "[batch,C_cell,512] after detrend and polyphase downsample",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    used = frame[frame.split.isin(["train", "val", "test"])].copy()
    for model_name in args.models:
        model_seeds = args.seeds[:1] if model_name.startswith("pca_") else args.seeds
        for seed in model_seeds:
            run_dir = args.output_dir / "runs" / f"{model_name}__{seed}"
            if (run_dir / "complete.json").exists():
                print(f"SKIP complete model={model_name} seed={seed}", flush=True)
                continue
            started = time.time()
            for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
                path = group_path(run_dir, key[0], key[1])
                if path.exists():
                    print(f"SKIP group model={model_name} seed={seed} cell={key}", flush=True)
                    continue
                print(f"model={model_name} seed={seed} cell={key}", flush=True)
                group = used[(used.setup == key[0]) & (used.excitation == key[1])]
                run_group(
                    model_name, seed, group_index, group, labels, scenario_rows,
                    args.raw_window, args.window_stride, args.downsample,
                    args.epochs, args.batch_size, device, path,
                )
            completed = collect(run_dir, sensor_map, len(scenario_rows))
            finish(model_name, seed, run_dir, completed)
            elapsed = time.time() - started
            info = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
            info["seconds"] = elapsed
            (run_dir / "complete.json").write_text(
                json.dumps(info, indent=2), encoding="utf-8"
            )
            refresh(args.output_dir)
            print(pd.read_csv(run_dir / "metrics.csv").to_string(index=False), flush=True)
    refresh(args.output_dir)
    print(pd.read_csv(args.output_dir / "summary.csv").to_string(index=False))
    print(f"Z24_MISSINGNESS_BASELINES_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

