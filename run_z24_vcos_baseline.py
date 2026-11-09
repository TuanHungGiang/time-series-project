"""Paper-faithful VCOS/MsFSVE baseline on the supervised Z24 split.

Reference
---------
Chen et al., "VCOS: Multi-scale information fusion to feature selection
using fuzzy rough combination entropy", Information Fusion 117 (2025)
102901.  DOI: 10.1016/j.inffus.2024.102901.

This is a clean Python equation reproduction checked against the public MATLAB
implementation (Binbincqnu/MsFSVE).  It intentionally contains no proposed
class balancing or group-stability term; those belong to later ablations.

Protocol
--------
* Complete MAT records are samples.
* Setups 1--5 train, 6--7 validation, 8--9 locked test.
* State 02 is excluded; states 01 and 08 are healthy; all others are damage.
* Feature extraction is deterministic and label-free.
* Min-max parameters, fuzzy scales and VCOS features are fitted on train only
  while selecting delta.  The selected delta is then refitted on train+val.
* The locked test is never used for feature or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, resample_poly, welch
from scipy.special import logsumexp
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


HEALTHY_STATES = {"01", "08"}
EXCLUDED_STATES = {"02"}
SPLIT_SETUPS = {"train": {1, 2, 3, 4, 5}, "val": {6, 7}, "test": {8, 9}}


def resolve_manifest(path: Path, data_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"state": str}).copy()
    needed = {"record_id", "state", "scenario", "excitation", "setup", "relative_path"}
    missing_columns = sorted(needed.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"manifest missing columns: {missing_columns}")
    frame["state"] = frame["state"].str.zfill(2)
    frame["setup"] = pd.to_numeric(frame["setup"], errors="raise").astype(int)
    frame = frame[~frame["state"].isin(EXCLUDED_STATES)].copy()
    frame["label"] = (~frame["state"].isin(HEALTHY_STATES)).astype(int)

    # Reconstruct the locked supervised protocol even when the input is the old
    # healthy-only manifest.  A generated audit manifest produces the same map.
    frame["split"] = "unassigned"
    for split, setups in SPLIT_SETUPS.items():
        frame.loc[frame["setup"].isin(setups), "split"] = split
    if (frame["split"] == "unassigned").any():
        unknown = sorted(frame.loc[frame["split"] == "unassigned", "setup"].unique())
        raise ValueError(f"unassigned setup values: {unknown}")

    frame["path"] = frame["relative_path"].map(lambda value: (data_root / str(value)).resolve())
    missing = frame.loc[~frame["path"].map(Path.exists), "path"]
    if len(missing):
        raise FileNotFoundError(f"missing {len(missing)} MAT files; first={missing.iloc[0]}")
    if frame["record_id"].duplicated().any():
        raise ValueError("record_id must be unique")

    expected = {
        "train": (20, 140),
        "val": (8, 56),
        "test": (8, 56),
    }
    for split, (healthy, damage) in expected.items():
        part = frame[frame["split"] == split]
        actual = (int((part["label"] == 0).sum()), int((part["label"] == 1).sum()))
        if actual != (healthy, damage):
            raise ValueError(f"unexpected {split} class counts {actual}; expected {(healthy, damage)}")
    return frame.sort_values(["split", "state", "excitation", "setup"]).reset_index(drop=True)


def trimmed_top_quarter(values: np.ndarray) -> np.ndarray:
    """Average the strongest 25% sensors after removing the single maximum."""
    count = max(2, int(math.ceil(values.shape[0] * 0.25)))
    top = np.sort(values, axis=0)[-count:]
    return top[:-1].mean(axis=0)


def spectral_entropy(power: np.ndarray) -> np.ndarray:
    probability = power / np.maximum(power.sum(axis=0, keepdims=True), 1e-20)
    return -np.sum(probability * np.log2(probability + 1e-20), axis=0)


def load_vertical(path: Path) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    labels = [str(value).strip() for value in mat["labelshulp"].tolist()]
    selected = [
        i
        for i, name in enumerate(labels)
        if name.endswith("V") and name[0].isdigit() and not name.startswith(("R", "DP"))
    ]
    values = np.asarray(mat["data"][:, selected], dtype=np.float64)
    if not 15 <= values.shape[1] <= 17:
        raise ValueError(f"expected 15--17 vertical sensors in {path}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"NaN/Inf values in {path}")
    return values


def record_features(path: Path, fs_raw: float, fs: float, fmax: float) -> tuple[np.ndarray, list[str]]:
    values = load_vertical(path)
    ratio = fs_raw / fs
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("--fs-raw/--fs must be an integer for deterministic resampling")
    down = int(round(ratio))
    if down > 1:
        values = resample_poly(values, 1, down, axis=0)
    values = detrend(values, axis=0, type="linear")

    std = np.std(values, axis=0)
    rms = np.sqrt(np.mean(values * values, axis=0))
    peak = np.max(np.abs(values), axis=0)
    q95 = np.quantile(np.abs(values), 0.95, axis=0)
    crest = peak / np.maximum(rms, 1e-12)
    sk = np.nan_to_num(skew(values, axis=0, bias=False))
    ku = np.nan_to_num(kurtosis(values, axis=0, fisher=False, bias=False))

    frequency, power = welch(
        values,
        fs=fs,
        axis=0,
        nperseg=min(2048, len(values)),
        noverlap=min(1024, len(values) // 2),
    )
    band = (frequency > 0) & (frequency <= fmax)
    f = frequency[band]
    p = power[band]
    total = np.maximum(p.sum(axis=0), 1e-20)
    probability = p / total
    dominant = f[np.argmax(p, axis=0)]
    centroid = np.sum(f[:, None] * probability, axis=0)
    spread = np.sqrt(np.sum((f[:, None] - centroid[None, :]) ** 2 * probability, axis=0))
    sentropy = spectral_entropy(p)
    edges = np.asarray([3.0, 6.0, 10.0, 15.0, fmax], dtype=float)
    band_energy = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (f >= left) & (f < right)
        band_energy.append(p[mask].sum(axis=0) / total)

    centered = values - values.mean(axis=0, keepdims=True)
    variance = np.sum(centered * centered, axis=0) + 1e-20
    acf1 = np.sum(centered[:-1] * centered[1:], axis=0) / variance
    lag = min(10, len(centered) - 1)
    acf10 = np.sum(centered[:-lag] * centered[lag:], axis=0) / variance

    descriptor_names = [
        "std",
        "rms",
        "abs_peak",
        "abs_q95",
        "crest_factor",
        "skewness",
        "kurtosis",
        "dominant_frequency",
        "spectral_centroid",
        "spectral_spread",
        "spectral_entropy",
        "energy_3_6Hz",
        "energy_6_10Hz",
        "energy_10_15Hz",
        "energy_15_20Hz",
        "acf_lag1",
        "acf_lag10",
    ]
    sensor = np.column_stack(
        [std, rms, peak, q95, crest, sk, ku, dominant, centroid, spread, sentropy, *band_energy, acf1, acf10]
    )
    sensor = np.nan_to_num(sensor, nan=0.0, posinf=0.0, neginf=0.0)
    pools = {
        "sensor_mean": sensor.mean(axis=0),
        "sensor_std": sensor.std(axis=0),
        "sensor_median": np.median(sensor, axis=0),
        "sensor_max": sensor.max(axis=0),
        "sensor_trimmed_top25": trimmed_top_quarter(sensor),
    }
    feature_names = [f"{pool}__{name}" for pool in pools for name in descriptor_names]
    vector = np.concatenate(list(pools.values())).astype(np.float64)
    return vector, feature_names


def build_feature_table(
    frame: pd.DataFrame,
    output: Path,
    fs_raw: float,
    fs: float,
    fmax: float,
    rebuild: bool,
) -> tuple[np.ndarray, list[str]]:
    cache_path = output / "z24_record_features.csv"
    if cache_path.exists() and not rebuild:
        cached = pd.read_csv(cache_path)
        if cached["record_id"].tolist() == frame["record_id"].tolist():
            names = [column for column in cached.columns if column.startswith("feature__")]
            return cached[names].to_numpy(dtype=np.float64), [name.removeprefix("feature__") for name in names]

    rows, names = [], None
    for index, row in enumerate(frame.itertuples(), 1):
        vector, current_names = record_features(Path(row.path), fs_raw, fs, fmax)
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("inconsistent feature names")
        rows.append(vector)
        if index == 1 or index % 24 == 0 or index == len(frame):
            print(f"features {index}/{len(frame)} {row.record_id}", flush=True)
    matrix = np.vstack(rows)
    table = frame[["record_id", "state", "scenario", "excitation", "setup", "split", "label"]].copy()
    for column, values in zip(names, matrix.T):
        table[f"feature__{column}"] = values
    table.to_csv(cache_path, index=False)
    return matrix, list(names)


@dataclass
class MinMaxState:
    minimum: np.ndarray
    scale: np.ndarray
    retained: np.ndarray


def fit_minmax(x: np.ndarray, tolerance: float = 1e-12) -> MinMaxState:
    minimum = x.min(axis=0)
    scale = x.max(axis=0) - minimum
    retained = scale > tolerance
    if not retained.any():
        raise ValueError("all features are constant in the fitting partition")
    return MinMaxState(minimum=minimum, scale=scale, retained=retained)


def minmax_transform(x: np.ndarray, state: MinMaxState) -> np.ndarray:
    z = (x[:, state.retained] - state.minimum[state.retained]) / state.scale[state.retained]
    return np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)


def fuzzy_relation(values: np.ndarray, alpha: float) -> np.ndarray:
    feature_range = float(values.max() - values.min())
    if feature_range <= 1e-15:
        return np.ones((len(values), len(values)), dtype=np.float64)
    difference = np.abs(values[:, None] - values[None, :]) / feature_range
    return np.clip(1.2 - alpha * difference, 0.0, 1.0)


def maximum_scale(values: np.ndarray, xi: float, alpha_limit: float) -> float:
    previous = None
    alpha = 1.2
    while alpha <= alpha_limit + 1e-9:
        granularity = float(fuzzy_relation(values, alpha).sum())
        if previous is not None and previous - granularity <= xi:
            return alpha
        previous = granularity
        alpha += 1.0
    return alpha_limit


def optimal_scale(values: np.ndarray, y: np.ndarray, delta: float, xi: float, alpha_limit: float) -> float:
    maximum = maximum_scale(values, xi, alpha_limit)
    decision = y[:, None] == y[None, :]
    candidates = np.arange(1.2, maximum + 1e-9, 5.0)
    if not len(candidates):
        candidates = np.asarray([1.2])
    selected = float(candidates[-1])
    for alpha in candidates:
        relation = fuzzy_relation(values, float(alpha))
        consistency = float(np.mean(relation <= decision.astype(float) + 1e-12))
        selected = float(alpha)
        if consistency >= delta:
            break
    return selected


def fuzzy_combination(value: np.ndarray | float) -> np.ndarray | float:
    return value * (value - 1.0) / 2.0


def frce_score(relation: np.ndarray, y: np.ndarray) -> float:
    """Fuzzy rough combination entropy from the paper and author MATLAB code."""
    n = len(y)
    lower, upper = [], []
    for label in np.unique(y):
        inside = y == label
        outside = ~inside
        if outside.any():
            lower.append(np.min(1.0 - relation[:, outside], axis=1))
        else:
            lower.append(np.ones(n))
        upper.append(np.max(relation[:, inside], axis=1))
    lower_max = np.max(np.vstack(lower), axis=0)
    upper_max = np.max(np.vstack(upper), axis=0)
    accuracy = float(lower_max.sum() / max(upper_max.sum(), 1e-15))

    full = float(fuzzy_combination(float(n)))
    if full <= 0:
        return 0.0
    row_cardinality = relation.sum(axis=1)
    combination_entropy = float(np.sum(full - fuzzy_combination(row_cardinality)) / (n * full))
    return accuracy * combination_entropy


@dataclass
class VCOSResult:
    selected: list[int]
    scales: np.ndarray
    trace: list[dict]


def fit_vcos(
    x: np.ndarray,
    y: np.ndarray,
    delta: float,
    epsilon: float,
    xi: float,
    alpha_limit: float,
    max_features: int | None,
) -> VCOSResult:
    scales = np.asarray(
        [optimal_scale(x[:, j], y, delta, xi, alpha_limit) for j in range(x.shape[1])],
        dtype=float,
    )
    relations = [fuzzy_relation(x[:, j], scales[j]) for j in range(x.shape[1])]
    selected: list[int] = []
    remaining = list(range(x.shape[1]))
    current_relation = np.ones((len(x), len(x)), dtype=np.float64)
    current_score = frce_score(current_relation, y)
    trace: list[dict] = []
    limit = x.shape[1] if max_features is None else min(max_features, x.shape[1])

    while remaining and len(selected) < limit:
        candidates = []
        for index in remaining:
            candidate_relation = np.minimum(current_relation, relations[index])
            score = frce_score(candidate_relation, y)
            candidates.append((score - current_score, score, index, candidate_relation))
        gain, score, winner, winner_relation = max(candidates, key=lambda item: (item[0], -item[2]))
        trace.append(
            {
                "step": len(selected) + 1,
                "feature_index": int(winner),
                "scale": float(scales[winner]),
                "frce_before": float(current_score),
                "frce_after": float(score),
                "significance": float(gain),
                "accepted": bool(gain > epsilon),
            }
        )
        if gain <= epsilon:
            break
        selected.append(winner)
        remaining.remove(winner)
        current_relation = winner_relation
        current_score = score

    if not selected:
        # The author demo falls back to all features when the reduct is empty.
        selected = list(range(x.shape[1]))
    return VCOSResult(selected=selected, scales=scales, trace=trace)


class KernelNaiveBayes:
    """Independent Gaussian-kernel NB matching MATLAB fitcnb(...,'kernel')."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelNaiveBayes":
        self.classes_ = np.unique(y)
        self.models_: dict[int, list[tuple[np.ndarray, float]]] = {}
        self.log_prior_: dict[int, float] = {}
        for label in self.classes_:
            values = np.asarray(x[y == label], dtype=np.float64)
            self.log_prior_[int(label)] = float(np.log(len(values) / len(x)))
            models = []
            for column in values.T:
                std = float(np.std(column, ddof=1)) if len(column) > 1 else 0.0
                bandwidth = 1.06 * max(std, 1e-3) * max(len(column), 2) ** (-0.2)
                models.append((column.copy(), max(bandwidth, 1e-3)))
            self.models_[int(label)] = models
        return self

    def _joint_log(self, x: np.ndarray, label: int) -> np.ndarray:
        result = np.full(len(x), self.log_prior_[label], dtype=np.float64)
        for j, (samples, bandwidth) in enumerate(self.models_[label]):
            z = (x[:, j, None] - samples[None, :]) / bandwidth
            result += logsumexp(-0.5 * z * z, axis=1) - math.log(len(samples) * bandwidth * math.sqrt(2 * math.pi))
        return result

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        joint = np.column_stack([self._joint_log(x, int(label)) for label in self.classes_])
        normalizer = logsumexp(joint, axis=1, keepdims=True)
        return np.exp(joint - normalizer)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probability = self.predict_proba(x)
        return self.classes_[np.argmax(probability, axis=1)]


def classifiers(seed: int) -> dict[str, object]:
    return {
        "kernel_nb": KernelNaiveBayes(),
        # MATLAB fitcecoc uses a linear SVM learner by default.  Decision scores
        # are sufficient for AUROC/AUPRC and avoid post-hoc probability fitting.
        "linear_svm": SVC(kernel="linear", C=1.0, probability=False, random_state=seed),
        "knn_k3": KNeighborsClassifier(n_neighbors=3),
        "cart": DecisionTreeClassifier(random_state=seed),
    }


def positive_score(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(x)
        classes = list(model.classes_)
        return probability[:, classes.index(1)]
    decision = model.decision_function(x)
    return np.asarray(decision, dtype=float)


def metric_row(y: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "damage_precision": float(precision_score(y, prediction, zero_division=0)),
        "damage_recall": float(recall_score(y, prediction, zero_division=0)),
        "damage_f1": float(f1_score(y, prediction, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def fit_evaluate(model: object, x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    model.fit(x_fit, y_fit)
    prediction = np.asarray(model.predict(x_eval), dtype=int)
    score = positive_score(model, x_eval)
    return metric_row(y_eval, prediction, score), prediction, score


def write_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_vcos_baseline"))
    parser.add_argument("--delta-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--scale-xi", type=float, default=0.01)
    parser.add_argument("--alpha-limit", type=float, default=100.0)
    parser.add_argument("--max-features", type=int, default=0, help="0 means no explicit cap")
    parser.add_argument("--fs-raw", type=float, default=100.0)
    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deltas = [float(value) for value in args.delta_grid.split(",")]
    if not deltas or any(not 0 <= value <= 1 for value in deltas):
        raise ValueError("--delta-grid values must be in [0,1]")

    started = time.time()
    frame = resolve_manifest(args.manifest, args.data_root)
    x_raw, all_feature_names = build_feature_table(
        frame, args.output, args.fs_raw, args.fs, args.fmax, args.rebuild_features
    )
    train_mask = frame["split"].eq("train").to_numpy()
    val_mask = frame["split"].eq("val").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    y = frame["label"].to_numpy(dtype=int)
    max_features = args.max_features or None

    train_scaler = fit_minmax(x_raw[train_mask])
    x_train = minmax_transform(x_raw[train_mask], train_scaler)
    x_val = minmax_transform(x_raw[val_mask], train_scaler)
    y_train, y_val = y[train_mask], y[val_mask]
    retained_names = np.asarray(all_feature_names)[train_scaler.retained]

    validation_rows, validation_reducts = [], {}
    for delta in deltas:
        fitted = fit_vcos(
            x_train,
            y_train,
            delta,
            args.epsilon,
            args.scale_xi,
            args.alpha_limit,
            max_features,
        )
        validation_reducts[delta] = fitted
        for name, model in classifiers(args.seed).items():
            metrics, _, _ = fit_evaluate(
                model,
                x_train[:, fitted.selected],
                y_train,
                x_val[:, fitted.selected],
                y_val,
            )
            validation_rows.append(
                {
                    "classifier": name,
                    "delta": delta,
                    "selected_features": len(fitted.selected),
                    **metrics,
                }
            )
        print(
            f"delta={delta:.1f} selected={len(fitted.selected)} "
            f"features={[str(retained_names[i]) for i in fitted.selected[:5]]}",
            flush=True,
        )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(args.output / "validation_delta_grid.csv", index=False)

    selected_hyperparameters = (
        validation.sort_values(
            ["classifier", "balanced_accuracy", "macro_f1", "mcc", "selected_features", "delta"],
            ascending=[True, False, False, False, True, True],
        )
        .groupby("classifier", as_index=False)
        .first()
    )
    selected_hyperparameters.to_csv(args.output / "selected_validation_configuration.csv", index=False)

    fit_mask = train_mask | val_mask
    final_scaler = fit_minmax(x_raw[fit_mask])
    x_fit = minmax_transform(x_raw[fit_mask], final_scaler)
    x_test = minmax_transform(x_raw[test_mask], final_scaler)
    y_fit, y_test = y[fit_mask], y[test_mask]
    final_names = np.asarray(all_feature_names)[final_scaler.retained]

    test_rows, prediction_rows, selected_rows, trace_rows = [], [], [], []
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, (classifier_name, selected_config) in zip(axes.ravel(), selected_hyperparameters.set_index("classifier").iterrows()):
        delta = float(selected_config["delta"])
        fitted = fit_vcos(
            x_fit,
            y_fit,
            delta,
            args.epsilon,
            args.scale_xi,
            args.alpha_limit,
            max_features,
        )
        model = classifiers(args.seed)[classifier_name]
        metrics, prediction, score = fit_evaluate(
            model,
            x_fit[:, fitted.selected],
            y_fit,
            x_test[:, fitted.selected],
            y_test,
        )
        test_rows.append(
            {
                "method": "VCOS_InformationFusion2025_equation_reproduction",
                "classifier": classifier_name,
                "selected_delta": delta,
                "selected_features": len(fitted.selected),
                **metrics,
            }
        )
        for order, feature_index in enumerate(fitted.selected, 1):
            selected_rows.append(
                {
                    "classifier": classifier_name,
                    "order": order,
                    "feature_index_after_constant_filter": int(feature_index),
                    "feature": str(final_names[feature_index]),
                    "scale": float(fitted.scales[feature_index]),
                }
            )
        for item in fitted.trace:
            trace_rows.append({"classifier": classifier_name, "delta": delta, **item})
        test_frame = frame.loc[test_mask, ["record_id", "state", "scenario", "excitation", "setup", "label"]].copy()
        test_frame["classifier"] = classifier_name
        test_frame["prediction"] = prediction
        test_frame["damage_score"] = score
        prediction_rows.append(test_frame)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            labels=[0, 1],
            display_labels=["healthy", "damage"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(classifier_name)

    test_results = pd.DataFrame(test_rows).sort_values("balanced_accuracy", ascending=False)
    test_results.to_csv(args.output / "locked_test_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(args.output / "locked_test_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output / "selected_features.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(args.output / "vcos_selection_trace.csv", index=False)
    figure.suptitle("VCOS baseline: locked-test confusion matrices", weight="bold")
    figure.savefig(args.output / "locked_test_confusion_matrices.png", dpi=180)
    plt.close(figure)

    compact_columns = [
        "classifier",
        "selected_delta",
        "selected_features",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "damage_recall",
        "specificity",
        "auroc",
        "auprc",
    ]
    report = f"""# Z24 VCOS baseline report

## Protocol

- Paper: Chen et al., Information Fusion 117 (2025), 102901.
- Task: supervised binary structural damage detection.
- Train setups: 1--5; validation setups: 6--7; locked-test setups: 8--9.
- Train/validation/test counts: 160/64/64 complete records.
- Healthy states: S01 and S08; S02 excluded.
- Delta selected using validation balanced accuracy, then macro-F1 and MCC.
- Final VCOS and classifier refitted on train+validation before one locked-test evaluation.
- No class-balanced or group-stable proposed term is active in this baseline.

## Locked-test results

{write_markdown_table(test_results[compact_columns])}

## Interpretation guard

This protocol measures transfer to held-out roving-sensor setups while retaining
the same structural damage states across partitions. It must not be described
as generalization to unseen damage types. Accuracy is secondary because damage
represents 87.5% of every partition; balanced accuracy, macro-F1 and MCC are the
primary metrics.
"""
    (args.output / "VCOS_BASELINE_REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "paper": "VCOS: Multi-scale information fusion to feature selection using fuzzy rough combination entropy",
        "doi": "10.1016/j.inffus.2024.102901",
        "author_code": "https://github.com/Binbincqnu/MsFSVE",
        "damage_used_for_supervised_fit": True,
        "locked_test_used_for_selection": False,
        "split": {name: sorted(values) for name, values in SPLIT_SETUPS.items()},
        "delta_grid": deltas,
        "epsilon": args.epsilon,
        "scale_xi": args.scale_xi,
        "alpha_limit": args.alpha_limit,
        "raw_features": len(all_feature_names),
        "seconds": time.time() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSELECTED ON VALIDATION", flush=True)
    print(selected_hyperparameters[["classifier", "delta", "selected_features", "balanced_accuracy", "macro_f1", "mcc"]].to_string(index=False), flush=True)
    print("\nLOCKED TEST RESULTS", flush=True)
    print(test_results[compact_columns].to_string(index=False), flush=True)
    print(f"outputs={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

