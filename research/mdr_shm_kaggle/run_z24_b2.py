from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, find_peaks, welch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


FS = 100.0
FMIN, FMAX = 0.5, 30.0
BANDS = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0))


def find_manifest(root: Path) -> Path:
    candidates = []
    for path in root.rglob("*.csv"):
        try:
            columns = set(pd.read_csv(path, nrows=2).columns)
        except Exception:
            continue
        if {"record_id", "role", "label"}.issubset(columns):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one record manifest with record_id/role/label; "
            f"found {candidates}"
        )
    return candidates[0]


def build_file_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*.mat"):
        index.setdefault(path.name.lower(), []).append(path)
    if not index:
        raise RuntimeError(f"No MAT files found below {root}")
    return index


def resolve_source(row: pd.Series, index: dict[str, list[Path]]) -> Path:
    raw = str(row.get("source_path", row.get("path", "")))
    direct = Path(raw)
    if direct.exists():
        return direct
    names = []
    if raw and raw.lower() != "nan":
        names.append(Path(raw).name.lower())
    names.extend([
        f"{row['record_id']}.mat".lower(),
        str(row["record_id"]).lower(),
    ])
    candidates: list[Path] = []
    for name in names:
        candidates.extend(index.get(name, []))
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        role = str(row["role"]).lower()
        role_aliases = ("val", "calibration") if role in ("val", "calibration") else (role,)
        narrowed = [
            p for p in candidates
            if any(alias in [part.lower() for part in p.parts] for alias in role_aliases)
        ]
        if len(narrowed) == 1:
            return narrowed[0]
    raise RuntimeError(f"Cannot uniquely resolve {row['record_id']} from {raw}: {candidates}")


def load_vertical(path: Path) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    data = np.asarray(mat["data"], dtype=np.float32)
    raw_labels = np.asarray(mat["labelshulp"]).squeeze().tolist()
    if not isinstance(raw_labels, list):
        raw_labels = [raw_labels]
    labels = [str(value).strip() for value in raw_labels]
    indices = [
        i for i, label in enumerate(labels)
        if label.endswith("V") and label[:1].isdigit()
        and not label.startswith(("R", "DP"))
    ]
    if not indices:
        raise RuntimeError(f"No vertical channels found in {path}; labels={labels[:5]}")
    signals = data[:, indices].T
    if signals.ndim != 2 or not np.isfinite(signals).all():
        raise RuntimeError(f"Invalid sensor matrix in {path}: {signals.shape}")
    return signals


def normalized_sensor_spectra(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = detrend(signals, axis=1, type="linear")
    frequencies, psd = welch(
        values, fs=FS, window="hann", nperseg=4096, noverlap=2048,
        axis=1, detrend=False, scaling="density"
    )
    keep = (frequencies >= FMIN) & (frequencies <= FMAX)
    frequencies, psd = frequencies[keep], psd[:, keep]
    total = np.trapezoid(psd, frequencies, axis=1)
    psd = psd / np.maximum(total[:, None], np.finfo(float).tiny)
    return frequencies, psd


def record_spectrum(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, sensor_psd = normalized_sensor_spectra(signals)
    return frequencies, np.median(sensor_psd, axis=0)


def learn_anchors(spectra: list[np.ndarray], frequencies: np.ndarray, count: int = 4) -> list[float]:
    reference = np.median(np.stack(spectra), axis=0)
    distance = max(1, int(round(0.75 / (frequencies[1] - frequencies[0]))))
    peaks, _ = find_peaks(reference, distance=distance)
    if len(peaks) < count:
        raise RuntimeError("Insufficient healthy spectral peaks")
    strongest = peaks[np.argsort(reference[peaks])[-count:]]
    return [float(v) for v in sorted(frequencies[strongest])]


def features(spectrum: np.ndarray, frequencies: np.ndarray, anchors: list[float]) -> np.ndarray:
    output = []
    for anchor in anchors:
        local = np.flatnonzero(np.abs(frequencies - anchor) <= 0.6)
        peak = local[int(np.argmax(spectrum[local]))]
        output.extend([float(frequencies[peak]), float(np.log10(spectrum[peak] + 1e-16))])
    for low, high in BANDS:
        keep = (frequencies >= low) & (frequencies < high)
        output.append(float(np.log10(np.trapezoid(spectrum[keep], frequencies[keep]) + 1e-16)))
    return np.asarray(output)


@dataclass
class Model:
    anchors: list[float]
    center: np.ndarray
    scale: np.ndarray
    mean: np.ndarray
    inv_cov: np.ndarray
    threshold: float = 0.0

    @classmethod
    def fit(cls, anchors: list[float], x: np.ndarray) -> "Model":
        center = np.median(x, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(x - center), axis=0), 1e-6)
        z = (x - center) / scale
        mean = z.mean(axis=0)
        empirical = np.cov(z, rowvar=False)
        covariance = 0.5 * empirical + 0.5 * np.diag(np.diag(empirical))
        covariance += np.eye(covariance.shape[0]) * 1e-5
        return cls(anchors, center, scale, mean, np.linalg.pinv(covariance))

    def raw_score(self, x: np.ndarray) -> float:
        delta = (x - self.center) / self.scale - self.mean
        return float(np.sqrt(max(delta @ self.inv_cov @ delta, 0.0)))

    def score(self, x: np.ndarray) -> float:
        if self.threshold <= 0:
            raise RuntimeError("Model is not calibrated")
        return self.raw_score(x) / self.threshold

    def as_dict(self) -> dict:
        return {
            "anchors": self.anchors,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "mean": self.mean.tolist(),
            "inverse_covariance": self.inv_cov.tolist(),
            "threshold": self.threshold,
        }


def metric_dict(labels: list[int], scores: list[float]) -> dict:
    y = np.asarray(labels)
    s = np.asarray(scores)
    pred = s > 1.0
    return {
        "auroc": float(roc_auc_score(y, s)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(recall_score(1 - y, ~pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def stable_rng(record_id: str, condition: str, replicate: int) -> np.random.Generator:
    token = f"mdr-shm-kaggle-v1|{record_id}|{condition}|{replicate}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))


def corrupt(signals: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    output = signals.copy()
    sensor_count, sample_count = output.shape
    affected_count = max(1, int(round(0.30 * sensor_count)))
    affected = rng.choice(sensor_count, size=affected_count, replace=False)
    if condition.startswith("missing_"):
        fraction = float(condition.split("_")[1])
        length = int(round(sample_count * fraction))
        coordinates = np.arange(sample_count)
        for sensor in affected:
            start = int(rng.integers(0, sample_count - length + 1))
            mask = np.ones(sample_count, dtype=bool)
            mask[start:start + length] = False
            output[sensor, ~mask] = np.interp(
                coordinates[~mask], coordinates[mask], output[sensor, mask]
            )
    elif condition.startswith("noise_"):
        snr = float(condition.split("_")[1])
        for sensor in affected:
            centered = output[sensor] - output[sensor].mean()
            power = float(np.mean(centered ** 2))
            sigma = np.sqrt(max(power, 1e-30) / 10 ** (snr / 10))
            output[sensor] += rng.normal(0, sigma, sample_count).astype(np.float32)
    elif condition.startswith("clip_"):
        quantile = float(condition.split("_")[1])
        for sensor in affected:
            center = float(np.median(output[sensor]))
            radius = float(np.quantile(np.abs(output[sensor] - center), quantile))
            output[sensor] = np.clip(output[sensor], center - radius, center + radius)
    else:
        raise ValueError(condition)
    return output


def excitation(row: pd.Series) -> str:
    value = str(row.get("excitation", "")).lower()
    if value in ("avt", "fvt"):
        return value
    token = " ".join(str(row.get(key, "")) for key in ("record_id", "source_path")).lower()
    if "fvt" in token:
        return "fvt"
    if "avt" in token:
        return "avt"
    raise RuntimeError(f"Cannot infer AVT/FVT for {row['record_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/mdr_shm_b2"))
    args = parser.parse_args()

    manifest_path = find_manifest(args.input_root)
    frame = pd.read_csv(manifest_path)
    frame["role"] = frame["role"].astype(str).str.lower().replace({"val": "calibration"})
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["role"].isin(["train", "calibration", "test"])].copy()
    if frame.empty:
        raise RuntimeError("Manifest contains no train/calibration/test rows")
    if frame.loc[frame["role"].isin(["train", "calibration"]), "label"].ne(0).any():
        raise RuntimeError("Leakage: non-healthy record found in train/calibration")
    index = build_file_index(args.input_root)
    frame["resolved_path"] = [str(resolve_source(row, index)) for _, row in frame.iterrows()]
    frame["excitation_resolved"] = [excitation(row) for _, row in frame.iterrows()]

    models: dict[str, Model] = {}
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group in ("avt", "fvt"):
        train = frame[(frame.role == "train") & (frame.excitation_resolved == group)]
        calibration = frame[(frame.role == "calibration") & (frame.excitation_resolved == group)]
        if train.empty or calibration.empty:
            raise RuntimeError(f"Missing healthy train/calibration records for {group}")
        spectra = []
        frequencies = None
        for _, row in train.iterrows():
            frequencies, spectrum = record_spectrum(load_vertical(Path(row.resolved_path)))
            cache[row.record_id] = (frequencies, spectrum)
            spectra.append(spectrum)
        assert frequencies is not None
        anchors = learn_anchors(spectra, frequencies)
        x = np.stack([features(spectrum, frequencies, anchors) for spectrum in spectra])
        model = Model.fit(anchors, x)
        cal_scores = []
        for _, row in calibration.iterrows():
            freq, spectrum = record_spectrum(load_vertical(Path(row.resolved_path)))
            cal_scores.append(model.raw_score(features(spectrum, freq, anchors)))
        model.threshold = float(max(cal_scores))
        models[group] = model

    test = frame[frame.role == "test"]
    clean_records = []
    for _, row in test.iterrows():
        freq, spectrum = record_spectrum(load_vertical(Path(row.resolved_path)))
        model = models[row.excitation_resolved]
        clean_records.append({
            "record_id": row.record_id,
            "label": int(row.label),
            "excitation": row.excitation_resolved,
            "score": model.score(features(spectrum, freq, model.anchors)),
        })
    clean = pd.DataFrame(clean_records)
    clean_metrics = metric_dict(clean.label.tolist(), clean.score.tolist())

    conditions = ["missing_0.10", "missing_0.30", "missing_0.50",
                  "noise_20", "noise_10", "noise_5",
                  "clip_0.99", "clip_0.95", "clip_0.90"]
    corruption_records = []
    for _, row in test.iterrows():
        signals = load_vertical(Path(row.resolved_path))
        model = models[row.excitation_resolved]
        for condition in conditions:
            for replicate in range(3):
                modified = corrupt(
                    signals, condition,
                    stable_rng(str(row.record_id), condition, replicate)
                )
                freq, spectrum = record_spectrum(modified)
                corruption_records.append({
                    "record_id": row.record_id,
                    "label": int(row.label),
                    "condition": condition,
                    "replicate": replicate,
                    "score": model.score(features(spectrum, freq, model.anchors)),
                })
    corrupted = pd.DataFrame(corruption_records)
    summaries = []
    for (condition, replicate), group in corrupted.groupby(["condition", "replicate"]):
        summaries.append({
            "condition": condition,
            "replicate": int(replicate),
            **metric_dict(group.label.tolist(), group.score.tolist()),
        })
    replicate_summary = pd.DataFrame(summaries)
    mean_summary = replicate_summary.groupby("condition").mean(numeric_only=True).reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output_dir / "clean_record_scores.csv", index=False)
    corrupted.to_csv(args.output_dir / "corruption_record_scores.csv", index=False)
    replicate_summary.to_csv(args.output_dir / "corruption_replicate_metrics.csv", index=False)
    mean_summary.to_csv(args.output_dir / "corruption_summary.csv", index=False)
    artifact = {
        "status": "healthy_only_model_frozen",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "clean_metrics": clean_metrics,
        "models": {key: model.as_dict() for key, model in models.items()},
    }
    (args.output_dir / "model_and_result.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print(json.dumps(clean_metrics, indent=2))
    print(mean_summary.to_string(index=False))
    print(f"PASS output={args.output_dir}")


if __name__ == "__main__":
    main()
