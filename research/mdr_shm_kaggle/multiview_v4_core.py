from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend, find_peaks, welch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score


FS = 100.0
FMIN, FMAX = 0.5, 30.0
BANDS = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0))


def sensor_spectra(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-area PSDs, one row per sensor."""
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError(f"Expected sensor x time matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Signals must be finite before spectral extraction")
    values = detrend(values, axis=1, type="linear")
    nperseg = min(4096, values.shape[1])
    if nperseg < 256:
        raise ValueError("A window must contain at least 256 samples")
    frequency, psd = welch(
        values, fs=FS, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, axis=1, detrend=False, scaling="density",
    )
    keep = (frequency >= FMIN) & (frequency <= FMAX)
    frequency, psd = frequency[keep], psd[:, keep]
    area = np.trapezoid(psd, frequency, axis=1)
    psd = psd / np.maximum(area[:, None], np.finfo(float).tiny)
    return frequency, psd


def learn_anchors(global_spectra: list[np.ndarray], frequency: np.ndarray, count: int = 4) -> list[float]:
    reference = np.median(np.stack(global_spectra), axis=0)
    spacing = max(1, int(round(0.75 / (frequency[1] - frequency[0]))))
    peaks, _ = find_peaks(reference, distance=spacing)
    if len(peaks) < count:
        raise RuntimeError("Insufficient healthy modal peaks")
    strongest = peaks[np.argsort(reference[peaks])[-count:]]
    return sorted(float(frequency[index]) for index in strongest)


def _global_features(global_psd: np.ndarray, frequency: np.ndarray, anchors: list[float]) -> np.ndarray:
    output: list[float] = []
    for anchor in anchors:
        local = np.flatnonzero(np.abs(frequency - anchor) <= 0.6)
        peak = local[int(np.argmax(global_psd[local]))]
        output.extend((float(frequency[peak]), float(np.log10(global_psd[peak] + 1e-16))))
    for low, high in BANDS:
        keep = (frequency >= low) & (frequency < high)
        output.append(float(np.log10(np.trapezoid(global_psd[keep], frequency[keep]) + 1e-16)))
    return np.asarray(output, dtype=float)


def _relational_features(sensor_psd: np.ndarray, frequency: np.ndarray, anchors: list[float]) -> np.ndarray:
    """Permutation-invariant sensor disagreement; dimension does not depend on sensor count."""
    output: list[float] = []
    for anchor in anchors:
        local = np.flatnonzero(np.abs(frequency - anchor) <= 0.6)
        peak_indices = local[np.argmax(sensor_psd[:, local], axis=1)]
        peak_frequencies = frequency[peak_indices]
        output.append(float(1.4826 * np.median(np.abs(peak_frequencies - np.median(peak_frequencies)))))
    for low, high in BANDS:
        keep = (frequency >= low) & (frequency < high)
        energy = np.trapezoid(sensor_psd[:, keep], frequency[keep], axis=1)
        log_energy = np.log10(energy + 1e-16)
        median = float(np.median(log_energy))
        mad = float(1.4826 * np.median(np.abs(log_energy - median)))
        iqr = float(np.quantile(log_energy, 0.75) - np.quantile(log_energy, 0.25))
        output.extend((mad, iqr))
    return np.asarray(output, dtype=float)


def extract_views(frequency: np.ndarray, psd: np.ndarray, anchors: list[float]) -> dict[str, np.ndarray]:
    global_psd = np.median(psd, axis=0)
    return {
        "global": _global_features(global_psd, frequency, anchors),
        "relational": _relational_features(psd, frequency, anchors),
    }


@dataclass
class HealthyDistance:
    center: np.ndarray
    scale: np.ndarray
    location: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "HealthyDistance":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or len(matrix) < 4:
            raise ValueError(f"Need >=4 healthy samples, got {matrix.shape}")
        center = np.median(matrix, axis=0)
        scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
        fallback = np.std(matrix, axis=0, ddof=1)
        scale = np.maximum(scale, np.maximum(0.25 * fallback, 1e-6))
        z = np.clip((matrix - center) / scale, -12.0, 12.0)
        covariance = LedoitWolf().fit(z)
        return cls(center, scale, covariance.location_, covariance.precision_)

    def score(self, vector: np.ndarray) -> float:
        z = np.clip((np.asarray(vector) - self.center) / self.scale, -20.0, 20.0)
        delta = z - self.location
        return float(np.sqrt(max(delta @ self.precision @ delta, 0.0)))


@dataclass
class MultiViewModel:
    global_model: HealthyDistance
    relational_model: HealthyDistance
    global_reference: tuple[float, float]
    relational_reference: tuple[float, float]
    global_weight: float = 0.65

    @classmethod
    def fit(cls, view_rows: list[dict[str, np.ndarray]], global_weight: float = 0.65) -> "MultiViewModel":
        global_model = HealthyDistance.fit(np.stack([row["global"] for row in view_rows]))
        relational_model = HealthyDistance.fit(np.stack([row["relational"] for row in view_rows]))
        global_scores = np.asarray([global_model.score(row["global"]) for row in view_rows])
        relational_scores = np.asarray([relational_model.score(row["relational"]) for row in view_rows])

        def reference(values: np.ndarray) -> tuple[float, float]:
            center = float(np.median(values))
            scale = float(max(1.4826 * np.median(np.abs(values - center)), np.std(values), 1e-6))
            return center, scale

        return cls(global_model, relational_model, reference(global_scores), reference(relational_scores), global_weight)

    def components(self, views: dict[str, np.ndarray]) -> dict[str, float]:
        global_raw = self.global_model.score(views["global"])
        relational_raw = self.relational_model.score(views["relational"])
        gc, gs = self.global_reference
        rc, rs = self.relational_reference
        global_z = (global_raw - gc) / gs
        relational_z = (relational_raw - rc) / rs
        ensemble = self.global_weight * global_z + (1.0 - self.global_weight) * relational_z
        # Disagreement is used as model-validity evidence, never as the damage score.
        domain_error = abs(global_z - relational_z)
        return {
            "global_raw": global_raw,
            "relational_raw": relational_raw,
            "global_z": float(global_z),
            "relational_z": float(relational_z),
            "score_raw": float(ensemble),
            "domain_error": float(domain_error),
        }


def conformal_threshold(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        raise ValueError("Empty calibration scores")
    rank = int(np.ceil((len(values) + 1) * (1.0 - alpha))) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


def normalized_score(raw: float, threshold: float, calibration: np.ndarray) -> float:
    calibration = np.asarray(calibration, dtype=float)
    scale = max(
        1.4826 * float(np.median(np.abs(calibration - np.median(calibration)))),
        float(np.std(calibration)), 1e-6,
    )
    return float(1.0 + (raw - threshold) / scale)


def signal_quality(signals: np.ndarray) -> dict[str, float]:
    values = np.asarray(signals, dtype=float)
    finite = np.isfinite(values)
    coverage = float(finite.mean())
    safe = np.where(finite, values, np.nan)
    sensor_std = np.nanstd(safe, axis=1)
    flat = float(np.mean(sensor_std <= max(float(np.nanmedian(sensor_std)) * 1e-4, 1e-12)))
    centered = safe - np.nanmedian(safe, axis=1, keepdims=True)
    radius = np.nanquantile(np.abs(centered), 0.99, axis=1, keepdims=True) + 1e-12
    clipped = float(np.nanmean(np.abs(centered) >= radius))
    delta = np.diff(safe, axis=1)
    impulse = float(np.nanquantile(np.abs(delta), 0.999) / (np.nanmedian(np.abs(delta)) + 1e-12))
    integrity_error = (1.0 - coverage) * 10.0 + flat * 5.0 + clipped + np.log1p(max(impulse - 25.0, 0.0))
    return {"coverage": coverage, "flat_sensor_fraction": flat, "impulse_ratio": impulse, "integrity_error": float(integrity_error)}


def metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = scores > 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }

