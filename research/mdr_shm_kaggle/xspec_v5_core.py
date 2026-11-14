from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend


FS = 100.0
FMIN, FMAX = 0.5, 30.0
BANDS = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0))


def _spatial_quantiles(values: np.ndarray, bins: int) -> np.ndarray:
    """Map a variable sensor count to a fixed, order-statistic representation."""
    grid = (np.arange(bins, dtype=float) + 0.5) / bins
    return np.quantile(np.asarray(values, dtype=float), grid, method="linear")


def cross_spectral_features(
    signals: np.ndarray, nperseg: int = 2048, spatial_bins: int = 12
) -> np.ndarray:
    """Sensor-resolved relative power and coherence features.

    The representation is insensitive to common excitation amplitude. Spatial
    order statistics retain much more information than mean/IQR summaries while
    producing a fixed dimension when Z24 setups contain different sensor counts.
    """
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 8:
        raise ValueError(f"Expected sensor x time with >=8 sensors, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Signals must be finite")
    values = detrend(values, axis=1, type="linear")
    width = min(nperseg, values.shape[1])
    if width < 512:
        raise ValueError("Need at least 512 time samples")
    hop = width // 2
    starts = np.arange(0, values.shape[1] - width + 1, hop)
    if len(starts) < 3:
        starts = np.asarray([0])
    window = np.hanning(width)
    scale = np.sqrt(np.sum(window**2))
    spectra = np.stack(
        [np.fft.rfft(values[:, start : start + width] * window, axis=1) / scale for start in starts],
        axis=0,
    )  # segment x sensor x frequency
    frequency = np.fft.rfftfreq(width, 1.0 / FS)
    keep = (frequency >= FMIN) & (frequency <= FMAX)
    frequency = frequency[keep]
    spectra = spectra[:, :, keep]
    auto = np.mean(np.abs(spectra) ** 2, axis=0) + 1e-18
    cross = np.einsum("asf,atf->stf", spectra, np.conjugate(spectra), optimize=True) / len(spectra)
    coherence = np.abs(cross) ** 2 / (auto[:, None, :] * auto[None, :, :])
    coherence = np.clip(coherence.real, 0.0, 1.0)
    sensor_count = values.shape[0]
    diagonal = np.eye(sensor_count, dtype=bool)

    output = []
    for low, high in BANDS:
        band = (frequency >= low) & (frequency < high)
        band_power = np.trapezoid(auto[:, band], frequency[band], axis=1)
        log_power = np.log10(band_power + 1e-18)
        relative_power = log_power - np.median(log_power)
        band_coherence = np.median(coherence[:, :, band], axis=2)
        masked = np.where(diagonal, np.nan, band_coherence)
        node_coherence = np.nanmedian(masked, axis=1)
        # Signed relative power preserves local information. Coherence is
        # logit-transformed to increase resolution near one.
        coherence_logit = np.log((node_coherence + 1e-4) / (1.0001 - node_coherence))
        output.extend(_spatial_quantiles(relative_power, spatial_bins).tolist())
        output.extend(_spatial_quantiles(coherence_logit, spatial_bins).tolist())
    return np.asarray(output, dtype=np.float64)


@dataclass
class SubspaceDetector:
    center: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    residual_scale: float
    score_center: float
    score_scale: float

    @classmethod
    def fit(cls, matrix: np.ndarray, max_components: int = 5) -> "SubspaceDetector":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) < 6:
            raise ValueError(f"Need >=6 healthy samples, got {matrix.shape}")
        center = np.median(matrix, axis=0)
        scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
        fallback = np.std(matrix, axis=0, ddof=1)
        scale = np.maximum(scale, np.maximum(0.25 * fallback, 1e-5))
        z = np.clip((matrix - center) / scale, -12.0, 12.0)
        z_centered = z - np.mean(z, axis=0)
        _, singular, vt = np.linalg.svd(z_centered, full_matrices=False)
        eigenvalues_all = singular**2 / max(len(z) - 1, 1)
        variance_ratio = eigenvalues_all / max(float(eigenvalues_all.sum()), 1e-12)
        explained_count = int(np.searchsorted(np.cumsum(variance_ratio), 0.85) + 1)
        count = min(max(2, explained_count), max_components, len(matrix) - 3, matrix.shape[1])
        components = vt[:count]
        eigenvalues = np.maximum(eigenvalues_all[:count], 1e-6)

        def unscaled_scores(values: np.ndarray) -> np.ndarray:
            latent = values @ components.T
            reconstruction = latent @ components
            t2 = np.sum(latent**2 / eigenvalues, axis=1)
            residual = np.sum((values - reconstruction) ** 2, axis=1)
            return np.log1p(t2) + np.log1p(residual)

        residual = z_centered - (z_centered @ components.T) @ components
        residual_scale = float(max(np.median(np.sum(residual**2, axis=1)), 1e-6))
        latent = z_centered @ components.T
        train_scores = np.log1p(np.sum(latent**2 / eigenvalues, axis=1)) + np.log1p(
            np.sum(residual**2, axis=1) / residual_scale
        )
        score_center = float(np.median(train_scores))
        score_scale = float(max(1.4826 * np.median(np.abs(train_scores - score_center)), np.std(train_scores), 1e-6))
        return cls(center, scale, components, eigenvalues, residual_scale, score_center, score_scale)

    def components_score(self, vector: np.ndarray) -> dict[str, float]:
        z = np.clip((np.asarray(vector) - self.center) / self.scale, -20.0, 20.0)
        latent = self.components @ z
        reconstruction = self.components.T @ latent
        t2 = float(np.sum(latent**2 / self.eigenvalues))
        spe = float(np.sum((z - reconstruction) ** 2) / self.residual_scale)
        raw = float(np.log1p(t2) + np.log1p(spe))
        return {"xspec_raw": raw, "xspec_z": (raw - self.score_center) / self.score_scale, "t2": t2, "spe": spe}


def robust_reference(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    scale = float(max(1.4826 * np.median(np.abs(values - center)), np.std(values), 1e-6))
    return center, scale


def standardize(value: float, reference: tuple[float, float]) -> float:
    return float((value - reference[0]) / reference[1])


def calibrate(values: np.ndarray, alpha: float) -> tuple[float, tuple[float, float]]:
    values = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(values) + 1) * (1.0 - alpha))) - 1
    threshold = float(values[min(max(rank, 0), len(values) - 1)])
    return threshold, robust_reference(values)


def normalize(value: float, calibration: tuple[float, tuple[float, float]]) -> float:
    threshold, reference = calibration
    return float(1.0 + (value - threshold) / reference[1])

