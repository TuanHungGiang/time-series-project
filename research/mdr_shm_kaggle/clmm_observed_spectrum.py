"""Healthy-reference score fusion for CLMM with a known channel mask.

Call score_pair with complete and masked/imputed USAD reference scores from
the same healthy fit windows. Keep calibration and test windows out of those
reference arrays. All returned columns use larger values for more anomalous.
"""

from __future__ import annotations

import numpy as np


def _location_and_scale(values, axis=0):
    center = np.median(values, axis=axis)
    deviations = np.abs(values - np.expand_dims(center, axis))
    mad = 1.4826 * np.median(deviations, axis=axis)
    scale = np.maximum(mad, 0.25 * np.std(values, axis=axis))
    return center, scale


def _log_power(windows):
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3 or windows.shape[-1] < 8 or not np.isfinite(windows).all():
        raise ValueError("Expected finite windows with shape [N, C, T] and T >= 8")
    centered = windows - windows.mean(axis=-1, keepdims=True)
    tapered = centered * np.hanning(windows.shape[-1])
    power = np.abs(np.fft.rfft(tapered, axis=-1)[..., 1:]) ** 2
    return np.log1p(power / windows.shape[-1])


class ObservedSpectrum:
    """Spectral deviation on available channels and mask-specific fit scaling."""

    def __init__(self, healthy_fit):
        self.fit_power = _log_power(healthy_fit)
        if len(self.fit_power) < 2:
            raise ValueError("At least two healthy fit windows are required")
        self.center, self.scale = _location_and_scale(self.fit_power)
        self.scale = np.maximum(self.scale, 0.05)

    def _deviation(self, power, available):
        deviation = (power - self.center[available]) / self.scale[available]
        return np.abs(deviation).mean(axis=(1, 2))

    def score_pair(self, windows, available, reference_usad, query_usad, *,
                   usad_direction, spectral_weight=0.5):
        """Return separate and fused scores, using only observed waveforms.

        reference_usad is scored on the healthy fit cohort after applying the
        same mask and completion used for query_usad. For clean queries pass
        an all-True mask and clean USAD scores. Set usad_direction to the
        preregistered score direction (+1 upper tail, -1 lower tail).
        """
        available = np.asarray(available, dtype=bool)
        windows = np.asarray(windows)
        if available.shape != (self.fit_power.shape[1],) or not available.any():
            raise ValueError("Expected one fixed, nonempty availability mask")
        if windows.ndim != 3 or windows.shape[1] != available.size:
            raise ValueError("Query windows must use the fit channel order")
        if usad_direction not in (-1, 1) or not 0 <= spectral_weight <= 1:
            raise ValueError("Invalid score direction or spectral weight")
        reference_usad = np.asarray(reference_usad, dtype=np.float64)
        query_usad = np.asarray(query_usad, dtype=np.float64)
        if reference_usad.shape != (len(self.fit_power),) or query_usad.shape != (len(windows),):
            raise ValueError("Expected one USAD score per fit and query window")
        if any(not np.isfinite(scores).all() or (scores < 0).any()
               for scores in (reference_usad, query_usad)):
            raise ValueError("Expected nonnegative finite raw USAD errors")

        reference_spectrum = self._deviation(self.fit_power[:, available], available)
        query_spectrum = self._deviation(_log_power(windows[:, available]), available)
        spec_center, spec_scale = _location_and_scale(reference_spectrum)
        spectrum = (query_spectrum - spec_center) / max(float(spec_scale), 0.05)
        usad_center, usad_scale = _location_and_scale(np.log1p(reference_usad))
        usad = usad_direction * (
            np.log1p(query_usad) - usad_center
        ) / max(float(usad_scale), 1e-4)
        fused = (1 - spectral_weight) * usad + spectral_weight * spectrum
        return {"mask_normalized_usad": usad, "observed_spectrum": spectrum, "fused": fused}
