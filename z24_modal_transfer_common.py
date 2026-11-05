"""Leakage-safe modal observations for the Z24 transfer experiments.

The two MSSP transfer papers operate on observations of natural frequencies,
not on raw acceleration records.  This module performs only that input adapter:
whole records remain the split unit, and modal anchors are estimated from S01
healthy training records only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, welch

import run_z24_strict_psd_scatterad as base


def record_spectra(path: Path, fs: float, window: int, stride: int,
                   fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    values = base.vertical_values(path)
    rows = []
    for start in range(0, len(values) - window + 1, stride):
        block = values[start:start + window]
        channel_psd = []
        for channel in range(block.shape[1]):
            x = block[:, channel] - block[:, channel].mean()
            f, p = welch(x, fs=fs, nperseg=window, noverlap=window // 2)
            channel_psd.append(p)
        spectrum = np.median(np.stack(channel_psd), axis=0)
        keep = (f >= fmin) & (f <= fmax)
        rows.append(spectrum[keep])
    if not rows:
        raise ValueError(f"Record shorter than modal window: {path}")
    return f[keep].astype(np.float32), np.asarray(rows, np.float32)


def learn_anchors(spectra: list[np.ndarray], frequencies: np.ndarray,
                  modes: int, minimum_separation_hz: float) -> np.ndarray:
    reference = np.median(np.concatenate(spectra, axis=0), axis=0)
    step = float(np.median(np.diff(frequencies)))
    peaks, _ = find_peaks(
        np.log10(reference + np.finfo(float).tiny),
        distance=max(1, int(round(minimum_separation_hz / step))),
        prominence=.05,
    )
    ranked = list(peaks[np.argsort(reference[peaks])[::-1]])
    if len(ranked) < modes:
        ranked.extend(i for i in np.argsort(reference)[::-1] if i not in ranked)
    selected = sorted(ranked[:modes], key=lambda i: frequencies[i])
    if len(selected) != modes:
        raise ValueError(f"Could not identify {modes} modal anchors")
    return frequencies[selected].astype(np.float32)


def track_modes(frequencies: np.ndarray, spectra: np.ndarray,
                anchors: np.ndarray, radius_hz: float) -> np.ndarray:
    observations = []
    for spectrum in spectra:
        modes = []
        for anchor in anchors:
            local = np.where(np.abs(frequencies - anchor) <= radius_hz)[0]
            if not len(local):
                raise ValueError(f"No frequency bin around modal anchor {anchor}")
            modes.append(float(frequencies[local[np.argmax(spectrum[local])]]))
        observations.append(modes)
    return np.asarray(observations, np.float32)


def robust_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - center), axis=0)
    scale[scale < 1e-6] = 1.
    return center.astype(np.float32), scale.astype(np.float32)


def scale(x: np.ndarray, center: np.ndarray, spread: np.ndarray) -> np.ndarray:
    return np.clip((x - center) / spread, -10., 10.).astype(np.float32)


def domain_for_state(state: str) -> str:
    """Z24 chronology: S01--S07 pre-lifting, S08+ post-lifting."""
    return "source" if int(state) <= 7 else "target"


def record_quantile(window_scores: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(window_scores, float), quantile))
