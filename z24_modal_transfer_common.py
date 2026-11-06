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
from scipy.linalg import eig

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


def track_modes_centroid(frequencies: np.ndarray, spectra: np.ndarray,
                         anchors: np.ndarray, radius_hz: float) -> np.ndarray:
    """Continuous peak-centroid modal estimates for cointegration.

    Hard argmax tracking is quantised at the Welch frequency resolution and
    can produce an exactly constant short record. A power-weighted centroid
    within the identical anchor radius retains sub-bin modal drift without
    using damage labels or changing the selected modal neighbourhood.
    """
    observations = []
    for spectrum in spectra:
        modes = []
        for anchor in anchors:
            local = np.where(np.abs(frequencies - anchor) <= radius_hz)[0]
            if not len(local): raise ValueError(f"No frequency bin around modal anchor {anchor}")
            power = np.asarray(spectrum[local], float)
            # Remove the local floor so broad-band energy does not pull every
            # estimate to the centre of the fixed search interval.
            weight = np.maximum(power - np.min(power), 0.)
            if float(weight.sum()) <= np.finfo(float).tiny:
                modes.append(float(frequencies[local[np.argmax(power)]]))
            else:
                modes.append(float(np.sum(frequencies[local] * weight) / np.sum(weight)))
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


class RegimeJohansenResidualizer:
    """Operational-regime Johansen-style cointegration residuals.

    The MSSP regime-switching paper uses latent regimes to handle nonlinear
    EOV behaviour.  The exported Z24 PDT manifest already identifies the
    operational regime (AVT/FVT), so this leak-free adapter fits one Johansen
    stationary subspace per known excitation using S01/train only.  It is an
    equation-based Z24 adaptation, not author code and not a hidden-Markov
    reproduction.
    """

    def __init__(self, rank: int = 3, ridge: float = 1e-5):
        self.rank = rank; self.ridge = ridge; self.models: dict[str, dict] = {}

    @staticmethod
    def _residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(z)), z])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        return y - design @ coef

    def _fit_regime(self, sequences: list[np.ndarray]) -> dict:
        pooled = np.concatenate(sequences)
        center, scale = robust_scaler(pooled)
        normalized = [(x - center) / scale for x in sequences]
        z0_parts, z1_parts, z2_parts = [], [], []
        for x in normalized:
            if len(x) < 4: continue
            dx = np.diff(x, axis=0)
            # VECM with one lag of differences: dX_t ~ X_{t-1} + dX_{t-1}.
            z0_parts.append(dx[1:]); z1_parts.append(x[1:-1]); z2_parts.append(dx[:-1])
        if not z0_parts: raise ValueError("Cointegration needs >=4 modal windows per healthy record")
        # Estimate one VECM per operational regime. Residualising each short
        # record separately would be underdetermined and collapse S00/S11.
        z0, z1, z2 = np.concatenate(z0_parts), np.concatenate(z1_parts), np.concatenate(z2_parts)
        r0, r1 = self._residualize(z0, z2), self._residualize(z1, z2)
        n = len(r0); eye = np.eye(r0.shape[1])
        s00 = r0.T @ r0 / n + self.ridge * eye
        s11 = r1.T @ r1 / n + self.ridge * eye
        s01 = r0.T @ r1 / n; s10 = s01.T
        matrix = np.linalg.solve(s11, s10 @ np.linalg.solve(s00, s01))
        values, vectors = eig(matrix)
        order = np.argsort(np.real(values))[::-1]
        if not np.isfinite(values).all() or float(np.max(np.real(values))) < 1e-8:
            raise ValueError("Degenerate cointegration system: no estimable common modal trend")
        rank = max(1, min(self.rank, vectors.shape[1] - 1))
        beta = np.real(vectors[:, order[:rank]])
        beta /= np.maximum(np.linalg.norm(beta, axis=0, keepdims=True), 1e-12)
        train_residual = np.concatenate([x @ beta for x in normalized])
        residual_center, residual_scale = robust_scaler(train_residual)
        return {"center": center, "scale": scale, "beta": beta.astype(np.float32),
                "residual_center": residual_center, "residual_scale": residual_scale,
                "eigenvalues": np.real(values[order[:rank]]).astype(np.float32)}

    def fit(self, features: dict[str, np.ndarray], rows) -> "RegimeJohansenResidualizer":
        for excitation, group in rows.groupby("excitation"):
            sequences = [features[x] for x in group.record_id]
            self.models[str(excitation).upper()] = self._fit_regime(sequences)
        if not self.models: raise ValueError("No healthy regimes supplied")
        return self

    def transform(self, x: np.ndarray, excitation: str) -> np.ndarray:
        key = str(excitation).upper()
        if key not in self.models: raise ValueError(f"Unseen operational regime: {key}")
        model = self.models[key]
        normalized = (x - model["center"]) / model["scale"]
        residual = normalized @ model["beta"]
        return np.clip((residual - model["residual_center"]) /
                       model["residual_scale"], -10., 10.).astype(np.float32)

    def audit(self) -> dict:
        return {key: {"rank": int(value["beta"].shape[1]),
                      "eigenvalues": value["eigenvalues"].tolist()}
                for key, value in self.models.items()}


def cointegration_features(features: dict[str, np.ndarray], frame, fit_rows,
                           rank: int = 3) -> tuple[dict[str, np.ndarray], RegimeJohansenResidualizer]:
    model = RegimeJohansenResidualizer(rank=rank).fit(features, fit_rows)
    lookup = frame.drop_duplicates("record_id").set_index("record_id")["excitation"].to_dict()
    transformed = {key: model.transform(value, lookup[key]) for key, value in features.items()}
    return transformed, model

