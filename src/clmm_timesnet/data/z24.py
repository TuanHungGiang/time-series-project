import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, resample_poly, welch


def load_manifest(root: Path) -> pd.DataFrame:
    path = root / "metadata" / "records_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["split"] = frame.split.astype(str).str.lower()
    frame["excitation"] = frame.excitation.astype(str).str.lower()
    frame["setup"] = frame.setup.astype(int)
    frame["label"] = frame.label.astype(int)
    frame["state"] = frame.state.astype(int)
    frame = frame[frame.split.isin(["train", "val", "test"])].copy()
    frame["path"] = [str(root / Path(value)) for value in frame.relative_path]
    counts = {
        "train_healthy": int(((frame.split == "train") & (frame.label == 0)).sum()),
        "calibration_healthy": int(((frame.split == "val") & (frame.label == 0)).sum()),
        "test_healthy": int(((frame.split == "test") & (frame.label == 0)).sum()),
        "test_damage": int(((frame.split == "test") & (frame.label == 1)).sum()),
    }
    if counts != {"train_healthy": 24, "calibration_healthy": 6, "test_healthy": 6, "test_damage": 126}:
        raise RuntimeError(f"Unexpected locked Z24 split: {counts}")
    if frame.loc[frame.split == "train", "label"].ne(0).any():
        raise RuntimeError("Damage leakage in training split")
    return frame.reset_index(drop=True)


def parse_labels(value: str) -> list[str]:
    return [item.strip() for item in str(value).split("|") if item.strip()]


def stable_sensor_map(frame: pd.DataFrame) -> dict[tuple[int, str], list[str]]:
    output = {}
    for key, group in frame.groupby(["setup", "excitation"], sort=True):
        sets = [set(parse_labels(value)) for value in group.vertical_labels]
        stable = sorted(set.intersection(*sets), key=lambda value: int(re.match(r"\d+", value).group()))
        if len(stable) < 15:
            raise RuntimeError(f"Too few stable sensors for {key}: {stable}")
        output[(int(key[0]), str(key[1]))] = stable
    expected = {(setup, excitation) for setup in range(1, 10) for excitation in ("avt", "fvt")}
    if set(output) != expected:
        raise RuntimeError(f"Missing setup/excitation cells: {expected.difference(output)}")
    return output


def read_stable_vertical(path: Path, labels: list[str]) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    data = np.asarray(mat["data"], dtype=np.float32)
    raw = np.asarray(mat["labelshulp"]).squeeze().tolist()
    if not isinstance(raw, list):
        raw = [raw]
    observed = [str(value).strip() for value in raw]
    lookup = {label: index for index, label in enumerate(observed)}
    missing = [label for label in labels if label not in lookup]
    if missing:
        raise RuntimeError(f"{path.name} misses stable labels {missing}")
    signals = data[:, [lookup[label] for label in labels]].T
    if signals.shape[0] != len(labels) or not np.isfinite(signals).all():
        raise RuntimeError(f"Invalid selected matrix {path}: {signals.shape}")
    return signals


def record_windows(signals: np.ndarray, raw_length: int, stride: int,
                   representation: str, downsample: int,
                   frequency_bins: int) -> np.ndarray:
    """Create raw-temporal primary inputs or spectral ablation inputs."""
    starts = list(range(0, signals.shape[1] - raw_length + 1, stride))
    if len(starts) < 16:
        raise RuntimeError(f"Too few complete windows: {signals.shape}")
    grid = np.linspace(0.5, 30.0, frequency_bins)
    windows = []
    for start in starts:
        values = signals[:, start:start + raw_length]
        values = detrend(values, axis=1, type="linear")
        if representation == "raw_temporal":
            # Anti-aliasing is part of scipy's polyphase resampler. No test or
            # damage statistic is involved in this transformation.
            feature = resample_poly(values, up=1, down=downsample, axis=1)
            windows.append(feature.astype(np.float32))
            continue
        frequency, psd = welch(
            values, fs=100.0, window="hann", nperseg=min(1024, raw_length),
            noverlap=min(512, raw_length // 2), axis=1, detrend=False,
            scaling="density",
        )
        keep = (frequency >= 0.5) & (frequency <= 30.0)
        frequency, psd = frequency[keep], psd[:, keep]
        total = np.trapezoid(psd, frequency, axis=1)
        psd = psd / np.maximum(total[:, None], np.finfo(float).tiny)
        feature = np.stack([
            np.interp(grid, frequency, np.log10(sensor + 1e-18))
            for sensor in psd
        ])
        # Retain spectral shape while preventing a single broadband amplitude
        # offset from dominating every reconstruction baseline.
        feature -= feature.mean(axis=1, keepdims=True)
        windows.append(feature.astype(np.float32))
    return np.stack(windows)


def split_training_windows(windows_by_record: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    fit, internal = [], []
    for windows in windows_by_record:
        boundary = int(np.floor(0.75 * len(windows)))
        # One complete raw window is a guard band between fit and validation.
        fit.append(windows[:max(1, boundary - 1)])
        internal.append(windows[boundary:])
    return np.concatenate(fit), np.concatenate(internal)
