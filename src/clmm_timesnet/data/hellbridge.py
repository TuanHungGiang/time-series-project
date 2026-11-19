from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import detrend, resample


def read_al40(group: h5py.Group) -> np.ndarray:
    acceleration = group["acceleration"]
    values = [np.asarray(acceleration[f"AL{i:02d}"]["z"], np.float32) for i in range(1, 41)]
    length = min(map(len, values))
    return np.stack([value[:length] for value in values])


def load_windows(h5_path: Path, manifest: pd.DataFrame, length: int) -> np.ndarray:
    output = np.empty((len(manifest), 40, length), np.float32)
    with h5py.File(h5_path, "r") as source:
        total = manifest.record_id.nunique()
        for position, (record_id, rows) in enumerate(manifest.groupby("record_id", sort=False), 1):
            recording = read_al40(source[record_id])
            for row in rows.itertuples():
                window = recording[:, int(row.start):int(row.stop)]
                window = detrend(window, axis=1, type="linear")
                output[row.Index] = resample(window, length, axis=1).astype(np.float32)
            print(f"load record={position}/{total} id={record_id} windows={len(rows)}", flush=True)
    return output


def development_split(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = manifest[manifest.role.eq("train_healthy")].sort_values(["record_id", "start"])
    if len(train) != 80 or train.label.ne(0).any() or train.record_id.nunique() != 2:
        raise ValueError("Expected 80 healthy development windows from two recordings")
    fit, calibration = [], []
    for _, group in train.groupby("record_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) != 40:
            raise ValueError("Expected 40 healthy windows per development recording")
        # Interleaved blocked allocation avoids putting calibration in only one time region.
        calibration.extend(indices.reshape(5, 8)[:, -2:].ravel())
        fit.extend(indices.reshape(5, 8)[:, :-2].ravel())
    test = manifest.index[manifest.role.isin(["test_healthy", "test_damage"])].to_numpy()
    if len(fit) != 60 or len(calibration) != 20 or len(test) != 720:
        raise RuntimeError("Unexpected fit/calibration/test sizes")
    return np.asarray(fit), np.asarray(calibration), test


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        mean = values.mean(axis=(0, 2), keepdims=True)
        scale = values.std(axis=(0, 2), keepdims=True)
        return cls(mean.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)
