import hashlib

import numpy as np


def sensor_mask(channels, rate, order):
    available = np.ones(channels, dtype=bool)
    if rate:
        count = min(channels - 1, max(1, round(channels * rate)))
        available[order[:count]] = False
    return available


def stable_rng(record_id: str, kind: str, replicate: int) -> np.random.Generator:
    # Deliberately exclude severity/rate: each replicate defines one latent
    # corruption pattern and larger rates reveal a strict superset of it.
    token = f"z24-missing-nested-v2|{record_id}|{kind}|{replicate}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(seed)
