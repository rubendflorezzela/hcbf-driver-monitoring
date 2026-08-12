from __future__ import annotations

import hashlib

import numpy as np


def deterministic_seed(*parts: object, base_seed: int = 20260725) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def gaussian_noise(
    image: np.ndarray,
    *,
    sigma: float,
    sample_id: str,
    severity: str | int,
    realization: int,
    base_seed: int = 20260725,
) -> np.ndarray:
    array = np.asarray(image)
    rng = np.random.default_rng(
        deterministic_seed(sample_id, "gaussian_noise", severity, realization, base_seed=base_seed)
    )
    noisy = array.astype(np.float32) + rng.normal(0.0, sigma, size=array.shape)
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(noisy, 0, 255).round().astype(array.dtype)
    return np.clip(noisy, 0.0, 1.0).astype(array.dtype)


def brightness(image: np.ndarray, *, factor: float) -> np.ndarray:
    array = np.asarray(image)
    result = array.astype(np.float32) * float(factor)
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(result, 0, 255).round().astype(array.dtype)
    return np.clip(result, 0.0, 1.0).astype(array.dtype)


def horizontal_motion_blur(image: np.ndarray, *, kernel_size: int) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    array = np.asarray(image)
    work = array.astype(np.float32)
    squeeze = work.ndim == 2
    if squeeze:
        work = work[..., None]
    pad = kernel_size // 2
    padded = np.pad(work, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
    cumulative = np.concatenate([np.zeros_like(cumulative[:, :1]), cumulative], axis=1)
    result = (cumulative[:, kernel_size:] - cumulative[:, :-kernel_size]) / kernel_size
    if squeeze:
        result = result[..., 0]
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(result, 0, 255).round().astype(array.dtype)
    return np.clip(result, 0.0, 1.0).astype(array.dtype)
