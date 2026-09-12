"""Tiny deterministic value-noise implementation (no external deps)."""
import math

import numpy as np


def _hash01(ix: int, iy: int, seed: int) -> float:
    """Deterministic pseudo-random float in [0, 1) for an integer lattice point."""
    h = ix * 374761393 + iy * 668265263 + seed * 2147483647
    h = (h ^ (h >> 13)) * 1274126177
    h &= 0xFFFFFFFF
    return (h ^ (h >> 16)) / 0xFFFFFFFF


def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def value_noise(x: float, y: float, seed: int) -> float:
    """Smoothly interpolated value noise in [0, 1)."""
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    sx, sy = _smooth(x - x0), _smooth(y - y0)

    n00 = _hash01(x0, y0, seed)
    n10 = _hash01(x1, y0, seed)
    n01 = _hash01(x0, y1, seed)
    n11 = _hash01(x1, y1, seed)

    ix0 = n00 + (n10 - n00) * sx
    ix1 = n01 + (n11 - n01) * sx
    return ix0 + (ix1 - ix0) * sy


def _hash01_vec(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Vectorized sibling of _hash01, for whole coordinate grids at once."""
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    h = ix * 374761393 + iy * 668265263 + seed * 2147483647
    h = h & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177
    h = h & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h.astype(np.float64) / 0xFFFFFFFF


def value_noise_vec(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Vectorized sibling of value_noise, operating elementwise on arrays."""
    x0, y0 = np.floor(x), np.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    sx, sy = _smooth(x - x0), _smooth(y - y0)

    n00 = _hash01_vec(x0, y0, seed)
    n10 = _hash01_vec(x1, y0, seed)
    n01 = _hash01_vec(x0, y1, seed)
    n11 = _hash01_vec(x1, y1, seed)

    ix0 = n00 + (n10 - n00) * sx
    ix1 = n01 + (n11 - n01) * sx
    return ix0 + (ix1 - ix0) * sy


def fbm(x: float, y: float, seed: int, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> float:
    """Fractal Brownian motion: layered value noise, normalized to [0, 1)."""
    amplitude = 1.0
    frequency = 1.0
    total = 0.0
    max_amp = 0.0
    for i in range(octaves):
        total += value_noise(x * frequency, y * frequency, seed + i * 101) * amplitude
        max_amp += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / max_amp
