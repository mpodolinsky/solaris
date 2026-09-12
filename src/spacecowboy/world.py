"""Infinite, deterministic, procedurally generated desert world.

Every integer grid cell's contents are a pure function of its coordinates
and the world seed — nothing is stored, so the world is unbounded and
never needs to be "generated" up front. A small LRU cache just keeps
repeated lookups (adjacent rays, adjacent frames) cheap.
"""
from functools import lru_cache

from . import config
from .noise import fbm, value_noise

_CLEAR_RADIUS = 3  # keep the spawn area walkable


class World:
    def __init__(self, seed: int):
        self.seed = seed
        self._get_cell_cached = lru_cache(maxsize=200_000)(self._compute_cell)

    def get_cell(self, gx: int, gy: int) -> int:
        """Returns 0 for open ground, or an obstacle type id (see config.OBSTACLE_TYPES)."""
        if gx * gx + gy * gy <= _CLEAR_RADIUS * _CLEAR_RADIUS:
            return 0
        return self._get_cell_cached(gx, gy)

    def _raw_feature(self, gx: int, gy: int) -> float:
        return value_noise(gx * 0.97, gy * 0.97, self.seed + 9001)

    def _compute_cell(self, gx: int, gy: int) -> int:
        # Trees disabled for now — none of the silhouette designs looked
        # good enough yet. World is empty ground until a new design lands.
        return 0

        # Low-frequency noise decides regional density: open flats vs. rare landmark clusters.
        density = fbm(gx * 0.06, gy * 0.06, self.seed, octaves=3)
        # High-frequency noise decides per-cell placement.
        feature = self._raw_feature(gx, gy)

        # Loosened, because non-maximum suppression below thins this out a lot.
        threshold = 0.72 - density * 0.16
        if feature < threshold:
            return 0

        # Non-maximum suppression: a cell only becomes an obstacle if it's the
        # single strongest candidate in its own 3x3 neighborhood. Without this,
        # regional density clustering fuses adjacent obstacle cells into one
        # big contiguous mass (a "wall" or "big rock") instead of scattered,
        # individually-readable landmarks.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if self._raw_feature(gx + dx, gy + dy) > feature:
                    return 0

        return 2  # dead tree — the only landmark type in this world now

    def is_solid(self, gx: int, gy: int) -> bool:
        return self.get_cell(gx, gy) != 0
