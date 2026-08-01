"""First-positive, token-normalized neuron novelty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ARR_SIZE = 1 << 22


@dataclass(frozen=True)
class NoveltyResult:
    """Novelty slopes and the newly recruited sparse mass for every row."""

    slopes: np.ndarray
    new_keys: list[np.ndarray]
    new_mass: list[np.ndarray]


@dataclass
class FirstSeenTracker:
    """Versioned first-seen tracker shared across responses."""

    seen_version: np.ndarray
    version: int = 0

    @classmethod
    def create(cls, size: int = ARR_SIZE) -> "FirstSeenTracker":
        return cls(np.zeros(int(size), dtype=np.int32))

    def begin_response(self) -> int:
        self.version += 1
        if self.version >= np.iinfo(np.int32).max:
            self.seen_version.fill(0)
            self.version = 1
        return self.version


def first_positive_novelty(
    row_keys: Sequence[np.ndarray],
    row_mass: Sequence[np.ndarray],
    token_counts: np.ndarray,
    *,
    tracker: FirstSeenTracker,
    version: int,
) -> NoveltyResult:
    """Compute newly recruited neurons per token for each retained row.

    A neuron is newly recruited at its first row with positive retained mass in
    the current response. Rows are assumed to have already been sanitized.
    """

    if not (len(row_keys) == len(row_mass) == len(token_counts)):
        raise ValueError("row_keys, row_mass, and token_counts must have equal length")

    slopes: list[float] = []
    new_keys: list[np.ndarray] = []
    new_mass: list[np.ndarray] = []
    for keys, mass, n_tokens in zip(row_keys, row_mass, token_counts, strict=True):
        n_tokens = int(n_tokens)
        if n_tokens <= 0:
            raise ValueError("token counts must be positive")
        keys = np.asarray(keys, dtype=np.int64)
        mass = np.asarray(mass, dtype=np.float32)
        mask = tracker.seen_version[keys] != int(version)
        tracker.seen_version[keys] = int(version)
        introduced_keys = keys[mask]
        introduced_mass = mass[mask]
        slopes.append(float(introduced_keys.size / n_tokens))
        new_keys.append(introduced_keys)
        new_mass.append(introduced_mass)

    return NoveltyResult(
        slopes=np.asarray(slopes, dtype=np.float32),
        new_keys=new_keys,
        new_mass=new_mass,
    )
