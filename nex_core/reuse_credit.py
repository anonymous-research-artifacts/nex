"""Explore-to-Exploit reuse credit and signed neuron weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .novelty import ARR_SIZE, NoveltyResult


EPS = 1e-6


@dataclass
class ReuseScratch:
    """Versioned membership mask for E-phase neuron sets."""

    marks: np.ndarray
    version: int = 0

    @classmethod
    def create(cls, size: int = ARR_SIZE) -> "ReuseScratch":
        return cls(np.zeros(int(size), dtype=np.int32))

    def mark(self, keys: np.ndarray) -> int:
        self.version += 1
        if self.version >= np.iinfo(np.int32).max:
            self.marks.fill(0)
            self.version = 1
        self.marks[keys] = self.version
        return self.version


def _explore_exploit_cycles(labels: np.ndarray) -> list[tuple[int, int, int, int]]:
    labels = np.asarray(labels, dtype=bool)
    cycles: list[tuple[int, int, int, int]] = []
    row = 0
    while row < labels.size:
        if not labels[row]:
            row += 1
            continue
        e_start = row
        while row < labels.size and labels[row]:
            row += 1
        e_end = row - 1
        if row >= labels.size:
            break
        x_start = row
        while row < labels.size and not labels[row]:
            row += 1
        x_end = row - 1
        if e_end >= e_start and x_end >= x_start:
            cycles.append((e_start, e_end, x_start, x_end))
    return cycles


def accumulate_reuse_credit(
    *,
    labels: np.ndarray,
    novelty: NoveltyResult,
    row_keys: Sequence[np.ndarray],
    row_mass: Sequence[np.ndarray],
    effective_mass: np.ndarray,
    redundant_mass: np.ndarray,
    scratch: ReuseScratch,
) -> None:
    """Accumulate the paper's signed reuse credit for one response."""

    slopes = np.asarray(novelty.slopes, dtype=np.float32)
    if not (
        len(labels)
        == len(slopes)
        == len(row_keys)
        == len(row_mass)
        == len(novelty.new_keys)
        == len(novelty.new_mass)
    ):
        raise ValueError("all row-level inputs must have equal length")

    cycles = _explore_exploit_cycles(labels)
    if not cycles:
        return

    row_totals = np.asarray([float(np.sum(mass)) for mass in row_mass], dtype=np.float64)
    median_slope = float(np.median(slopes))
    reuse_shares: list[float] = []
    consolidations: list[float] = []
    strengths: list[float] = []

    for e_start, e_end, x_start, x_end in cycles:
        slope_e = float(np.median(slopes[e_start : e_end + 1]))
        slope_x = float(np.median(slopes[x_start : x_end + 1]))
        strength = slope_e - median_slope
        consolidation = (
            float(np.clip(1.0 - slope_x / (slope_e + EPS), 0.0, 1.0))
            if slope_e > 0.0
            else 0.0
        )
        parts = [
            novelty.new_keys[row]
            for row in range(e_start, e_end + 1)
            if novelty.new_keys[row].size
        ]
        introduced = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if strength <= 0.0 or introduced.size == 0:
            reuse_shares.append(float("nan"))
            consolidations.append(consolidation)
            strengths.append(strength)
            continue

        mark_version = scratch.mark(introduced)
        reused_mass = 0.0
        for row in range(x_start, x_end + 1):
            keys = np.asarray(row_keys[row], dtype=np.int64)
            mass = np.asarray(row_mass[row], dtype=np.float32)
            reused_mass += float(np.sum(mass[scratch.marks[keys] == mark_version]))
        total_mass = float(np.sum(row_totals[x_start : x_end + 1]))
        reuse_shares.append(
            reused_mass / (total_mass + EPS) if total_mass > 0.0 else float("nan")
        )
        consolidations.append(consolidation)
        strengths.append(strength)

    finite = np.asarray(
        [value for value in reuse_shares if np.isfinite(value)], dtype=np.float32
    )
    if finite.size == 0:
        return
    median_reuse = float(np.median(finite))

    for index, (e_start, e_end, _x_start, _x_end) in enumerate(cycles):
        reuse = reuse_shares[index]
        if not np.isfinite(reuse) or strengths[index] <= 0.0:
            continue
        progress = float(reuse - median_reuse)
        if not np.isfinite(progress) or progress == 0.0:
            continue
        effective = progress > 0.0 and consolidations[index] > 0.0
        scale = progress * consolidations[index] if effective else abs(progress)
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        target = effective_mass if effective else redundant_mass
        for row in range(e_start, e_end + 1):
            keys = novelty.new_keys[row]
            if keys.size:
                target[keys] += novelty.new_mass[row] * scale


def neuron_weights(effective_mass: np.ndarray, redundant_mass: np.ndarray) -> np.ndarray:
    """Map accumulated effective/redundant mass to weights in `[-1, 1]`."""

    effective_mass = np.asarray(effective_mass, dtype=np.float32)
    redundant_mass = np.asarray(redundant_mass, dtype=np.float32)
    if effective_mass.shape != redundant_mass.shape:
        raise ValueError("effective_mass and redundant_mass must have equal shape")
    return np.tanh(
        np.log((effective_mass + EPS) / (redundant_mass + EPS))
    ).astype(np.float32)
