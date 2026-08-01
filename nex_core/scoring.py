"""Sparse-cache orchestration and Good-Mass scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .novelty import ARR_SIZE, FirstSeenTracker, first_positive_novelty
from .reuse_credit import ReuseScratch, accumulate_reuse_credit, neuron_weights
from .segmentation import sticky_hmm_segments


def _require_cache_files(cache_dir: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [name for name in relative_paths if not (cache_dir / name).is_file()]
    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(f"cache is missing required file(s): {formatted}")


def _retained_row(keys: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.uint32)
    mass = np.asarray(mass, dtype=np.float32)
    valid = (keys >> 16) != 255
    if not np.all(valid):
        keys = keys[valid]
        mass = mass[valid]
    if keys.size == 0 or int(keys.max()) >= ARR_SIZE:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    mass = np.where(np.isfinite(mass), mass, 0.0)
    mass = np.clip(mass, 0.0, None)
    return keys.astype(np.int64, copy=False), mass


def learn_nex_weights(
    cache_dir: Path, *, rho: float = 0.95, min_run: int = 2, max_responses: int = 0
) -> np.ndarray:
    """Learn candidate-specific signed neuron weights from an unlabeled cache."""

    cache_dir = Path(cache_dir)
    _require_cache_files(
        cache_dir,
        (
            "rows/sample_row_ptr.int64",
            "rows/row_ptr.int64",
            "rows/token_row_ptr.int64",
            "rows/keys.uint32",
            "rows/w_sum.float16",
        ),
    )
    sample_row_ptr = np.memmap(
        cache_dir / "rows/sample_row_ptr.int64", dtype=np.int64, mode="r"
    )
    row_ptr = np.memmap(cache_dir / "rows/row_ptr.int64", dtype=np.int64, mode="r")
    token_row_ptr = np.memmap(
        cache_dir / "rows/token_row_ptr.int64", dtype=np.int64, mode="r"
    )
    keys_mem = np.memmap(cache_dir / "rows/keys.uint32", dtype=np.uint32, mode="r")
    mass_mem = np.memmap(cache_dir / "rows/w_sum.float16", dtype=np.float16, mode="r")

    n_responses = len(sample_row_ptr) - 1
    if int(max_responses) > 0:
        n_responses = min(n_responses, int(max_responses))

    tracker = FirstSeenTracker.create()
    reuse_scratch = ReuseScratch.create()
    effective_mass = np.zeros(ARR_SIZE, dtype=np.float32)
    redundant_mass = np.zeros(ARR_SIZE, dtype=np.float32)

    for response in range(n_responses):
        version = tracker.begin_response()
        first_row = int(sample_row_ptr[response])
        last_row = int(sample_row_ptr[response + 1])
        if last_row - first_row < 3:
            continue

        row_keys: list[np.ndarray] = []
        row_mass: list[np.ndarray] = []
        token_counts: list[int] = []
        for row in range(first_row, last_row):
            start, end = int(row_ptr[row]), int(row_ptr[row + 1])
            n_tokens = int(token_row_ptr[row + 1] - token_row_ptr[row])
            if end <= start or n_tokens <= 0:
                continue
            keys, mass = _retained_row(keys_mem[start:end], mass_mem[start:end])
            if keys.size == 0 or float(np.sum(mass)) <= 0.0:
                continue
            row_keys.append(keys)
            row_mass.append(mass)
            token_counts.append(n_tokens)

        if len(row_keys) < 3:
            continue
        novelty = first_positive_novelty(
            row_keys,
            row_mass,
            np.asarray(token_counts, dtype=np.int32),
            tracker=tracker,
            version=version,
        )
        labels = sticky_hmm_segments(novelty.slopes, rho=float(rho), min_run=int(min_run))
        accumulate_reuse_credit(
            labels=labels,
            novelty=novelty,
            row_keys=row_keys,
            row_mass=row_mass,
            effective_mass=effective_mass,
            redundant_mass=redundant_mass,
            scratch=reuse_scratch,
        )

    return neuron_weights(effective_mass, redundant_mass)


def score_responses(cache_dir: Path, weights: np.ndarray) -> np.ndarray:
    """Return the Good-Mass Fraction for every cached response."""

    cache_dir = Path(cache_dir)
    _require_cache_files(
        cache_dir,
        ("base/row_ptr.int64", "base/keys.uint32", "base/w_sum.float16"),
    )
    response_ptr = np.memmap(cache_dir / "base/row_ptr.int64", dtype=np.int64, mode="r")
    keys_mem = np.memmap(cache_dir / "base/keys.uint32", dtype=np.uint32, mode="r")
    mass_mem = np.memmap(cache_dir / "base/w_sum.float16", dtype=np.float16, mode="r")

    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 1 or weights.size < ARR_SIZE:
        raise ValueError(f"weights must be a one-dimensional array of size at least {ARR_SIZE}")
    positive_weights = np.clip(weights, 0.0, None)
    absolute_weights = np.abs(weights)

    scores = np.full(len(response_ptr) - 1, np.nan, dtype=np.float32)
    for response in range(len(scores)):
        start, end = int(response_ptr[response]), int(response_ptr[response + 1])
        if end <= start:
            continue
        keys, mass = _retained_row(keys_mem[start:end], mass_mem[start:end])
        if keys.size == 0 or float(np.sum(mass)) <= 0.0:
            continue
        positive_mass = float(np.sum(mass * positive_weights[keys]))
        credible_mass = float(np.sum(mass * absolute_weights[keys]))
        scores[response] = positive_mass / credible_mass if credible_mass > 0.0 else 0.0
    return scores


def score_cache(cache_dir: Path, weights: np.ndarray) -> float:
    """Return NEX: the mean per-response Good-Mass Fraction."""

    scores = score_responses(cache_dir, weights)
    return float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else float("nan")
