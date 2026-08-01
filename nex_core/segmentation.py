"""Sticky two-state HMM segmentation of a novelty trajectory."""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture


EPS = 1e-6


def _enforce_min_run(labels: np.ndarray, *, min_run: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool).copy()
    if labels.size == 0 or int(min_run) <= 1:
        return labels
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and bool(labels[end]) == bool(labels[start]):
            end += 1
        if end - start < int(min_run):
            previous = bool(labels[start - 1]) if start > 0 else None
            following = bool(labels[end]) if end < labels.size else None
            if previous is not None or following is not None:
                labels[start:end] = previous if previous is not None else following
        start = end
    return labels


def _preprocess_novelty(slopes: np.ndarray) -> tuple[np.ndarray, bool]:
    slopes = np.asarray(slopes, dtype=np.float64)
    if slopes.ndim != 1:
        raise ValueError("slopes must be one-dimensional")
    if slopes.size == 0:
        return slopes.copy(), False

    transformed = np.log1p(np.clip(slopes, 0.0, None))
    time = np.arange(transformed.size, dtype=np.float64)
    design = np.stack((np.ones_like(time), np.log1p(time)), axis=1)
    try:
        coefficients, *_ = np.linalg.lstsq(design, transformed, rcond=None)
        residual = transformed - design @ coefficients
    except Exception:
        residual = transformed - float(np.median(transformed))

    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if not np.isfinite(mad) or mad <= 0.0:
        return residual - median, False
    return (residual - median) / (mad + EPS), True


def preprocess_novelty(slopes: np.ndarray) -> np.ndarray:
    """Apply log transform, log-time detrending, and median/MAD scaling."""

    observations, _ = _preprocess_novelty(slopes)
    return observations


def sticky_hmm_segments(
    slopes: np.ndarray, *, rho: float = 0.95, min_run: int = 2
) -> np.ndarray:
    """Return `True` for Explore rows and `False` for Exploit rows."""

    slopes = np.asarray(slopes, dtype=np.float64)
    if slopes.ndim != 1:
        raise ValueError("slopes must be one-dimensional")
    if int(min_run) < 1:
        raise ValueError("min_run must be at least one")
    if not np.isfinite(rho) or not 0.0 < float(rho) < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    if slopes.size < 3:
        return np.zeros(slopes.size, dtype=bool)

    observations, has_nonzero_mad = _preprocess_novelty(slopes)
    if not has_nonzero_mad:
        return _enforce_min_run(observations > 0.0, min_run=int(min_run))

    try:
        values = observations.reshape(-1, 1)
        mixture = GaussianMixture(
            n_components=2,
            covariance_type="diag",
            random_state=0,
            max_iter=50,
            reg_covar=1e-6,
        )
        mixture.fit(values)
        means = mixture.means_.reshape(-1)
        variances = np.clip(mixture.covariances_.reshape(-1), 1e-6, None)

        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            init_params="",
            params="",
        )
        model.startprob_ = np.asarray([0.5, 0.5], dtype=np.float64)
        model.transmat_ = np.asarray(
            [[rho, 1.0 - rho], [1.0 - rho, rho]], dtype=np.float64
        )
        model.means_ = means.reshape(-1, 1)
        model.covars_ = variances.reshape(-1, 1)
        _, states = model.decode(values, algorithm="viterbi")
        explore_state = int(np.argmax(means))
        explore = states == explore_state
    except Exception:
        explore = observations > 0.0

    return _enforce_min_run(explore, min_run=int(min_run))
