#!/usr/bin/env python3
"""Command-line entry point for training and evaluating the NEX core score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nex_core import learn_nex_weights, score_responses


def _save_weights(path: Path, weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=np.asarray(weights, dtype=np.float32))


def _load_weights(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "weights" not in archive:
            raise KeyError(f"{path} does not contain a 'weights' array")
        return np.asarray(archive["weights"], dtype=np.float32)


def _write_score_outputs(
    output_dir: Path,
    scores: np.ndarray,
    *,
    rho: float | None = None,
    min_run: int | None = None,
    log_gap_output: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "response_scores.npy", scores.astype(np.float32))
    finite = scores[np.isfinite(scores)]
    nex_score = float(np.mean(finite)) if finite.size else None
    if log_gap_output and nex_score is not None:
        remaining_mass = max(
            1.0 - float(np.clip(nex_score, 0.0, 1.0)),
            1e-12,
        )
        nex_score = float(-np.log10(remaining_mass))

    summary: dict[str, int | float | None] = {
        "n_responses": int(scores.size),
        "n_scored_responses": int(finite.size),
        "nex_score": nex_score,
    }
    if rho is not None:
        summary["rho"] = float(rho)
    if min_run is not None:
        summary["min_run"] = int(min_run)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--min-run", type=int, default=2)
    parser.add_argument(
        "--max-responses",
        type=int,
        default=0,
        help="Maximum calibration responses; 0 uses all responses.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Learn NEX neuron weights.")
    _add_training_arguments(train)
    train.add_argument("--weights-out", type=Path, required=True)

    score = commands.add_parser("score", help="Score a cache using saved weights.")
    score.add_argument("--cache-dir", type=Path, required=True)
    score.add_argument("--weights", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument(
        "--log-gap-output",
        action="store_true",
        help="Report -log10(1-NEX) as the final nex_score.",
    )

    run = commands.add_parser(
        "run", help="Learn weights and score the same candidate cache."
    )
    _add_training_arguments(run)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--log-gap-output",
        action="store_true",
        help="Report -log10(1-NEX) as the final nex_score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        weights = learn_nex_weights(
            args.cache_dir,
            rho=args.rho,
            min_run=args.min_run,
            max_responses=args.max_responses,
        )
        _save_weights(args.weights_out, weights)
        print(f"Saved weights to {args.weights_out}")
        return

    if args.command == "score":
        scores = score_responses(args.cache_dir, _load_weights(args.weights))
        _write_score_outputs(
            args.output_dir,
            scores,
            log_gap_output=args.log_gap_output,
        )
        return

    weights = learn_nex_weights(
        args.cache_dir,
        rho=args.rho,
        min_run=args.min_run,
        max_responses=args.max_responses,
    )
    _save_weights(args.output_dir / "weights.npz", weights)
    scores = score_responses(args.cache_dir, weights)
    _write_score_outputs(
        args.output_dir,
        scores,
        rho=args.rho,
        min_run=args.min_run,
        log_gap_output=args.log_gap_output,
    )


if __name__ == "__main__":
    main()
