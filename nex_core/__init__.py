"""Anonymous reference implementation of the NEX core score."""

from .novelty import ARR_SIZE, FirstSeenTracker, NoveltyResult, first_positive_novelty
from .reuse_credit import ReuseScratch, accumulate_reuse_credit, neuron_weights
from .scoring import learn_nex_weights, score_cache, score_responses
from .segmentation import preprocess_novelty, sticky_hmm_segments

__all__ = [
    "ARR_SIZE",
    "FirstSeenTracker",
    "NoveltyResult",
    "ReuseScratch",
    "accumulate_reuse_credit",
    "first_positive_novelty",
    "learn_nex_weights",
    "neuron_weights",
    "preprocess_novelty",
    "score_cache",
    "score_responses",
    "sticky_hmm_segments",
]
