"""Offline evaluation: ranking metrics, significance tests, counterfactual evaluation."""

from neorec.eval.metrics import (
    coverage,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    novelty,
    recall_at_k,
)

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "hit_rate_at_k",
    "coverage",
    "novelty",
]
