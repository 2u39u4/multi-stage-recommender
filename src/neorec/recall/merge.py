"""Fuse multi-channel recall outputs into a single candidate pool.

Three strategies (compared in the ablation study):
    * ``norm_weighted`` — min-max normalize each channel's scores, then weighted sum.
    * ``rrf``           — Reciprocal Rank Fusion: score = Σ_c 1 / (k_rrf + rank_c).
    * ``learned``       — train a small LR on (channel score, channel rank) features.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from neorec.recall.base import RecallResult


def merge_rrf(
    results: Iterable[RecallResult],
    k_rrf: int = 60,
    candidate_pool_size: int = 1000,
) -> RecallResult:
    """Reciprocal Rank Fusion."""
    raise NotImplementedError  # TODO(W2 Day 14)


def merge_norm_weighted(
    results: Iterable[RecallResult],
    weights: dict[str, float],
    candidate_pool_size: int = 1000,
) -> RecallResult:
    """Min-max normalize per channel, then weighted sum."""
    raise NotImplementedError  # TODO(W2 Day 14)


def merge(
    results: Iterable[RecallResult],
    strategy: str = "rrf",
    **kwargs: object,
) -> RecallResult:
    """Dispatch to the requested fusion strategy."""
    if strategy == "rrf":
        return merge_rrf(results, **kwargs)  # type: ignore[arg-type]
    if strategy == "norm_weighted":
        return merge_norm_weighted(results, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown merge strategy: {strategy}")


__all__ = ["merge", "merge_rrf", "merge_norm_weighted"]
