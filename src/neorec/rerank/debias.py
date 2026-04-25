"""Popularity / exposure debias via inverse propensity re-weighting."""

from __future__ import annotations

from collections.abc import Sequence


def ips_rerank(
    candidate_ids: Sequence[int],
    candidate_scores: Sequence[float],
    item_popularity: dict[int, float],
    clip: tuple[float, float] = (0.01, 10.0),
    k: int | None = None,
) -> list[int]:
    """Re-rank candidates by IPS-adjusted score: ``score / clip(popularity)``."""
    raise NotImplementedError  # TODO(W4 Day 22-23)
