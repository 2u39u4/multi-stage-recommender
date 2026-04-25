"""Business-rule filters applied after MMR / debias.

    * drop items already in the user's history
    * cap the number of items per genre
    * cap per year_bucket
"""

from __future__ import annotations

from collections.abc import Sequence


def apply_rules(
    candidate_ids: Sequence[int],
    user_history: set[int],
    item_meta: dict[int, dict[str, object]],
    max_per_genre_ratio: float = 0.5,
    max_per_year_bucket: int = 3,
    filter_already_watched: bool = True,
    k: int = 10,
) -> list[int]:
    """Apply business rules greedily, preserving the input order."""
    raise NotImplementedError  # TODO(W4 Day 22-23)
