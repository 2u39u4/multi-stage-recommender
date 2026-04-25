"""Popularity recall — heuristic baseline.

Two variants:
    * raw count
    * time-decayed count: weight_i = exp(-ln(2) * age_days / half_life)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from neorec.recall.base import BaseRecaller, RecallResult


class PopularityRecaller(BaseRecaller):
    name = "popularity"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._top_items: list[int] = []
        self._top_scores: list[float] = []

    def fit(self, interactions_path: str | Path) -> None:
        """TODO(W2 Day 13):
        * groupby(item_id).size() [.multiply(decay_weight) if time_decay]
        * cache top cfg.output.top_k items
        """
        raise NotImplementedError  # TODO(W2)

    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        """Return the same top-K list for every user (filter already-watched later)."""
        raise NotImplementedError  # TODO(W2)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)
