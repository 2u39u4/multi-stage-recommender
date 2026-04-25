"""Cold-start recall — content-based fallback for users with too little history.

Strategy:
    1. Build TF-IDF over item genres (and year_bucket).
    2. Represent a user by the mean TF-IDF of items they've watched.
    3. For a cold user (<=5 interactions), recall nearest items by cosine sim.
    4. If the user has literally zero history, fall back to popularity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from neorec.recall.base import BaseRecaller, RecallResult


class ColdStartRecaller(BaseRecaller):
    name = "cold_start"

    def fit(self, interactions_path: str | Path) -> None:
        """Build item TF-IDF and user-profile centroids."""
        raise NotImplementedError  # TODO(W2)

    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        raise NotImplementedError  # TODO(W2)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)
