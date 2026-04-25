"""Feature store — unified offline / online feature lookup.

Offline: parquet files under ``cfg.paths.data_processed``.
Online:  Redis-backed cache (see ``neorec.serving.feature_cache``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UserFeatures:
    user_id: int
    age_bucket: int
    gender: int
    occupation: int
    history: list[int]


@dataclass
class ItemFeatures:
    item_id: int
    genres: list[int]
    year_bucket: int
    popularity_bucket: int


class FeatureStore:
    """Thin wrapper that can switch between offline parquet and online Redis."""

    def __init__(self, processed_dir: str | Path) -> None:
        self.processed_dir = Path(processed_dir)
        # TODO(W1): lazy-load parquet tables into in-memory dicts on first access.

    def get_user(self, user_id: int) -> UserFeatures:
        raise NotImplementedError  # TODO(W1)

    def get_item(self, item_id: int) -> ItemFeatures:
        raise NotImplementedError  # TODO(W1)

    def batch_get_items(self, item_ids: list[int]) -> list[ItemFeatures]:
        raise NotImplementedError  # TODO(W1)
