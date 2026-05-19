"""Feature store — unified offline / online feature lookup.

Offline: parquet files under ``cfg.paths.data_processed``.
Online:  Redis-backed cache (see ``neorec.serving.feature_cache``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
        self._users: dict[int, UserFeatures] | None = None
        self._items: dict[int, ItemFeatures] | None = None
        self._histories: dict[int, list[int]] | None = None

    def get_user(self, user_id: int) -> UserFeatures:
        self._load_users()
        assert self._users is not None
        try:
            return self._users[int(user_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown user_id={user_id}") from exc

    def get_item(self, item_id: int) -> ItemFeatures:
        self._load_items()
        assert self._items is not None
        try:
            return self._items[int(item_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id={item_id}") from exc

    def batch_get_items(self, item_ids: list[int]) -> list[ItemFeatures]:
        return [self.get_item(item_id) for item_id in item_ids]

    def _load_users(self) -> None:
        if self._users is not None:
            return
        users_path = self.processed_dir / "user_features.parquet"
        seq_path = self.processed_dir / "sequences.parquet"
        if not users_path.exists():
            raise FileNotFoundError(f"Missing user feature table: {users_path}")

        histories: dict[int, list[int]] = {}
        if seq_path.exists():
            seq_df = pd.read_parquet(seq_path)
            histories = {
                int(row.user_id): [int(x) for x in row.history]
                for row in seq_df.itertuples(index=False)
            }

        users_df = pd.read_parquet(users_path)
        self._histories = histories
        self._users = {
            int(row.user_id): UserFeatures(
                user_id=int(row.user_id),
                age_bucket=int(row.age_bucket),
                gender=int(row.gender),
                occupation=int(row.occupation),
                history=histories.get(int(row.user_id), []),
            )
            for row in users_df.itertuples(index=False)
        }

    def _load_items(self) -> None:
        if self._items is not None:
            return
        items_path = self.processed_dir / "item_features.parquet"
        if not items_path.exists():
            raise FileNotFoundError(f"Missing item feature table: {items_path}")

        items_df = pd.read_parquet(items_path)
        self._items = {
            int(row.item_id): ItemFeatures(
                item_id=int(row.item_id),
                genres=[int(x) for x in row.genres],
                year_bucket=int(row.year_bucket),
                popularity_bucket=int(row.popularity_bucket),
            )
            for row in items_df.itertuples(index=False)
        }
