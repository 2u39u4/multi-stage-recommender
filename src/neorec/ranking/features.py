"""Shared feature schema + featurizer for all ranking models.

All ranking models (LR / GBDT / DeepFM / DIN) read the same parquet artefacts
produced by ``data/preprocess.py`` and emit a common set of NumPy arrays.

Feature groups
--------------
* **sparse**       : user_id, item_id, gender, age_bucket, occupation,
                     year_bucket, popularity_bucket   →  embed (or one-hot for LR/GBDT)
* **multi-hot**    : item genres (up to ``max_genres`` ids, padded with 0)
                     →  embed + mean for deep models, multi-hot for LR/GBDT
* **sequence**     : user's chronological positive history (padded /
                     truncated to ``max_seq_len``)
                     →  used by DIN only

The featurizer is **stateless after construction** and **vectorised**:
``featurize(user_ids, item_ids)`` is constant-time per row, no Python
loops in the hot path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Padding token reserved across multi-hot and sequence features.
PAD_GENRE = 0
PAD_ITEM = 0  # caller is responsible for handling the collision if needed
SEQ_PAD_VALUE = -1  # use -1 because item_id=0 is a real item


@dataclass
class FeatureSchema:
    """Cardinalities for every embedding table the ranking models will build.

    Cardinality values are *inclusive of the padding slot* where applicable,
    so a model can do ``nn.Embedding(schema.num_genres + 1, d, padding_idx=0)``.
    """

    num_users: int
    num_items: int
    num_genders: int = 2
    num_age_buckets: int = 0
    num_occupations: int = 0
    num_year_buckets: int = 0
    num_popularity_buckets: int = 0
    num_genres: int = 0  # one-hot vocabulary (genre ids start at 1, 0 = pad)
    max_genres: int = 6
    max_seq_len: int = 50
    sparse_cols: list[str] = field(
        default_factory=lambda: [
            "user_id",
            "item_id",
            "gender",
            "age_bucket",
            "occupation",
            "year_bucket",
            "popularity_bucket",
        ]
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_genders": self.num_genders,
            "num_age_buckets": self.num_age_buckets,
            "num_occupations": self.num_occupations,
            "num_year_buckets": self.num_year_buckets,
            "num_popularity_buckets": self.num_popularity_buckets,
            "num_genres": self.num_genres,
            "max_genres": self.max_genres,
            "max_seq_len": self.max_seq_len,
            "sparse_cols": self.sparse_cols,
        }


@dataclass
class FeatureBatch:
    """One ``featurize`` output, columns aligned by row index."""

    sparse: dict[str, np.ndarray]         # each (N,)
    genres: np.ndarray                    # (N, max_genres)
    genres_mask: np.ndarray               # (N, max_genres) — 1=real, 0=pad
    history: np.ndarray | None = None     # (N, max_seq_len)
    history_mask: np.ndarray | None = None  # (N, max_seq_len)

    def __len__(self) -> int:
        return len(next(iter(self.sparse.values())))


class RankingFeaturizer:
    """Loads parquet artefacts and assembles feature batches for any (user, item) pair list."""

    def __init__(
        self,
        processed_dir: Path,
        max_genres: int = 6,
        max_seq_len: int = 50,
        train_interactions_path: Path | None = None,
    ) -> None:
        processed_dir = Path(processed_dir)
        users = pd.read_parquet(processed_dir / "user_features.parquet")
        items = pd.read_parquet(processed_dir / "item_features.parquet")
        id_maps = json.loads((processed_dir / "id_maps.json").read_text())

        num_users = int(users["user_id"].max()) + 1
        num_items = int(items["item_id"].max()) + 1

        # Build lookup tables — densely indexed by user_id / item_id for O(1) access.
        self.user_gender = np.zeros(num_users, dtype=np.int64)
        self.user_age_bucket = np.zeros(num_users, dtype=np.int64)
        self.user_occupation = np.zeros(num_users, dtype=np.int64)
        self.user_gender[users["user_id"].to_numpy()] = users["gender"].to_numpy()
        self.user_age_bucket[users["user_id"].to_numpy()] = users["age_bucket"].to_numpy()
        self.user_occupation[users["user_id"].to_numpy()] = users["occupation"].to_numpy()

        self.item_year_bucket = np.zeros(num_items, dtype=np.int64)
        self.item_pop_bucket = np.zeros(num_items, dtype=np.int64)
        self.item_year_bucket[items["item_id"].to_numpy()] = items["year_bucket"].to_numpy()
        self.item_pop_bucket[items["item_id"].to_numpy()] = items["popularity_bucket"].to_numpy()

        # Genres come in as numpy arrays of ids → pad to ``max_genres``.
        self.max_genres = int(max_genres)
        self.item_genres = np.zeros((num_items, max_genres), dtype=np.int64)
        self.item_genres_mask = np.zeros((num_items, max_genres), dtype=np.float32)
        for item_id, genres in zip(items["item_id"].to_numpy(), items["genres"], strict=True):
            g = np.asarray(genres, dtype=np.int64)[: max_genres]
            self.item_genres[item_id, : len(g)] = g
            self.item_genres_mask[item_id, : len(g)] = 1.0

        # Behaviour sequences (only available for DIN; build lazily from train df).
        self.max_seq_len = int(max_seq_len)
        self._train_interactions_path = train_interactions_path
        self._user_history: np.ndarray | None = None  # (num_users, max_seq_len)
        self._user_history_mask: np.ndarray | None = None

        self.schema = FeatureSchema(
            num_users=num_users,
            num_items=num_items,
            num_genders=int(users["gender"].max()) + 1,
            num_age_buckets=int(users["age_bucket"].max()) + 1,
            num_occupations=int(users["occupation"].max()) + 1,
            num_year_buckets=int(items["year_bucket"].max()) + 1,
            num_popularity_buckets=int(items["popularity_bucket"].max()) + 1,
            num_genres=len(id_maps["genre_map"]) + 1,  # +1 for padding slot 0
            max_genres=max_genres,
            max_seq_len=max_seq_len,
        )

        log.info(
            "RankingFeaturizer ready: %d users, %d items, %d genres, max_seq_len=%d",
            num_users, num_items, self.schema.num_genres, max_seq_len,
        )

    # ------------------------------------------------------------------
    # Sequence handling
    # ------------------------------------------------------------------
    def build_sequences(self, train_df: pd.DataFrame) -> None:
        """Build per-user chronological history from training positives.

        Each user's history is the **most recent** ``max_seq_len`` items, left-padded
        with ``SEQ_PAD_VALUE`` (= -1). Items are stored in chronological order
        (oldest first, newest last) — which matches DIN's typical input convention.
        """
        n_users = self.schema.num_users
        L = self.max_seq_len

        history = np.full((n_users, L), SEQ_PAD_VALUE, dtype=np.int64)
        mask = np.zeros((n_users, L), dtype=np.float32)

        df = train_df.sort_values(["user_id", "ts"]).reset_index(drop=True)
        for user_id, group in df.groupby("user_id", sort=False):
            items = group["item_id"].to_numpy()
            if len(items) >= L:
                items = items[-L:]
                history[user_id] = items
                mask[user_id] = 1.0
            else:
                history[user_id, -len(items):] = items
                mask[user_id, -len(items):] = 1.0

        self._user_history = history
        self._user_history_mask = mask
        log.info("Built user history table: shape=%s, non-pad ratio=%.2f",
                 history.shape, float(mask.mean()))

    # ------------------------------------------------------------------
    # Featurise one batch of (user, item) pairs
    # ------------------------------------------------------------------
    def featurize(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        include_history: bool = False,
    ) -> FeatureBatch:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        if user_ids.shape != item_ids.shape:
            raise ValueError(f"shape mismatch: users={user_ids.shape}, items={item_ids.shape}")

        sparse = {
            "user_id":           user_ids,
            "item_id":           item_ids,
            "gender":            self.user_gender[user_ids],
            "age_bucket":        self.user_age_bucket[user_ids],
            "occupation":        self.user_occupation[user_ids],
            "year_bucket":       self.item_year_bucket[item_ids],
            "popularity_bucket": self.item_pop_bucket[item_ids],
        }
        genres = self.item_genres[item_ids]
        genres_mask = self.item_genres_mask[item_ids]

        if include_history:
            if self._user_history is None:
                raise RuntimeError(
                    "Call build_sequences(train_df) before featurize(..., include_history=True)"
                )
            history = self._user_history[user_ids]
            history_mask = self._user_history_mask[user_ids]
        else:
            history = None
            history_mask = None

        return FeatureBatch(
            sparse=sparse,
            genres=genres,
            genres_mask=genres_mask,
            history=history,
            history_mask=history_mask,
        )

    # ------------------------------------------------------------------
    # Cardinality dict (handy for embedding tables)
    # ------------------------------------------------------------------
    def cardinalities(self) -> dict[str, int]:
        return {
            "user_id":           self.schema.num_users,
            "item_id":           self.schema.num_items,
            "gender":            self.schema.num_genders,
            "age_bucket":        self.schema.num_age_buckets,
            "occupation":        self.schema.num_occupations,
            "year_bucket":       self.schema.num_year_buckets,
            "popularity_bucket": self.schema.num_popularity_buckets,
        }


# ===========================================================================
# Training-time negative sampling
# ===========================================================================
def build_training_pairs(
    train_df: pd.DataFrame,
    num_items: int,
    user_seen: dict[int, set[int]],
    negative_ratio: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Expand training positives into a flat (user, item, label) frame.

    For each positive (u, i, label=1), sample ``negative_ratio`` items uniformly
    at random from items the user has **not** interacted with → (u, j, label=0).

    Returns
    -------
    pd.DataFrame with columns ``user_id``, ``item_id``, ``label`` (int8).
    Rows are shuffled so SGD batches see a mix of users.
    """
    rng = np.random.default_rng(seed)
    pos_users = train_df["user_id"].to_numpy(dtype=np.int64)
    pos_items = train_df["item_id"].to_numpy(dtype=np.int64)
    n_pos = len(pos_users)

    neg_users = np.repeat(pos_users, negative_ratio)
    neg_items = np.empty(n_pos * negative_ratio, dtype=np.int64)
    for i in range(n_pos * negative_ratio):
        u = neg_users[i]
        seen = user_seen.get(int(u), set())
        for _ in range(20):  # rejection sampling — very low collision on ML-1M
            cand = int(rng.integers(num_items))
            if cand not in seen:
                neg_items[i] = cand
                break
        else:
            neg_items[i] = int(rng.integers(num_items))

    users = np.concatenate([pos_users, neg_users])
    items = np.concatenate([pos_items, neg_items])
    labels = np.concatenate(
        [np.ones(n_pos, dtype=np.int8), np.zeros(n_pos * negative_ratio, dtype=np.int8)]
    )

    out = pd.DataFrame({"user_id": users, "item_id": items, "label": labels})
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
