"""Two-Tower deep retrieval (DSSM / YouTubeDNN style).

Reference:
    * Huang et al. Learning Deep Structured Semantic Models. CIKM 2013.
    * Covington et al. Deep Neural Networks for YouTube Recommendations. RecSys 2016.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from neorec.recall.base import BaseRecaller, RecallResult


class UserTower(nn.Module):
    """MLP over (user_id, age_bucket, gender, occupation, recent_items_mean)."""

    def __init__(self, num_users: int, embedding_dim: int, hidden: list[int]) -> None:
        super().__init__()
        # TODO(W2): implement
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.mlp: nn.Module = nn.Identity()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError  # TODO(W2)


class ItemTower(nn.Module):
    """MLP over (item_id, genres_multi_hot, year_bucket, popularity_bucket)."""

    def __init__(self, num_items: int, embedding_dim: int, hidden: list[int]) -> None:
        super().__init__()
        # TODO(W2): implement
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.mlp: nn.Module = nn.Identity()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError  # TODO(W2)


class TwoTowerModel(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(
        self,
        user_batch: dict[str, torch.Tensor],
        item_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return cosine similarity between user and item embeddings."""
        raise NotImplementedError  # TODO(W2)


class TwoTowerRecaller(BaseRecaller):
    """Train Two-Tower with in-batch sampled softmax, then FAISS for retrieval."""

    name = "two_tower"

    def fit(self, interactions_path: str | Path) -> None:
        """TODO(W2 Day 8-10):
        * build DataLoader
        * loss: -log softmax_temp( u·i / τ ) with in-batch negatives
        * early stopping on valid Recall@100
        * save user / item embeddings as .npy
        """
        raise NotImplementedError  # TODO(W2)

    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        """FAISS HNSW search over item index."""
        raise NotImplementedError  # TODO(W2)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)
