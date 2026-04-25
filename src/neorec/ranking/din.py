"""Deep Interest Network (DIN) — attention-based fine-ranking.

Reference: Zhou et al. Deep Interest Network for Click-Through Rate Prediction.
KDD 2018.

Key idea: attention weights between the target item and each element of the
user's historical behavior sequence make the user representation
target-dependent.

The ablation switch ``cfg.model.use_attention`` downgrades DIN to sum pooling
so we can directly quantify the lift from attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from neorec.ranking.base import BaseRanker, RankResult


class AttentionUnit(nn.Module):
    """MLP over [h_i ; target ; h_i - target ; h_i * target]."""

    def __init__(self, embedding_dim: int, hidden: list[int]) -> None:
        super().__init__()
        # TODO(W3 Day 18-20): MLP + dice/prelu activation
        self.mlp: nn.Module = nn.Identity()

    def forward(
        self, hist: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Return attention-weighted sum of hist w.r.t. target."""
        raise NotImplementedError  # TODO(W3)


class DIN(nn.Module):
    """Full DIN network: embedding lookup + attention + MLP head."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        attention_hidden: list[int] | None = None,
        dnn_hidden: list[int] | None = None,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        # TODO(W3 Day 18-20): build modules
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return CTR logit for each (user, target_item) pair."""
        raise NotImplementedError  # TODO(W3)

    def attention_weights(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the (B, L) attention matrix — used by the notebook viz."""
        raise NotImplementedError  # TODO(W3)


class DINRanker(BaseRanker):
    name = "din"
    stage = "fine_rank"

    def fit(self, train_path: str | Path, valid_path: str | Path) -> None:
        """TODO(W3): training loop, BCE, MLflow logging, save attention snapshots."""
        raise NotImplementedError  # TODO(W3)

    def predict(
        self,
        user_ids: Sequence[int],
        candidate_items: Sequence[Sequence[int]],
    ) -> RankResult:
        raise NotImplementedError  # TODO(W3)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W3)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W3)
