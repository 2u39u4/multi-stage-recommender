"""Behavior Sequence Transformer (BST-style) — optional fine-ranker.

Reference: Chen et al. Behavior Sequence Transformer for E-commerce
Recommendation. DLP-KDD 2019.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from neorec.ranking.base import BaseRanker, RankResult


class BSTModel(nn.Module):
    """Target-aware Transformer encoder over user behavior sequence."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 32,
        num_blocks: int = 2,
        num_heads: int = 4,
        ffn_hidden: int = 128,
        seq_max_len: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # TODO(W3 optional): implement
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError  # TODO(W3 optional)


class TransformerCTRRanker(BaseRanker):
    name = "transformer_ctr"
    stage = "fine_rank"

    def fit(self, train_path: str | Path, valid_path: str | Path) -> None:
        raise NotImplementedError  # TODO(W3 optional)

    def predict(
        self,
        user_ids: Sequence[int],
        candidate_items: Sequence[Sequence[int]],
    ) -> RankResult:
        raise NotImplementedError  # TODO(W3 optional)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W3 optional)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W3 optional)
