"""SASRec — Self-Attentive Sequential Recommendation.

Reference: Kang & McAuley. Self-Attentive Sequential Recommendation. ICDM 2018.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from neorec.recall.base import BaseRecaller, RecallResult


class SASRec(nn.Module):
    """Item + positional embeddings → N self-attention blocks → next-item logits."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_seq_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # TODO(W2 Day 11-12): encoder blocks with causal mask
        self.blocks = nn.ModuleList()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) of item ids; returns (B, L, D) hidden states."""
        raise NotImplementedError  # TODO(W2)


class SASRecRecaller(BaseRecaller):
    name = "sasrec"

    def fit(self, interactions_path: str | Path) -> None:
        """TODO(W2 Day 11-12):
        * build user sequence dataset
        * next-item prediction with sampled softmax or full softmax
        * early stopping on valid Recall@10
        """
        raise NotImplementedError  # TODO(W2)

    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        """Encode user's last-N history, then inner-product with all item embeddings."""
        raise NotImplementedError  # TODO(W2)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W2)
