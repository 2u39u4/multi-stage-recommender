"""Behavior Sequence Transformer (BST-style) — optional fine-ranker.

Reference: Chen et al. Behavior Sequence Transformer for E-commerce
Recommendation. DLP-KDD 2019.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from neorec.ranking.base import BaseRanker


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
        self.seq_max_len = int(seq_max_len)
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_max_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history"]
        target = batch["item_id"]
        positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0)
        x = self.item_emb(history) + self.pos_emb(positions)
        pad_mask = batch.get("history_mask")
        key_padding_mask = pad_mask == 0 if pad_mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if pad_mask is None:
            pooled = encoded[:, -1, :]
        else:
            weights = pad_mask.float()
            pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        target_vec = self.item_emb(target)
        return self.head(torch.cat([pooled, target_vec], dim=-1)).squeeze(-1)


class TransformerCTRRanker(BaseRanker):
    name = "transformer_ctr"
    stage = "fine_rank"
    needs_history = True

    def __init__(self, cfg, featurizer) -> None:
        super().__init__(cfg, featurizer)
        model_cfg = cfg.rank.model
        self.device = torch.device(str(model_cfg.get("device", "cpu")))
        self.model = BSTModel(
            num_items=featurizer.schema.num_items,
            embedding_dim=int(model_cfg.embedding_dim),
            num_blocks=int(model_cfg.num_blocks),
            num_heads=int(model_cfg.num_heads),
            ffn_hidden=int(model_cfg.ffn_hidden),
            seq_max_len=int(model_cfg.seq_max_len),
            dropout=float(model_cfg.dropout),
        ).to(self.device)

    def _batch_tensors(self, pairs: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.featurizer.featurize(
            pairs["user_id"].to_numpy(dtype=np.int64),
            pairs["item_id"].to_numpy(dtype=np.int64),
            include_history=True,
        )
        assert batch.history is not None
        history = np.where(batch.history < 0, -1, batch.history) + 1
        items = pairs["item_id"].to_numpy(dtype=np.int64) + 1
        labels = pairs["label"].to_numpy(dtype=np.float32)
        return (
            torch.as_tensor(history, dtype=torch.long),
            torch.as_tensor(items, dtype=torch.long),
            torch.as_tensor(labels, dtype=torch.float32),
        )

    def fit(self, train_pairs: pd.DataFrame, valid_pairs: pd.DataFrame) -> dict[str, float]:
        train_history, train_items, train_labels = self._batch_tensors(train_pairs)
        ds = TensorDataset(train_history, train_items, train_labels)
        loader = DataLoader(
            ds,
            batch_size=int(self.cfg.rank.train.batch_size),
            shuffle=True,
        )
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.rank.train.lr),
            weight_decay=float(self.cfg.rank.train.weight_decay),
        )

        self.model.train()
        losses: list[float] = []
        for _ in range(int(self.cfg.rank.train.epochs)):
            for history, items, labels in loader:
                payload = {
                    "history": history.to(self.device),
                    "item_id": items.to(self.device),
                    "history_mask": (history > 0).float().to(self.device),
                }
                logits = self.model(payload)
                loss = F.binary_cross_entropy_with_logits(logits, labels.to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))

        valid_scores = self.score(
            valid_pairs["user_id"].to_numpy(dtype=np.int64),
            valid_pairs["item_id"].to_numpy(dtype=np.int64),
        )
        valid_labels = valid_pairs["label"].to_numpy(dtype=np.float32)
        valid_bce = F.binary_cross_entropy(
            torch.as_tensor(np.clip(valid_scores, 1e-7, 1 - 1e-7), dtype=torch.float32),
            torch.as_tensor(valid_labels, dtype=torch.float32),
        )
        return {
            "final_train_bce": float(np.mean(losses)) if losses else 0.0,
            "final_valid_bce": float(valid_bce),
        }

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        pairs = pd.DataFrame({"user_id": user_ids, "item_id": item_ids, "label": 0})
        history, items, _ = self._batch_tensors(pairs)
        self.model.eval()
        scores: list[np.ndarray] = []
        batch_size = int(self.cfg.rank.train.get("inference_batch_size", 4096))
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                h = history[start:start + batch_size].to(self.device)
                it = items[start:start + batch_size].to(self.device)
                logits = self.model({
                    "history": h,
                    "item_id": it,
                    "history_mask": (h > 0).float(),
                })
                scores.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(scores).astype(np.float32) if scores else np.empty(0, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "meta.json").write_text(json.dumps({"name": self.name}, indent=2))

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path) / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
