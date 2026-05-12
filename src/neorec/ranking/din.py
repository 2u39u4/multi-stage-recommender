"""Deep Interest Network (DIN) — attention-based fine-ranking.

Reference
---------
Zhou, Zhu, Song et al. *Deep Interest Network for Click-Through Rate
Prediction.* KDD 2018.

Core idea
---------
A user's interest is **target-aware**: when scoring movie X, only the parts
of the user's history that are *similar to X* should contribute to the user
representation. DIN learns this via an **attention unit** that takes the
target item embedding and each history item embedding as input:

.. math::

   a(h_i, t) = \\mathrm{MLP}([\\, h_i \\,;\\, t \\,;\\, h_i - t \\,;\\, h_i \\odot t \\,])

The attention weights are **not softmax-ed** (unlike vanilla self-attention)
— the paper argues this preserves the magnitude of "interest intensity".
Padding positions get a mask of 0.

Final user representation::

    u_repr = Σ_i a(h_i, t) · h_i        (over non-pad history positions)

Then the CTR logit is::

    logit = MLP([ user_id_emb ; target_emb ; u_repr ; side_features ])

Ablation
--------
``cfg.rank.model.use_attention=False`` degrades the attention unit to a
constant 1, yielding plain sum pooling — directly quantifies the gain
from attention.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from neorec.ranking.base import BaseRanker
from neorec.ranking.features import SEQ_PAD_VALUE, RankingFeaturizer

log = logging.getLogger(__name__)


# ===========================================================================
# Attention unit
# ===========================================================================
class AttentionUnit(nn.Module):
    """Local activation unit: MLP over [h ; target ; h-target ; h*target]."""

    def __init__(self, embedding_dim: int, hidden: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        in_dim = 4 * embedding_dim
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hist: torch.Tensor,        # (B, L, d)
        target: torch.Tensor,      # (B, d)
        mask: torch.Tensor,        # (B, L) 1=real, 0=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weights (B, L), pooled (B, d))``."""
        B, L, d = hist.shape
        tgt = target.unsqueeze(1).expand(-1, L, -1)  # (B, L, d)
        feats = torch.cat([hist, tgt, hist - tgt, hist * tgt], dim=-1)  # (B, L, 4d)
        score = self.mlp(feats).squeeze(-1)  # (B, L)
        # Mask padding before any aggregation.
        score = score.masked_fill(mask == 0, 0.0)
        # Weighted sum, **not** softmax-normalised — DIN's design choice.
        pooled = (score.unsqueeze(-1) * hist).sum(dim=1)  # (B, d)
        return score, pooled


# ===========================================================================
# Full DIN network
# ===========================================================================
class DIN(nn.Module):
    def __init__(
        self,
        cardinalities: dict[str, int],
        num_genres: int,
        embedding_dim: int = 32,
        attention_hidden: tuple[int, ...] = (64, 32),
        dnn_hidden: tuple[int, ...] = (200, 80),
        dropout: float = 0.3,
        use_attention: bool = True,
        n_recall_features: int = 0,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.cardinalities = dict(cardinalities)
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        self.n_recall_features = int(n_recall_features)
        self.side_fields = [f for f in cardinalities if f not in ("user_id", "item_id")]

        self.user_emb = nn.Embedding(cardinalities["user_id"], embedding_dim)
        self.item_emb = nn.Embedding(cardinalities["item_id"], embedding_dim)
        self.side_emb = nn.ModuleDict({
            f: nn.Embedding(cardinalities[f], embedding_dim) for f in self.side_fields
        })
        self.genre_emb = nn.Embedding(num_genres, embedding_dim, padding_idx=0)

        self.attention = AttentionUnit(embedding_dim, attention_hidden) if use_attention else None

        # MLP head: [user, target, user_repr_from_history, side(*), genre_pooled, (recall)].
        n_side = len(self.side_fields)
        head_in = embedding_dim * (3 + n_side + 1)  # +1 for genre
        head_in += self.n_recall_features            # raw recall-score + mask cols
        layers: list[nn.Module] = []
        for h in dnn_hidden:
            layers += [nn.Linear(head_in, h), nn.PReLU(), nn.Dropout(dropout)]
            head_in = h
        layers.append(nn.Linear(head_in, 1))
        self.head = nn.Sequential(*layers)

        # Wide-style linear path for recall scores — guarantees DIN at least
        # matches the recall-layer's own ordering even before attention learns.
        if self.n_recall_features > 0:
            self.recall_linear = nn.Linear(self.n_recall_features, 1)
        else:
            self.recall_linear = None

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in [self.user_emb, self.item_emb, self.genre_emb] + list(self.side_emb.values()):
            nn.init.normal_(emb.weight, std=1e-2)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        if self.recall_linear is not None:
            nn.init.zeros_(self.recall_linear.weight)
            nn.init.zeros_(self.recall_linear.bias)

    # ------------------------------------------------------------------
    def _embed_history(
        self, history: torch.Tensor, history_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # SEQ_PAD_VALUE is -1; replace with 0 (any valid item) and zero-out via mask.
        safe = history.clamp(min=0)
        emb = self.item_emb(safe)                          # (B, L, d)
        emb = emb * history_mask.unsqueeze(-1)              # zero padded rows
        return emb, history_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_e = self.user_emb(batch["user_id"])
        target_e = self.item_emb(batch["item_id"])

        # History
        hist_e, hist_mask = self._embed_history(batch["history"], batch["history_mask"])
        if self.use_attention and self.attention is not None:
            _, user_repr = self.attention(hist_e, target_e, hist_mask)
        else:
            # Sum pooling baseline.
            user_repr = hist_e.sum(dim=1)

        # Side features
        side_vecs = [self.side_emb[f](batch[f]) for f in self.side_fields]

        # Genres mean-pool.
        g_emb = self.genre_emb(batch["genres"])
        g_mask = batch["genres_mask"].unsqueeze(-1)
        g_pooled = (g_emb * g_mask).sum(dim=1) / g_mask.sum(dim=1).clamp(min=1.0)

        parts = [user_e, target_e, user_repr, *side_vecs, g_pooled]
        if self.n_recall_features > 0 and "recall_scores" in batch:
            parts.append(batch["recall_scores"])
        x = torch.cat(parts, dim=-1)
        logit = self.head(x).squeeze(-1)
        if self.recall_linear is not None and "recall_scores" in batch:
            logit = logit + self.recall_linear(batch["recall_scores"]).squeeze(-1)
        return logit

    @torch.no_grad()
    def attention_weights(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return (B, L) attention weights for the notebook visualisation."""
        if not self.use_attention or self.attention is None:
            raise RuntimeError("Attention disabled — no weights to extract.")
        target_e = self.item_emb(batch["item_id"])
        hist_e, hist_mask = self._embed_history(batch["history"], batch["history_mask"])
        weights, _ = self.attention(hist_e, target_e, hist_mask)
        return weights


# ===========================================================================
# Ranker wrapper
# ===========================================================================
class DINRanker(BaseRanker):
    name = "din"
    stage = "fine_rank"
    needs_history = True

    def __init__(self, cfg, featurizer: RankingFeaturizer) -> None:
        super().__init__(cfg, featurizer)
        self.device = torch.device(cfg.rank.model.get("device", "cpu"))
        cards = featurizer.cardinalities()
        n_recall_features = (
            featurizer.recall_store.n_features if featurizer.recall_store is not None else 0
        )
        self.model = DIN(
            cardinalities=cards,
            num_genres=featurizer.schema.num_genres,
            embedding_dim=int(cfg.rank.model.embedding_dim),
            attention_hidden=tuple(cfg.rank.model.attention_hidden),
            dnn_hidden=tuple(cfg.rank.model.dnn_hidden),
            dropout=float(cfg.rank.model.dnn_dropout),
            use_attention=bool(cfg.rank.model.use_attention),
            n_recall_features=n_recall_features,
        ).to(self.device)

    # ------------------------------------------------------------------
    def _batch_to_tensors(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> dict[str, torch.Tensor]:
        b = self.featurizer.featurize(user_ids, item_ids, include_history=True)
        tensors: dict[str, torch.Tensor] = {}
        for col in self.featurizer.schema.sparse_cols:
            tensors[col] = torch.from_numpy(b.sparse[col]).long().to(self.device)
        tensors["genres"] = torch.from_numpy(b.genres).long().to(self.device)
        tensors["genres_mask"] = torch.from_numpy(b.genres_mask).float().to(self.device)
        # ``history`` may contain -1 padding tokens; we keep them and rely on the mask.
        tensors["history"] = torch.from_numpy(b.history).long().to(self.device)
        tensors["history_mask"] = torch.from_numpy(b.history_mask).float().to(self.device)
        if b.recall_scores is not None and self.model.n_recall_features > 0:
            tensors["recall_scores"] = torch.from_numpy(b.recall_scores).float().to(self.device)
        return tensors

    # ------------------------------------------------------------------
    def fit(
        self,
        train_pairs: pd.DataFrame,
        valid_pairs: pd.DataFrame,
    ) -> dict[str, float]:
        epochs = int(self.cfg.rank.train.epochs)
        bs = int(self.cfg.rank.train.batch_size)
        lr = float(self.cfg.rank.train.lr)
        wd = float(self.cfg.rank.train.get("weight_decay", 0.0))
        patience = int(self.cfg.rank.train.get("early_stopping_patience", 3))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        users_tr = train_pairs["user_id"].to_numpy(np.int64)
        items_tr = train_pairs["item_id"].to_numpy(np.int64)
        labels_tr = train_pairs["label"].to_numpy(np.float32)
        users_v = valid_pairs["user_id"].to_numpy(np.int64)
        items_v = valid_pairs["item_id"].to_numpy(np.int64)
        labels_v = valid_pairs["label"].to_numpy(np.float32)

        loader = DataLoader(
            TensorDataset(torch.arange(len(users_tr))),
            batch_size=bs, shuffle=True, num_workers=0,
        )

        best = float("inf")
        bad = 0
        for ep in range(epochs):
            self.model.train()
            losses: list[float] = []
            for (idx,) in loader:
                idx = idx.numpy()
                tensors = self._batch_to_tensors(users_tr[idx], items_tr[idx])
                y = torch.from_numpy(labels_tr[idx]).to(self.device)
                logit = self.model(tensors)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.cfg.rank.train.get("grad_clip", 5.0)),
                )
                optimizer.step()
                losses.append(loss.item())

            # Valid
            self.model.eval()
            with torch.no_grad():
                v_losses: list[float] = []
                for i in range(0, len(users_v), bs):
                    j = min(i + bs, len(users_v))
                    tensors = self._batch_to_tensors(users_v[i:j], items_v[i:j])
                    y = torch.from_numpy(labels_v[i:j]).to(self.device)
                    logit = self.model(tensors)
                    v_losses.append(F.binary_cross_entropy_with_logits(logit, y).item())
            v_loss = float(np.mean(v_losses))
            log.info("ep %d/%d  tr_loss=%.4f  va_loss=%.4f", ep + 1, epochs,
                     float(np.mean(losses)), v_loss)

            if v_loss + 1e-5 < best:
                best = v_loss
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    log.info("Early stopping at epoch %d.", ep + 1)
                    break

        return {"final_valid_bce": float(best)}

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        self.model.eval()
        bs = int(self.cfg.rank.train.get("inference_batch_size",
                                          self.cfg.rank.train.batch_size))
        out = np.zeros(len(user_ids), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(user_ids), bs):
                j = min(i + bs, len(user_ids))
                tensors = self._batch_to_tensors(user_ids[i:j], item_ids[i:j])
                logit = self.model(tensors)
                out[i:j] = torch.sigmoid(logit).cpu().numpy()
        return out

    def attention_for_users(
        self,
        user_ids: np.ndarray,
        target_items: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(weights, history, mask)`` for visualisation."""
        self.model.eval()
        with torch.no_grad():
            tensors = self._batch_to_tensors(user_ids, target_items)
            weights = self.model.attention_weights(tensors)
        return (
            weights.cpu().numpy(),
            tensors["history"].cpu().numpy(),
            tensors["history_mask"].cpu().numpy(),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "din.pt")
        (path / "meta.json").write_text(json.dumps({
            "name": self.name,
            "cardinalities": self.model.cardinalities,
            "num_genres": self.model.num_genres,
            "embedding_dim": self.model.embedding_dim,
            "use_attention": self.model.use_attention,
            "n_recall_features": self.model.n_recall_features,
        }, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.model.load_state_dict(torch.load(path / "din.pt", map_location=self.device))
