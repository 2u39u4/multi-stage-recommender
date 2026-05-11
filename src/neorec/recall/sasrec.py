"""SASRec — Self-Attentive Sequential Recommendation.

Architecture (Kang & McAuley, ICDM 2018)
----------------------------------------
For each user we treat their chronologically-sorted training history as a
sequence ``[i_1, …, i_T]``.  At every position ``t`` the model uses only items
up to ``t`` (causal mask) and predicts ``i_{t+1}``.

    input target    = [pad, …, i_1, i_2, …, i_{T-1}]   →   [i_1, …, i_T]

    item_emb (V+1×D, pad=V)   ┐
                              ├── add ── dropout ── N × {causal-MHA + FFN}
    pos_emb (L×D)             ┘
                                                       │
                                                       LayerNorm
                                                       │  (B, L, D)

At inference, we take the hidden state at the **last non-pad** position and
inner-product it against every item embedding to produce the recall score.

Loss
----
BPR with one uniform negative per non-pad position (rejection-sampled against
the user's own training-set seen items).  Per-position BCE is the original
SASRec choice; BPR is empirically very similar and lets us share the same
sampler / loss code path as the two-tower channel.

Notes on conventions
--------------------
* All item ids in our dataset are 0-indexed in ``[0, num_items)``.  Inside the
  model we use ``num_items`` itself as the padding token, which gives us a
  clean ``padding_idx`` without ever colliding with real items.
* Sequences are **left-padded** to ``max_seq_len`` so the most recent item is
  always at position ``L-1`` — that's the hidden state we read at inference.

References
----------
Kang, McAuley.  *Self-Attentive Sequential Recommendation.*  ICDM 2018.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from neorec.recall.base import BaseRecaller, RecallResult
from neorec.utils.io import ensure_dir, read_json, write_json

log = logging.getLogger(__name__)


# ===========================================================================
# Dataset
# ===========================================================================
class _SASRecDataset(Dataset):
    """One sample per user: full training history → (input, target, negative).

    Stored layout per sample (all length = ``max_seq_len``, left-padded):

        input  : [pad, …, pad, i_1,   i_2,   …, i_{T-1}]
        target : [pad, …, pad, i_2,   i_3,   …, i_T   ]
        neg    : [pad, …, pad, j_1,   j_2,   …, j_{T-1}]

    ``target == pad`` positions are skipped by the loss.
    """

    def __init__(
        self,
        user_seq: dict[int, list[int]],
        num_items: int,
        max_seq_len: int,
        rng_seed: int = 42,
    ) -> None:
        self.users = [u for u, seq in user_seq.items() if len(seq) >= 2]
        self.user_seq = user_seq
        self.user_seen = {u: set(s) for u, s in user_seq.items()}
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.pad_idx = num_items
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negative(self, seen: set[int]) -> int:
        for _ in range(8):
            cand = int(self._rng.integers(0, self.num_items))
            if cand not in seen:
                return cand
        return cand  # accept the rare collision

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u = self.users[idx]
        full = self.user_seq[u]
        # Trim to max_seq_len + 1 so input length ≤ max_seq_len
        if len(full) > self.max_seq_len + 1:
            full = full[-(self.max_seq_len + 1):]

        input_seq = full[:-1]
        target_seq = full[1:]
        seen = self.user_seen[u]
        neg_seq = [self._sample_negative(seen) for _ in input_seq]

        L = self.max_seq_len
        pad = L - len(input_seq)
        if pad > 0:
            input_seq = [self.pad_idx] * pad + input_seq
            target_seq = [self.pad_idx] * pad + target_seq
            neg_seq = [self.pad_idx] * pad + neg_seq

        return {
            "input":  torch.tensor(input_seq, dtype=torch.long),
            "target": torch.tensor(target_seq, dtype=torch.long),
            "neg":    torch.tensor(neg_seq, dtype=torch.long),
        }


# ===========================================================================
# Model
# ===========================================================================
class _SASRecBlock(nn.Module):
    """Pre-LN transformer block — more stable than post-LN at small depth."""

    def __init__(self, dim: int, num_heads: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.ff_ln = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_n = self.attn_ln(x)
        attn_out, _ = self.attn(
            x_n, x_n, x_n,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ff_ln(x)))
        return x


class SASRecModel(nn.Module):
    """Item & positional embeddings → N causal self-attention blocks → (B,L,D)."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_seq_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.pad_idx = num_items

        # +1 slot reserved for the padding token (index = num_items).
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=num_items)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.input_ln = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.blocks = nn.ModuleList(
            _SASRecBlock(embedding_dim, num_heads, dropout, layer_norm_eps)
            for _ in range(num_blocks)
        )

        # Causal mask is constant across the batch — register it once.
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", mask, persistent=False)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[self.pad_idx].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) of item ids (pad_idx for padding). Returns (B, L, D)."""
        L = seq.size(1)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand_as(seq)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.input_dropout(x)

        pad_mask = seq == self.pad_idx  # (B, L)
        # nn.MultiheadAttention dislikes rows that are *fully* padded — replace
        # those with all-False so attention computes (and we just ignore the
        # output for those positions in the loss).
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad] = False

        for block in self.blocks:
            x = block(x, self.causal_mask[:L, :L], pad_mask)
        return self.input_ln(x)


# ===========================================================================
# Recaller
# ===========================================================================
class SASRecRecaller(BaseRecaller):
    name = "sasrec"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.model: SASRecModel | None = None
        self.device: torch.device = torch.device("cpu")
        self.num_items = 0
        self.max_seq_len = 50
        self.embedding_dim = 64
        self.num_blocks = 2
        self.num_heads = 2
        self.dropout = 0.2
        self.layer_norm_eps = 1e-12

        self._user_seq: dict[int, list[int]] = {}
        self._user_seen: dict[int, set[int]] = {}
        self._item_emb: np.ndarray | None = None  # cached for fast inference

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, interactions_path: str | Path) -> None:
        interactions_path = Path(interactions_path)
        processed_dir = interactions_path.parent

        train_df = pd.read_parquet(interactions_path)
        id_maps = read_json(processed_dir / "id_maps.json")
        self.num_items = int(id_maps["item_id_map_size"])
        log.info("SASRec setup: items=%d, train rows=%d",
                 self.num_items, len(train_df))

        self.max_seq_len = int(self.cfg.recall.model.max_seq_len)
        self.embedding_dim = int(self.cfg.recall.model.embedding_dim)
        self.num_blocks = int(self.cfg.recall.model.num_blocks)
        self.num_heads = int(self.cfg.recall.model.num_heads)
        self.dropout = float(self.cfg.recall.model.dropout)
        self.layer_norm_eps = float(self.cfg.recall.model.layer_norm_eps)

        # Build per-user training sequence directly from train_df (no leakage:
        # train_df already excludes the LOO test item).
        train_df = train_df.sort_values(["user_id", "ts"])
        self._user_seq = (
            train_df.groupby("user_id")["item_id"].apply(list).to_dict()
        )
        self._user_seen = {u: set(s) for u, s in self._user_seq.items()}
        log.info("Users with ≥2 train items: %d", sum(1 for s in self._user_seq.values() if len(s) >= 2))

        device_cfg = str(self.cfg.recall.model.get("device", "auto")).lower()
        if device_cfg == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_cfg)
        log.info("Using device: %s", self.device)

        self.model = SASRecModel(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_seq_len,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Model params: %s", f"{n_params:,}")

        ds = _SASRecDataset(
            user_seq=self._user_seq,
            num_items=self.num_items,
            max_seq_len=self.max_seq_len,
            rng_seed=int(self.cfg.seed),
        )
        loader = DataLoader(
            ds,
            batch_size=int(self.cfg.recall.train.batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.recall.train.lr),
            weight_decay=float(self.cfg.recall.train.weight_decay),
        )
        epochs = int(self.cfg.recall.train.epochs)
        grad_clip = float(self.cfg.recall.train.grad_clip)
        log.info("Training BPR-per-position: epochs=%d batch=%d lr=%.1e",
                 epochs, int(self.cfg.recall.train.batch_size),
                 self.cfg.recall.train.lr)

        pad_idx = self.model.pad_idx
        self.model.train()
        for epoch in range(epochs):
            losses: list[float] = []
            for batch in loader:
                input_ids = batch["input"].to(self.device)    # (B, L)
                target_ids = batch["target"].to(self.device)
                neg_ids = batch["neg"].to(self.device)

                hidden = self.model(input_ids)               # (B, L, D)
                # Look up positives/negatives from the same item embedding.
                pos_v = self.model.item_emb(target_ids)
                neg_v = self.model.item_emb(neg_ids)

                pos_score = (hidden * pos_v).sum(dim=-1)     # (B, L)
                neg_score = (hidden * neg_v).sum(dim=-1)

                valid = (target_ids != pad_idx).float()
                per_pos = F.softplus(neg_score - pos_score) * valid
                denom = valid.sum().clamp(min=1.0)
                loss = per_pos.sum() / denom

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                optim.step()
                losses.append(loss.item())
            log.info("epoch %d/%d  bpr_loss=%.4f",
                     epoch + 1, epochs, float(np.mean(losses)))

        self._cache_item_embeddings()

    # ------------------------------------------------------------------
    def _cache_item_embeddings(self) -> None:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            self._item_emb = self.model.item_emb.weight[: self.num_items].detach().cpu().numpy()
        log.info("Cached item embeddings: %s", self._item_emb.shape)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def _build_input_batch(self, user_ids: np.ndarray) -> torch.Tensor:
        """Build a left-padded (B, L) input tensor from each user's history."""
        L = self.max_seq_len
        pad_idx = self.num_items
        batch = np.full((len(user_ids), L), pad_idx, dtype=np.int64)
        for row, uid in enumerate(user_ids):
            seq = self._user_seq.get(int(uid), [])
            if not seq:
                continue
            seq = seq[-L:]
            batch[row, L - len(seq):] = seq
        return torch.from_numpy(batch)

    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        assert self.model is not None and self._item_emb is not None, "fit() first"

        users = np.asarray(list(user_ids), dtype=np.int32)
        # Tolerate being driven by a parent cfg (e.g. MergeRecaller) that
        # doesn't carry the SASRec training section any more.
        batch_size = int(
            OmegaConf.select(self.cfg, "recall.train.inference_batch_size", default=512)
        )
        all_scores = np.empty((len(users), self.num_items), dtype=np.float32)

        self.model.eval()
        item_emb_t = torch.from_numpy(self._item_emb).to(self.device)  # (V, D)

        with torch.no_grad():
            for start in range(0, len(users), batch_size):
                chunk = users[start : start + batch_size]
                input_t = self._build_input_batch(chunk).to(self.device)
                hidden = self.model(input_t)            # (B, L, D)
                last = hidden[:, -1, :]                 # (B, D)
                scores = last @ item_emb_t.T            # (B, V)
                all_scores[start : start + len(chunk)] = scores.cpu().numpy()

        # Mask already-seen items
        for row, uid in enumerate(users):
            seen = self._user_seen.get(int(uid))
            if seen:
                all_scores[row, list(seen)] = -np.inf

        if k >= all_scores.shape[1]:
            idx = np.argsort(-all_scores, axis=1)[:, :k]
        else:
            partial = np.argpartition(-all_scores, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(all_scores.shape[0])[:, None]
            partial_scores = all_scores[row_idx, partial]
            order = np.argsort(-partial_scores, axis=1)
            idx = partial[row_idx, order]
        out_scores = np.take_along_axis(all_scores, idx, axis=1)
        out_scores = np.where(np.isinf(out_scores), 0.0, out_scores)

        return RecallResult(
            user_ids=users,
            item_ids=idx.astype(np.int32),
            scores=out_scores.astype(np.float32),
            channel=self.name,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        assert self.model is not None and self._item_emb is not None
        out = ensure_dir(path)
        torch.save(self.model.state_dict(), out / "model.pt")
        np.save(out / "item_emb.npy", self._item_emb)
        seen_rows = [
            {"user_id": int(u), "item_id": int(i)}
            for u, items in self._user_seen.items()
            for i in items
        ]
        pd.DataFrame(seen_rows).to_parquet(out / "user_seen.parquet", index=False)
        # Persist the training sequences too — needed at recall() time to encode the user.
        seq_rows = [
            {"user_id": int(u), "history": list(map(int, s))}
            for u, s in self._user_seq.items()
        ]
        pd.DataFrame(seq_rows).to_parquet(out / "user_sequences.parquet", index=False)
        write_json(
            {
                "num_items":      self.num_items,
                "embedding_dim":  self.embedding_dim,
                "max_seq_len":    self.max_seq_len,
                "num_blocks":     self.num_blocks,
                "num_heads":      self.num_heads,
                "dropout":        self.dropout,
                "layer_norm_eps": self.layer_norm_eps,
            },
            out / "meta.json",
        )
        log.info("Saved SASRec artefacts to %s", out)

    def load(self, path: str | Path) -> None:
        in_dir = Path(path)
        meta = read_json(in_dir / "meta.json")
        self.num_items = meta["num_items"]
        self.embedding_dim = meta["embedding_dim"]
        self.max_seq_len = meta["max_seq_len"]
        self.num_blocks = meta["num_blocks"]
        self.num_heads = meta["num_heads"]
        self.dropout = meta["dropout"]
        self.layer_norm_eps = meta["layer_norm_eps"]

        self.model = SASRecModel(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_seq_len,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.model.load_state_dict(torch.load(in_dir / "model.pt", map_location="cpu"))
        self.model.eval()
        self.device = torch.device("cpu")
        self._item_emb = np.load(in_dir / "item_emb.npy")

        seen_df = pd.read_parquet(in_dir / "user_seen.parquet")
        self._user_seen = (
            seen_df.groupby("user_id")["item_id"].agg(lambda s: set(s.tolist())).to_dict()
        )
        seq_df = pd.read_parquet(in_dir / "user_sequences.parquet")
        self._user_seq = {
            int(u): list(map(int, h)) for u, h in zip(seq_df["user_id"], seq_df["history"])
        }
        log.info("Loaded SASRec artefacts from %s", in_dir)
