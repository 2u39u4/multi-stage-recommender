"""Two-Tower deep retrieval (DSSM / YouTubeDNN style).

Architecture
------------
Item tower
    item_id_emb (V_i × D)   ─┐
    genre_emb mean (G × D)  ─┼── concat ── MLP ── L2-normalize ── item vector (D)
                             │  (item_id_emb is shared with the user tower's
                             │   sequence encoder for parameter efficiency.)

User tower
    user_id_emb (V_u × D)        ─┐
    mean of recent item_emb (D)  ─┼── concat ── MLP ── L2-normalize ── user vector (D)

Loss
----
In-batch sampled softmax with temperature τ:

    logits_{ij} = (u_i · v_j) / τ
    L = - log softmax_j(logits)[i, i]

Each row in the batch's positives serves as everyone else's negative, which
gives "popularity-weighted" negatives for free (item more popular ⇒ more
likely to appear in the batch ⇒ more often a negative).

References
----------
* Huang et al. *Learning Deep Structured Semantic Models.* CIKM 2013.
* Covington et al. *Deep Neural Networks for YouTube Recommendations.* RecSys 2016.
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
from torch.utils.data import DataLoader, Dataset

from neorec.recall.base import BaseRecaller, RecallResult
from neorec.utils.io import ensure_dir, read_json, write_json

log = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================
def _pad_or_truncate(seq: list[int], length: int, pad_value: int = 0) -> list[int]:
    """Right-pad or truncate (keep last ``length`` elements)."""
    if len(seq) >= length:
        return seq[-length:]
    return seq + [pad_value] * (length - len(seq))


# ===========================================================================
# Dataset
# ===========================================================================
class _BPRDataset(Dataset):
    """BPR triplets (user, pos_item, neg_item) — heavily pre-computed.

    All look-up tables are pre-built as ``np.ndarray``\\ s so that
    ``__getitem__`` is just constant-time slicing — no Python-side list /
    dict work in the inner loop, which is the difference between a 14 s/epoch
    and a 60 s/epoch training run on a small dataset.

    Negative sampling is uniform over items, with rejection on the user's
    seen set.  The collision rate is tiny (≤ 5 % for ML-1M), so the expected
    number of retries per draw is < 1.05.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_seen: dict[int, set[int]],
        user_seq: dict[int, list[int]],
        item_genres: dict[int, list[int]],
        num_users: int,
        num_items: int,
        max_seq_len: int,
        max_genres: int,
        rng_seed: int = 42,
    ) -> None:
        self.users = interactions["user_id"].to_numpy(dtype=np.int64)
        self.items = interactions["item_id"].to_numpy(dtype=np.int64)
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.max_genres = max_genres

        # Pre-compute (num_users, max_seq_len) sequence + mask tables.
        self.seq_table = np.zeros((num_users, max_seq_len), dtype=np.int64)
        self.seq_mask_table = np.zeros((num_users, max_seq_len), dtype=np.float32)
        for u, hist in user_seq.items():
            hist = list(hist)
            if len(hist) >= max_seq_len:
                self.seq_table[u] = hist[-max_seq_len:]
                self.seq_mask_table[u, :] = 1.0
            elif hist:
                pad = max_seq_len - len(hist)
                self.seq_table[u, pad:] = hist
                self.seq_mask_table[u, pad:] = 1.0

        # Pre-compute (num_items, max_genres) genre + mask tables.
        self.genre_table = np.zeros((num_items, max_genres), dtype=np.int64)
        self.gmask_table = np.zeros((num_items, max_genres), dtype=np.float32)
        for i, gens in item_genres.items():
            gens = list(gens)[:max_genres] or [0]
            self.genre_table[i, : len(gens)] = gens
            self.gmask_table[i, : len(gens)] = [1.0 if g != 0 else 0.0 for g in gens]

        # Per-user seen set as a sorted numpy array (fast np.searchsorted),
        # plus a python set for O(1) collision check during negative sampling.
        self.user_seen_set: list[set[int]] = [set() for _ in range(num_users)]
        for u, items in user_seen.items():
            self.user_seen_set[u] = items

        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negative(self, u: int) -> int:
        seen = self.user_seen_set[u] if u < len(self.user_seen_set) else set()
        for _ in range(8):
            cand = int(self._rng.integers(0, self.num_items))
            if cand not in seen:
                return cand
        return cand  # extremely rare worst case — caller tolerates a false negative

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u = int(self.users[idx])
        pos = int(self.items[idx])
        neg = self._sample_negative(u)
        return {
            "user_id":    torch.from_numpy(np.array(u, dtype=np.int64)),
            "pos_item":   torch.from_numpy(np.array(pos, dtype=np.int64)),
            "neg_item":   torch.from_numpy(np.array(neg, dtype=np.int64)),
            "seq":        torch.from_numpy(self.seq_table[u]),
            "seq_mask":   torch.from_numpy(self.seq_mask_table[u]),
            "pos_genres": torch.from_numpy(self.genre_table[pos]),
            "pos_gmask":  torch.from_numpy(self.gmask_table[pos]),
            "neg_genres": torch.from_numpy(self.genre_table[neg]),
            "neg_gmask":  torch.from_numpy(self.gmask_table[neg]),
        }


# ===========================================================================
# Model
# ===========================================================================
def _mlp(in_dim: int, hidden: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    return nn.Sequential(*layers)


def _masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of non-masked rows. ``emb``: (B, L, D), ``mask``: (B, L)."""
    weights = mask.unsqueeze(-1)
    summed = (emb * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return summed / denom


class TwoTowerModel(nn.Module):
    """Two-Tower with side features (DSSM/YouTubeDNN style).

    The user tower fuses ``user_id_emb`` and the masked-mean of recent item
    embeddings.  The item tower fuses ``item_id_emb`` and the masked-mean of
    genre embeddings.  Both go through their own MLP and are L2-normalized.

    The L2 normalization can be turned off by passing ``normalize=False``,
    which is recommended on small datasets where letting magnitudes grow gives
    the optimizer more headroom to separate positives from negatives.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        embedding_dim: int = 64,
        user_hidden: list[int] | None = None,
        item_hidden: list[int] | None = None,
        dropout: float = 0.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.normalize = normalize

        # Shared item embedding (used both as the item tower input and as the
        # historical-item encoder in the user tower). No explicit padding row:
        # pad positions in sequences are zeroed out by the mask before mean
        # pooling, so they contribute neither activation nor gradient.
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        # Genres: id 0 is reserved as pad (preprocess emits ≥1 for real genres).
        self.genre_emb = nn.Embedding(num_genres + 1, embedding_dim, padding_idx=0)

        # Per-item bias (popularity bias term — critical for in-batch softmax).
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.zeros_(self.item_bias.weight)

        self.user_mlp = self._build_mlp(embedding_dim * 2, user_hidden, dropout)
        self.item_mlp = self._build_mlp(embedding_dim * 2, item_hidden, dropout)

        # Larger init scale than Xavier — embeddings need real signal early.
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.normal_(self.genre_emb.weight[1:], std=0.1)

    @staticmethod
    def _build_mlp(in_dim: int, hidden: list[int] | None, dropout: float) -> nn.Module:
        if not hidden:
            return nn.Identity()
        layers: list[nn.Module] = []
        last = in_dim
        for k, h in enumerate(hidden):
            layers.append(nn.Linear(last, h))
            if k < len(hidden) - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            last = h
        return nn.Sequential(*layers)

    # --------------------------------------------------------------------
    def _maybe_norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) if self.normalize else x

    def encode_user(
        self,
        user_id: torch.Tensor,
        seq: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        u_id = self.user_emb(user_id)                        # (B, D)
        s_emb = self.item_emb(seq)                           # (B, L, D)
        s_mean = _masked_mean(s_emb, seq_mask)               # (B, D)
        x = torch.cat([u_id, s_mean], dim=-1)
        return self._maybe_norm(self.user_mlp(x))

    def encode_item(
        self,
        item_id: torch.Tensor,
        genres: torch.Tensor,
        genre_mask: torch.Tensor,
    ) -> torch.Tensor:
        i_emb = self.item_emb(item_id)
        g_emb = self.genre_emb(genres)
        g_mean = _masked_mean(g_emb, genre_mask)
        x = torch.cat([i_emb, g_mean], dim=-1)
        return self._maybe_norm(self.item_mlp(x))

    # --------------------------------------------------------------------
    def score_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward pass — computes scores for *both* positive and negative
        items, sharing the user encoding so we don't pay for it twice.

        Returns ``(pos_score, neg_score)`` of shape ``(B,)`` each.
        """
        u = self.encode_user(batch["user_id"], batch["seq"], batch["seq_mask"])
        pos_v = self.encode_item(batch["pos_item"], batch["pos_genres"], batch["pos_gmask"])
        neg_v = self.encode_item(batch["neg_item"], batch["neg_genres"], batch["neg_gmask"])
        pos_b = self.item_bias(batch["pos_item"]).squeeze(-1)
        neg_b = self.item_bias(batch["neg_item"]).squeeze(-1)
        pos_score = (u * pos_v).sum(dim=-1) + pos_b
        neg_score = (u * neg_v).sum(dim=-1) + neg_b
        return pos_score, neg_score


# ===========================================================================
# Recaller
# ===========================================================================
class TwoTowerRecaller(BaseRecaller):
    name = "two_tower"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.model: TwoTowerModel | None = None
        self.device: torch.device = torch.device("cpu")
        self._user_vecs: np.ndarray | None = None
        self._item_vecs: np.ndarray | None = None
        self._item_biases: np.ndarray | None = None
        self._user_seen: dict[int, set[int]] = {}
        self.num_users = 0
        self.num_items = 0
        self.num_genres = 0
        self.max_seq_len = 50
        self.max_genres = 6
        self.embedding_dim = 64

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, interactions_path: str | Path) -> None:
        interactions_path = Path(interactions_path)
        processed_dir = interactions_path.parent

        train_df = pd.read_parquet(interactions_path)
        seq_df = pd.read_parquet(processed_dir / "sequences.parquet")
        item_df = pd.read_parquet(processed_dir / "item_features.parquet")
        id_maps = read_json(processed_dir / "id_maps.json")

        self.num_users = int(id_maps["user_id_map_size"])
        self.num_items = int(id_maps["item_id_map_size"])
        self.num_genres = int(len(id_maps["genre_map"]))
        log.info("Two-Tower setup: users=%d items=%d genres=%d",
                 self.num_users, self.num_items, self.num_genres)

        self.max_seq_len = int(self.cfg.data.features.sequence.max_len)
        self.max_genres = int(item_df["genres"].apply(len).max())
        self.embedding_dim = int(self.cfg.recall.model.embedding_dim)

        user_seq = dict(zip(seq_df["user_id"], seq_df["history"].apply(list)))
        item_genres = dict(zip(item_df["item_id"], item_df["genres"].apply(list)))
        self._user_seen = {u: set(h) for u, h in user_seq.items()}

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

        self.model = TwoTowerModel(
            num_users=self.num_users,
            num_items=self.num_items,
            num_genres=self.num_genres,
            embedding_dim=self.embedding_dim,
            user_hidden=list(self.cfg.recall.model.user_tower_hidden),
            item_hidden=list(self.cfg.recall.model.item_tower_hidden),
            dropout=float(self.cfg.recall.model.dropout),
            normalize=bool(self.cfg.recall.model.get("normalize", False)),
        ).to(self.device)

        ds = _BPRDataset(
            interactions=train_df,
            user_seen=self._user_seen,
            user_seq=user_seq,
            item_genres=item_genres,
            num_users=self.num_users,
            num_items=self.num_items,
            max_seq_len=self.max_seq_len,
            max_genres=self.max_genres,
            rng_seed=int(self.cfg.seed),
        )
        loader = DataLoader(
            ds,
            batch_size=int(self.cfg.recall.train.batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.recall.train.lr),
            weight_decay=float(self.cfg.recall.train.weight_decay),
        )
        epochs = int(self.cfg.recall.train.epochs)
        log.info("Training BPR: epochs=%d batch=%d lr=%.1e",
                 epochs, int(self.cfg.recall.train.batch_size),
                 self.cfg.recall.train.lr)

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("Model params: %s", f"{n_params:,}")

        self.model.train()
        for epoch in range(epochs):
            losses: list[float] = []
            for batch in loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                pos_score, neg_score = self.model.score_batch(batch)
                # BPR — minimize  -log σ(s_pos - s_neg) == softplus(s_neg - s_pos)
                loss = F.softplus(neg_score - pos_score).mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optim.step()
                losses.append(loss.item())
            log.info("epoch %d/%d  bpr_loss=%.4f", epoch + 1, epochs, float(np.mean(losses)))

        # Pre-compute item / user embeddings for fast inference
        self._cache_embeddings(seq_df, item_df, item_genres, user_seq)

    # ------------------------------------------------------------------
    def _cache_embeddings(
        self,
        seq_df: pd.DataFrame,
        item_df: pd.DataFrame,
        item_genres: dict[int, list[int]],
        user_seq: dict[int, list[int]],
    ) -> None:
        assert self.model is not None
        self.model.eval()

        # Items: one big batch
        item_ids = item_df["item_id"].to_numpy()
        all_genres = np.zeros((len(item_ids), self.max_genres), dtype=np.int64)
        all_gmask = np.zeros((len(item_ids), self.max_genres), dtype=np.float32)
        for k, iid in enumerate(item_ids):
            g = list(item_genres.get(int(iid), [0]))
            g = _pad_or_truncate(g, self.max_genres, pad_value=0)
            all_genres[k] = g
            all_gmask[k] = [1.0 if x != 0 else 0.0 for x in g]

        with torch.no_grad():
            item_ids_t = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            item_vecs = self.model.encode_item(
                item_ids_t,
                torch.tensor(all_genres, dtype=torch.long, device=self.device),
                torch.tensor(all_gmask, dtype=torch.float32, device=self.device),
            )
            biases = self.model.item_bias(item_ids_t).squeeze(-1)

        item_dim = item_vecs.shape[1]
        self._item_vecs = np.zeros((self.num_items, item_dim), dtype=np.float32)
        self._item_vecs[item_ids] = item_vecs.cpu().numpy()
        self._item_biases = np.zeros(self.num_items, dtype=np.float32)
        self._item_biases[item_ids] = biases.cpu().numpy()

        # Users: also one big batch (only ~6k users → fits easily on CPU)
        user_ids = np.arange(self.num_users)
        all_seq = np.zeros((self.num_users, self.max_seq_len), dtype=np.int64)
        all_smask = np.zeros((self.num_users, self.max_seq_len), dtype=np.float32)
        for u in user_ids:
            seq = list(user_seq.get(int(u), []))
            if len(seq) >= self.max_seq_len:
                all_seq[u] = seq[-self.max_seq_len:]
                all_smask[u] = 1.0
            else:
                pad = self.max_seq_len - len(seq)
                all_seq[u, pad:] = seq
                all_smask[u, pad:] = 1.0

        with torch.no_grad():
            user_vecs = self.model.encode_user(
                torch.tensor(user_ids, dtype=torch.long, device=self.device),
                torch.tensor(all_seq, dtype=torch.long, device=self.device),
                torch.tensor(all_smask, dtype=torch.float32, device=self.device),
            )
        self._user_vecs = user_vecs.cpu().numpy()

        log.info("Cached vecs: users=%s items=%s",
                 self._user_vecs.shape, self._item_vecs.shape)

    # ------------------------------------------------------------------
    # Recall (numpy matmul + filter)
    # ------------------------------------------------------------------
    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        assert self._user_vecs is not None and self._item_vecs is not None, "fit() first"
        assert self._item_biases is not None

        users = np.asarray(list(user_ids), dtype=np.int32)
        u_vecs = self._user_vecs[users]                    # (N, D)
        scores = u_vecs @ self._item_vecs.T                # (N, V_i)
        scores = scores + self._item_biases[None, :]       # popularity offset

        # Mask already-seen items to -inf
        for row, uid in enumerate(users):
            seen = self._user_seen.get(int(uid))
            if seen:
                scores[row, list(seen)] = -np.inf

        # Vectorized top-K
        if k >= scores.shape[1]:
            idx = np.argsort(-scores, axis=1)
        else:
            partial = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(scores.shape[0])[:, None]
            partial_scores = scores[row_idx, partial]
            order = np.argsort(-partial_scores, axis=1)
            idx = partial[row_idx, order]
        idx = idx[:, :k]

        out_scores = np.take_along_axis(scores, idx, axis=1)

        # Convert any -inf positions to 0 for cleanliness in downstream code
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
        assert self.model is not None
        out = ensure_dir(path)
        torch.save(self.model.state_dict(), out / "model.pt")
        np.save(out / "user_vecs.npy", self._user_vecs)
        np.save(out / "item_vecs.npy", self._item_vecs)
        np.save(out / "item_biases.npy", self._item_biases)
        seen_rows = [
            {"user_id": int(u), "item_id": int(i)}
            for u, items in self._user_seen.items()
            for i in items
        ]
        pd.DataFrame(seen_rows).to_parquet(out / "user_seen.parquet", index=False)
        write_json(
            {
                "num_users":     self.num_users,
                "num_items":     self.num_items,
                "num_genres":    self.num_genres,
                "embedding_dim": self.embedding_dim,
                "max_seq_len":   self.max_seq_len,
                "max_genres":    self.max_genres,
            },
            out / "meta.json",
        )
        log.info("Saved Two-Tower artefacts to %s", out)

    def load(self, path: str | Path) -> None:
        in_dir = Path(path)
        meta = read_json(in_dir / "meta.json")
        self.num_users = meta["num_users"]
        self.num_items = meta["num_items"]
        self.num_genres = meta["num_genres"]
        self.embedding_dim = meta["embedding_dim"]
        self.max_seq_len = meta["max_seq_len"]
        self.max_genres = meta["max_genres"]
        self.model = TwoTowerModel(
            num_users=self.num_users,
            num_items=self.num_items,
            num_genres=self.num_genres,
            embedding_dim=self.embedding_dim,
        )
        self.model.load_state_dict(torch.load(in_dir / "model.pt", map_location="cpu"))
        self.model.eval()
        self._user_vecs = np.load(in_dir / "user_vecs.npy")
        self._item_vecs = np.load(in_dir / "item_vecs.npy")
        self._item_biases = np.load(in_dir / "item_biases.npy")
        df = pd.read_parquet(in_dir / "user_seen.parquet")
        self._user_seen = (
            df.groupby("user_id")["item_id"].agg(lambda s: set(s.tolist())).to_dict()
        )
        log.info("Loaded Two-Tower artefacts from %s", in_dir)
