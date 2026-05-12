"""DeepFM — factorisation-machine + deep tower for CTR prediction (pre-ranking).

Reference
---------
Guo, Tang, Ye, Li, He. *DeepFM: A Factorization-Machine based Neural Network
for CTR Prediction.* IJCAI 2017.

Architecture
------------
For each row (user, item) we look up an embedding for every sparse field
(``user_id``, ``item_id``, gender, age_bucket, occupation, year_bucket,
``popularity_bucket``, and a mean-pooled genre embedding). All embeddings
share the same dimension ``d``.

* **FM component**:
    1st-order weight per field (a bias)::

        y_fm_1 = b + Σ_i w_i

    2nd-order cross — efficient O(K·d) closed form::

        y_fm_2 = 0.5 * Σ_d ( (Σ_i e_i^d)^2 - Σ_i (e_i^d)^2 )

* **Deep component**: concat all embeddings, feed into an MLP::

        y_deep = MLP(concat(e_1, …, e_K))

* **Output**::

        logit = y_fm_1 + y_fm_2 + y_deep
        p     = sigmoid(logit)

Implementation is **pure PyTorch** so we own the training loop, can ablate
FM-only or Deep-only by toggling ``cfg.rank.model.use_fm/use_deep``, and
don't ship a heavy ``deepctr-torch`` dependency.
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
from neorec.ranking.features import RankingFeaturizer

log = logging.getLogger(__name__)


# ===========================================================================
# Model
# ===========================================================================
class DeepFM(nn.Module):
    """The factorisation-machine + deep tower."""

    def __init__(
        self,
        cardinalities: dict[str, int],
        num_genres: int,
        embedding_dim: int = 16,
        dnn_hidden: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        use_fm: bool = True,
        use_deep: bool = True,
    ) -> None:
        super().__init__()
        if not (use_fm or use_deep):
            raise ValueError("At least one of use_fm / use_deep must be True.")
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.cardinalities = dict(cardinalities)
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        self.sparse_fields = list(cardinalities.keys())

        # Embedding tables for sparse fields + genres.
        self.emb_2nd = nn.ModuleDict({
            name: nn.Embedding(card, embedding_dim)
            for name, card in cardinalities.items()
        })
        self.emb_2nd_genre = nn.Embedding(num_genres, embedding_dim, padding_idx=0)

        # 1st-order weights (each field gets a single scalar lookup).
        self.emb_1st = nn.ModuleDict({
            name: nn.Embedding(card, 1)
            for name, card in cardinalities.items()
        })
        self.emb_1st_genre = nn.Embedding(num_genres, 1, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(1))

        # Deep tower.
        if use_deep:
            n_fields = len(self.sparse_fields) + 1  # +1 for genre mean
            layers: list[nn.Module] = []
            in_dim = n_fields * embedding_dim
            for h in dnn_hidden:
                layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.dnn = nn.Sequential(*layers)
        else:
            self.dnn = None

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.emb_2nd.values():
            nn.init.normal_(emb.weight, std=1e-2)
        nn.init.normal_(self.emb_2nd_genre.weight, std=1e-2)
        for emb in self.emb_1st.values():
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.emb_1st_genre.weight)
        if self.dnn is not None:
            for m in self.dnn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Gather 2nd-order embeddings — (B, K, d) tensor.
        emb_list = [self.emb_2nd[name](batch[name]) for name in self.sparse_fields]
        # Genres: (B, max_genres) → embed → masked mean → (B, d).
        g_emb = self.emb_2nd_genre(batch["genres"])            # (B, G, d)
        g_mask = batch["genres_mask"].unsqueeze(-1)             # (B, G, 1)
        g_pooled = (g_emb * g_mask).sum(dim=1) / g_mask.sum(dim=1).clamp(min=1.0)
        emb_list.append(g_pooled)
        emb_2nd = torch.stack(emb_list, dim=1)                  # (B, K+1, d)

        logit = self.bias.expand(emb_2nd.size(0))

        # FM 1st-order: sum of scalar look-ups per field.
        if self.use_fm:
            order1 = torch.zeros(emb_2nd.size(0), device=emb_2nd.device)
            for name in self.sparse_fields:
                order1 = order1 + self.emb_1st[name](batch[name]).squeeze(-1)
            g1 = self.emb_1st_genre(batch["genres"]).squeeze(-1)  # (B, G)
            g1 = (g1 * batch["genres_mask"]).sum(dim=1)
            order1 = order1 + g1
            # FM 2nd-order: (Σ e)^2 - Σ e^2  per latent dim, summed.
            summed = emb_2nd.sum(dim=1)                  # (B, d)
            summed_sq = summed * summed                  # (B, d)
            sq_summed = (emb_2nd * emb_2nd).sum(dim=1)   # (B, d)
            order2 = 0.5 * (summed_sq - sq_summed).sum(dim=1)
            logit = logit + order1 + order2

        if self.use_deep and self.dnn is not None:
            flat = emb_2nd.flatten(1)
            deep_out = self.dnn(flat).squeeze(-1)
            logit = logit + deep_out

        return logit  # raw logit; sigmoid in loss / scoring


# ===========================================================================
# Ranker wrapper
# ===========================================================================
class DeepFMRanker(BaseRanker):
    name = "deepfm"
    stage = "pre_rank"

    def __init__(self, cfg, featurizer: RankingFeaturizer) -> None:
        super().__init__(cfg, featurizer)
        self.device = torch.device(cfg.rank.model.get("device", "cpu"))
        cards = featurizer.cardinalities()
        self.model = DeepFM(
            cardinalities=cards,
            num_genres=featurizer.schema.num_genres,
            embedding_dim=int(cfg.rank.model.embedding_dim),
            dnn_hidden=tuple(cfg.rank.model.dnn_hidden),
            dropout=float(cfg.rank.model.dnn_dropout),
            use_fm=bool(cfg.rank.model.use_fm),
            use_deep=bool(cfg.rank.model.get("use_deep", True)),
        ).to(self.device)

    # ------------------------------------------------------------------
    # Batch → torch tensors on device
    # ------------------------------------------------------------------
    def _batch_to_tensors(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> dict[str, torch.Tensor]:
        b = self.featurizer.featurize(user_ids, item_ids, include_history=False)
        tensors: dict[str, torch.Tensor] = {}
        for col in self.model.sparse_fields:
            tensors[col] = torch.from_numpy(b.sparse[col]).long().to(self.device)
        tensors["genres"] = torch.from_numpy(b.genres).long().to(self.device)
        tensors["genres_mask"] = torch.from_numpy(b.genres_mask).float().to(self.device)
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

        # Pre-allocate (small) tensors for train pairs.
        users_tr = train_pairs["user_id"].to_numpy(np.int64)
        items_tr = train_pairs["item_id"].to_numpy(np.int64)
        labels_tr = train_pairs["label"].to_numpy(np.float32)
        idx_ds = TensorDataset(torch.arange(len(users_tr)))
        loader = DataLoader(idx_ds, batch_size=bs, shuffle=True, num_workers=0)

        users_v = valid_pairs["user_id"].to_numpy(np.int64)
        items_v = valid_pairs["item_id"].to_numpy(np.int64)
        labels_v = valid_pairs["label"].to_numpy(np.float32)

        best_val = float("inf")
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
                optimizer.step()
                losses.append(loss.item())

            # Valid pass.
            self.model.eval()
            with torch.no_grad():
                v_losses: list[float] = []
                v_iter = range(0, len(users_v), bs)
                for i in v_iter:
                    j = min(i + bs, len(users_v))
                    tensors = self._batch_to_tensors(users_v[i:j], items_v[i:j])
                    y = torch.from_numpy(labels_v[i:j]).to(self.device)
                    logit = self.model(tensors)
                    v_losses.append(F.binary_cross_entropy_with_logits(logit, y).item())
            v_loss = float(np.mean(v_losses))
            log.info("ep %d/%d  tr_loss=%.4f  va_loss=%.4f", ep + 1, epochs,
                     float(np.mean(losses)), v_loss)

            if v_loss + 1e-5 < best_val:
                best_val = v_loss
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    log.info("Early stopping at epoch %d (patience=%d).", ep + 1, patience)
                    break

        return {"final_valid_bce": float(best_val)}

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "deepfm.pt")
        (path / "meta.json").write_text(json.dumps({
            "name": self.name,
            "cardinalities": self.model.cardinalities,
            "num_genres": self.model.num_genres,
            "embedding_dim": self.model.embedding_dim,
        }, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.model.load_state_dict(torch.load(path / "deepfm.pt", map_location=self.device))
