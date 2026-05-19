"""Cold-start recall — content-based fallback for users with little/no history.

Why a dedicated channel
-----------------------
Collaborative-filtering channels (iALS, Two-Tower, SASRec) all degenerate on
cold users because every user-side signal they consume — the user id
embedding, the history sequence — is either uninitialised or empty.  A
content-based channel sidesteps that by representing each item with its
*intrinsic* features (here: genres + year bucket), so the recall quality for
cold users is bounded by *how good your item features are*, not *how many
interactions you've collected*.

Pipeline
--------
1. **Build item TF-IDF.**
   Each item is a "document" whose terms are its genres (token = ``"g{id}"``)
   and year bucket (token = ``"yb{id}"``).  TF is 0/1 (we just check presence,
   no repetition), IDF is the usual ``log(N / df)``.  After L2 normalization
   the dot product between two item vectors is cosine similarity.

2. **User profile = mean of liked-item vectors.**
   For each training user, take the mean (= centroid) of their item TF-IDF
   vectors and re-normalize.  This is the simplest plausible user encoder.

3. **Recall = item centroids × all-items matrix.**
   One dense ``(U_eval, T) @ (T, V)`` matmul.  Top-K after filtering items the
   user has already seen.

4. **Cold fallback.**
   If a user has *no* training history at all (``profile = 0``), we fall
   back to the global popularity top-K — the same logic as
   ``PopularityRecaller``.

Why TF-IDF and not "just average a genre embedding"
---------------------------------------------------
TF-IDF gives a closed-form, training-free, deterministic baseline.  Whatever
you build later (learned content embeddings, two-tower with cold-friendly
features) should clearly outperform this — and if it doesn't, the new model
isn't actually using the content signal.

Cold-evaluation
---------------
Definition of "cold user" is configurable via
``cfg.recall.eval.cold_user_max_interactions``.  When evaluating, ``train.py``
just calls ``recall()`` on the same global test users; the channel's
relative strength on cold users is read out by the analysis notebook.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix

from neorec.recall.base import BaseRecaller, RecallResult
from neorec.utils.io import ensure_dir, read_json, write_json

log = logging.getLogger(__name__)


class ColdStartRecaller(BaseRecaller):
    name = "cold_start"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._item_vecs: np.ndarray | None = None    # (V, T) L2-normalized TF-IDF
        self._user_vecs: np.ndarray | None = None    # (U, T) user centroids
        self._popular_items: np.ndarray | None = None  # (V,) item ids sorted desc
        self._popular_scores: np.ndarray | None = None
        self._user_seen: dict[int, set[int]] = {}
        self._token_map: dict[str, int] = {}         # token -> column index
        self.num_users = 0
        self.num_items = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, interactions_path: str | Path) -> None:
        interactions_path = Path(interactions_path)
        processed_dir = interactions_path.parent

        train_df = pd.read_parquet(interactions_path)
        item_df = pd.read_parquet(processed_dir / "item_features.parquet")
        id_maps = read_json(processed_dir / "id_maps.json")
        self.num_users = int(id_maps["user_id_map_size"])
        self.num_items = int(id_maps["item_id_map_size"])
        log.info("Cold-start setup: users=%d items=%d", self.num_users, self.num_items)

        features: list[str] = list(self.cfg.recall.model.features)
        log.info("Content features: %s", features)

        # 1) Build item × token TF-IDF matrix
        item_vecs = self._build_item_tfidf(item_df, features)
        self._item_vecs = item_vecs

        # 2) User centroids = mean of L2-normalized item vectors
        self._user_seen = (
            train_df.groupby("user_id")["item_id"].agg(lambda s: set(s.tolist())).to_dict()
        )
        self._user_vecs = self._build_user_centroids(train_df)

        # 3) Popularity fallback (for users with zero training history)
        item_counts = train_df["item_id"].value_counts()
        order = item_counts.sort_values(ascending=False)
        self._popular_items = order.index.to_numpy(dtype=np.int32)
        self._popular_scores = order.to_numpy(dtype=np.float32)
        log.info(
            "Top-5 popular fallback items: %s",
            list(zip(self._popular_items[:5], self._popular_scores[:5].astype(int))),
        )

    # ------------------------------------------------------------------
    def _build_item_tfidf(
        self, item_df: pd.DataFrame, features: list[str]
    ) -> np.ndarray:
        """Build the (V, T) TF-IDF matrix. ``T`` = total unique tokens across features."""
        # Collect tokens per item (token strings keep features namespaced)
        item_tokens: dict[int, list[str]] = {}
        for _, row in item_df.iterrows():
            iid = int(row["item_id"])
            toks: list[str] = []
            if "genres" in features:
                for g in row["genres"]:
                    if int(g) != 0:
                        toks.append(f"g{int(g)}")
            if "year_bucket" in features and "year_bucket" in row:
                toks.append(f"yb{int(row['year_bucket'])}")
            item_tokens[iid] = toks or ["unk"]  # avoid empty rows

        # Build vocabulary
        vocab = sorted({t for toks in item_tokens.values() for t in toks})
        self._token_map = {t: i for i, t in enumerate(vocab)}
        T = len(vocab)
        V = self.num_items
        log.info("Vocab size: %d (items=%d)", T, V)

        # Document frequency
        df_count = np.zeros(T, dtype=np.float32)
        for toks in item_tokens.values():
            seen_tokens = set(toks)
            for t in seen_tokens:
                df_count[self._token_map[t]] += 1.0
        idf = np.log((V + 1) / (df_count + 1)) + 1.0  # smoothed IDF

        # Build sparse item-token then materialize dense
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        if not bool(self.cfg.recall.model.tfidf_on_genres):
            # binary indicator only — for ablation
            tf_value = 1.0
            for iid, toks in item_tokens.items():
                for t in set(toks):
                    rows.append(iid)
                    cols.append(self._token_map[t])
                    vals.append(tf_value)
        else:
            for iid, toks in item_tokens.items():
                for t in set(toks):
                    rows.append(iid)
                    cols.append(self._token_map[t])
                    vals.append(idf[self._token_map[t]])

        sparse = csr_matrix(
            (vals, (rows, cols)), shape=(V, T), dtype=np.float32
        )
        dense = sparse.toarray()
        # L2 normalize rows
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        np.divide(dense, norms, out=dense, where=norms > 0)
        return dense.astype(np.float32)

    # ------------------------------------------------------------------
    def _build_user_centroids(self, train_df: pd.DataFrame) -> np.ndarray:
        """Average L2-normalized item vectors per user, then re-normalize."""
        assert self._item_vecs is not None
        T = self._item_vecs.shape[1]
        out = np.zeros((self.num_users, T), dtype=np.float32)

        groups = train_df.groupby("user_id")["item_id"].apply(list)
        for u, items in groups.items():
            vecs = self._item_vecs[items]                  # (k, T)
            out[int(u)] = vecs.mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        zero = int((norms.squeeze(-1) == 0).sum())
        log.info("User centroids built (%d users with zero history → popularity fallback)", zero)
        return out

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        assert (
            self._item_vecs is not None and self._user_vecs is not None
            and self._popular_items is not None and self._popular_scores is not None
        ), "fit() first"

        users = np.asarray(list(user_ids), dtype=np.int32)
        out_items = np.full((len(users), k), -1, dtype=np.int32)
        out_scores = np.zeros((len(users), k), dtype=np.float32)

        # Tolerate being driven by a merge cfg without the cold-start
        # output section — default to "yes, fall back to popularity".
        fallback_to_pop = bool(
            OmegaConf.select(
                self.cfg, "recall.output.fallback_to_popularity", default=True
            )
        )

        u_vecs = self._user_vecs[users]                  # (N, T)
        # For users with all-zero profile we'll overwrite the result below.
        scores = u_vecs @ self._item_vecs.T              # (N, V)  cosine similarity

        for row, uid in enumerate(users):
            uvec = u_vecs[row]
            if np.linalg.norm(uvec) == 0:
                if fallback_to_pop:
                    take = min(k, len(self._popular_items))
                    out_items[row, :take] = self._popular_items[:take]
                    out_scores[row, :take] = self._popular_scores[:take]
                continue
            seen = self._user_seen.get(int(uid))
            row_scores = scores[row]
            if seen:
                row_scores = row_scores.copy()
                row_scores[list(seen)] = -np.inf
            if k >= len(row_scores):
                idx = np.argsort(-row_scores)[:k]
            else:
                part = np.argpartition(-row_scores, kth=k - 1)[:k]
                idx = part[np.argsort(-row_scores[part])]
            kept_scores = row_scores[idx]
            # Drop any -inf positions (happens when seen items > catalog - k)
            valid = ~np.isinf(kept_scores)
            n = int(valid.sum())
            out_items[row, :n] = idx[valid]
            out_scores[row, :n] = kept_scores[valid]

        return RecallResult(
            user_ids=users,
            item_ids=out_items,
            scores=out_scores.astype(np.float32),
            channel=self.name,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        assert self._item_vecs is not None and self._user_vecs is not None
        assert self._popular_items is not None and self._popular_scores is not None
        out_dir = ensure_dir(path)
        np.save(out_dir / "item_vecs.npy", self._item_vecs)
        np.save(out_dir / "user_vecs.npy", self._user_vecs)
        np.save(out_dir / "popular_items.npy", self._popular_items)
        np.save(out_dir / "popular_scores.npy", self._popular_scores)
        seen_rows = [
            {"user_id": int(u), "item_id": int(i)}
            for u, items in self._user_seen.items()
            for i in items
        ]
        pd.DataFrame(seen_rows).to_parquet(out_dir / "user_seen.parquet", index=False)
        write_json(
            {
                "num_users":     self.num_users,
                "num_items":     self.num_items,
                "vocab_size":    int(self._item_vecs.shape[1]),
                "features":      list(self.cfg.recall.model.features),
                "token_map":     self._token_map,
            },
            out_dir / "meta.json",
        )
        log.info("Saved cold-start artefacts to %s", out_dir)

    def load(self, path: str | Path) -> None:
        in_dir = Path(path)
        meta = read_json(in_dir / "meta.json")
        self.num_users = meta["num_users"]
        self.num_items = meta["num_items"]
        self._token_map = dict(meta["token_map"])
        self._item_vecs = np.load(in_dir / "item_vecs.npy")
        self._user_vecs = np.load(in_dir / "user_vecs.npy")
        self._popular_items = np.load(in_dir / "popular_items.npy")
        self._popular_scores = np.load(in_dir / "popular_scores.npy")
        df = pd.read_parquet(in_dir / "user_seen.parquet")
        self._user_seen = (
            df.groupby("user_id")["item_id"].agg(lambda s: set(s.tolist())).to_dict()
        )
        log.info("Loaded cold-start artefacts from %s", in_dir)
