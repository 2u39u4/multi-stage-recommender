"""Logistic Regression baseline — the cheapest ranking model in the comparison.

Trains a single linear layer on one-hot-encoded sparse features and a mean-pooled
multi-hot genre vector. Serves as a sanity floor: anything that doesn't beat LR
isn't worth shipping.

We use the **hashing trick** for the highest-cardinality fields (user_id,
item_id) so the model stays small (≤ 16 K weights) and trains in seconds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

from neorec.ranking.base import BaseRanker
from neorec.ranking.features import RankingFeaturizer

log = logging.getLogger(__name__)


class LRRanker(BaseRanker):
    name = "lr"
    stage = "baseline"

    def __init__(self, cfg, featurizer: RankingFeaturizer) -> None:
        super().__init__(cfg, featurizer)
        self.model: LogisticRegression | None = None
        self.hash_dim_user = int(cfg.rank.model.get("hash_dim_user", 4096))
        self.hash_dim_item = int(cfg.rank.model.get("hash_dim_item", 2048))

    # ------------------------------------------------------------------
    # Feature → sparse matrix
    # ------------------------------------------------------------------
    def _make_X(self, user_ids: np.ndarray, item_ids: np.ndarray) -> csr_matrix:
        batch = self.featurizer.featurize(user_ids, item_ids, include_history=False)
        s = self.featurizer.schema
        n = len(user_ids)

        # Hashing trick for very high-cardinality user/item ids.
        u_hash = (user_ids % self.hash_dim_user).astype(np.int64)
        i_hash = (item_ids % self.hash_dim_item).astype(np.int64)

        rows = np.arange(n, dtype=np.int64)
        blocks = []
        # user hash
        blocks.append(csr_matrix(
            (np.ones(n, dtype=np.float32), (rows, u_hash)),
            shape=(n, self.hash_dim_user),
        ))
        # item hash
        blocks.append(csr_matrix(
            (np.ones(n, dtype=np.float32), (rows, i_hash)),
            shape=(n, self.hash_dim_item),
        ))
        # gender / age / occupation / year / pop bucket → small one-hot
        for col, card in [
            ("gender",            s.num_genders),
            ("age_bucket",        s.num_age_buckets),
            ("occupation",        s.num_occupations),
            ("year_bucket",       s.num_year_buckets),
            ("popularity_bucket", s.num_popularity_buckets),
        ]:
            v = batch.sparse[col]
            blocks.append(csr_matrix(
                (np.ones(n, dtype=np.float32), (rows, v)),
                shape=(n, card),
            ))
        # genres multi-hot
        genre_mat = np.zeros((n, s.num_genres), dtype=np.float32)
        for col in range(s.max_genres):
            valid = batch.genres_mask[:, col] > 0
            genre_mat[np.where(valid)[0], batch.genres[valid, col]] = 1.0
        blocks.append(csr_matrix(genre_mat))

        # Recall-layer scores (Scheme A) — appended as dense numeric columns.
        # The :class:`RecallFeatureStore` already z-scored each channel using
        # global statistics, so the columns live on a comparable scale to the
        # one-hot indicators and we can pass them straight through.
        if batch.recall_scores is not None:
            recall_block = np.nan_to_num(
                batch.recall_scores, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            blocks.append(csr_matrix(recall_block))

        return hstack(blocks, format="csr")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def fit(
        self,
        train_pairs: pd.DataFrame,
        valid_pairs: pd.DataFrame,
    ) -> dict[str, float]:
        Xtr = self._make_X(
            train_pairs["user_id"].to_numpy(np.int64),
            train_pairs["item_id"].to_numpy(np.int64),
        )
        ytr = train_pairs["label"].to_numpy(np.int8)
        log.info("LR fitting on %d rows, feature dim=%d", Xtr.shape[0], Xtr.shape[1])

        self.model = LogisticRegression(
            C=float(self.cfg.rank.model.get("C", 1.0)),
            solver=str(self.cfg.rank.model.get("solver", "liblinear")),
            max_iter=int(self.cfg.rank.train.get("max_iter", 200)),
        )
        self.model.fit(Xtr, ytr)
        train_auc = float(self.model.score(Xtr, ytr))
        return {"train_accuracy": train_auc}

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LR model not fitted.")
        X = self._make_X(user_ids, item_ids)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self.model, path / "lr.joblib")
        (path / "meta.json").write_text(json.dumps({
            "name": self.name,
            "hash_dim_user": self.hash_dim_user,
            "hash_dim_item": self.hash_dim_item,
        }, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        import joblib
        self.model = joblib.load(path / "lr.joblib")
        meta = json.loads((path / "meta.json").read_text())
        self.hash_dim_user = int(meta["hash_dim_user"])
        self.hash_dim_item = int(meta["hash_dim_item"])
