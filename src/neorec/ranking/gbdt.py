"""Gradient-boosted trees baseline.

We use scikit-learn's :class:`HistGradientBoostingClassifier` (histogram-based
gradient boosting, the same algorithmic family as LightGBM / XGBoost).
It ships with sklearn and avoids the ``libomp`` system-library dependency
that LightGBM's macOS wheel needs.

GBDT is the *de facto* CTR baseline at most large-scale recommenders
(Meta, Pinterest, Yahoo had GBDT in production for years). Anything we build
that doesn't beat GBDT on AUC isn't worth the complexity.

Features are kept as **dense integer-coded categoricals** so the histogram
splitter does its own efficient binning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from neorec.ranking.base import BaseRanker
from neorec.ranking.features import RankingFeaturizer

log = logging.getLogger(__name__)


class GBDTRanker(BaseRanker):
    name = "gbdt"
    stage = "baseline"

    def __init__(self, cfg, featurizer: RankingFeaturizer) -> None:
        super().__init__(cfg, featurizer)
        self.model: HistGradientBoostingClassifier | None = None
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Feature → dense pandas frame
    # ------------------------------------------------------------------
    def _make_X(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        # Histogram-GBDT caps categorical cardinality at 255, so we drop the
        # raw user_id / item_id and rely on side features + per-user/per-item
        # popularity statistics. This makes GBDT a "side-feature only" baseline
        # — exactly the kind of model that motivates deep models with embeddings.
        batch = self.featurizer.featurize(user_ids, item_ids, include_history=False)
        df = pd.DataFrame({
            "gender":            batch.sparse["gender"].astype(np.int32),
            "age_bucket":        batch.sparse["age_bucket"].astype(np.int32),
            "occupation":        batch.sparse["occupation"].astype(np.int32),
            "year_bucket":       batch.sparse["year_bucket"].astype(np.int32),
            "popularity_bucket": batch.sparse["popularity_bucket"].astype(np.int32),
        })
        for k in range(min(3, self.featurizer.schema.max_genres)):
            df[f"genre_{k}"] = batch.genres[:, k].astype(np.int32)
        # Lightweight user statistics from the pre-built history table.
        u_hist = self.featurizer._user_history          # (num_users, max_seq_len)
        u_mask = self.featurizer._user_history_mask
        if u_hist is not None and u_mask is not None:
            user_hist_len = u_mask.sum(axis=1).astype(np.int32)
            df["user_hist_len"] = user_hist_len[user_ids]
        else:
            df["user_hist_len"] = np.zeros(len(user_ids), dtype=np.int32)

        # Recall-layer scores (Scheme A): each channel's score + "found" mask.
        # GBDT can natively split on continuous features, so we just stick them
        # in as ordinary numeric columns — no normalisation needed.
        if batch.recall_scores is not None:
            for c, name in enumerate(batch.recall_score_cols):
                df[f"recall__{name}"] = batch.recall_scores[:, c].astype(np.float32)
        return df

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
        self.feature_cols = list(Xtr.columns)

        # All side features are bounded: gender(2), age(6), occupation(21),
        # year(5), pop_bucket(10), genres(0..18), user_hist_len → ordinal.
        # Only the categorical ones get the categorical_features flag.
        cat_cols = ["gender", "age_bucket", "occupation", "year_bucket"]
        cat_cols += [c for c in self.feature_cols if c.startswith("genre_")]
        categorical_features = [self.feature_cols.index(c) for c in cat_cols]

        log.info("HistGBDT fitting on %d rows, %d features (%d categorical)",
                 Xtr.shape[0], Xtr.shape[1], len(categorical_features))

        self.model = HistGradientBoostingClassifier(
            max_iter=int(self.cfg.rank.train.get("num_boost_round", 300)),
            learning_rate=float(self.cfg.rank.train.get("lr", 0.05)),
            max_leaf_nodes=int(self.cfg.rank.model.get("num_leaves", 63)),
            min_samples_leaf=int(self.cfg.rank.model.get("min_data_in_leaf", 100)),
            l2_regularization=float(self.cfg.rank.model.get("l2_reg", 0.0)),
            categorical_features=categorical_features,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=int(self.cfg.rank.train.get("early_stopping_rounds", 30)),
            random_state=int(self.cfg.get("seed", 42)),
            verbose=0,
        )
        self.model.fit(Xtr, ytr)
        n_iter = int(self.model.n_iter_)
        log.info("Best iteration: %d", n_iter)
        return {"n_iter": float(n_iter)}

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GBDT model not fitted.")
        X = self._make_X(user_ids, item_ids)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "gbdt.joblib")
        (path / "meta.json").write_text(json.dumps({
            "name": self.name,
            "feature_cols": self.feature_cols,
        }, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.model = joblib.load(path / "gbdt.joblib")
        self.feature_cols = list(json.loads((path / "meta.json").read_text())["feature_cols"])
