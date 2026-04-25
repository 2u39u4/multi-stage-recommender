"""ALS (iALS) collaborative filtering — classical CF baseline.

Reference: Hu, Koren, Volinsky. *Collaborative Filtering for Implicit Feedback
Datasets.* ICDM 2008.

Wraps ``implicit.als.AlternatingLeastSquares``.  Confidence weight follows the
original paper: ``c_ui = 1 + alpha * r_ui`` where ``r_ui`` is the rating
or interaction count.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from neorec.recall.base import BaseRecaller, RecallResult
from neorec.utils.io import ensure_dir, read_json, write_json

log = logging.getLogger(__name__)


class ALSRecaller(BaseRecaller):
    name = "als"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.model: object | None = None
        self.user_item_csr: sparse.csr_matrix | None = None
        self.num_users: int = 0
        self.num_items: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, interactions_path: str | Path) -> None:
        from implicit.als import AlternatingLeastSquares

        interactions_path = Path(interactions_path)
        df = pd.read_parquet(interactions_path)
        log.info("Loaded interactions: %s rows", f"{len(df):,}")

        if "split" in df.columns:
            df = df[df["split"] == "train"]
            log.info("Filtered to split=train: %s rows", f"{len(df):,}")

        users = df["user_id"].to_numpy(dtype=np.int32)
        items = df["item_id"].to_numpy(dtype=np.int32)
        # If no explicit "rating" column (e.g. split.parquet only has ids), use 1
        if "rating" in df.columns:
            ratings = df["rating"].to_numpy(dtype=np.float32)
        else:
            ratings = np.ones(len(df), dtype=np.float32)

        alpha = float(self.cfg.recall.model.alpha)
        confidence = (1.0 + alpha * ratings).astype(np.float32)

        self.num_users = int(users.max() + 1)
        self.num_items = int(items.max() + 1)

        self.user_item_csr = sparse.csr_matrix(
            (confidence, (users, items)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32,
        )

        log.info(
            "Sparse matrix: shape=%s, nnz=%d, density=%.4f%%",
            self.user_item_csr.shape,
            self.user_item_csr.nnz,
            100 * self.user_item_csr.nnz / (self.num_users * self.num_items),
        )

        self.model = AlternatingLeastSquares(
            factors=int(self.cfg.recall.model.factors),
            regularization=float(self.cfg.recall.model.regularization),
            iterations=int(self.cfg.recall.model.iterations),
            use_gpu=bool(self.cfg.recall.model.get("use_gpu", False)),
            random_state=int(self.cfg.recall.model.get("random_state", 42)),
        )
        log.info("Fitting iALS (factors=%d, reg=%.4f, iters=%d, alpha=%.1f) …",
                 self.cfg.recall.model.factors,
                 self.cfg.recall.model.regularization,
                 self.cfg.recall.model.iterations,
                 alpha)
        self.model.fit(self.user_item_csr, show_progress=True)
        log.info("ALS training done. user_factors=%s item_factors=%s",
                 tuple(self.model.user_factors.shape),  # type: ignore[attr-defined]
                 tuple(self.model.item_factors.shape))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        assert self.model is not None and self.user_item_csr is not None, "fit() first"

        user_ids_arr = np.asarray(list(user_ids), dtype=np.int32)
        item_ids_out = np.full((len(user_ids_arr), k), -1, dtype=np.int32)
        scores_out = np.zeros((len(user_ids_arr), k), dtype=np.float32)

        # implicit.recommend supports batch mode in modern versions
        user_items_subset = self.user_item_csr[user_ids_arr]
        ids, scores = self.model.recommend(  # type: ignore[attr-defined]
            user_ids_arr,
            user_items_subset,
            N=k,
            filter_already_liked_items=True,
        )

        item_ids_out[:, : ids.shape[1]] = ids
        scores_out[:, : scores.shape[1]] = scores

        return RecallResult(
            user_ids=user_ids_arr,
            item_ids=item_ids_out,
            scores=scores_out,
            channel=self.name,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        assert self.model is not None
        out_dir = ensure_dir(path)
        np.save(out_dir / "user_factors.npy", self.model.user_factors)  # type: ignore[attr-defined]
        np.save(out_dir / "item_factors.npy", self.model.item_factors)  # type: ignore[attr-defined]
        sparse.save_npz(out_dir / "user_item.npz", self.user_item_csr)
        write_json(
            {
                "factors":        int(self.cfg.recall.model.factors),
                "regularization": float(self.cfg.recall.model.regularization),
                "alpha":          float(self.cfg.recall.model.alpha),
                "iterations":     int(self.cfg.recall.model.iterations),
                "num_users":      self.num_users,
                "num_items":      self.num_items,
            },
            out_dir / "meta.json",
        )
        log.info("Saved ALS artefacts to %s", out_dir)

    def load(self, path: str | Path) -> None:
        from implicit.als import AlternatingLeastSquares

        in_dir = Path(path)
        meta = read_json(in_dir / "meta.json")
        self.num_users = int(meta["num_users"])
        self.num_items = int(meta["num_items"])
        self.user_item_csr = sparse.load_npz(in_dir / "user_item.npz")

        self.model = AlternatingLeastSquares(
            factors=meta["factors"],
            regularization=meta["regularization"],
            iterations=meta["iterations"],
        )
        self.model.user_factors = np.load(in_dir / "user_factors.npy")  # type: ignore[attr-defined]
        self.model.item_factors = np.load(in_dir / "item_factors.npy")  # type: ignore[attr-defined]
        log.info("Loaded ALS artefacts from %s", in_dir)
