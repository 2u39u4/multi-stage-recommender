"""ALS (iALS) collaborative filtering — classical CF baseline.

Reference: Hu, Koren, Volinsky. Collaborative Filtering for Implicit Feedback
Datasets. ICDM 2008.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neorec.recall.base import BaseRecaller, RecallResult

log = logging.getLogger(__name__)


class ALSRecaller(BaseRecaller):
    """Wraps ``implicit.als.AlternatingLeastSquares``."""

    name = "als"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.model = None
        self.user_item_matrix = None

    def fit(self, interactions_path: str | Path) -> None:
        """Build a sparse (user × item) CSR, then fit iALS.

        TODO(W1 Day 4):
            * load interactions.parquet
            * build scipy.sparse CSR with confidence = 1 + alpha * count
            * self.model = implicit.als.AlternatingLeastSquares(
                factors=cfg.model.factors, ...)
            * self.model.fit(user_item_csr)
        """
        raise NotImplementedError  # TODO(W1)

    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        """Score all items for each user via factor product; return top-K."""
        raise NotImplementedError  # TODO(W1)

    def save(self, path: str | Path) -> None:
        """Save user/item factors + model hyper-params."""
        raise NotImplementedError  # TODO(W1)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W1)
