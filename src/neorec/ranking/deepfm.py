"""DeepFM — factorization-machine + deep tower for CTR prediction (pre-ranking).

Reference: Guo, Tang, Ye, Li, He. DeepFM: A Factorization-Machine based Neural
Network for CTR Prediction. IJCAI 2017.

Implementation uses ``deepctr-torch`` for speed, but features are
built by this project's feature pipeline so we keep control over the schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from neorec.ranking.base import BaseRanker, RankResult


class DeepFMRanker(BaseRanker):
    name = "deepfm"
    stage = "pre_rank"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.model = None
        self.feature_columns = None

    def fit(self, train_path: str | Path, valid_path: str | Path) -> None:
        """TODO(W3 Day 15-17):
        * build SparseFeat / DenseFeat columns
        * instantiate deepctr_torch.models.DeepFM
        * BCE loss, early stopping on valid AUC
        * log to MLflow
        """
        raise NotImplementedError  # TODO(W3)

    def predict(
        self,
        user_ids: Sequence[int],
        candidate_items: Sequence[Sequence[int]],
    ) -> RankResult:
        """Batch CTR scoring for (user, candidates) pairs, return top-K sorted."""
        raise NotImplementedError  # TODO(W3)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W3)

    def load(self, path: str | Path) -> None:
        raise NotImplementedError  # TODO(W3)
