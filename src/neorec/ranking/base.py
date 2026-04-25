"""Abstract base class for ranking models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig


@dataclass
class RankResult:
    """Per-user re-scored and reordered items."""

    user_ids: np.ndarray
    item_ids: np.ndarray      # (N, K) reordered
    scores: np.ndarray        # (N, K) CTR probabilities


class BaseRanker(ABC):
    """Interface for both pre-rankers and fine-rankers."""

    name: str = "base"
    stage: str = "rank"        # {pre_rank, fine_rank}

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def fit(self, train_path: str | Path, valid_path: str | Path) -> None: ...

    @abstractmethod
    def predict(
        self,
        user_ids: Sequence[int],
        candidate_items: Sequence[Sequence[int]],
    ) -> RankResult: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> None: ...
