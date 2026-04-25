"""Abstract base class for all recall channels.

Every channel (ALS, Two-Tower, SASRec, Popularity, Cold-start) must implement:
    * fit(train_interactions, cfg) -> None
    * recall(user_ids, k) -> RecallResult
    * save(path) / load(path)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig


@dataclass
class RecallResult:
    """Container for one channel's output, one row per user.

    Attributes
    ----------
    user_ids:   (N,)     target users
    item_ids:   (N, K)   recalled item ids (-1 for padding if fewer than K)
    scores:     (N, K)   raw channel scores (arbitrary scale)
    channel:    name of the producing channel
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    scores: np.ndarray
    channel: str

    def __len__(self) -> int:
        return len(self.user_ids)


class BaseRecaller(ABC):
    """Common interface."""

    name: str = "base"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def fit(self, interactions_path: str | Path) -> None: ...

    @abstractmethod
    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> None: ...
