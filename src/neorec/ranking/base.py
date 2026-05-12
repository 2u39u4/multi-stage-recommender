"""Abstract base class for ranking models (pre-rank + fine-rank)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from neorec.ranking.features import RankingFeaturizer


@dataclass
class RankResult:
    """Per-user re-scored and reordered items."""

    user_ids: np.ndarray
    item_ids: np.ndarray      # (N, K) reordered
    scores: np.ndarray        # (N, K) CTR probabilities


class BaseRanker(ABC):
    """Common interface for LR / GBDT / DeepFM / DIN.

    Lifecycle
    ---------
    ``__init__(cfg, featurizer)``       — store config & shared feature pipeline.
    ``fit(train_pairs, valid_pairs)``   — train, return dict of valid metrics.
    ``score(user_ids, item_ids)``       — vectorised per-pair CTR prob in [0, 1].
    ``predict(user_ids, candidate_items, k)``
                                        — default impl: score all candidates
                                          per user, sort, return top-K.
    ``save(path)`` / ``load(path)``     — disk round-trip for artefacts.

    Subclasses **must** override ``fit``, ``score``, ``save``, ``load``.
    """

    name: str = "base"
    stage: str = "rank"        # {pre_rank, fine_rank}
    needs_history: bool = False

    def __init__(self, cfg: DictConfig, featurizer: RankingFeaturizer) -> None:
        self.cfg = cfg
        self.featurizer = featurizer

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        train_pairs: pd.DataFrame,
        valid_pairs: pd.DataFrame,
    ) -> dict[str, float]: ...

    @abstractmethod
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Return predicted CTR probability in ``[0, 1]`` for each pair."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> None: ...

    # ------------------------------------------------------------------
    # Generic top-K shortlist (vectorised candidate scoring per user)
    # ------------------------------------------------------------------
    def predict(
        self,
        user_ids: Sequence[int],
        candidate_items: Sequence[Sequence[int]],
        k: int | None = None,
    ) -> RankResult:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        # Flatten (user, candidate) pairs for one vectorised score call.
        flat_users: list[int] = []
        flat_items: list[int] = []
        offsets: list[int] = [0]
        for u, cands in zip(user_ids, candidate_items, strict=True):
            cands_arr = np.asarray(cands, dtype=np.int64)
            flat_users.extend([int(u)] * len(cands_arr))
            flat_items.extend(cands_arr.tolist())
            offsets.append(offsets[-1] + len(cands_arr))
        flat_scores = self.score(
            np.asarray(flat_users, dtype=np.int64),
            np.asarray(flat_items, dtype=np.int64),
        )

        # Re-group by user, sort desc, truncate to k.
        max_len = max(o2 - o1 for o1, o2 in zip(offsets[:-1], offsets[1:]))
        k_eff = k if k is not None else max_len
        out_items = np.zeros((len(user_ids), k_eff), dtype=np.int64)
        out_scores = np.zeros((len(user_ids), k_eff), dtype=np.float32)
        for row, (o1, o2) in enumerate(zip(offsets[:-1], offsets[1:])):
            cands = np.asarray(candidate_items[row], dtype=np.int64)
            scores = flat_scores[o1:o2]
            order = np.argsort(-scores)[:k_eff]
            picked = cands[order]
            out_items[row, : len(picked)] = picked
            out_scores[row, : len(picked)] = scores[order]
        return RankResult(user_ids=user_ids, item_ids=out_items, scores=out_scores)
