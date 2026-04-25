"""Online inference pipeline: recall → pre-rank → fine-rank → rerank.

Usage (from api.py)::

    pipeline = OnlinePipeline.from_config(cfg)
    response = pipeline.recommend(user_id=123, k=10, diversity=0.3)

The pipeline exposes per-stage latency so the API response can advertise
``latency_ms.total`` and its breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecommendationItem:
    item_id: int
    score: float
    channel: str
    explain: str | None = None


@dataclass
class RecommendationResponse:
    user_id: int
    items: list[RecommendationItem]
    latency_ms: dict[str, float] = field(default_factory=dict)


class OnlinePipeline:
    """Thread-safe online inference orchestrator."""

    def __init__(
        self,
        cfg: Any,
        *,
        recallers: list[Any] | None = None,
        pre_ranker: Any | None = None,
        fine_ranker: Any | None = None,
        feature_cache: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.recallers = recallers or []
        self.pre_ranker = pre_ranker
        self.fine_ranker = fine_ranker
        self.feature_cache = feature_cache

    @classmethod
    def from_config(cls, cfg: Any) -> "OnlinePipeline":
        """Hydrate all components from a Hydra config."""
        raise NotImplementedError  # TODO(W5 Day 29)

    def recommend(
        self, user_id: int, k: int = 10, diversity: float = 0.5
    ) -> RecommendationResponse:
        """Run the full funnel for one user and return top-K with latency breakdown."""
        raise NotImplementedError  # TODO(W5 Day 29-30)
