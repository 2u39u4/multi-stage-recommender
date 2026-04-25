"""Maximal Marginal Relevance (MMR) re-ranking for diversity.

    MMR(d) = λ · relevance(d) - (1 - λ) · max_{d' ∈ S} sim(d, d')

Parameters are surfaced via ``configs/rerank/mmr.yaml``. The ablation sweep
over ``λ ∈ {0, 0.3, 0.5, 0.7, 1.0}`` produces the accuracy-diversity Pareto
curve plotted in the README.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mmr_rerank(
    candidate_ids: Sequence[int],
    candidate_scores: Sequence[float],
    item_embeddings: np.ndarray,
    k: int,
    lam: float = 0.5,
) -> list[int]:
    """Return ``k`` items selected greedily under the MMR objective.

    ``item_embeddings`` must be indexable by item id and L2-normalized
    (cosine similarity == dot product).
    """
    raise NotImplementedError  # TODO(W4 Day 22-23)
