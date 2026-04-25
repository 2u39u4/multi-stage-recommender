"""Top-K retrieval and ranking metrics.

All functions accept:
    * ``y_true``: list/ndarray of ground-truth item ids per user (variable length)
    * ``y_pred``: list/ndarray of predicted item ids per user, ordered by score desc
    * ``k``:     cutoff

Implementations are kept dependency-light (pure NumPy) and vectorized per user.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Array = Sequence[Sequence[int]] | np.ndarray


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
def recall_at_k(y_true: Array, y_pred: Array, k: int) -> float:
    """Mean Recall@K over users.

    For each user, Recall@K = |{relevant} ∩ top-K| / |{relevant}|.
    """
    _check_same_length(y_true, y_pred)
    scores: list[float] = []
    for gt, pred in zip(y_true, y_pred, strict=True):
        if len(gt) == 0:
            continue
        topk = set(pred[:k])
        scores.append(len(topk & set(gt)) / len(gt))
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(y_true: Array, y_pred: Array, k: int) -> float:
    """Fraction of users for whom at least one relevant item appears in top-K."""
    _check_same_length(y_true, y_pred)
    hits = 0
    total = 0
    for gt, pred in zip(y_true, y_pred, strict=True):
        if len(gt) == 0:
            continue
        total += 1
        if set(pred[:k]) & set(gt):
            hits += 1
    return hits / total if total else 0.0


def ndcg_at_k(y_true: Array, y_pred: Array, k: int) -> float:
    """Mean NDCG@K with binary relevance."""
    _check_same_length(y_true, y_pred)
    scores: list[float] = []
    for gt, pred in zip(y_true, y_pred, strict=True):
        if len(gt) == 0:
            continue
        gt_set = set(gt)
        dcg = 0.0
        for i, item in enumerate(pred[:k]):
            if item in gt_set:
                dcg += 1.0 / np.log2(i + 2)
        ideal_hits = min(len(gt_set), k)
        idcg = float(np.sum(1.0 / np.log2(np.arange(ideal_hits) + 2))) if ideal_hits else 0.0
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def mean_reciprocal_rank(y_true: Array, y_pred: Array, k: int | None = None) -> float:
    """Mean Reciprocal Rank (optionally truncated to top-K)."""
    _check_same_length(y_true, y_pred)
    rrs: list[float] = []
    for gt, pred in zip(y_true, y_pred, strict=True):
        if len(gt) == 0:
            continue
        gt_set = set(gt)
        rr = 0.0
        seq = pred if k is None else pred[:k]
        for i, item in enumerate(seq, start=1):
            if item in gt_set:
                rr = 1.0 / i
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------
def coverage(y_pred: Array, catalog_size: int, k: int) -> float:
    """Catalog coverage: fraction of items that appear in some user's top-K."""
    seen: set[int] = set()
    for pred in y_pred:
        seen.update(pred[:k])
    return len(seen) / catalog_size if catalog_size else 0.0


def novelty(y_pred: Array, item_popularity: dict[int, float], k: int) -> float:
    """Mean self-information (in bits) of recommended items — higher = more novel."""
    total_pop = sum(item_popularity.values()) or 1.0
    scores: list[float] = []
    for pred in y_pred:
        user_scores: list[float] = []
        for item in pred[:k]:
            p = item_popularity.get(item, 0.0) / total_pop
            if p > 0:
                user_scores.append(-float(np.log2(p)))
        if user_scores:
            scores.append(float(np.mean(user_scores)))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_same_length(a: Array, b: Array) -> None:
    if len(a) != len(b):  # type: ignore[arg-type]
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")  # type: ignore[arg-type]
