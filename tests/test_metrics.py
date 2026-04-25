"""Unit tests for ranking metrics.

These verify closed-form values on hand-crafted inputs — the first line of
defense against silent regressions when refactoring.
"""

from __future__ import annotations

import math

import pytest

from neorec.eval.metrics import (
    coverage,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_simple() -> None:
    y_true = [[1, 2, 3]]
    y_pred = [[1, 9, 2, 10, 3]]
    # K=3 → {1, 9, 2} ∩ {1,2,3} = {1, 2}   → 2/3
    assert recall_at_k(y_true, y_pred, k=3) == pytest.approx(2 / 3)
    # K=5 → all three hit → 1.0
    assert recall_at_k(y_true, y_pred, k=5) == pytest.approx(1.0)


def test_hit_rate_at_k() -> None:
    y_true = [[1, 2, 3], [4]]
    y_pred = [[9, 8, 1], [10, 11, 12]]
    # user 0 hits at rank 3, user 1 misses → HR@3 = 0.5
    assert hit_rate_at_k(y_true, y_pred, k=3) == pytest.approx(0.5)


def test_ndcg_at_k_single_user() -> None:
    y_true = [[10]]
    y_pred = [[1, 2, 10]]                       # position 3 (i=2)
    # DCG = 1 / log2(2+2) = 1/2,  IDCG = 1 / log2(0+2) = 1
    assert ndcg_at_k(y_true, y_pred, k=3) == pytest.approx(1 / 2)


def test_ndcg_perfect_top() -> None:
    y_true = [[1, 2]]
    y_pred = [[1, 2, 3, 4]]
    # dcg = 1 + 1/log2(3) ; idcg same → 1.0
    assert ndcg_at_k(y_true, y_pred, k=4) == pytest.approx(1.0)


def test_mrr_truncation() -> None:
    y_true = [[5]]
    y_pred = [[1, 2, 5]]
    assert mean_reciprocal_rank(y_true, y_pred, k=5) == pytest.approx(1 / 3)
    # If K=2 the relevant item is not in the cutoff → 0
    assert mean_reciprocal_rank(y_true, y_pred, k=2) == pytest.approx(0.0)


def test_coverage() -> None:
    y_pred = [[1, 2, 3], [2, 3, 4]]
    # K=2 → {1,2,2,3} = {1,2,3} out of catalog 5 → 3/5
    assert coverage(y_pred, catalog_size=5, k=2) == pytest.approx(3 / 5)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k([[1]], [[1], [2]], k=1)


@pytest.mark.parametrize("k", [1, 5, 10])
def test_recall_bounds(k: int) -> None:
    """Recall is always in [0, 1]."""
    y_true = [[1, 2, 3]]
    y_pred = [list(range(20))]
    r = recall_at_k(y_true, y_pred, k=k)
    assert 0.0 <= r <= 1.0
    assert not math.isnan(r)
