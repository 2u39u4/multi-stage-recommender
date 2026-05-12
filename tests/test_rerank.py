"""Unit tests for rerank primitives and statistical helpers.

These verify closed-form behaviour on hand-crafted inputs — the rerank
math is small enough that we can write expected outputs by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from neorec.eval.significance import (
    bootstrap_ci,
    compare_models,
    paired_bootstrap,
    paired_t_test,
)
from neorec.rerank.debias import (
    ips_rerank,
    item_popularity_from_interactions,
)
from neorec.rerank.mmr import intra_list_similarity, mmr_rerank
from neorec.rerank.rules import apply_rules


# ===========================================================================
# MMR
# ===========================================================================
def test_mmr_lambda_one_preserves_relevance_order() -> None:
    """λ=1.0 means pure relevance — MMR must reproduce the argsort order."""
    ids = [0, 1, 2, 3]
    scores = [0.1, 0.9, 0.5, 0.7]
    emb = np.eye(4, dtype=np.float32)  # orthogonal → no diversity penalty
    out = mmr_rerank(ids, scores, emb, k=4, lam=1.0)
    assert out == [1, 3, 2, 0]


def test_mmr_lambda_zero_prefers_diverse() -> None:
    """λ=0.0 — first pick is highest-relevance (tie-break by relevance),
    second pick should avoid items similar to the first."""
    ids = [0, 1, 2]
    scores = [0.6, 0.5, 0.4]
    emb = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.14],   # very similar to item 0
            [0.0, 1.0],     # orthogonal to item 0
        ],
        dtype=np.float32,
    )
    # L2-normalise the rows (test invariant).
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = mmr_rerank(ids, scores, emb, k=2, lam=0.0)
    assert out[0] == 0           # highest relevance gets the first slot
    assert out[1] == 2           # diversity prefers the orthogonal item


def test_mmr_truncation() -> None:
    ids = [10, 20, 30, 40]
    scores = [0.4, 0.3, 0.2, 0.1]
    emb = np.eye(50, dtype=np.float32)
    out = mmr_rerank(ids, scores, emb, k=2, lam=0.5)
    assert len(out) == 2
    assert set(out).issubset({10, 20, 30, 40})


def test_mmr_invalid_lambda_raises() -> None:
    with pytest.raises(ValueError):
        mmr_rerank([0], [0.5], np.eye(2, dtype=np.float32), k=1, lam=1.5)


def test_intra_list_similarity_orthogonal_is_zero() -> None:
    emb = np.eye(5, dtype=np.float32)
    assert intra_list_similarity([0, 1, 2], emb) == pytest.approx(0.0)


def test_intra_list_similarity_identical_is_one() -> None:
    emb = np.zeros((5, 3), dtype=np.float32)
    emb[:] = np.array([1.0, 0.0, 0.0])
    assert intra_list_similarity([0, 1, 2], emb) == pytest.approx(1.0)


# ===========================================================================
# IPS
# ===========================================================================
def test_ips_demotes_popular_items() -> None:
    """A super-popular tied-score item should drop behind a long-tail one."""
    ids = [0, 1]
    scores = [0.5, 0.5]
    pops = {0: 100.0, 1: 1.0}
    out = ips_rerank(ids, scores, pops, clip=(0.1, 1000.0))
    assert out == [1, 0]


def test_ips_clipping_caps_boost() -> None:
    """Item with pop below clip-min should be treated as clip-min."""
    ids = [0, 1]
    scores = [0.5, 0.5]
    pops = {0: 0.0001, 1: 10.0}  # 0 would be 5000× boost without clip
    out = ips_rerank(ids, scores, pops, clip=(1.0, 100.0))
    assert out == [0, 1]


def test_ips_exponent_zero_is_identity() -> None:
    ids = [0, 1, 2]
    scores = [0.1, 0.5, 0.3]
    pops = {0: 100, 1: 1, 2: 50}
    out = ips_rerank(ids, scores, pops, exponent=0.0)
    assert out == [1, 2, 0]


def test_item_popularity_helper() -> None:
    counts = item_popularity_from_interactions([1, 1, 2, 3, 3, 3], smooth=0.0)
    assert counts == {1: 2.0, 2: 1.0, 3: 3.0}


# ===========================================================================
# Rules
# ===========================================================================
def test_rules_filter_watched() -> None:
    cands = [0, 1, 2, 3]
    history = {1, 3}
    meta = {i: {"genres": [], "year_bucket": i} for i in cands}
    out = apply_rules(cands, history, meta, k=4)
    assert out == [0, 2]


def test_rules_genre_cap() -> None:
    """With max_per_genre_ratio=0.5 and k=4, at most 2 items per genre."""
    cands = [0, 1, 2, 3, 4]
    meta = {
        0: {"genres": [1], "year_bucket": 0},
        1: {"genres": [1], "year_bucket": 1},
        2: {"genres": [1], "year_bucket": 2},  # blocked by genre cap
        3: {"genres": [2], "year_bucket": 3},
        4: {"genres": [2], "year_bucket": 4},
    }
    out = apply_rules(cands, set(), meta, max_per_genre_ratio=0.5, k=4)
    assert out == [0, 1, 3, 4]


def test_rules_year_bucket_cap() -> None:
    cands = [0, 1, 2, 3]
    meta = {i: {"genres": [], "year_bucket": 7} for i in cands}
    out = apply_rules(cands, set(), meta, max_per_year_bucket=2, k=4)
    assert out == [0, 1]


# ===========================================================================
# Statistical significance
# ===========================================================================
def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=0.5, scale=0.1, size=200)
    pt, lo, hi = bootstrap_ci(scores, n_boot=500, alpha=0.05)
    assert lo <= pt <= hi
    assert pt == pytest.approx(scores.mean(), rel=1e-6)
    assert lo < 0.55 and hi > 0.45


def test_paired_bootstrap_detects_real_diff() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.55, scale=0.1, size=300)
    b = rng.normal(loc=0.45, scale=0.1, size=300)
    p = paired_bootstrap(a, b, n_boot=500)
    assert p < 0.05


def test_paired_bootstrap_identical_arrays_high_p() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    p = paired_bootstrap(a, a.copy(), n_boot=200)
    assert p >= 0.5


def test_paired_t_test_matches_bootstrap_direction() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(loc=0.6, scale=0.1, size=300)
    b = rng.normal(loc=0.5, scale=0.1, size=300)
    p_t = paired_t_test(a, b)
    p_boot = paired_bootstrap(a, b, n_boot=300)
    assert p_t < 0.05 and p_boot < 0.05


def test_compare_models_diagonal() -> None:
    rng = np.random.default_rng(4)
    res = {
        "a": rng.normal(size=100),
        "b": rng.normal(loc=0.3, size=100),
    }
    mat = compare_models(res, method="paired_bootstrap", n_boot=200)
    assert mat["a"]["a"] == 1.0 and mat["b"]["b"] == 1.0
    # Off-diagonal should be symmetric (within bootstrap sampling noise) and
    # below conventional thresholds for a clear mean shift.
    assert mat["a"]["b"] < 0.05 and mat["b"]["a"] < 0.05
