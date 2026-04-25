"""Statistical significance utilities for model comparison.

Provided:
    * bootstrap_ci        — percentile bootstrap confidence interval for a metric
    * paired_bootstrap    — paired bootstrap test between two model outputs
    * paired_t_test       — paired t-test (fallback / sanity check)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    per_user_scores: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) of a percentile bootstrap CI."""
    raise NotImplementedError  # TODO(W4): implement in Week 4


def paired_bootstrap(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> float:
    """Return a two-sided p-value for H0: mean(a) == mean(b)."""
    raise NotImplementedError  # TODO(W4)


def paired_t_test(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
) -> float:
    """Return two-sided p-value from a paired t-test."""
    raise NotImplementedError  # TODO(W4)


def compare_models(
    results: dict[str, Sequence[float]],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, dict[str, float]]:
    """Pairwise comparison over all models in ``results``.

    Returns a matrix of p-values keyed by model name.
    """
    raise NotImplementedError  # TODO(W4)
