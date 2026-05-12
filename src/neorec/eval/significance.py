"""Statistical significance utilities for model comparison.

Provided:
    * bootstrap_ci        — percentile bootstrap confidence interval for a metric
    * paired_bootstrap    — paired bootstrap test between two per-user score vectors
    * paired_t_test       — paired t-test (fallback / sanity check)
    * compare_models      — pairwise p-value matrix over a dict of models

All inputs are *per-user* score arrays (e.g. per-user Recall@10). Aggregation
to the dataset-level metric is done inside each routine using the user-paired
resampling protocol from Boytsov & Naumov (RecSys 2019).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
def _as_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D scores, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
def bootstrap_ci(
    per_user_scores: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any user-aggregated statistic.

    Returns ``(point_estimate, lo, hi)`` for the two-sided CI at level
    ``1 - alpha``. The point estimate is the statistic on the original
    sample (not the bootstrap mean), since the latter has lower coverage
    for skewed metrics.
    """
    scores = _as_array(per_user_scores)
    rng = np.random.default_rng(seed)
    n = scores.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0

    point = float(statistic(scores))
    if n_boot <= 0:
        return point, point, point

    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(statistic(scores[idx]))
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return point, lo, hi


# ---------------------------------------------------------------------------
def paired_bootstrap(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> float:
    """Two-sided paired-bootstrap p-value for H0: stat(a) == stat(b).

    The procedure resamples *user indices* and computes the statistic
    difference ``stat(a[idx]) − stat(b[idx])`` for each bootstrap draw,
    then reports the fraction whose sign differs from the observed
    difference (doubled for two-sidedness).
    """
    a = _as_array(a_scores)
    b = _as_array(b_scores)
    if a.shape != b.shape:
        raise ValueError(f"paired_bootstrap requires aligned arrays; got {a.shape} vs {b.shape}")
    n = a.shape[0]
    if n == 0:
        return 1.0

    observed = float(statistic(a) - statistic(b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[k] = float(statistic(a[idx]) - statistic(b[idx]))

    # Centre under H0 (subtract observed) and count tail extremes.
    centred = diffs - observed
    extreme = float(np.mean(np.abs(centred) >= abs(observed)))
    return min(max(extreme, 1.0 / (n_boot + 1)), 1.0)


# ---------------------------------------------------------------------------
def paired_t_test(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
) -> float:
    """Two-sided paired-sample t-test p-value (Student's t with df=n-1)."""
    a = _as_array(a_scores)
    b = _as_array(b_scores)
    if a.shape != b.shape:
        raise ValueError(
            f"paired_t_test requires aligned arrays; got {a.shape} vs {b.shape}"
        )
    n = a.shape[0]
    if n < 2:
        return 1.0
    d = a - b
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd < 1e-12:
        return 1.0 if mean == 0 else 0.0
    t_stat = mean / (sd / np.sqrt(n))
    # Two-sided p-value via the survival function — defer to scipy if
    # available, otherwise fall back to a closed-form approximation.
    try:
        from scipy.stats import t as student_t

        return float(2 * (1 - student_t.cdf(abs(t_stat), df=n - 1)))
    except ImportError:  # pragma: no cover — scipy is part of project deps
        # Standard-normal approximation; perfectly fine for n > ~30.
        from math import erf, sqrt

        return float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))


# ---------------------------------------------------------------------------
def compare_models(
    results: Mapping[str, Sequence[float]],
    method: str = "paired_bootstrap",
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Pairwise p-value matrix over the models in ``results``.

    Parameters
    ----------
    results
        Mapping ``model_name -> per_user_scores`` (all vectors must share
        the same user ordering and length).
    method
        ``'paired_bootstrap'`` (default) or ``'paired_t'``.
    """
    names = list(results)
    out: dict[str, dict[str, float]] = {n: {} for n in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                out[a][b] = 1.0
                continue
            if method == "paired_bootstrap":
                p = paired_bootstrap(
                    results[a], results[b],
                    n_boot=n_boot, seed=seed, statistic=statistic,
                )
            elif method == "paired_t":
                p = paired_t_test(results[a], results[b])
            else:
                raise ValueError(f"Unknown method: {method!r}")
            out[a][b] = float(p)
    return out


__all__ = [
    "bootstrap_ci",
    "paired_bootstrap",
    "paired_t_test",
    "compare_models",
]
