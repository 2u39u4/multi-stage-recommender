"""Counterfactual offline evaluation — simulates A/B testing on logged data.

Supports:
    * IPS (Inverse Propensity Scoring)
    * SNIPS (self-normalized IPS)
    * Doubly-robust estimator (stretch goal)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _importance_weights(
    policy_probs: Sequence[float],
    logging_probs: Sequence[float],
    clip: tuple[float, float] | None,
) -> np.ndarray:
    policy = np.asarray(policy_probs, dtype=np.float64)
    logging = np.asarray(logging_probs, dtype=np.float64)
    if policy.shape != logging.shape:
        raise ValueError(f"shape mismatch: policy={policy.shape}, logging={logging.shape}")
    if np.any(logging <= 0):
        raise ValueError("logging_probs must be strictly positive")

    weights = policy / logging
    if clip is not None:
        lo, hi = clip
        if lo <= 0 or hi < lo:
            raise ValueError(f"invalid clip range: {clip}")
        weights = np.clip(weights, lo, hi)
    return weights


def ips_estimator(
    rewards: Sequence[float],
    policy_probs: Sequence[float],
    logging_probs: Sequence[float],
    clip: tuple[float, float] | None = (0.01, 10.0),
) -> float:
    """Importance-weighted estimate of the target policy value."""
    r = np.asarray(rewards, dtype=np.float64)
    w = _importance_weights(policy_probs, logging_probs, clip)
    if r.shape != w.shape:
        raise ValueError(f"shape mismatch: rewards={r.shape}, weights={w.shape}")
    return float(np.mean(w * r)) if r.size else 0.0


def snips_estimator(
    rewards: Sequence[float],
    policy_probs: Sequence[float],
    logging_probs: Sequence[float],
    clip: tuple[float, float] | None = (0.01, 10.0),
) -> float:
    """Self-normalized IPS (SNIPS) — lower variance, slightly biased."""
    r = np.asarray(rewards, dtype=np.float64)
    w = _importance_weights(policy_probs, logging_probs, clip)
    if r.shape != w.shape:
        raise ValueError(f"shape mismatch: rewards={r.shape}, weights={w.shape}")
    denom = float(np.sum(w))
    return float(np.sum(w * r) / denom) if denom > 0 else 0.0
