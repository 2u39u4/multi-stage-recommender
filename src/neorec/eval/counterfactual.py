"""Counterfactual offline evaluation — simulates A/B testing on logged data.

Supports:
    * IPS (Inverse Propensity Scoring)
    * SNIPS (self-normalized IPS)
    * Doubly-robust estimator (stretch goal)
"""

from __future__ import annotations

from collections.abc import Sequence


def ips_estimator(
    rewards: Sequence[float],
    policy_probs: Sequence[float],
    logging_probs: Sequence[float],
    clip: tuple[float, float] | None = (0.01, 10.0),
) -> float:
    """Importance-weighted estimate of the target policy value."""
    raise NotImplementedError  # TODO(W4)


def snips_estimator(
    rewards: Sequence[float],
    policy_probs: Sequence[float],
    logging_probs: Sequence[float],
    clip: tuple[float, float] | None = (0.01, 10.0),
) -> float:
    """Self-normalized IPS (SNIPS) — lower variance, slightly biased."""
    raise NotImplementedError  # TODO(W4)
