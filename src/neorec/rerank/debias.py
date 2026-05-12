"""Popularity / exposure debias via inverse propensity re-weighting.

For each candidate item, divide its relevance by its (clipped) marginal
popularity. This down-weights head items and surfaces long-tail items that
would otherwise be drowned out by the popularity bias baked into both the
training data and most recall models.

References
----------
* Schnabel et al. *Recommendations as Treatments: Debiasing Learning and
  Evaluation.* ICML 2016.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def ips_rerank(
    candidate_ids: Sequence[int],
    candidate_scores: Sequence[float],
    item_popularity: Mapping[int, float],
    clip: tuple[float, float] = (0.01, 10.0),
    k: int | None = None,
    exponent: float = 1.0,
) -> list[int]:
    """Re-order candidates by ``score / clip(popularity ** exponent)``.

    Parameters
    ----------
    candidate_ids
        Item ids (typically the top-K output of a ranker).
    candidate_scores
        Relevance scores aligned with ``candidate_ids``.
    item_popularity
        Dict from item id to a non-negative popularity proxy (interaction
        count, exposure rate, etc.). The exact units don't matter; only the
        ratio across items does. Missing items are treated as having
        popularity ``clip[0]`` (i.e. the maximum possible boost — they get
        the benefit of the doubt).
    clip
        ``(min, max)`` clamp for the propensity to bound the IPS weight.
        Without clipping, a single very rare item can completely dominate.
    k
        Optional cutoff. If ``None``, return all candidates in re-ranked
        order.
    exponent
        Power applied to the popularity (a.k.a. *β* in some papers).
        ``β=0`` is identity (no debias), ``β=1`` is plain IPS, larger
        values are more aggressive.

    Returns
    -------
    list[int]
        Re-ordered item ids.
    """
    if clip[0] <= 0 or clip[1] <= clip[0]:
        raise ValueError(f"clip must satisfy 0 < min < max; got {clip}")

    ids = np.asarray(candidate_ids, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if ids.shape != scores.shape:
        raise ValueError(
            f"candidate_ids and candidate_scores must have the same length "
            f"({ids.shape[0]} vs {scores.shape[0]})"
        )
    if ids.size == 0:
        return []

    pops = np.array(
        [float(item_popularity.get(int(i), clip[0])) for i in ids],
        dtype=np.float64,
    )
    if exponent != 1.0:
        pops = pops ** float(exponent)
    pops = np.clip(pops, clip[0], clip[1])

    adjusted = scores / pops
    order = np.argsort(-adjusted)
    if k is not None:
        order = order[: int(k)]
    return ids[order].tolist()


def item_popularity_from_interactions(
    item_ids: Sequence[int],
    smooth: float = 1.0,
) -> dict[int, float]:
    """Convenience helper — count appearances of each item id.

    ``smooth`` is added to every count (Laplace-smoothing-style) so that
    items with zero training exposure don't get a div-by-zero boost.
    """
    counts: dict[int, float] = {}
    for i in item_ids:
        counts[int(i)] = counts.get(int(i), 0.0) + 1.0
    if smooth > 0:
        for k in counts:
            counts[k] += float(smooth)
    return counts
