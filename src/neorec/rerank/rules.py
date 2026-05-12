"""Business-rule filters applied after MMR / IPS.

Three rules ship by default:

* drop items the user has already interacted with (``filter_already_watched``);
* cap each genre at ``max_per_genre_ratio`` of the final list;
* cap each ``year_bucket`` at ``max_per_year_bucket`` items.

Rules are applied **greedily** in input order so the upstream MMR /
relevance ranking is preserved as much as possible — the i-th item in the
output is always the highest-rank survivor among the inputs ≤ i.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def apply_rules(
    candidate_ids: Sequence[int],
    user_history: set[int] | frozenset[int] | Iterable[int],
    item_meta: Mapping[int, Mapping[str, object]],
    max_per_genre_ratio: float = 0.5,
    max_per_year_bucket: int = 3,
    filter_already_watched: bool = True,
    k: int = 10,
) -> list[int]:
    """Greedy rule pass over an ordered candidate list.

    Parameters
    ----------
    candidate_ids
        Item ids in **score-descending** order — the rule engine will keep
        as many high-rank items as it can without violating quotas.
    user_history
        Set-like of items already seen by the user. Items in this set are
        dropped if ``filter_already_watched`` is true.
    item_meta
        Per-item metadata dict. The known keys are ``genres`` (list[int])
        and ``year_bucket`` (int). Missing items are treated as having
        empty genres / a unique year bucket — they're never blocked by the
        per-bucket cap.
    max_per_genre_ratio
        Maximum fraction of the *final* list that may share any single
        genre. ``0.5`` ≅ "no genre takes over more than half the output".
    max_per_year_bucket
        Hard cap on items per year bucket.
    filter_already_watched
        Whether to drop items in ``user_history``.
    k
        Maximum output size.

    Returns
    -------
    list[int]
        Up to ``k`` items, preserving the input order amongst survivors.
    """
    history = set(int(x) for x in user_history) if user_history else set()
    genre_cap = int(max_per_genre_ratio * k) if max_per_genre_ratio > 0 else k
    year_cap = int(max_per_year_bucket) if max_per_year_bucket > 0 else k

    out: list[int] = []
    genre_counts: dict[int, int] = {}
    year_counts: dict[int, int] = {}

    for raw_id in candidate_ids:
        if len(out) >= k:
            break
        iid = int(raw_id)
        if filter_already_watched and iid in history:
            continue

        meta = item_meta.get(iid, {})
        genres = meta.get("genres", [])
        year_bucket = meta.get("year_bucket", None)

        if year_bucket is not None and year_cap > 0:
            yb = int(year_bucket)
            if year_counts.get(yb, 0) >= year_cap:
                continue

        if genres and genre_cap > 0:
            genre_list = [int(g) for g in genres]
            if any(genre_counts.get(g, 0) >= genre_cap for g in genre_list):
                continue
        else:
            genre_list = []

        out.append(iid)
        for g in genre_list:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        if year_bucket is not None:
            yb = int(year_bucket)
            year_counts[yb] = year_counts.get(yb, 0) + 1

    return out
