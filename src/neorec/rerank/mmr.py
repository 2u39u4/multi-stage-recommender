"""Maximal Marginal Relevance (MMR) re-ranking for diversity.

    MMR(d) = λ · relevance(d) - (1 - λ) · max_{d' ∈ S} sim(d, d')

References
----------
* Carbonell & Goldstein. *The Use of MMR, Diversity-Based Reranking for
  Reordering Documents and Producing Summaries.* SIGIR 1998.

The function is fully vectorised over the candidate pool (one row per
candidate, similarity is a length-K vector against the already-selected
set), so each user takes O(K · pool) time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max scale into [0, 1] so MMR weighs relevance and similarity on the
    same scale. We return a constant 0.5 array if all scores are equal — this
    falls back to a pure-diversity selection.
    """
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - lo) / (hi - lo)


def mmr_rerank(
    candidate_ids: Sequence[int],
    candidate_scores: Sequence[float],
    item_embeddings: np.ndarray,
    k: int,
    lam: float = 0.5,
    normalize_relevance: bool = True,
) -> list[int]:
    """Greedy MMR re-ranking.

    Parameters
    ----------
    candidate_ids
        Item ids (must index ``item_embeddings`` directly).
    candidate_scores
        Relevance scores (e.g. CTR probabilities). Same length as
        ``candidate_ids``.
    item_embeddings
        Item-vector matrix; row ``i`` is ``embeddings[i]``. Should be
        L2-normalized so that ``embeddings @ embeddings.T`` equals cosine
        similarity (Two-Tower's ``item_vecs.npy`` satisfies this by
        construction).
    k
        Number of items to return.
    lam
        Trade-off in ``[0, 1]``. ``1.0`` = pure relevance, ``0.0`` = pure
        diversity. Values around 0.5–0.7 are common in production.
    normalize_relevance
        Min-max scale relevance before mixing it with similarity. Off-by-
        default behaviour would let one large CTR value dominate the
        objective regardless of ``lam``.

    Returns
    -------
    list[int]
        Up to ``k`` re-ordered item ids (deterministic).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    ids = np.asarray(candidate_ids, dtype=np.int64)
    rel = np.asarray(candidate_scores, dtype=np.float64)
    if ids.shape != rel.shape:
        raise ValueError(
            f"candidate_ids and candidate_scores must have the same length "
            f"({ids.shape[0]} vs {rel.shape[0]})"
        )
    if ids.size == 0:
        return []
    k = min(int(k), ids.size)
    if k <= 0:
        return []

    if normalize_relevance:
        rel = _normalize(rel)

    cand_emb = item_embeddings[ids]  # (N, d)
    # Greedy selection.
    selected_pos: list[int] = []
    remaining = np.ones(ids.size, dtype=bool)
    max_sim = np.full(ids.size, -np.inf, dtype=np.float64)

    # First pick: highest relevance (MMR(d) = lam · rel(d) since the selected
    # set is empty and we treat sim_to_empty = 0).
    first = int(np.argmax(rel))
    selected_pos.append(first)
    remaining[first] = False
    # Update max-sim for everyone else against the newly selected item.
    sims_new = cand_emb @ cand_emb[first]
    max_sim = np.maximum(max_sim, sims_new)
    max_sim[~remaining] = -np.inf  # frozen

    while len(selected_pos) < k:
        rem_idx = np.where(remaining)[0]
        if rem_idx.size == 0:
            break
        sim_clipped = np.maximum(max_sim[rem_idx], 0.0)
        mmr_scores = lam * rel[rem_idx] - (1.0 - lam) * sim_clipped
        chosen = int(rem_idx[int(np.argmax(mmr_scores))])
        selected_pos.append(chosen)
        remaining[chosen] = False
        sims_new = cand_emb @ cand_emb[chosen]
        max_sim = np.maximum(max_sim, sims_new)
        max_sim[~remaining] = -np.inf

    return ids[selected_pos].tolist()


def intra_list_similarity(
    item_ids: Sequence[int],
    item_embeddings: np.ndarray,
) -> float:
    """Average pairwise cosine similarity inside a list — the "diversity"
    half of the MMR-λ Pareto plot. Lower = more diverse.

    Returns 0.0 if fewer than two items are supplied.
    """
    ids = np.asarray(item_ids, dtype=np.int64)
    if ids.size < 2:
        return 0.0
    emb = item_embeddings[ids]
    sim = emb @ emb.T  # (k, k)
    # Upper triangle, excluding diagonal.
    iu = np.triu_indices(ids.size, k=1)
    pairs = sim[iu]
    return float(pairs.mean())
