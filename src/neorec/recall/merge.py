"""Multi-channel recall fusion.

Three strategies are implemented; the ablation study compares them:

* ``rrf``           — Reciprocal Rank Fusion ``score = Σ_c 1 / (k_rrf + rank_c)``
                      (Cormack et al., SIGIR 2009).  Score-free — only the
                      *rank* in each channel matters, which makes it robust to
                      channels that emit scores on wildly different scales
                      (e.g. dot-product logits vs. raw popularity counts).
* ``norm_weighted`` — Per-channel min-max normalize each user's scores into
                      ``[0, 1]``, then weighted sum.  Sensitive to the score
                      distribution of each channel but cheap to interpret.
* ``learned``       — *Stub* for future work: train a logistic regression on
                      ``(channel_score, channel_rank)`` features.

The :class:`MergeRecaller` is a regular :class:`BaseRecaller` that loads
*already-trained* per-channel artefacts from ``artifacts/recall/<name>/`` and
combines them on the fly at ``recall()`` time.  This keeps ``train.py``'s
"fit → evaluate → save" flow unchanged: training the merge channel is just
loading-and-validating the other channels.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from neorec.recall.base import BaseRecaller, RecallResult
from neorec.utils.io import ensure_dir, read_json, write_json

log = logging.getLogger(__name__)


# Map channel name → (module path, class name) for loading saved artefacts.
_CHANNEL_REGISTRY: dict[str, str] = {
    "als":        "neorec.recall.als:ALSRecaller",
    "two_tower":  "neorec.recall.two_tower:TwoTowerRecaller",
    "sasrec":     "neorec.recall.sasrec:SASRecRecaller",
    "popularity": "neorec.recall.popularity:PopularityRecaller",
    "cold_start": "neorec.recall.cold_start:ColdStartRecaller",
}


# ===========================================================================
# Pure fusion functions
# ===========================================================================
def merge_rrf(
    results: Iterable[RecallResult],
    k_rrf: int = 60,
    candidate_pool_size: int = 1000,
) -> RecallResult:
    """Reciprocal Rank Fusion over per-channel top-K lists.

    For every user::

        score(item) = Σ_c 1 / (k_rrf + rank_c(item))

    Items not present in a channel's list contribute 0 to the sum from that
    channel (equivalent to giving them rank → ∞).
    """
    results = list(results)
    if not results:
        raise ValueError("merge_rrf called with zero channels")

    users = results[0].user_ids
    n_users = len(users)
    out_items = np.full((n_users, candidate_pool_size), -1, dtype=np.int32)
    out_scores = np.zeros((n_users, candidate_pool_size), dtype=np.float32)

    for u_idx in range(n_users):
        scores: dict[int, float] = {}
        for res in results:
            # Every result was produced from the same user list — defensively
            # use u_idx, not user_id lookup, since order is guaranteed by
            # the merge driver.
            row = res.item_ids[u_idx]
            for rank, item in enumerate(row, start=1):
                if item < 0:
                    continue
                scores[int(item)] = scores.get(int(item), 0.0) + 1.0 / (k_rrf + rank)
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for j, (item, score) in enumerate(ranked[:candidate_pool_size]):
            out_items[u_idx, j] = item
            out_scores[u_idx, j] = score

    return RecallResult(
        user_ids=users,
        item_ids=out_items,
        scores=out_scores,
        channel="merge_rrf",
    )


def merge_norm_weighted(
    results: Iterable[RecallResult],
    weights: dict[str, float] | None = None,
    candidate_pool_size: int = 1000,
) -> RecallResult:
    """Min-max normalize each channel's per-user scores, then weighted sum.

    Weights default to ``1.0`` for any channel not in ``weights``.
    Items only present in one channel still contribute their normalized score
    from that channel (other channels contribute 0).
    """
    results = list(results)
    if not results:
        raise ValueError("merge_norm_weighted called with zero channels")
    weights = weights or {}

    users = results[0].user_ids
    n_users = len(users)
    out_items = np.full((n_users, candidate_pool_size), -1, dtype=np.int32)
    out_scores = np.zeros((n_users, candidate_pool_size), dtype=np.float32)

    for u_idx in range(n_users):
        scores: dict[int, float] = {}
        for res in results:
            row_items = res.item_ids[u_idx]
            row_scores = res.scores[u_idx].astype(np.float64)
            mask = row_items >= 0
            if mask.sum() == 0:
                continue
            valid_scores = row_scores[mask]
            s_min, s_max = float(valid_scores.min()), float(valid_scores.max())
            denom = max(s_max - s_min, 1e-12)
            w = float(weights.get(res.channel, 1.0))
            for item, sc, ok in zip(row_items, row_scores, mask):
                if not ok:
                    continue
                norm = (float(sc) - s_min) / denom
                scores[int(item)] = scores.get(int(item), 0.0) + w * norm
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for j, (item, score) in enumerate(ranked[:candidate_pool_size]):
            out_items[u_idx, j] = item
            out_scores[u_idx, j] = score

    return RecallResult(
        user_ids=users,
        item_ids=out_items,
        scores=out_scores.astype(np.float32),
        channel="merge_norm",
    )


def merge(
    results: Iterable[RecallResult],
    strategy: str = "rrf",
    **kwargs: object,
) -> RecallResult:
    """Dispatch to the requested fusion strategy."""
    if strategy == "rrf":
        return merge_rrf(results, **kwargs)  # type: ignore[arg-type]
    if strategy == "norm_weighted":
        return merge_norm_weighted(results, **kwargs)  # type: ignore[arg-type]
    if strategy == "learned":
        raise NotImplementedError("learned fusion is reserved for the W4 ablation")
    raise ValueError(f"Unknown merge strategy: {strategy!r}")


# ===========================================================================
# MergeRecaller — loads other channels' artefacts and combines them
# ===========================================================================
def _instantiate_channel(name: str, cfg) -> BaseRecaller:
    """Build a recaller for an *individual* channel, re-using its own config.

    The merge channel's cfg only carries the channel name; for instantiation
    we still need a recaller that exposes ``load(path)``.  The recaller's
    own training-time config knobs aren't used in load-only mode, so we just
    pass the merge config through (each recaller's ``__init__`` only stashes
    ``cfg`` without resolving any keys eagerly).
    """
    if name not in _CHANNEL_REGISTRY:
        raise ValueError(
            f"Unknown channel for merge: {name!r}. Known: {list(_CHANNEL_REGISTRY)}"
        )
    mod_path, cls_name = _CHANNEL_REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(cfg)


class MergeRecaller(BaseRecaller):
    """Loads N already-trained recall channels and combines them per-user."""

    name = "merge"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._channels: dict[str, BaseRecaller] = {}
        self._per_channel_depth: int = 500
        self._weights: dict[str, float] = {}
        self._strategy: str = "rrf"
        self._k_rrf: int = 60
        self._pool_size: int = 1000

    # ------------------------------------------------------------------
    def fit(self, interactions_path: str | Path) -> None:
        """Load each enabled channel's artefacts. The ``interactions_path``
        arg is unused — we keep it to match the BaseRecaller signature.

        When ``cfg.data.oof_split`` is true, artefacts are loaded from
        ``artifacts/recall_oof/`` instead of ``artifacts/recall/`` so merged
        recall matches the leakage-free OOF training path.
        """
        del interactions_path  # silence linter
        oof = False
        if "data" in self.cfg:
            oof = bool(self.cfg.data.get("oof_split", False))
        subdir = "recall_oof" if oof else "recall"
        artefacts_root = Path(self.cfg.paths.artifacts) / subdir

        self._strategy = str(self.cfg.recall.strategy)
        self._k_rrf = int(self.cfg.recall.rrf.k)
        self._pool_size = int(self.cfg.recall.output.candidate_pool_size)
        self._per_channel_depth = int(self.cfg.recall.output.per_channel_depth)

        for ch_name, ch_cfg in self.cfg.recall.channels.items():
            if not bool(ch_cfg.enabled):
                continue
            path = artefacts_root / ch_name
            if not path.exists():
                log.warning(
                    "Skipping merge channel %s: artefacts %s not found "
                    "(run `neorec train recall recall=%s` first).",
                    ch_name, path, ch_name,
                )
                continue
            recaller = _instantiate_channel(ch_name, self.cfg)
            recaller.load(path)
            self._channels[ch_name] = recaller
            self._weights[ch_name] = float(ch_cfg.weight)
        if not self._channels:
            raise RuntimeError(
                "No channels could be loaded for merging — train at least one base "
                "channel first."
            )
        log.info(
            "Merge ready (%s) over %d channels: %s; per-channel depth=%d, pool=%d",
            self._strategy, len(self._channels), list(self._channels),
            self._per_channel_depth, self._pool_size,
        )

    # ------------------------------------------------------------------
    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        if not self._channels:
            raise RuntimeError("fit() first to load channels.")

        per_channel: list[RecallResult] = []
        n_users = len(user_ids)
        users_arr = np.asarray(list(user_ids), dtype=np.int32)
        depth = max(self._per_channel_depth, k)
        for name, recaller in self._channels.items():
            res = recaller.recall(user_ids, k=depth)
            # Guarantee the merged results all use the same user ordering.
            if not np.array_equal(res.user_ids, users_arr):
                raise RuntimeError(
                    f"Channel {name} returned mismatched user order — "
                    "merge requires consistent per-user rows."
                )
            per_channel.append(res)
            log.debug("Channel %s: top-%d ready", name, depth)

        pool_size = max(self._pool_size, k)
        if self._strategy == "rrf":
            merged = merge_rrf(per_channel, k_rrf=self._k_rrf,
                               candidate_pool_size=pool_size)
        elif self._strategy == "norm_weighted":
            merged = merge_norm_weighted(per_channel, weights=self._weights,
                                         candidate_pool_size=pool_size)
        else:
            raise ValueError(f"Unsupported merge strategy: {self._strategy!r}")

        # Truncate to top-k
        return RecallResult(
            user_ids=merged.user_ids,
            item_ids=merged.item_ids[:, :k].astype(np.int32),
            scores=merged.scores[:, :k].astype(np.float32),
            channel=merged.channel,
        )

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """The merge channel itself has no learnable state; we only record
        provenance so the run is reproducible."""
        out_dir = ensure_dir(path)
        meta = {
            "strategy":           self._strategy,
            "k_rrf":              self._k_rrf,
            "pool_size":          self._pool_size,
            "per_channel_depth":  self._per_channel_depth,
            "channels":           list(self._channels.keys()),
            "weights":            self._weights,
        }
        write_json(meta, out_dir / "meta.json")
        log.info("Saved merge meta to %s", out_dir)

    def load(self, path: str | Path) -> None:
        meta = read_json(Path(path) / "meta.json")
        # Just record the historical settings — the actual channels still need
        # to be re-loaded by calling fit() against current cfg.
        log.info(
            "Loaded merge meta: %s channels=%s strategy=%s",
            path, meta.get("channels"), meta.get("strategy"),
        )


__all__ = [
    "merge",
    "merge_rrf",
    "merge_norm_weighted",
    "MergeRecaller",
]
