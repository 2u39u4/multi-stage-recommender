"""Recall-score lookup features for ranking models.

Why this exists
---------------
A point-wise ranker that only consumes (user_id, item_id, side_features)
**throws away** the work the recall layer already did.  Every candidate at
ranking time already has a score from each upstream channel (ALS,
Two-Tower, SASRec, popularity, cold-start) plus the merged RRF score, and
those scores carry strong CF / sequential / content signal that side
features alone can't reproduce.

This module exposes a :class:`RecallFeatureStore` that:

1.  Calls each base recaller's :meth:`recall(user_ids, k=depth)`,
2.  Builds, **per channel**, a row-sorted ``(n_users, depth)`` table of
    item ids + scores so per-pair lookup is a single vectorised
    :func:`np.searchsorted`,
3.  Provides ``lookup_batch(user_ids, item_ids) -> ndarray(N, n_channels+2)``
    that returns the per-channel score + an "in-pool" mask for every
    requested pair.

A pair that doesn't appear in a channel's top-``depth`` gets a 0 in that
column (with the mask set to 0 too — so the ranker can learn to treat
"missing" differently from "low score"). Random training negatives mostly
fall into this bucket, which itself is informative ("not in any recall
pool" ⇒ probably not a click).

Storage layout
--------------
For each channel ``c``:

* ``sorted_items[c]``   — ``(n_users, depth)`` int32, sorted ascending per row
* ``sorted_scores[c]``  — ``(n_users, depth)`` float32, aligned with the
                          sorted-items row

Memory at depth=500 × 6 channels × 6 034 users on ML-1M
≈ 144 MB — manageable.

A precompute-and-cache pass takes ~30 s on CPU, after which lookup is
sub-millisecond per batch of 2 048 pairs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from neorec.recall.base import BaseRecaller, RecallResult

log = logging.getLogger(__name__)


# Channel order is canonical — must match across save/load round-trips.
DEFAULT_CHANNELS: tuple[str, ...] = (
    "als",
    "two_tower",
    "sasrec",
    "popularity",
    "cold_start",
    "merge_rrf",
)


@dataclass
class _ChannelTable:
    """One channel's per-user sorted lookup table."""

    sorted_items: np.ndarray   # (n_users, K) int32, sorted ascending per row
    sorted_scores: np.ndarray  # (n_users, K) float32

    @classmethod
    def from_result(cls, result: RecallResult, n_users: int, depth: int) -> "_ChannelTable":
        """Convert a ``RecallResult`` (already top-K sorted by score) into
        the row-sorted-by-item-id layout we need for fast binary-search lookup.

        The original ordering by score is irrelevant here — we just need to
        find a given (u, i) quickly.
        """
        users = result.user_ids
        items = result.item_ids
        scores = result.scores
        K = items.shape[1]
        if K < depth:
            raise ValueError(f"depth {depth} exceeds channel top-K {K}")

        # Map user_id -> row position (assumes recaller returned the same user
        # ordering as we requested; merge.py already asserts this).
        out_items = np.full((n_users, depth), -1, dtype=np.int32)
        out_scores = np.zeros((n_users, depth), dtype=np.float32)
        for row_idx, user_id in enumerate(users):
            u = int(user_id)
            row_items = items[row_idx, :depth]
            row_scores = scores[row_idx, :depth]
            # Some channels emit -1 padding at the tail — strip them so the
            # binary search doesn't match item_id == -1 accidentally.
            valid = row_items >= 0
            row_items = row_items[valid]
            row_scores = row_scores[valid]
            order = np.argsort(row_items, kind="stable")
            n_valid = len(row_items)
            out_items[u, :n_valid] = row_items[order]
            out_scores[u, :n_valid] = row_scores[order]
        return cls(sorted_items=out_items, sorted_scores=out_scores)

    # Vectorised batched lookup — single numpy pass.
    def lookup_batch(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, mask)`` aligned with the input rows.

        * ``scores[i]`` = this channel's score for (user_ids[i], item_ids[i]),
          or 0 if the item wasn't in the user's top-K.
        * ``mask[i]``   = 1 if the item *was* found in the user's top-K, else 0.
        """
        rows_items = self.sorted_items[user_ids]            # (N, K)
        rows_scores = self.sorted_scores[user_ids]          # (N, K)
        # `np.searchsorted` along axis 1 isn't natively vectorised — but the
        # rows are independent and small, so we use the standard "elementwise
        # equality after binary search" trick via take_along_axis.
        K = rows_items.shape[1]
        # ``positions[i] ∈ [0, K]`` is the insertion index for item_ids[i] in
        # the sorted row.  If positions[i] < K and rows_items[i, positions[i]]
        # == item_ids[i], we have a hit.
        positions = np.empty(len(user_ids), dtype=np.int64)
        for i in range(len(user_ids)):
            positions[i] = np.searchsorted(rows_items[i], item_ids[i])
        clipped = np.clip(positions, 0, K - 1)
        rows_idx = np.arange(len(user_ids))
        found_items = rows_items[rows_idx, clipped]
        mask = (found_items == item_ids) & (positions < K)
        scores = np.where(mask, rows_scores[rows_idx, clipped], 0.0).astype(np.float32)
        return scores, mask.astype(np.float32)


class RecallFeatureStore:
    """Holds one :class:`_ChannelTable` per channel + lookup orchestration."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        channels: Iterable[str] = DEFAULT_CHANNELS,
        depth: int = 500,
    ) -> None:
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.channels = tuple(channels)
        self.depth = int(depth)
        self._tables: dict[str, _ChannelTable] = {}

    @property
    def n_features(self) -> int:
        """One score column per channel + one "found" mask column per channel."""
        return 2 * len(self.channels)

    @property
    def channel_score_cols(self) -> list[str]:
        return [f"score_{c}" for c in self.channels]

    @property
    def channel_mask_cols(self) -> list[str]:
        return [f"mask_{c}" for c in self.channels]

    @property
    def feature_cols(self) -> list[str]:
        return self.channel_score_cols + self.channel_mask_cols

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(
        self,
        cfg,
        user_ids: np.ndarray | None = None,
    ) -> "RecallFeatureStore":
        """Run each base recaller + MergeRecaller, populate the tables.

        ``cfg`` is the global config; we only need ``cfg.paths`` and
        ``cfg.data`` to build a MergeRecaller — and that MergeRecaller
        re-loads every base channel's artefacts via its standard ``fit()``.
        """
        from neorec.recall.merge import MergeRecaller

        if user_ids is None:
            user_ids = np.arange(self.n_users, dtype=np.int64)
        user_ids = np.asarray(user_ids, dtype=np.int64)

        # Compose recall=merge config (mirrors what train.py does).
        repo_root = Path(__file__).resolve().parents[3]
        merge_yaml = OmegaConf.load(repo_root / "configs" / "recall" / "merge.yaml")
        full = OmegaConf.create({"recall": merge_yaml, "paths": cfg.paths, "data": cfg.data})

        recaller = MergeRecaller(full)
        recaller.fit("")
        per_channel: dict[str, BaseRecaller] = dict(recaller._channels)  # type: ignore[attr-defined]

        log.info(
            "Building RecallFeatureStore: %d users × depth=%d × %d base channels + merge_rrf",
            len(user_ids), self.depth, len(per_channel),
        )

        # 1) Per-base-channel recall().
        for ch_name in self.channels:
            if ch_name == "merge_rrf":
                continue
            if ch_name not in per_channel:
                log.warning("Channel %s not loaded in MergeRecaller — skipping.", ch_name)
                continue
            log.info("Channel %s: recalling top-%d…", ch_name, self.depth)
            res = per_channel[ch_name].recall(user_ids.tolist(), k=self.depth)
            self._tables[ch_name] = _ChannelTable.from_result(res, self.n_users, self.depth)

        # 2) MergeRecaller itself — gives us the fused score.
        if "merge_rrf" in self.channels:
            log.info("Merge_rrf: recalling top-%d…", self.depth)
            merged_res = recaller.recall(user_ids.tolist(), k=self.depth)
            # Override channel name for clarity in serialisation.
            merged_res = RecallResult(
                user_ids=merged_res.user_ids,
                item_ids=merged_res.item_ids,
                scores=merged_res.scores,
                channel="merge_rrf",
            )
            self._tables["merge_rrf"] = _ChannelTable.from_result(
                merged_res, self.n_users, self.depth
            )

        # Normalise scores per channel so every column lives on a comparable
        # scale.  We z-score using statistics computed over the *present*
        # entries only (absent items stay at 0 — both score and mask).  This
        # frozen normalisation is what every consumer (LR / GBDT / DeepFM /
        # DIN) sees at both train and inference time, which removes the
        # batch-dependent standardisation foot-gun.
        for ch_name, table in self._tables.items():
            table.sorted_scores = self._normalise(table.sorted_items, table.sorted_scores)
        return self

    @staticmethod
    def _normalise(items: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Per-channel z-score; clip extreme tails to ±5 σ for stability."""
        present_mask = items >= 0
        present_scores = scores[present_mask]
        if present_scores.size == 0:
            return np.zeros_like(scores, dtype=np.float32)
        # Some channels (cosine, dot-product) can return inf/nan for cold items.
        present_scores = np.nan_to_num(present_scores, nan=0.0, posinf=0.0, neginf=0.0)
        mu = float(present_scores.mean())
        sigma = float(present_scores.std()) + 1e-6
        normed = np.where(
            present_mask,
            np.clip((np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0) - mu) / sigma, -5.0, 5.0),
            0.0,
        ).astype(np.float32)
        return normed

    # ------------------------------------------------------------------
    # Top-K by score — needed for hard-negative sampling at training time.
    # ------------------------------------------------------------------
    def top_items_by_score(
        self,
        user_ids: np.ndarray | None = None,
        channel: str = "merge_rrf",
        k: int | None = None,
    ) -> np.ndarray:
        """Return per-user item ids ordered by **descending** channel score.

        Shape ``(N_users, k)``; padded with -1 if the channel returned fewer
        than ``k`` items for a user.  This is the natural source of *hard*
        negatives for ranker training: items the recall layer ranked highly
        but that aren't the user's hold-out positive.
        """
        if channel not in self._tables:
            raise KeyError(f"channel {channel!r} not in store ({list(self._tables)})")
        table = self._tables[channel]
        if user_ids is None:
            user_ids = np.arange(self.n_users, dtype=np.int64)
        user_ids = np.asarray(user_ids, dtype=np.int64)
        k = int(k or self.depth)

        rows_items = table.sorted_items[user_ids]   # (N, D)  ascending by item id
        rows_scores = table.sorted_scores[user_ids]  # (N, D)  z-scored (or 0 if absent)
        # `sorted_scores` is 0 for absent items, so argsort descending naturally
        # pushes them to the back.
        order = np.argsort(-rows_scores, axis=1)
        top_items = np.take_along_axis(rows_items, order, axis=1)[:, :k]
        return top_items.astype(np.int32)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def lookup_batch(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        """Return a ``(N, 2 × n_channels)`` float32 matrix:
        ``[score_ch1, …, score_chC, mask_ch1, …, mask_chC]``.

        Missing channels (e.g. cold_start not loaded) contribute 0-columns.
        """
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        N = len(user_ids)
        C = len(self.channels)

        scores = np.zeros((N, C), dtype=np.float32)
        masks = np.zeros((N, C), dtype=np.float32)
        for c, ch in enumerate(self.channels):
            if ch not in self._tables:
                continue
            s, m = self._tables[ch].lookup_batch(user_ids, item_ids)
            scores[:, c] = s
            masks[:, c] = m
        return np.concatenate([scores, masks], axis=1)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        for ch, table in self._tables.items():
            payload[f"{ch}__items"] = table.sorted_items
            payload[f"{ch}__scores"] = table.sorted_scores
        payload["__meta_channels__"] = np.asarray(self.channels)
        payload["__meta_n_users__"] = np.asarray([self.n_users], dtype=np.int64)
        payload["__meta_n_items__"] = np.asarray([self.n_items], dtype=np.int64)
        payload["__meta_depth__"] = np.asarray([self.depth], dtype=np.int64)
        np.savez_compressed(path, **payload)
        log.info("Saved RecallFeatureStore → %s (%d channels, depth=%d)",
                 path, len(self._tables), self.depth)

    @classmethod
    def load(cls, path: str | Path) -> "RecallFeatureStore":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        channels = tuple(data["__meta_channels__"].tolist())
        store = cls(
            n_users=int(data["__meta_n_users__"][0]),
            n_items=int(data["__meta_n_items__"][0]),
            channels=channels,
            depth=int(data["__meta_depth__"][0]),
        )
        for ch in channels:
            items_key = f"{ch}__items"
            scores_key = f"{ch}__scores"
            if items_key in data and scores_key in data:
                store._tables[ch] = _ChannelTable(
                    sorted_items=data[items_key],
                    sorted_scores=data[scores_key],
                )
        log.info("Loaded RecallFeatureStore from %s (%d channels)", path, len(store._tables))
        return store


__all__ = ["RecallFeatureStore", "DEFAULT_CHANNELS"]
