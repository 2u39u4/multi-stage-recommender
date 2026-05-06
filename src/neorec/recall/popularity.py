"""Popularity recall — heuristic baseline + cold-start fallback.

Two variants controlled by ``cfg.recall.model.time_decay``:

* **raw count** — ``score_i = |{u : (u,i) ∈ train}|``
* **time-decayed count** — half-life decay weighted by interaction age:
  ``score_i = Σ exp(-ln(2) · age_days / half_life)``

The same global ranking is returned for every user, with the user's already-seen
items filtered out (so two users who have watched different things will see
slightly different lists). New / unknown users transparently get the global
top-K — that is the cold-start fallback.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from neorec.recall.base import BaseRecaller, RecallResult
from neorec.utils.io import ensure_dir, read_json, write_json

log = logging.getLogger(__name__)


class PopularityRecaller(BaseRecaller):
    name = "popularity"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._sorted_items: np.ndarray | None = None  # (num_items,) item_ids sorted desc by score
        self._sorted_scores: np.ndarray | None = None
        self._user_seen: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, interactions_path: str | Path) -> None:
        df = pd.read_parquet(interactions_path)
        log.info("Loaded interactions: %s rows", f"{len(df):,}")

        if "split" in df.columns:
            df = df[df["split"] == "train"]

        time_decay = bool(self.cfg.recall.model.time_decay)
        if time_decay:
            half_life_days = float(self.cfg.recall.model.decay_half_life_days)
            now_ts = int(df["ts"].max())
            age_days = (now_ts - df["ts"].to_numpy()) / 86_400.0
            weights = np.exp(-math.log(2.0) * age_days / half_life_days)
            scored = (
                pd.Series(weights, index=df["item_id"].to_numpy())
                .groupby(level=0).sum()
            )
            log.info("Using time-decay popularity (half-life=%.1f days)", half_life_days)
        else:
            scored = df["item_id"].value_counts()
            log.info("Using raw count popularity")

        scored = scored.sort_values(ascending=False)
        self._sorted_items = scored.index.to_numpy(dtype=np.int32)
        self._sorted_scores = scored.to_numpy(dtype=np.float32)

        # Build per-user "seen" set so we can filter at recall() time
        self._user_seen = (
            df.groupby("user_id")["item_id"]
            .agg(lambda s: set(s.tolist()))
            .to_dict()
        )
        log.info(
            "Top-5 popular items: %s",
            list(zip(self._sorted_items[:5], self._sorted_scores[:5].round(1))),
        )

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(self, user_ids: Sequence[int], k: int) -> RecallResult:
        assert self._sorted_items is not None, "fit() first"

        users = np.asarray(list(user_ids), dtype=np.int32)
        out_items = np.full((len(users), k), -1, dtype=np.int32)
        out_scores = np.zeros((len(users), k), dtype=np.float32)

        # If we'd need more than the catalog size, just take everything we have
        candidate_ceiling = min(len(self._sorted_items), k * 4 + 200)
        cand_items = self._sorted_items[:candidate_ceiling]
        cand_scores = self._sorted_scores[:candidate_ceiling]

        for row, uid in enumerate(users):
            seen = self._user_seen.get(int(uid), set())
            if not seen:
                # cold-start: just take global top-K
                take = min(k, len(self._sorted_items))
                out_items[row, :take] = self._sorted_items[:take]
                out_scores[row, :take] = self._sorted_scores[:take]
                continue
            mask = ~np.isin(cand_items, list(seen))
            kept_items = cand_items[mask][:k]
            kept_scores = cand_scores[mask][:k]
            n = len(kept_items)
            out_items[row, :n] = kept_items
            out_scores[row, :n] = kept_scores

        return RecallResult(
            user_ids=users, item_ids=out_items, scores=out_scores, channel=self.name
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        assert self._sorted_items is not None and self._sorted_scores is not None
        out_dir = ensure_dir(path)
        np.save(out_dir / "sorted_items.npy", self._sorted_items)
        np.save(out_dir / "sorted_scores.npy", self._sorted_scores)
        # Persist seen sets as a flat parquet (much smaller than json for many users)
        seen_rows = [
            {"user_id": int(u), "item_id": int(i)}
            for u, items in self._user_seen.items()
            for i in items
        ]
        pd.DataFrame(seen_rows).to_parquet(out_dir / "user_seen.parquet", index=False)
        write_json(
            {
                "time_decay": bool(self.cfg.recall.model.time_decay),
                "num_items": int(len(self._sorted_items)),
            },
            out_dir / "meta.json",
        )
        log.info("Saved popularity artefacts to %s", out_dir)

    def load(self, path: str | Path) -> None:
        in_dir = Path(path)
        _ = read_json(in_dir / "meta.json")
        self._sorted_items = np.load(in_dir / "sorted_items.npy")
        self._sorted_scores = np.load(in_dir / "sorted_scores.npy")
        df = pd.read_parquet(in_dir / "user_seen.parquet")
        self._user_seen = (
            df.groupby("user_id")["item_id"].agg(lambda s: set(s.tolist())).to_dict()
        )
        log.info("Loaded popularity artefacts from %s", in_dir)
