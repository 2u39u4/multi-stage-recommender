"""Online inference pipeline: recall → pre-rank → fine-rank → rerank.

Usage (from api.py)::

    pipeline = OnlinePipeline.from_config(cfg)
    response = pipeline.recommend(user_id=123, k=10, diversity=0.3)

The pipeline exposes per-stage latency so the API response can advertise
``latency_ms.total`` and its breakdown.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neorec.ranking.features import RankingFeaturizer
from neorec.ranking.train import _instantiate, _load_data
from neorec.recall.merge import MergeRecaller
from neorec.rerank.mmr import mmr_rerank
from neorec.rerank.rules import apply_rules
from neorec.serving.feature_cache import RedisFeatureCache

log = logging.getLogger(__name__)


@dataclass
class RecommendationItem:
    item_id: int
    score: float
    channel: str
    title: str | None = None
    explain: str | None = None


@dataclass
class RecommendationResponse:
    user_id: int
    items: list[RecommendationItem]
    latency_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


class _Timer:
    def __enter__(self) -> "_Timer":
        self.t0 = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0


def _item_meta_lookup(processed_dir: Path) -> tuple[dict[int, dict[str, object]], dict[int, str]]:
    items = pd.read_parquet(processed_dir / "item_features.parquet")
    meta: dict[int, dict[str, object]] = {}
    titles: dict[int, str] = {}
    for row in items.itertuples(index=False):
        iid = int(row.item_id)
        meta[iid] = {
            "genres": list(row.genres),
            "year_bucket": int(row.year_bucket),
            "popularity_bucket": int(row.popularity_bucket),
        }
        titles[iid] = str(getattr(row, "title", f"item-{iid}"))
    return meta, titles


def _load_item_embeddings(cfg: DictConfig, n_items: int) -> np.ndarray:
    oof = bool(cfg.data.get("oof_split", False))
    sub = "recall_oof" if oof else "recall"
    root = Path(cfg.paths.artifacts) / sub
    for fname in ("two_tower/item_vecs.npy", "als/item_factors.npy"):
        path = root / fname
        if path.exists():
            vecs = np.load(path).astype(np.float32)
            vecs = vecs[:n_items]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.clip(norms, 1e-8, None)
    log.warning("No item embeddings found under %s; MMR falls back to identity.", root)
    return np.eye(n_items, dtype=np.float32)


def _build_seen(train_df: pd.DataFrame, extra_df: pd.DataFrame | None = None) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for df in (train_df, extra_df):
        if df is None:
            continue
        for u, i in zip(df["user_id"].to_numpy(), df["item_id"].to_numpy(), strict=True):
            seen[int(u)].add(int(i))
    return seen


class OnlinePipeline:
    """Thread-safe online inference orchestrator."""

    def __init__(
        self,
        cfg: Any,
        *,
        recallers: list[Any] | None = None,
        pre_ranker: Any | None = None,
        fine_ranker: Any | None = None,
        feature_cache: Any | None = None,
        featurizer: RankingFeaturizer | None = None,
        item_embeddings: np.ndarray | None = None,
        item_meta: dict[int, dict[str, object]] | None = None,
        item_titles: dict[int, str] | None = None,
        user_seen: dict[int, set[int]] | None = None,
        fallback_items: list[int] | None = None,
    ) -> None:
        self.cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg
        self.recallers = recallers or []
        self.pre_ranker = pre_ranker
        self.fine_ranker = fine_ranker
        self.feature_cache = feature_cache
        self.featurizer = featurizer
        self.item_embeddings = item_embeddings
        self.item_meta = item_meta or {}
        self.item_titles = item_titles or {}
        self.user_seen = user_seen or {}
        self.fallback_items = fallback_items or []

    @classmethod
    def from_config(cls, cfg: Any) -> "OnlinePipeline":
        """Hydrate all components from a Hydra config."""
        cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg
        oof = bool(cfg.data.get("oof_split", True))
        cfg.data.oof_split = oof
        processed, train_history_df, ranker_positives_df, _ = _load_data(cfg)
        featurizer = RankingFeaturizer(
            processed_dir=processed,
            max_genres=int(cfg.rank.input.get("max_genres", 6)),
            max_seq_len=int(cfg.rank.input.get("max_seq_len", 50)),
        )
        featurizer.build_sequences(train_history_df)

        item_meta, item_titles = _item_meta_lookup(processed)
        item_embeddings = _load_item_embeddings(cfg, featurizer.schema.num_items)

        # Merge recaller loads all trained recall channels. If artefacts are missing,
        # we keep the API alive and rely on popularity fallback.
        recaller = None
        try:
            merge_yaml = OmegaConf.load(
                Path(__file__).resolve().parents[3] / "configs" / "recall" / "merge.yaml"
            )
            recall_cfg = OmegaConf.create({"recall": merge_yaml, "paths": cfg.paths, "data": cfg.data})
            recaller = MergeRecaller(recall_cfg)
            recaller.fit("")
        except Exception:
            log.exception("MergeRecaller failed to hydrate; serving will use fallback items.")

        artifacts_root = Path(cfg.paths.artifacts)
        rank_subdir = "rank_oof" if oof else "rank"
        pre_ranker = None
        fine_ranker = None
        for name, attr in (("deepfm", "pre_ranker"), ("din", "fine_ranker")):
            try:
                base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                OmegaConf.set_struct(base_cfg, False)
                model_cfg = OmegaConf.merge(base_cfg, {"rank": OmegaConf.load(
                    Path(__file__).resolve().parents[3] / "configs" / "rank" / f"{name}.yaml"
                )})
                model = _instantiate(name, model_cfg, featurizer)
                model_dir = artifacts_root / rank_subdir / name
                model.load(model_dir)
                if attr == "pre_ranker":
                    pre_ranker = model
                else:
                    fine_ranker = model
            except Exception:
                log.exception("%s ranker failed to hydrate; serving will skip that stage.", name)

        cache = None
        if str(cfg.serving.cache.get("backend", "redis")).lower() == "redis":
            try:
                cache = RedisFeatureCache(
                    host=str(cfg.serving.cache.host),
                    port=int(cfg.serving.cache.port),
                    db=int(cfg.serving.cache.db),
                    namespace=str(cfg.serving.cache.namespace),
                    ttl_seconds=int(cfg.serving.cache.ttl_seconds),
                )
                cache._connect()
            except Exception:
                log.warning("Redis unavailable; continuing with in-process lookups.", exc_info=True)

        pop = (
            pd.read_parquet(processed / "item_features.parquet")
            .sort_values("popularity", ascending=False)["item_id"]
            .head(1000)
            .astype(int)
            .tolist()
        )
        seen = _build_seen(train_history_df, ranker_positives_df if oof else None)
        return cls(
            cfg=cfg,
            recallers=[recaller] if recaller is not None else [],
            pre_ranker=pre_ranker,
            fine_ranker=fine_ranker,
            feature_cache=cache,
            featurizer=featurizer,
            item_embeddings=item_embeddings,
            item_meta=item_meta,
            item_titles=item_titles,
            user_seen=seen,
            fallback_items=pop,
        )

    def _recall(self, user_id: int, k: int) -> tuple[list[int], list[float], str]:
        if self.recallers:
            result = self.recallers[0].recall([user_id], k=k)
            items = [int(i) for i in result.item_ids[0].tolist() if int(i) >= 0]
            scores = [float(s) for s, i in zip(result.scores[0].tolist(), result.item_ids[0].tolist()) if int(i) >= 0]
            return items, scores, str(result.channel)
        seen = self.user_seen.get(user_id, set())
        items = [i for i in self.fallback_items if i not in seen][:k]
        scores = [1.0 / (r + 1) for r in range(len(items))]
        return items, scores, "popularity_fallback"

    def _rank(
        self,
        ranker: Any | None,
        user_id: int,
        candidates: list[int],
        scores: list[float],
        k: int,
        channel: str,
    ) -> tuple[list[int], list[float], str]:
        if ranker is None or not candidates:
            return candidates[:k], scores[:k], f"{channel}->skip"
        result = ranker.predict([user_id], [candidates], k=min(k, len(candidates)))
        return (
            [int(i) for i in result.item_ids[0].tolist()],
            [float(s) for s in result.scores[0].tolist()],
            ranker.name,
        )

    def recommend(
        self, user_id: int, k: int = 10, diversity: float = 0.5
    ) -> RecommendationResponse:
        """Run the full funnel for one user and return top-K with latency breakdown."""
        if self.featurizer is not None and not (0 <= int(user_id) < self.featurizer.schema.num_users):
            raise KeyError(f"Unknown user_id={user_id}")

        latency: dict[str, float] = {}
        debug: dict[str, Any] = {}
        with _Timer() as t:
            cands, scores, recall_channel = self._recall(user_id, k=1000)
        latency["recall"] = t.elapsed_ms
        debug["recall_channel"] = recall_channel
        debug["recall_candidates"] = len(cands)

        with _Timer() as t:
            pre_items, pre_scores, pre_channel = self._rank(
                self.pre_ranker, user_id, cands, scores, k=100, channel=recall_channel
            )
        latency["pre_rank"] = t.elapsed_ms
        debug["pre_rank_channel"] = pre_channel

        with _Timer() as t:
            fine_items, fine_scores, fine_channel = self._rank(
                self.fine_ranker, user_id, pre_items, pre_scores, k=20, channel=pre_channel
            )
        latency["fine_rank"] = t.elapsed_ms
        debug["fine_rank_channel"] = fine_channel

        with _Timer() as t:
            if fine_items and self.item_embeddings is not None:
                ordered = mmr_rerank(
                    fine_items,
                    fine_scores,
                    self.item_embeddings,
                    k=len(fine_items),
                    lam=float(diversity),
                )
                score_lookup = dict(zip(fine_items, fine_scores, strict=True))
                ordered = apply_rules(
                    ordered,
                    user_history=self.user_seen.get(user_id, set()),
                    item_meta=self.item_meta,
                    max_per_genre_ratio=float(self.cfg.rerank.rules.max_per_genre_ratio),
                    max_per_year_bucket=int(self.cfg.rerank.rules.max_per_year_bucket),
                    filter_already_watched=bool(self.cfg.rerank.rules.filter_already_watched),
                    k=k,
                )
                final_scores = [float(score_lookup.get(i, 0.0)) for i in ordered]
            else:
                ordered, final_scores = fine_items[:k], fine_scores[:k]
        latency["rerank"] = t.elapsed_ms
        latency["total"] = sum(latency.values())

        items = [
            RecommendationItem(
                item_id=i,
                title=self.item_titles.get(i),
                score=s,
                channel=fine_channel,
                explain=(
                    f"recall={recall_channel}; pre_rank={pre_channel}; "
                    f"fine_rank={fine_channel}; MMR lambda={diversity:.2f}"
                ),
            )
            for i, s in zip(ordered, final_scores, strict=True)
        ]
        return RecommendationResponse(user_id=user_id, items=items, latency_ms=latency, debug=debug)
