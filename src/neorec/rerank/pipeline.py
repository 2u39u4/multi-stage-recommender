"""Rerank orchestrator — recall → rank → rerank end-to-end evaluation.

Pipeline
--------
1. Load the trained recall ``MergeRecaller`` and produce per-user
   ``candidate_pool_size`` candidates.
2. Load the trained ranker (LR / GBDT / DeepFM / DIN) and re-score those
   candidates, keeping the top ``top_k_after_rank``.
3. Apply optional rerank passes:
   * **MMR** for diversity, using L2-normalised Two-Tower item vectors.
   * **IPS** debias, using interaction counts from the training data.
   * **Business rules**: filter watched, genre quota, year-bucket cap.
4. Compute Recall@K / NDCG@K / MRR + diversity (intra-list similarity, ILS)
   + coverage @K. Log everything to MLflow.

The whole thing is driven by ``configs/rerank/mmr.yaml`` and a *base ranker*
config (``rank=din`` by default). Outputs land in
``artifacts/rerank/{ranker}_{strategy}/`` for downstream notebooks.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neorec.eval.metrics import (
    coverage,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from neorec.ranking.features import RankingFeaturizer
from neorec.ranking.train import _build_merge_candidates, _instantiate, _load_data
from neorec.rerank.debias import ips_rerank
from neorec.rerank.mmr import intra_list_similarity, mmr_rerank
from neorec.rerank.rules import apply_rules
from neorec.utils.io import ensure_dir, write_json
from neorec.utils.mlflow_utils import mlflow_run

log = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================
def _load_item_embeddings(cfg: DictConfig, n_items: int) -> np.ndarray:
    """Load Two-Tower item vectors and L2-normalise them.

    Falls back to ALS factors (then to one-hot zeros) when Two-Tower
    artefacts aren't available — MMR still works but with cruder similarity.
    """
    oof = bool(cfg.data.get("oof_split", False))
    sub = "recall_oof" if oof else "recall"
    root = Path(cfg.paths.artifacts) / sub

    for fname in ("two_tower/item_vecs.npy", "als/item_factors.npy"):
        path = root / fname
        if path.exists():
            vecs = np.load(path).astype(np.float32)
            if vecs.shape[0] >= n_items:
                vecs = vecs[:n_items]
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.clip(norms, 1e-8, None)
                log.info("Loaded item embeddings from %s (%s)", path, vecs.shape)
                return vecs
    log.warning(
        "No item embeddings found under %s — falling back to identity. "
        "MMR will degrade to pure relevance ranking.", root,
    )
    return np.eye(n_items, dtype=np.float32)


def _item_popularity(train_df: pd.DataFrame) -> dict[int, float]:
    counts = train_df["item_id"].value_counts()
    return {int(k): float(v) for k, v in counts.items()}


def _item_meta_lookup(processed_dir: Path) -> dict[int, dict[str, object]]:
    items = pd.read_parquet(processed_dir / "item_features.parquet")
    meta: dict[int, dict[str, object]] = {}
    for row in items.itertuples(index=False):
        meta[int(row.item_id)] = {
            "genres": list(row.genres),
            "year_bucket": int(row.year_bucket),
        }
    return meta


# ===========================================================================
# Stage-3 rerank passes
# ===========================================================================
def _apply_rerank_stack(
    cands: list[int],
    scores: list[float],
    *,
    cfg: DictConfig,
    item_embeddings: np.ndarray,
    item_popularity: dict[int, float],
    item_meta: dict[int, dict[str, object]],
    user_history: set[int],
    k_final: int,
) -> tuple[list[int], list[float]]:
    """Run the configured passes (mmr / debias / rules) over one user's
    candidate list and return (top-k items, aligned scores).
    """
    strategy = str(cfg.rerank.strategy).lower()
    score_lookup = dict(zip(cands, scores, strict=True))

    if strategy == "mmr":
        lam = float(cfg.rerank.mmr["lambda"])
        ordered = mmr_rerank(
            candidate_ids=cands,
            candidate_scores=scores,
            item_embeddings=item_embeddings,
            k=len(cands),
            lam=lam,
        )
    elif strategy == "debias":
        ordered = ips_rerank(
            candidate_ids=cands,
            candidate_scores=scores,
            item_popularity=item_popularity,
            clip=(
                float(cfg.rerank.debias.clip_min),
                float(cfg.rerank.debias.clip_max),
            ),
        )
    elif strategy == "composite":
        # MMR first → IPS over MMR-ordered list → rules.
        lam = float(cfg.rerank.mmr["lambda"])
        ordered = mmr_rerank(
            candidate_ids=cands,
            candidate_scores=scores,
            item_embeddings=item_embeddings,
            k=len(cands),
            lam=lam,
        )
        if bool(cfg.rerank.debias.enabled):
            ordered = ips_rerank(
                candidate_ids=ordered,
                candidate_scores=[score_lookup[i] for i in ordered],
                item_popularity=item_popularity,
                clip=(
                    float(cfg.rerank.debias.clip_min),
                    float(cfg.rerank.debias.clip_max),
                ),
            )
    elif strategy == "rules":
        ordered = list(cands)
    elif strategy == "none":
        ordered = list(cands)
    else:
        raise ValueError(f"Unknown rerank strategy: {strategy!r}")

    if cfg.rerank.rules:
        ordered = apply_rules(
            candidate_ids=ordered,
            user_history=user_history,
            item_meta=item_meta,
            max_per_genre_ratio=float(cfg.rerank.rules.max_per_genre_ratio),
            max_per_year_bucket=int(cfg.rerank.rules.max_per_year_bucket),
            filter_already_watched=bool(cfg.rerank.rules.filter_already_watched),
            k=k_final,
        )
    else:
        ordered = ordered[:k_final]

    final_scores = [float(score_lookup.get(i, 0.0)) for i in ordered]
    return ordered, final_scores


# ===========================================================================
# Entry point
# ===========================================================================
def run(cfg: DictConfig) -> dict[str, float]:
    """Recall → rank → rerank end-to-end evaluation.

    The base ranker is whatever ``cfg.rank`` resolves to (``rank=din`` by
    default). We *load* the previously-trained ranker artefact rather than
    re-training, so the rerank pass is cheap (~30 s on ML-1M CPU).
    """
    ranker_name = str(cfg.rank.name)
    oof = bool(cfg.data.get("oof_split", False))
    strategy = str(cfg.rerank.strategy)
    k_final = int(cfg.rerank.output.top_k_final)
    log.info(
        "=== Rerank pipeline: ranker=%s, strategy=%s, oof=%s, k_final=%d ===",
        ranker_name, strategy, oof, k_final,
    )

    # ---- Load data --------------------------------------------------------
    processed, train_history_df, ranker_positives_df, test_df = _load_data(cfg)
    artifacts_root = Path(cfg.paths.artifacts)
    rank_subdir = "rank_oof" if oof else "rank"
    ranker_dir = artifacts_root / rank_subdir / ranker_name
    if not ranker_dir.exists():
        raise FileNotFoundError(
            f"Ranker artefacts not found at {ranker_dir}. "
            f"Run `neorec train rank rank={ranker_name}` first."
        )

    featurizer = RankingFeaturizer(
        processed_dir=processed,
        max_genres=int(cfg.rank.input.get("max_genres", 6)),
        max_seq_len=int(cfg.rank.input.get("max_seq_len", 50)),
    )
    featurizer.build_sequences(train_history_df)

    # ---- Build candidate pool from MergeRecaller --------------------------
    pool_size = int(cfg.rank.input.get("candidate_pool_size", 1000))
    extra_seen: dict[int, set[int]] | None = None
    if oof:
        seen: dict[int, set[int]] = defaultdict(set)
        for u, i in zip(
            ranker_positives_df["user_id"].to_numpy(),
            ranker_positives_df["item_id"].to_numpy(),
        ):
            seen[int(u)].add(int(i))
        extra_seen = seen

    test_users = test_df["user_id"].tolist()
    candidates = _build_merge_candidates(
        cfg=cfg,
        test_user_ids=test_users,
        pool_size=pool_size,
        extra_user_seen=extra_seen,
    )

    # ---- Load ranker and score candidates → top-K ------------------------
    model = _instantiate(ranker_name, cfg, featurizer)
    model.load(ranker_dir)
    log.info("Loaded %s ranker from %s", ranker_name, ranker_dir)

    top_k_after = int(cfg.rank.output.get("top_k_after_rerank", 100))
    user_ids_used: list[int] = []
    cand_lists: list[list[int]] = []
    truths: list[int] = []
    for u, gt in zip(test_df["user_id"].tolist(), test_df["item_id"].tolist()):
        if u in candidates and candidates[u]:
            user_ids_used.append(int(u))
            cand_lists.append(list(candidates[u]))
            truths.append(int(gt))
    log.info(
        "Scoring %d users × %d candidates with %s …",
        len(user_ids_used), pool_size, ranker_name,
    )
    t0 = time.perf_counter()
    rank_result = model.predict(user_ids_used, cand_lists, k=top_k_after)
    rank_elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "Ranker scored: %.0f ms (%.2f ms/user)",
        rank_elapsed_ms, rank_elapsed_ms / max(len(user_ids_used), 1),
    )

    # ---- Re-rank pass -----------------------------------------------------
    item_emb = _load_item_embeddings(cfg, n_items=featurizer.schema.num_items)
    item_pop = _item_popularity(train_history_df)
    item_meta = _item_meta_lookup(processed)
    user_history = defaultdict(set)
    for u, i in zip(
        train_history_df["user_id"].to_numpy(),
        train_history_df["item_id"].to_numpy(),
    ):
        user_history[int(u)].add(int(i))

    log.info("Rerank stack: strategy=%s, λ=%s, debias=%s, rules=%s",
             strategy,
             cfg.rerank.mmr["lambda"],
             cfg.rerank.debias.enabled,
             cfg.rerank.rules is not None)

    rerank_items: list[list[int]] = []
    rerank_scores: list[list[float]] = []
    t0 = time.perf_counter()
    for row_idx, u in enumerate(user_ids_used):
        cands_row = rank_result.item_ids[row_idx].tolist()
        scores_row = rank_result.scores[row_idx].tolist()
        # Drop the zero-padding slots that ``BaseRanker.predict`` left behind.
        valid = [(c, s) for c, s in zip(cands_row, scores_row, strict=True) if c >= 0]
        if not valid:
            rerank_items.append([])
            rerank_scores.append([])
            continue
        cands_clean, scores_clean = map(list, zip(*valid, strict=True))
        picked, picked_scores = _apply_rerank_stack(
            cands=cands_clean,
            scores=scores_clean,
            cfg=cfg,
            item_embeddings=item_emb,
            item_popularity=item_pop,
            item_meta=item_meta,
            user_history=user_history.get(u, set()),
            k_final=k_final,
        )
        rerank_items.append(picked)
        rerank_scores.append(picked_scores)
    rerank_elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "Rerank pass: %.0f ms (%.2f ms/user)",
        rerank_elapsed_ms, rerank_elapsed_ms / max(len(user_ids_used), 1),
    )

    # ---- Metrics ----------------------------------------------------------
    y_true = [[t] for t in truths]
    y_pred = rerank_items
    k_list = list(cfg.rerank.get("eval", {}).get("k_list", [1, 5, 10]))
    if k_final not in k_list:
        k_list.append(k_final)
    k_list = sorted(set(int(k) for k in k_list))
    metrics: dict[str, float] = {}
    for k in k_list:
        metrics[f"recall@{k}"]   = recall_at_k(y_true, y_pred, k=k)
        metrics[f"ndcg@{k}"]     = ndcg_at_k(y_true, y_pred, k=k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(y_true, y_pred, k=k)
        metrics[f"mrr@{k}"]      = mean_reciprocal_rank(y_true, y_pred, k=k)
        metrics[f"coverage@{k}"] = coverage(y_pred, catalog_size=featurizer.schema.num_items, k=k)

    # Intra-list similarity (lower = more diverse) — only at the final cutoff
    ils_vals = [intra_list_similarity(ids, item_emb) for ids in y_pred if ids]
    metrics[f"ils@{k_final}"] = float(np.mean(ils_vals)) if ils_vals else 0.0
    metrics["rank_latency_ms_per_user"]   = rank_elapsed_ms / max(len(user_ids_used), 1)
    metrics["rerank_latency_ms_per_user"] = rerank_elapsed_ms / max(len(user_ids_used), 1)
    metrics["users_evaluated"]            = float(len(user_ids_used))

    log.info("Rerank metrics: %s", metrics)

    # ---- Persist + MLflow -------------------------------------------------
    out_dir = ensure_dir(artifacts_root / "rerank" / f"{ranker_name}_{strategy}")
    write_json(
        {
            "ranker": ranker_name,
            "strategy": strategy,
            "oof_split": oof,
            "k_final": k_final,
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg.rerank, resolve=True),
        },
        out_dir / "metrics.json",
    )
    # Dump per-user rerank lists for downstream notebooks (Sankey funnel etc.).
    pd.DataFrame({
        "user_id": user_ids_used,
        "truth":   truths,
        "rerank_topk": [json.dumps(x) for x in rerank_items],
    }).to_parquet(out_dir / "predictions.parquet", index=False)

    flat_params = {
        "stage":      "rerank",
        "ranker":     ranker_name,
        "strategy":   strategy,
        "oof_split":  oof,
        "k_final":    k_final,
        "mmr.lambda": float(cfg.rerank.mmr["lambda"]),
        "debias.enabled": bool(cfg.rerank.debias.enabled),
        "debias.clip":  f"[{cfg.rerank.debias.clip_min},{cfg.rerank.debias.clip_max}]",
        "rules.filter_watched":   bool(cfg.rerank.rules.filter_already_watched),
        "rules.max_per_genre":    float(cfg.rerank.rules.max_per_genre_ratio),
        "rules.max_per_year":     int(cfg.rerank.rules.max_per_year_bucket),
    }
    mlflow_metrics = {k.replace("@", "_at_"): v for k, v in metrics.items()}
    with mlflow_run(
        experiment=cfg.mlflow.experiment_name,
        run_name=f"rerank.{ranker_name}.{strategy}" + (".oof" if oof else ""),
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={
            "stage": "rerank",
            "ranker": ranker_name,
            "strategy": strategy,
            "dataset": cfg.data.name,
            "split_mode": "oof" if oof else "full",
        },
    ) as mlf:
        mlf.log_params(flat_params)
        mlf.log_metrics(mlflow_metrics)

    log.info("=== Done. Artefacts: %s ===", out_dir)
    return metrics
