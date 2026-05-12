"""Unified training + evaluation entry point for ranking models.

Reads ``cfg.rank.name`` and dispatches to one of:
    lr, gbdt, deepfm, din

Pipeline
--------
1. Load processed parquet artefacts + the shared :class:`RankingFeaturizer`.
2. Build (user, item, label) training pairs with 1:N random negatives.
3. Split into train / valid (last 10% by timestamp).
4. Fit the model; track AUC + LogLoss on valid.
5. **End-to-end evaluation**: load merge channel's per-user candidate pool
   (top-N from RRF), re-score with the trained model, compute
   Recall@K / NDCG@K / MRR@K against the held-out test positives.
6. Log everything (params, metrics, artefacts) to MLflow.
"""

from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import log_loss, roc_auc_score

from neorec.eval.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from neorec.ranking.features import RankingFeaturizer, build_training_pairs
from neorec.ranking.recall_features import DEFAULT_CHANNELS, RecallFeatureStore
from neorec.utils.io import ensure_dir
from neorec.utils.mlflow_utils import mlflow_run
from neorec.utils.timer import Timer

log = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "lr":              "neorec.ranking.lr:LRRanker",
    "gbdt":            "neorec.ranking.gbdt:GBDTRanker",
    "deepfm":          "neorec.ranking.deepfm:DeepFMRanker",
    "din":             "neorec.ranking.din:DINRanker",
    "transformer_ctr": "neorec.ranking.transformer_ctr:TransformerCTRRanker",
}


def _instantiate(name: str, cfg: DictConfig, featurizer: RankingFeaturizer):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown ranker: {name}. Known: {list(_REGISTRY)}")
    mod_path, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(cfg, featurizer)


# ===========================================================================
# Data preparation
# ===========================================================================
def _load_data(cfg: DictConfig):
    processed = Path(cfg.paths.data_processed) / cfg.data.name
    interactions = pd.read_parquet(processed / "interactions.parquet")
    split = pd.read_parquet(processed / "split.parquet")
    train_df = (
        interactions.merge(
            split[["user_id", "item_id", "split"]],
            on=["user_id", "item_id"],
            how="inner",
        )
        .query("split == 'train'")
        .reset_index(drop=True)
    )
    test_df = split.query("split == 'test'").reset_index(drop=True)
    return processed, train_df, test_df


def _build_user_seen(train_df: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(train_df["user_id"].to_numpy(), train_df["item_id"].to_numpy()):
        seen[int(u)].add(int(i))
    return seen


def _time_split_valid(pairs: pd.DataFrame, valid_ratio: float = 0.1, seed: int = 42):
    """Random valid split — pairs are already shuffled inside ``build_training_pairs``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n_valid = int(len(pairs) * valid_ratio)
    valid_idx = perm[:n_valid]
    train_idx = perm[n_valid:]
    return pairs.iloc[train_idx].reset_index(drop=True), pairs.iloc[valid_idx].reset_index(drop=True)


# ===========================================================================
# Candidate pool — call the saved MergeRecaller on demand (matches serving).
# ===========================================================================
def _build_merge_candidates(
    cfg: DictConfig,
    test_user_ids: list[int],
    pool_size: int,
) -> dict[int, list[int]]:
    """Re-instantiate MergeRecaller using a synthesised cfg with merge defaults.

    We can't reuse the original ``cfg.recall`` here because ranking jobs are
    invoked with ``rank=…`` not ``recall=…``. So we compose a minimal recall cfg
    in-memory and let MergeRecaller load all base-channel artefacts on its own.
    """
    # Compose a recall=merge config in-memory and call fit() to load all channels.
    from neorec.recall.merge import MergeRecaller
    repo_root = Path(__file__).resolve().parents[3]
    merge_yaml = OmegaConf.load(repo_root / "configs" / "recall" / "merge.yaml")
    full = OmegaConf.create({"recall": merge_yaml, "paths": cfg.paths, "data": cfg.data})

    log.info("Instantiating MergeRecaller to generate %d candidates per user…", pool_size)
    recaller = MergeRecaller(full)
    recaller.fit("")  # loads all per-channel artefacts
    result = recaller.recall(test_user_ids, k=pool_size)
    out: dict[int, list[int]] = {}
    for u, row in zip(result.user_ids.tolist(), result.item_ids.tolist(), strict=True):
        out[int(u)] = [int(x) for x in row[:pool_size] if x >= 0]
    return out


# ===========================================================================
# Evaluation
# ===========================================================================
def _eval_classification(model, valid_pairs: pd.DataFrame) -> dict[str, float]:
    """AUC + LogLoss + accuracy on the held-out training pairs."""
    users = valid_pairs["user_id"].to_numpy(dtype=np.int64)
    items = valid_pairs["item_id"].to_numpy(dtype=np.int64)
    y_true = valid_pairs["label"].to_numpy(dtype=np.int8)
    y_score = model.score(users, items)
    y_score = np.clip(y_score, 1e-7, 1 - 1e-7)
    return {
        "valid_auc":      float(roc_auc_score(y_true, y_score)),
        "valid_logloss":  float(log_loss(y_true, y_score)),
        "valid_accuracy": float(((y_score > 0.5) == y_true).mean()),
        "valid_pos_rate": float(y_true.mean()),
    }


def _eval_end_to_end(
    model,
    test_df: pd.DataFrame,
    candidates: dict[int, list[int]],
    k_list: list[int],
    top_k_after: int,
) -> dict[str, float]:
    """Re-rank candidates with the model and compute Recall/NDCG/MRR@K."""
    user_ids = test_df["user_id"].tolist()
    truths = test_df["item_id"].tolist()

    keep = [u in candidates for u in user_ids]
    user_ids_used = [u for u, k in zip(user_ids, keep) if k]
    truths_used = [t for t, k in zip(truths, keep) if k]
    cand_lists = [candidates[u] for u in user_ids_used]

    if not user_ids_used:
        return {f"recall@{k}": 0.0 for k in k_list}

    with Timer("rank.predict") as t:
        result = model.predict(user_ids_used, cand_lists, k=top_k_after)
    log.info("End-to-end re-rank: %.1f ms (%.2f ms/user)",
             t.elapsed_ms, t.elapsed_ms / max(len(user_ids_used), 1))

    y_pred = result.item_ids.tolist()
    y_true = [[t] for t in truths_used]
    metrics: dict[str, float] = {}
    for k in k_list:
        metrics[f"recall@{k}"]   = recall_at_k(y_true, y_pred, k=k)
        metrics[f"ndcg@{k}"]     = ndcg_at_k(y_true, y_pred, k=k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(y_true, y_pred, k=k)
        metrics[f"mrr@{k}"]      = mean_reciprocal_rank(y_true, y_pred, k=k)
    metrics["rerank_latency_ms_per_user"] = t.elapsed_ms / max(len(user_ids_used), 1)
    metrics["test_users_evaluated"] = float(len(user_ids_used))
    return metrics


# ===========================================================================
# Main entry
# ===========================================================================
def run(cfg: DictConfig) -> dict[str, float]:
    name = str(cfg.rank.name)
    log.info("=== Training ranking model: %s ===", name)

    processed, train_df, test_df = _load_data(cfg)
    artifacts_root = Path(cfg.paths.artifacts)

    user_seen = _build_user_seen(train_df)

    featurizer = RankingFeaturizer(
        processed_dir=processed,
        max_genres=int(cfg.rank.input.get("max_genres", 6)),
        max_seq_len=int(cfg.rank.input.get("max_seq_len", 50)),
    )

    # Most models don't need sequences, but DIN does. Build once, use if needed.
    featurizer.build_sequences(train_df)

    # -- Recall-score store (Scheme A + hard negatives) -----------------------
    # The :class:`RecallFeatureStore` is needed if **either** the ranker wants
    # recall-score inputs (``use_recall_features``) **or** training wants
    # hard-negative mining (``hard_negative_ratio>0``).  We build/load once and
    # only attach to the featurizer when the ranker actually consumes them.
    use_recall_features = bool(cfg.rank.input.get("use_recall_features", True))
    hard_neg_ratio = int(cfg.rank.input.get("hard_negative_ratio", 0))
    need_store = use_recall_features or hard_neg_ratio > 0
    recall_store: RecallFeatureStore | None = None
    if need_store:
        depth = int(cfg.rank.input.get("recall_feature_depth", 500))
        rfs_path = (
            artifacts_root / "rank" / "recall_features" /
            f"recall_features_d{depth}.npz"
        )
        if rfs_path.exists():
            log.info("Loading pre-computed RecallFeatureStore from %s", rfs_path)
            recall_store = RecallFeatureStore.load(rfs_path)
        else:
            log.info("Building RecallFeatureStore (depth=%d) — first run is ~30-60 s…", depth)
            recall_store = RecallFeatureStore(
                n_users=featurizer.schema.num_users,
                n_items=featurizer.schema.num_items,
                channels=DEFAULT_CHANNELS,
                depth=depth,
            ).build(cfg)
            recall_store.save(rfs_path)

    if use_recall_features and recall_store is not None:
        featurizer.recall_store = recall_store
        log.info(
            "Recall features enabled: %d columns = %d channels × (score + mask)",
            recall_store.n_features, len(recall_store.channels),
        )
    else:
        log.info("Recall-score features as ranker input: DISABLED.")

    # Build (user, item, label) pairs.
    #
    # Mixed negatives (Scheme A companion): without hard negatives, every
    # ranker that consumes recall-score features collapses to the trivial
    # "is this in the recall pool?" decision, because inference candidates
    # are 100 % drawn from the pool. Mixed sampling repairs the distribution
    # gap.
    neg_ratio = int(cfg.rank.input.get("negative_ratio", 4))
    if hard_neg_ratio > 0 and recall_store is None:
        log.warning(
            "hard_negative_ratio>0 but recall_store unavailable — falling back to random only."
        )
        hard_neg_ratio = 0

    hard_candidates: np.ndarray | None = None
    if hard_neg_ratio > 0 and recall_store is not None:
        hard_pool = int(cfg.rank.input.get("hard_pool_size", 200))
        log.info(
            "Building hard-negative pool (merge_rrf top-%d) for %d users…",
            hard_pool, featurizer.schema.num_users,
        )
        hard_candidates = recall_store.top_items_by_score(
            user_ids=np.arange(featurizer.schema.num_users, dtype=np.int64),
            channel="merge_rrf",
            k=hard_pool,
        )

    log.info(
        "Building %d positives + %d:1 random + %d:1 hard negatives…",
        len(train_df), neg_ratio, hard_neg_ratio,
    )
    pairs = build_training_pairs(
        train_df=train_df,
        num_items=featurizer.schema.num_items,
        user_seen=user_seen,
        negative_ratio=neg_ratio,
        seed=int(cfg.get("seed", 42)),
        hard_candidates=hard_candidates,
        hard_negative_ratio=hard_neg_ratio,
    )
    valid_ratio = float(cfg.rank.input.get("valid_ratio", 0.1))
    train_pairs, valid_pairs = _time_split_valid(
        pairs, valid_ratio=valid_ratio, seed=int(cfg.get("seed", 42))
    )
    log.info("Train pairs=%d, valid pairs=%d", len(train_pairs), len(valid_pairs))

    # Instantiate, fit, log.
    model = _instantiate(name, cfg, featurizer)

    with Timer("rank.fit") as t_fit:
        train_metrics = model.fit(train_pairs, valid_pairs)
    log.info("Fit: %.2f s, train metrics: %s",
             t_fit.elapsed_ms / 1000.0, train_metrics)

    cls_metrics = _eval_classification(model, valid_pairs)
    log.info("Classification metrics: %s", cls_metrics)

    # End-to-end: rerank merge channel's candidates and compute Recall@K.
    pool_size = int(cfg.rank.input.get("candidate_pool_size", 1000))
    top_k_after = int(cfg.rank.output.get("top_k_after_rerank", 100))
    k_list = list(cfg.rank.eval.get("k_list", [10, 50, 100]))
    log.info("Loading merge candidate pool (pool_size=%d, top_k_after=%d)…",
             pool_size, top_k_after)
    candidates = _build_merge_candidates(
        cfg=cfg,
        test_user_ids=test_df["user_id"].tolist(),
        pool_size=pool_size,
    )
    ranking_metrics = _eval_end_to_end(
        model=model,
        test_df=test_df,
        candidates=candidates,
        k_list=k_list,
        top_k_after=top_k_after,
    )
    log.info("End-to-end metrics: %s", ranking_metrics)

    # Save artefacts
    out_dir = ensure_dir(artifacts_root / "rank" / name)
    model.save(out_dir)

    # MLflow
    all_metrics = {**train_metrics, **cls_metrics, **ranking_metrics,
                   "fit_seconds": t_fit.elapsed_ms / 1000.0}
    flat_params: dict[str, object] = {
        "model":             name,
        "stage":             str(cfg.rank.stage),
        "dataset":           str(cfg.data.name),
        "negative_ratio":    neg_ratio,
        "candidate_pool":    pool_size,
        "top_k_after":       top_k_after,
    }
    model_cfg = OmegaConf.to_container(cfg.rank.model, resolve=True)  # type: ignore[union-attr]
    if isinstance(model_cfg, dict):
        for k, v in model_cfg.items():
            flat_params[f"model.{k}"] = (
                str(v) if isinstance(v, (list, tuple, dict)) else v
            )

    mlflow_metrics = {k.replace("@", "_at_"): v for k, v in all_metrics.items()}
    with mlflow_run(
        experiment=cfg.mlflow.experiment_name,
        run_name=f"rank.{name}",
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={"stage": "rank", "model": name, "dataset": cfg.data.name},
    ) as mlf:
        mlf.log_params(flat_params)
        mlf.log_metrics(mlflow_metrics)

    log.info("=== Done. Artefacts: %s ===", out_dir)
    return all_metrics
