"""Unified training entrypoint for recall channels.

Reads the channel name from ``cfg.recall.name``, instantiates the right
:class:`BaseRecaller`, fits on the train split, evaluates on test, and
logs everything to MLflow.
"""

from __future__ import annotations

import importlib
import logging
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
from neorec.utils.io import ensure_dir
from neorec.utils.mlflow_utils import mlflow_run
from neorec.utils.timer import Timer

log = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "als":        "neorec.recall.als:ALSRecaller",
    "two_tower":  "neorec.recall.two_tower:TwoTowerRecaller",
    "sasrec":     "neorec.recall.sasrec:SASRecRecaller",
    "popularity": "neorec.recall.popularity:PopularityRecaller",
    "cold_start": "neorec.recall.cold_start:ColdStartRecaller",
}


def _instantiate(name: str, cfg: DictConfig):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown recall channel: {name}. Known: {list(_REGISTRY)}")
    mod_path, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(cfg)


def _evaluate(
    recaller,
    test_df: pd.DataFrame,
    k_list: list[int],
    catalog_size: int,
) -> dict[str, float]:
    """Score test users at the largest K, then compute metrics for every K."""
    user_ids = test_df["user_id"].tolist()
    y_true = [[item] for item in test_df["item_id"].tolist()]

    max_k = max(k_list)
    log.info("Evaluating on %d users at K=%d (max)…", len(user_ids), max_k)
    with Timer("recall.predict") as t:
        result = recaller.recall(user_ids, k=max_k)
    log.info("Inference: %.1f ms (%.2f ms/user)",
             t.elapsed_ms, t.elapsed_ms / max(len(user_ids), 1))

    y_pred = result.item_ids.tolist()

    metrics: dict[str, float] = {}
    for k in k_list:
        metrics[f"recall@{k}"]   = recall_at_k(y_true, y_pred, k=k)
        metrics[f"ndcg@{k}"]     = ndcg_at_k(y_true, y_pred, k=k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(y_true, y_pred, k=k)
        metrics[f"mrr@{k}"]      = mean_reciprocal_rank(y_true, y_pred, k=k)
        metrics[f"coverage@{k}"] = coverage(y_pred, catalog_size, k=k)

    metrics["latency_ms_per_user"] = t.elapsed_ms / max(len(user_ids), 1)
    return metrics


def run(cfg: DictConfig) -> dict[str, float]:
    """Train + evaluate one recall channel; log to MLflow."""
    name = str(cfg.recall.name)
    log.info("=== Training recall channel: %s ===", name)

    processed = Path(cfg.paths.data_processed) / cfg.data.name
    interactions_path = processed / "interactions.parquet"
    split_path = processed / "split.parquet"

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"Run preprocess first: missing {interactions_path}\n"
            "Hint: neorec data preprocess"
        )

    interactions = pd.read_parquet(interactions_path)
    split = pd.read_parquet(split_path)
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

    catalog_size = int(interactions["item_id"].max() + 1)
    log.info("Train rows=%d, test rows=%d, catalog_size=%d",
             len(train_df), len(test_df), catalog_size)

    train_path = processed / "train_interactions.parquet"
    train_df[["user_id", "item_id", "ts", "rating", "label"]].to_parquet(
        train_path, index=False
    )

    recaller = _instantiate(name, cfg)
    with Timer("fit") as t_fit:
        recaller.fit(train_path)
    log.info("Fit: %.2f s", t_fit.elapsed_ms / 1000)

    k_list = list(cfg.recall.eval.k_list)
    metrics = _evaluate(recaller, test_df, k_list=k_list, catalog_size=catalog_size)
    metrics["fit_seconds"] = t_fit.elapsed_ms / 1000.0
    log.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    artefacts_dir = ensure_dir(Path(cfg.paths.artifacts) / "recall" / name)
    recaller.save(artefacts_dir)

    flat_params = {
        "channel":          name,
        "dataset":          str(cfg.data.name),
        "split_strategy":   str(cfg.data.split.strategy),
        "rating_threshold": float(cfg.data.feedback.rating_threshold),
        **{f"model.{k}": v for k, v in OmegaConf.to_container(  # type: ignore[union-attr]
            cfg.recall.model, resolve=True
        ).items()},
    }
    # MLflow forbids '@' in metric names; convert e.g. recall@10 -> recall_at_10.
    mlflow_metrics = {k.replace("@", "_at_"): v for k, v in metrics.items()}

    with mlflow_run(
        experiment=cfg.mlflow.experiment_name,
        run_name=f"recall.{name}",
        tracking_uri=cfg.mlflow.tracking_uri,
        tags={"stage": "recall", "channel": name, "dataset": cfg.data.name},
    ) as mlf:
        mlf.log_params(flat_params)
        mlf.log_metrics(mlflow_metrics)

    log.info("=== Done. Artefacts: %s ===", artefacts_dir)
    return metrics
