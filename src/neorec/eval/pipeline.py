"""End-to-end evaluation orchestrator — runs the full funnel and logs to MLflow."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from neorec.rerank.pipeline import run as run_rerank

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> dict[str, float]:
    """Run recall → prerank → rank → rerank on the test split and log metrics.

    The function is expected to:

    1. Load trained models from ``cfg.paths.artifacts``.
    2. For each user in the test split, generate candidates from every recall
       channel, fuse them, then run pre-rank → fine-rank → rerank.
    3. Compute Recall@K, NDCG@K, MRR, HitRate, Coverage, ILS, for K in
       {1, 5, 10, 20, 50, 100}.
    4. Log everything to MLflow under a single run.
    5. Dump a markdown table to ``experiments/results/end_to_end.md``.
    """
    log.info("Delegating full-funnel evaluation to the rerank orchestrator.")
    return run_rerank(cfg)
