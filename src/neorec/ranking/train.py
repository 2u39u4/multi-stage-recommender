"""Training entry point for ranking models."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "deepfm":          "neorec.ranking.deepfm:DeepFMRanker",
    "din":             "neorec.ranking.din:DINRanker",
    "transformer_ctr": "neorec.ranking.transformer_ctr:TransformerCTRRanker",
}


def run(cfg: DictConfig) -> None:
    """Instantiate the ranker declared in ``cfg.rank.name``, fit, evaluate, log.

    TODO(W3): implement dynamic import, MLflow run, artefact upload.
    """
    raise NotImplementedError  # TODO(W3)
