"""Unified training entrypoint for recall channels — dispatched from the CLI."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "als":        "neorec.recall.als:ALSRecaller",
    "two_tower":  "neorec.recall.two_tower:TwoTowerRecaller",
    "sasrec":     "neorec.recall.sasrec:SASRecRecaller",
    "popularity": "neorec.recall.popularity:PopularityRecaller",
    "cold_start": "neorec.recall.cold_start:ColdStartRecaller",
}


def run(cfg: DictConfig) -> None:
    """Select the channel declared in ``cfg.recall.name`` and train it.

    TODO(W1–W2): dynamic import, instantiation, fit, eval, MLflow log.
    """
    raise NotImplementedError  # TODO(W1–W2)
