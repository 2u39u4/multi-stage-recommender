"""Download MovieLens datasets.

Uses ``cfg.data.url`` and writes to ``cfg.paths.data_raw``.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Download and extract the raw dataset if not already present.

    TODO(W1 Day 3):
        * urllib.request.urlretrieve(cfg.data.url, ...)
        * unzip to cfg.paths.data_raw / cfg.data.name
        * sha256-verify and skip if already on disk
    """
    raise NotImplementedError  # TODO(W1)
