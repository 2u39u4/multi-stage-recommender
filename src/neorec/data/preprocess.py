"""Clean, re-index, split, and persist MovieLens ratings as parquet.

Output tables (under ``cfg.paths.data_processed``):
    * interactions.parquet     — (user_id, item_id, ts, label)
    * user_features.parquet
    * item_features.parquet
    * sequence.parquet         — (user_id, [item_id history])
    * split.parquet            — (user_id, item_id, split ∈ {train, valid, test})
    * id_maps/{user,item}.json
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Entry point called by the CLI.

    TODO(W1 Day 3):
        * read ratings.dat / ratings.csv
        * reindex user / item ids to dense contiguous integers
        * implicit feedback: rating >= cfg.data.feedback.rating_threshold → 1
        * leave-one-out or time-based split
        * write parquet tables + id maps
    """
    raise NotImplementedError  # TODO(W1)
