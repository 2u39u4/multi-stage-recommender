"""Feature engineering for ranking models.

Produces categorical + numeric feature columns, plus the user behavior
sequence used by DIN / SASRec / Transformer CTR.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def build_user_features(cfg: DictConfig) -> None:
    """age_bucket, gender, occupation, active_days, avg_rating, ..."""
    raise NotImplementedError  # TODO(W1)


def build_item_features(cfg: DictConfig) -> None:
    """genres (multi-hot), year_bucket, popularity_bucket, avg_rating, ..."""
    raise NotImplementedError  # TODO(W1)


def build_sequences(cfg: DictConfig) -> None:
    """Chronological item sequence per user, truncated to ``seq.max_len``."""
    raise NotImplementedError  # TODO(W1)
