"""Feature engineering for ranking models.

Produces categorical + numeric feature columns, plus the user behavior
sequence used by DIN / SASRec / Transformer CTR.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def _processed_dir(cfg: DictConfig) -> Path:
    return Path(cfg.paths.data_processed) / cfg.data.name


def build_user_features(cfg: DictConfig) -> None:
    """age_bucket, gender, occupation, active_days, avg_rating, ..."""
    out_dir = _processed_dir(cfg)
    src = out_dir / "user_features.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}; run `neorec data preprocess` first.")
    df = pd.read_parquet(src)
    dst = out_dir / "user_features_enriched.parquet"
    df.to_parquet(dst, index=False)
    log.info("User features already materialized by preprocess; wrote %s", dst)


def build_item_features(cfg: DictConfig) -> None:
    """genres (multi-hot), year_bucket, popularity_bucket, avg_rating, ..."""
    out_dir = _processed_dir(cfg)
    src = out_dir / "item_features.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}; run `neorec data preprocess` first.")
    df = pd.read_parquet(src)
    dst = out_dir / "item_features_enriched.parquet"
    df.to_parquet(dst, index=False)
    log.info("Item features already materialized by preprocess; wrote %s", dst)


def build_sequences(cfg: DictConfig) -> None:
    """Chronological item sequence per user, truncated to ``seq.max_len``."""
    out_dir = _processed_dir(cfg)
    split_path = out_dir / "split.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}; run `neorec data preprocess` first.")

    split = pd.read_parquet(split_path)
    train = split.loc[split["split"] == "train", ["user_id", "item_id", "ts"]]
    max_len = int(cfg.data.features.sequence.max_len)
    sequences = (
        train.sort_values(["user_id", "ts"])
        .groupby("user_id")["item_id"]
        .apply(lambda s: [int(x) for x in s.tolist()[-max_len:]])
        .reset_index()
        .rename(columns={"item_id": "history"})
    )
    dst = out_dir / "sequences.parquet"
    sequences.to_parquet(dst, index=False)
    log.info("Wrote %d user sequences to %s", len(sequences), dst)
