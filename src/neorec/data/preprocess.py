"""Clean, re-index, split and persist MovieLens-1M / 20M as parquet.

Outputs (under ``cfg.paths.data_processed / cfg.data.name``):

    interactions.parquet     user_id, item_id, ts, rating, label
    user_features.parquet    user_id, gender, age_bucket, occupation
    item_features.parquet    item_id, year_bucket, popularity_bucket, genres (list[int])
    sequences.parquet        user_id, history (list[int])      # train-only
    split.parquet            user_id, item_id, split           # {train,valid,test}
    id_maps.json             {user_id_map, item_id_map, genre_map}
    stats.json               summary statistics
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from neorec.utils.io import ensure_dir, write_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema readers
# ---------------------------------------------------------------------------
_RATINGS_COLS = ["user_id", "item_id", "rating", "ts"]
_USERS_COLS = ["user_id", "gender", "age", "occupation", "zip"]
_MOVIES_COLS = ["item_id", "title", "genres"]


def _read_ml_1m(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=_RATINGS_COLS,
        encoding="latin-1",
    )
    users = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=_USERS_COLS,
        encoding="latin-1",
    )
    movies = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=_MOVIES_COLS,
        encoding="latin-1",
    )
    return ratings, users, movies


def _read_ml_20m(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        raw_dir / "ratings.csv",
        names=_RATINGS_COLS,
        header=0,
    )
    movies = pd.read_csv(raw_dir / "movies.csv", names=_MOVIES_COLS, header=0)
    # ml-20m has no users.dat; create a minimal table
    users = pd.DataFrame(
        {
            "user_id": ratings["user_id"].unique(),
            "gender": "U",
            "age": -1,
            "occupation": -1,
            "zip": "",
        }
    )
    return ratings, users, movies


def _read_dataset(cfg: DictConfig, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cfg.data.size == "1m":
        return _read_ml_1m(raw_dir)
    if cfg.data.size == "20m":
        return _read_ml_20m(raw_dir)
    raise ValueError(f"Unsupported dataset size: {cfg.data.size}")


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------
_AGE_BUCKETS = [(0, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 200)]


def _age_bucket(age: int) -> int:
    if age is None or age < 0:
        return len(_AGE_BUCKETS)  # unknown bucket
    for i, (lo, hi) in enumerate(_AGE_BUCKETS):
        if lo <= age < hi:
            return i
    return len(_AGE_BUCKETS) - 1


def _year_from_title(title: str) -> int | None:
    if not isinstance(title, str):
        return None
    if title.endswith(")") and "(" in title[-7:]:
        try:
            return int(title[-5:-1])
        except ValueError:
            return None
    return None


def _year_bucket(year: int | None) -> int:
    if year is None:
        return 0
    if year < 1970:
        return 1
    if year < 1980:
        return 2
    if year < 1990:
        return 3
    if year < 2000:
        return 4
    if year < 2010:
        return 5
    return 6


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(cfg: DictConfig) -> dict[str, Path]:
    """Entry point invoked by the CLI."""
    raw_root = Path(cfg.paths.data_raw) / cfg.data.name
    inner = next((p for p in raw_root.iterdir() if p.is_dir()), raw_root)
    log.info("Reading raw data from %s", inner)

    ratings, users, movies = _read_dataset(cfg, inner)
    log.info(
        "Loaded ratings=%d, users=%d, movies=%d",
        len(ratings), len(users), len(movies),
    )

    # 1. Implicit feedback label
    threshold = float(cfg.data.feedback.rating_threshold)
    ratings["label"] = (ratings["rating"] >= threshold).astype(np.int8)
    pos = ratings[ratings["label"] == 1].copy()
    log.info("Positives at rating >= %.1f: %d (%.2f%%)",
             threshold, len(pos), 100 * len(pos) / max(len(ratings), 1))

    # 2. Drop very cold users (defined on positives)
    min_int = int(cfg.data.split.min_interactions_per_user)
    user_counts = pos.groupby("user_id").size()
    keep_users = user_counts[user_counts >= min_int].index
    pos = pos[pos["user_id"].isin(keep_users)].copy()
    log.info("After min_interactions=%d filter: users=%d, interactions=%d",
             min_int, pos["user_id"].nunique(), len(pos))

    # 3. Reindex user / item ids to dense [0, N)
    user_id_map = {u: i for i, u in enumerate(sorted(pos["user_id"].unique()))}
    item_id_map = {it: i for i, it in enumerate(sorted(pos["item_id"].unique()))}
    pos["user_id"] = pos["user_id"].map(user_id_map).astype(np.int32)
    pos["item_id"] = pos["item_id"].map(item_id_map).astype(np.int32)

    # 4. Sort by (user, ts) for downstream splits and sequences
    pos = pos.sort_values(["user_id", "ts"]).reset_index(drop=True)

    # 5. Split
    strategy = cfg.data.split.strategy
    if strategy == "leave_one_out":
        split = _leave_one_out(pos)
    elif strategy == "time_based":
        split = _time_based_split(pos, cfg.data.split.valid_ratio, cfg.data.split.test_ratio)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    log.info("Split sizes: train=%d, valid=%d, test=%d",
             (split["split"] == "train").sum(),
             (split["split"] == "valid").sum(),
             (split["split"] == "test").sum())

    # 6. Train-only behavior sequences (used by DIN / SASRec downstream)
    train_pos = split.loc[split["split"] == "train", ["user_id", "item_id", "ts"]]
    train_pos = train_pos.sort_values(["user_id", "ts"])
    seq_max_len = int(cfg.data.features.sequence.max_len)
    sequences = (
        train_pos.groupby("user_id")["item_id"]
        .apply(lambda s: s.tolist()[-seq_max_len:])
        .reset_index()
        .rename(columns={"item_id": "history"})
    )

    # 7. User features
    if "age" in users.columns:
        users["age_bucket"] = users["age"].apply(_age_bucket).astype(np.int8)
    else:
        users["age_bucket"] = np.int8(0)

    gender_map = {g: i for i, g in enumerate(sorted(users["gender"].astype(str).unique()))}
    users["gender"] = users["gender"].astype(str).map(gender_map).astype(np.int8)

    users = users[users["user_id"].isin(user_id_map.keys())].copy()
    users["user_id"] = users["user_id"].map(user_id_map).astype(np.int32)
    users = users[["user_id", "gender", "age_bucket", "occupation"]]
    users = users.sort_values("user_id").reset_index(drop=True)

    # 8. Item features (genres + year)
    movies["year"] = movies["title"].apply(_year_from_title)
    movies["year_bucket"] = movies["year"].apply(_year_bucket).astype(np.int8)

    all_genres = sorted({g for line in movies["genres"].fillna("").astype(str)
                         for g in line.split("|") if g})
    genre_map = {g: i + 1 for i, g in enumerate(all_genres)}  # 0 = unknown / pad

    def _encode_genres(s: object) -> list[int]:
        if not isinstance(s, str) or not s:
            return [0]
        return [genre_map[g] for g in s.split("|") if g in genre_map] or [0]

    movies["genres"] = movies["genres"].apply(_encode_genres)
    movies = movies[movies["item_id"].isin(item_id_map.keys())].copy()
    movies["item_id"] = movies["item_id"].map(item_id_map).astype(np.int32)
    movies = movies[["item_id", "title", "year", "year_bucket", "genres"]]

    # popularity bucket (based on train counts)
    train_item_counts = (
        split.loc[split["split"] == "train", "item_id"].value_counts().to_dict()
    )
    movies["popularity"] = movies["item_id"].map(train_item_counts).fillna(0).astype(np.int32)
    movies["popularity_bucket"] = pd.qcut(
        movies["popularity"].rank(method="first"), q=10, labels=False, duplicates="drop"
    ).fillna(0).astype(np.int8)
    movies = movies.sort_values("item_id").reset_index(drop=True)

    # 9. Persist
    out_dir = ensure_dir(Path(cfg.paths.data_processed) / cfg.data.name)
    paths = {
        "interactions":   out_dir / "interactions.parquet",
        "user_features":  out_dir / "user_features.parquet",
        "item_features":  out_dir / "item_features.parquet",
        "sequences":      out_dir / "sequences.parquet",
        "split":          out_dir / "split.parquet",
        "id_maps":        out_dir / "id_maps.json",
        "stats":          out_dir / "stats.json",
    }

    pos.to_parquet(paths["interactions"], index=False)
    users.to_parquet(paths["user_features"], index=False)
    movies.to_parquet(paths["item_features"], index=False)
    sequences.to_parquet(paths["sequences"], index=False)
    split.to_parquet(paths["split"], index=False)

    write_json(
        {
            "user_id_map_size": len(user_id_map),
            "item_id_map_size": len(item_id_map),
            "genre_map": genre_map,
        },
        paths["id_maps"],
    )

    stats = {
        "num_users":        int(pos["user_id"].nunique()),
        "num_items":        int(pos["item_id"].nunique()),
        "num_interactions": int(len(pos)),
        "num_train":        int((split["split"] == "train").sum()),
        "num_valid":        int((split["split"] == "valid").sum()),
        "num_test":         int((split["split"] == "test").sum()),
        "num_genres":       len(genre_map),
        "split_strategy":   strategy,
        "rating_threshold": threshold,
    }
    write_json(stats, paths["stats"])
    log.info("Stats: %s", stats)

    return paths


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------
def _leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """Last interaction per user → test, second-to-last → valid, rest → train."""
    df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)
    rank_desc = df.groupby("user_id").cumcount(ascending=False)
    split = np.where(rank_desc == 0, "test", np.where(rank_desc == 1, "valid", "train"))
    out = df[["user_id", "item_id", "ts"]].copy()
    out["split"] = split
    return out


def _time_based_split(df: pd.DataFrame, valid_ratio: float, test_ratio: float) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_valid = int(n * valid_ratio)
    split = np.array(["train"] * n, dtype=object)
    split[-n_test:] = "test"
    split[-(n_test + n_valid):-n_test] = "valid"
    out = df[["user_id", "item_id", "ts"]].copy()
    out["split"] = split
    return out
