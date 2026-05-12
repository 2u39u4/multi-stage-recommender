"""Smoke tests for ranking models.

These tests build a tiny synthetic dataset (10 users × 8 items × 4 genres)
in a temporary directory, fit each ranker for one epoch / a handful of
boosting rounds, and assert that:

* training emits a valid metric dict;
* ``score(...)`` returns a 1-D array of finite probabilities in [0, 1];
* ``predict(...)`` reorders candidate items deterministically;
* the DIN attention unit produces an attention map of the right shape;
* the DIN ``use_attention=False`` ablation runs without crashing.

We **deliberately** don't assert numerical thresholds — these are smoke
tests, not benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neorec.ranking.features import RankingFeaturizer, build_training_pairs


# ===========================================================================
# Synthetic dataset on disk
# ===========================================================================
N_USERS = 12
N_ITEMS = 8
N_GENRES = 4


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)

    # user features
    users = pd.DataFrame({
        "user_id":   np.arange(N_USERS, dtype=np.int32),
        "gender":    rng.integers(0, 2, N_USERS, dtype=np.int8),
        "age_bucket": rng.integers(0, 6, N_USERS, dtype=np.int8),
        "occupation": rng.integers(0, 4, N_USERS, dtype=np.int64),
    })
    users.to_parquet(out / "user_features.parquet", index=False)

    # item features
    item_rows = []
    for i in range(N_ITEMS):
        n_g = rng.integers(1, 3)
        genres = rng.choice(np.arange(1, N_GENRES + 1), size=int(n_g), replace=False)
        item_rows.append({
            "item_id": int(i),
            "title": f"item-{i}",
            "year": 1990 + int(i),
            "year_bucket": int(i % 4),
            "genres": np.asarray(genres, dtype=np.int64),
            "popularity": int(rng.integers(1, 100)),
            "popularity_bucket": int(i % 5),
        })
    items = pd.DataFrame(item_rows)
    items.to_parquet(out / "item_features.parquet", index=False)

    # id_maps
    (out / "id_maps.json").write_text(json.dumps({
        "genre_map": {f"g{i}": i for i in range(1, N_GENRES + 1)},
    }))

    # tiny interactions
    rows = []
    for u in range(N_USERS):
        for i in rng.choice(N_ITEMS, size=4, replace=False):
            rows.append((u, int(i), 1_700_000_000 + int(i), 5.0, 1))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "ts", "rating", "label"])
    inter.to_parquet(out / "interactions.parquet", index=False)
    return out


@pytest.fixture(scope="module")
def featurizer(processed_dir: Path) -> RankingFeaturizer:
    f = RankingFeaturizer(processed_dir=processed_dir, max_genres=4, max_seq_len=6)
    train_df = pd.read_parquet(processed_dir / "interactions.parquet")
    f.build_sequences(train_df)
    return f


@pytest.fixture(scope="module")
def pairs(featurizer: RankingFeaturizer, processed_dir: Path):
    train_df = pd.read_parquet(processed_dir / "interactions.parquet")
    user_seen = {u: set(g["item_id"].tolist()) for u, g in train_df.groupby("user_id")}
    pairs = build_training_pairs(
        train_df=train_df,
        num_items=featurizer.schema.num_items,
        user_seen=user_seen,
        negative_ratio=2,
        seed=1,
    )
    valid_cut = len(pairs) // 5
    return pairs.iloc[valid_cut:].reset_index(drop=True), pairs.iloc[:valid_cut].reset_index(drop=True)


# ===========================================================================
# Featurizer
# ===========================================================================
def test_featurizer_shapes(featurizer: RankingFeaturizer) -> None:
    user_ids = np.array([0, 1, 2], dtype=np.int64)
    item_ids = np.array([0, 1, 2], dtype=np.int64)
    batch = featurizer.featurize(user_ids, item_ids, include_history=True)
    assert batch.genres.shape == (3, featurizer.max_genres)
    assert batch.history is not None
    assert batch.history.shape == (3, featurizer.max_seq_len)
    assert set(batch.sparse) >= {"user_id", "item_id", "gender", "age_bucket"}


def test_build_training_pairs_balance(featurizer: RankingFeaturizer, processed_dir: Path) -> None:
    train_df = pd.read_parquet(processed_dir / "interactions.parquet")
    user_seen = {u: set(g["item_id"].tolist()) for u, g in train_df.groupby("user_id")}
    pairs = build_training_pairs(train_df, featurizer.schema.num_items, user_seen,
                                 negative_ratio=3, seed=0)
    pos_rate = pairs["label"].mean()
    assert 0.2 < pos_rate < 0.3, f"expected ~0.25 pos rate, got {pos_rate:.3f}"


# ===========================================================================
# LR baseline
# ===========================================================================
def test_lr_fit_score(featurizer: RankingFeaturizer, pairs):
    from neorec.ranking.lr import LRRanker

    cfg = OmegaConf.create({"seed": 0, "rank": {
        "name": "lr", "stage": "baseline",
        "model": {"hash_dim_user": 64, "hash_dim_item": 32, "C": 1.0, "solver": "liblinear"},
        "train": {"max_iter": 50},
    }})
    train_pairs, valid_pairs = pairs
    ranker = LRRanker(cfg, featurizer)
    metrics = ranker.fit(train_pairs, valid_pairs)
    assert "train_accuracy" in metrics

    scores = ranker.score(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1)), "probabilities must be in [0, 1]"


def test_lr_predict_topk(featurizer: RankingFeaturizer, pairs, tmp_path: Path):
    from neorec.ranking.lr import LRRanker
    cfg = OmegaConf.create({"seed": 0, "rank": {
        "name": "lr", "stage": "baseline",
        "model": {"hash_dim_user": 64, "hash_dim_item": 32, "C": 1.0, "solver": "liblinear"},
        "train": {"max_iter": 50},
    }})
    train_pairs, valid_pairs = pairs
    ranker = LRRanker(cfg, featurizer)
    ranker.fit(train_pairs, valid_pairs)

    res = ranker.predict(
        user_ids=[0, 1],
        candidate_items=[[0, 1, 2, 3], [4, 5, 6, 7]],
        k=2,
    )
    assert res.item_ids.shape == (2, 2)
    assert res.scores.shape == (2, 2)
    # Round-trip save/load.
    ranker.save(tmp_path / "lr")
    ranker2 = LRRanker(cfg, featurizer)
    ranker2.load(tmp_path / "lr")
    s1 = ranker.score(np.array([0]), np.array([3]))
    s2 = ranker2.score(np.array([0]), np.array([3]))
    np.testing.assert_allclose(s1, s2, atol=1e-6)


# ===========================================================================
# GBDT baseline
# ===========================================================================
def test_gbdt_fit_score(featurizer: RankingFeaturizer, pairs):
    from neorec.ranking.gbdt import GBDTRanker

    cfg = OmegaConf.create({"seed": 0, "rank": {
        "name": "gbdt", "stage": "baseline",
        "model": {"num_leaves": 8, "min_data_in_leaf": 5},
        "train": {"lr": 0.1, "num_boost_round": 10, "early_stopping_rounds": 5},
    }})
    train_pairs, valid_pairs = pairs
    ranker = GBDTRanker(cfg, featurizer)
    metrics = ranker.fit(train_pairs, valid_pairs)
    assert "n_iter" in metrics

    scores = ranker.score(np.array([0, 1]), np.array([0, 1]))
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))


# ===========================================================================
# DeepFM
# ===========================================================================
def test_deepfm_forward_shapes(featurizer: RankingFeaturizer, pairs):
    from neorec.ranking.deepfm import DeepFMRanker

    cfg = OmegaConf.create({"seed": 0, "rank": {
        "name": "deepfm", "stage": "pre_rank",
        "model": {
            "embedding_dim": 4, "dnn_hidden": [16, 8], "dnn_dropout": 0.1,
            "use_fm": True, "use_deep": True, "device": "cpu",
        },
        "train": {"epochs": 1, "batch_size": 16, "inference_batch_size": 64,
                  "lr": 1e-3, "weight_decay": 0.0, "early_stopping_patience": 1},
    }})
    train_pairs, valid_pairs = pairs
    ranker = DeepFMRanker(cfg, featurizer)
    metrics = ranker.fit(train_pairs, valid_pairs)
    assert "final_valid_bce" in metrics

    scores = ranker.score(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))


# ===========================================================================
# DIN
# ===========================================================================
def test_din_attention_shape(featurizer: RankingFeaturizer, pairs):
    from neorec.ranking.din import DINRanker

    cfg = OmegaConf.create({"seed": 0, "rank": {
        "name": "din", "stage": "fine_rank",
        "model": {
            "embedding_dim": 8, "attention_hidden": [8], "dnn_hidden": [16, 8],
            "dnn_dropout": 0.1, "use_attention": True, "device": "cpu",
        },
        "train": {"epochs": 1, "batch_size": 16, "inference_batch_size": 64,
                  "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 5.0,
                  "early_stopping_patience": 1},
    }})
    train_pairs, valid_pairs = pairs
    ranker = DINRanker(cfg, featurizer)
    ranker.fit(train_pairs, valid_pairs)

    weights, history, mask = ranker.attention_for_users(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
    )
    assert weights.shape == (3, featurizer.max_seq_len)
    assert mask.shape == (3, featurizer.max_seq_len)
    # Padded positions are zero-weighted (we mask before pooling).
    assert np.all(weights[mask == 0] == 0)


def test_din_ablation_falls_back_to_sum_pooling(featurizer: RankingFeaturizer, pairs):
    from neorec.ranking.din import DINRanker

    cfg = OmegaConf.create({"seed": 0, "rank": {
        "name": "din", "stage": "fine_rank",
        "model": {
            "embedding_dim": 8, "attention_hidden": [8], "dnn_hidden": [16, 8],
            "dnn_dropout": 0.1, "use_attention": False, "device": "cpu",
        },
        "train": {"epochs": 1, "batch_size": 16, "inference_batch_size": 64,
                  "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 5.0,
                  "early_stopping_patience": 1},
    }})
    train_pairs, valid_pairs = pairs
    ranker = DINRanker(cfg, featurizer)
    ranker.fit(train_pairs, valid_pairs)

    scores = ranker.score(np.array([0, 1]), np.array([0, 1]))
    assert scores.shape == (2,)
    assert ranker.model.use_attention is False

    with pytest.raises(RuntimeError):
        ranker.attention_for_users(np.array([0]), np.array([0]))
