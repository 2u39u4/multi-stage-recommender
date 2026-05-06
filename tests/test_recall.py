"""Smoke tests for recall channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neorec.recall.als import ALSRecaller
from neorec.recall.popularity import PopularityRecaller


def _make_tiny_interactions(path: Path, n_users: int = 30, n_items: int = 20) -> Path:
    """Generate a deterministic synthetic interactions parquet."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(n_users):
        # Each user "likes" 5 items, biased so ALS can recover the structure.
        liked = rng.choice(n_items, size=5, replace=False)
        for it in liked:
            rows.append((u, int(it), 1_700_000_000 + int(it), 5.0, 1))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts", "rating", "label"])
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def als_cfg():
    """Minimal cfg slice the recaller needs."""
    return OmegaConf.create(
        {
            "recall": {
                "name": "als",
                "model": {
                    "factors": 8,
                    "regularization": 0.01,
                    "alpha": 10.0,
                    "iterations": 5,
                    "use_gpu": False,
                    "random_state": 42,
                },
            }
        }
    )


def test_als_fits_and_recalls(als_cfg, tmp_path: Path) -> None:
    interactions_path = _make_tiny_interactions(tmp_path / "interactions.parquet")

    recaller = ALSRecaller(als_cfg)
    recaller.fit(interactions_path)

    assert recaller.model is not None
    assert recaller.num_users == 30
    assert recaller.num_items == 20

    result = recaller.recall(user_ids=list(range(5)), k=10)
    assert result.item_ids.shape == (5, 10)
    assert result.scores.shape == (5, 10)
    assert (result.user_ids == np.arange(5)).all()
    valid = result.item_ids[result.item_ids != -1]
    assert valid.min() >= 0
    assert valid.max() < 20


def test_als_save_load_roundtrip(als_cfg, tmp_path: Path) -> None:
    interactions_path = _make_tiny_interactions(tmp_path / "interactions.parquet")

    a = ALSRecaller(als_cfg)
    a.fit(interactions_path)
    save_dir = tmp_path / "als_artefacts"
    a.save(save_dir)
    assert (save_dir / "user_factors.npy").exists()
    assert (save_dir / "item_factors.npy").exists()

    b = ALSRecaller(als_cfg)
    b.load(save_dir)
    res_a = a.recall([0, 1, 2], k=5)
    res_b = b.recall([0, 1, 2], k=5)
    assert (res_a.item_ids == res_b.item_ids).all()


@pytest.fixture
def pop_cfg():
    return OmegaConf.create(
        {
            "recall": {
                "name": "popularity",
                "model": {"time_decay": False, "decay_half_life_days": 30.0},
            }
        }
    )


def test_popularity_filters_seen_items(pop_cfg, tmp_path: Path) -> None:
    """User who has watched item 0 ten times should never see it as a recall candidate."""
    rows = []
    for u in range(5):
        # Everybody has seen item 0 — heavy popularity skew.
        for _ in range(2):
            rows.append((u, 0, 1_700_000_000, 5.0, 1))
        # Each user also has another unique positive
        rows.append((u, u + 1, 1_700_000_001, 5.0, 1))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts", "rating", "label"])
    path = tmp_path / "interactions.parquet"
    df.to_parquet(path, index=False)

    rec = PopularityRecaller(pop_cfg)
    rec.fit(path)

    res = rec.recall(user_ids=[0, 1, 2], k=4)
    for row, uid in enumerate(res.user_ids):
        assert 0 not in res.item_ids[row], f"user {uid} got the seen item back"


def test_popularity_cold_start_uses_global_top(pop_cfg, tmp_path: Path) -> None:
    """A cold user (id 999, never seen in train) should get the global top-K."""
    rows = [
        (u, item, 1_700_000_000, 5.0, 1)
        for u in range(10)
        for item in [3, 3, 3, 7, 7, 11]  # item 3 is most popular
    ]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts", "rating", "label"])
    path = tmp_path / "interactions.parquet"
    df.to_parquet(path, index=False)

    rec = PopularityRecaller(pop_cfg)
    rec.fit(path)

    res = rec.recall(user_ids=[999], k=3)
    assert res.item_ids[0, 0] == 3  # most popular returned first
    assert -1 not in res.item_ids[0]


@pytest.fixture
def tt_cfg():
    """Tiny Two-Tower cfg fast enough to run inside a unit test."""
    return OmegaConf.create(
        {
            "seed": 0,
            "recall": {
                "name": "two_tower",
                "model": {
                    "embedding_dim": 8,
                    "user_tower_hidden": [],
                    "item_tower_hidden": [],
                    "dropout": 0.0,
                    "normalize": False,
                    "device": "cpu",
                    "temperature": 1.0,
                },
                "train": {
                    "epochs": 1,
                    "batch_size": 32,
                    "lr": 1e-2,
                    "weight_decay": 0.0,
                },
            },
            "data": {"features": {"sequence": {"max_len": 5}}},
        }
    )


def test_two_tower_smoke_runs(tt_cfg, tmp_path: Path) -> None:
    """One-epoch fit + recall on a tiny synthetic dataset must produce sane shapes."""
    pytest.importorskip("torch")
    from neorec.recall.two_tower import TwoTowerRecaller
    from neorec.utils.io import write_json

    n_users, n_items, n_genres = 12, 8, 3
    # Build all the parquet artefacts the recaller expects in the same dir.
    interactions = []
    user_seq: dict[int, list[int]] = {}
    rng = np.random.default_rng(0)
    for u in range(n_users):
        liked = rng.choice(n_items, size=4, replace=False).tolist()
        user_seq[u] = liked
        for it in liked:
            interactions.append((u, it, 1_700_000_000 + it, 5.0, 1))

    inter_df = pd.DataFrame(interactions,
                            columns=["user_id", "item_id", "ts", "rating", "label"])
    inter_df.to_parquet(tmp_path / "train_interactions.parquet", index=False)

    pd.DataFrame({"user_id": list(user_seq.keys()),
                  "history": list(user_seq.values())}
                 ).to_parquet(tmp_path / "sequences.parquet", index=False)

    pd.DataFrame({
        "item_id": list(range(n_items)),
        "title":   [f"item-{i}" for i in range(n_items)],
        "year":    [2000 + i for i in range(n_items)],
        "year_bucket": [3] * n_items,
        "genres":  [[(i % n_genres) + 1] for i in range(n_items)],
        "popularity": [1] * n_items,
        "popularity_bucket": [0] * n_items,
    }).to_parquet(tmp_path / "item_features.parquet", index=False)

    write_json(
        {
            "user_id_map_size": n_users,
            "item_id_map_size": n_items,
            "genre_map": {f"g{k}": k for k in range(1, n_genres + 1)},
        },
        tmp_path / "id_maps.json",
    )

    rec = TwoTowerRecaller(tt_cfg)
    rec.fit(tmp_path / "train_interactions.parquet")
    res = rec.recall(user_ids=[0, 1, 2], k=4)
    assert res.item_ids.shape == (3, 4)
    assert (res.item_ids >= 0).all()
    assert (res.item_ids < n_items).all()
