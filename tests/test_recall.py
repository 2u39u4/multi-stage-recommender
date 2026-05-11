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


# ===========================================================================
# SASRec
# ===========================================================================
@pytest.fixture
def sasrec_cfg():
    return OmegaConf.create(
        {
            "seed": 0,
            "recall": {
                "name": "sasrec",
                "model": {
                    "embedding_dim": 8,
                    "max_seq_len": 6,
                    "num_blocks": 1,
                    "num_heads": 2,
                    "dropout": 0.0,
                    "layer_norm_eps": 1e-12,
                    "device": "cpu",
                },
                "train": {
                    "epochs": 2,
                    "batch_size": 8,
                    "lr": 1e-2,
                    "weight_decay": 0.0,
                    "grad_clip": 5.0,
                    "inference_batch_size": 16,
                },
            },
        }
    )


def _build_tiny_sequential(tmp_path: Path, n_users: int, n_items: int) -> Path:
    """Build a minimal train_interactions + id_maps that SASRec can ingest."""
    from neorec.utils.io import write_json

    rng = np.random.default_rng(0)
    rows = []
    for u in range(n_users):
        # Each user has 5–7 interactions with monotonically increasing ts.
        n = int(rng.integers(5, 8))
        items = rng.choice(n_items, size=n, replace=False)
        for i, it in enumerate(items):
            rows.append((u, int(it), 1_700_000_000 + i, 5.0, 1))
    pd.DataFrame(
        rows, columns=["user_id", "item_id", "ts", "rating", "label"]
    ).to_parquet(tmp_path / "train_interactions.parquet", index=False)
    write_json(
        {
            "user_id_map_size": n_users,
            "item_id_map_size": n_items,
            "genre_map": {},
        },
        tmp_path / "id_maps.json",
    )
    return tmp_path / "train_interactions.parquet"


def test_sasrec_smoke_runs(sasrec_cfg, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from neorec.recall.sasrec import SASRecRecaller

    n_users, n_items = 10, 20
    path = _build_tiny_sequential(tmp_path, n_users, n_items)

    rec = SASRecRecaller(sasrec_cfg)
    rec.fit(path)
    res = rec.recall(user_ids=list(range(5)), k=4)
    assert res.item_ids.shape == (5, 4)
    valid = res.item_ids[res.item_ids != -1]
    assert valid.min() >= 0
    assert valid.max() < n_items


def test_sasrec_save_load_roundtrip(sasrec_cfg, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from neorec.recall.sasrec import SASRecRecaller

    n_users, n_items = 8, 12
    path = _build_tiny_sequential(tmp_path, n_users, n_items)

    a = SASRecRecaller(sasrec_cfg)
    a.fit(path)
    save_dir = tmp_path / "sasrec_artefacts"
    a.save(save_dir)
    assert (save_dir / "model.pt").exists()
    assert (save_dir / "item_emb.npy").exists()
    assert (save_dir / "user_sequences.parquet").exists()

    b = SASRecRecaller(sasrec_cfg)
    b.load(save_dir)
    res_a = a.recall([0, 1, 2], k=4)
    res_b = b.recall([0, 1, 2], k=4)
    # Same architecture + weights => identical recall output.
    assert (res_a.item_ids == res_b.item_ids).all()


# ===========================================================================
# Cold-start
# ===========================================================================
@pytest.fixture
def cs_cfg():
    return OmegaConf.create(
        {
            "recall": {
                "name": "cold_start",
                "model": {
                    "features": ["genres", "year_bucket"],
                    "similarity": "cosine",
                    "tfidf_on_genres": True,
                },
                "output": {"top_k": 50, "fallback_to_popularity": True},
            },
        }
    )


def test_cold_start_recovers_genre_neighbours(cs_cfg, tmp_path: Path) -> None:
    """Items sharing a unique genre should be ranked above unrelated items."""
    from neorec.recall.cold_start import ColdStartRecaller
    from neorec.utils.io import write_json

    # 6 items split into two clearly separable groups by genre tag.
    item_rows = [
        {"item_id": 0, "title": "i0", "year": 2000, "year_bucket": 3, "genres": [1]},  # group A
        {"item_id": 1, "title": "i1", "year": 2001, "year_bucket": 3, "genres": [1]},  # group A
        {"item_id": 2, "title": "i2", "year": 2002, "year_bucket": 3, "genres": [1]},  # group A
        {"item_id": 3, "title": "i3", "year": 2003, "year_bucket": 3, "genres": [2]},  # group B
        {"item_id": 4, "title": "i4", "year": 2004, "year_bucket": 3, "genres": [2]},  # group B
        {"item_id": 5, "title": "i5", "year": 2005, "year_bucket": 3, "genres": [2]},  # group B
    ]
    pd.DataFrame(item_rows).to_parquet(tmp_path / "item_features.parquet", index=False)

    # User 0 has watched A-items 0 and 1 → cold-start should rank item 2 (still A) above any B item.
    interactions = pd.DataFrame(
        [
            (0, 0, 1_700_000_000, 5.0, 1),
            (0, 1, 1_700_000_001, 5.0, 1),
            (1, 3, 1_700_000_002, 5.0, 1),  # different user, separate signal
        ],
        columns=["user_id", "item_id", "ts", "rating", "label"],
    )
    train_path = tmp_path / "train_interactions.parquet"
    interactions.to_parquet(train_path, index=False)

    write_json(
        {
            "user_id_map_size": 2,
            "item_id_map_size": 6,
            "genre_map": {"g1": 1, "g2": 2},
        },
        tmp_path / "id_maps.json",
    )

    rec = ColdStartRecaller(cs_cfg)
    rec.fit(train_path)
    res = rec.recall(user_ids=[0], k=3)
    # The only un-seen A-item for user 0 is item 2 — it should be ranked first.
    assert res.item_ids[0, 0] == 2


def test_cold_start_fallback_to_popularity(cs_cfg, tmp_path: Path) -> None:
    """Unknown user with zero history → global popularity top-K."""
    from neorec.recall.cold_start import ColdStartRecaller
    from neorec.utils.io import write_json

    item_rows = [
        {"item_id": i, "title": f"i{i}", "year": 2000, "year_bucket": 3, "genres": [1]}
        for i in range(4)
    ]
    pd.DataFrame(item_rows).to_parquet(tmp_path / "item_features.parquet", index=False)

    # item 2 is the most popular by far.
    inter_rows = [
        (u, 2, 1_700_000_000, 5.0, 1) for u in range(5)
    ] + [(0, 1, 1_700_000_001, 5.0, 1)]
    inter_df = pd.DataFrame(
        inter_rows, columns=["user_id", "item_id", "ts", "rating", "label"]
    )
    train_path = tmp_path / "train_interactions.parquet"
    inter_df.to_parquet(train_path, index=False)

    # n_users in id_map is 10 → users 5-9 have no profile in train data
    write_json(
        {
            "user_id_map_size": 10,
            "item_id_map_size": 4,
            "genre_map": {"g1": 1},
        },
        tmp_path / "id_maps.json",
    )

    rec = ColdStartRecaller(cs_cfg)
    rec.fit(train_path)
    res = rec.recall(user_ids=[9], k=3)
    assert res.item_ids[0, 0] == 2  # popularity fallback gives most popular first


# ===========================================================================
# Multi-channel merge (pure functions)
# ===========================================================================
def test_merge_rrf_combines_two_channels() -> None:
    """An item ranked #1 by both channels should dominate either single-channel top."""
    from neorec.recall.base import RecallResult
    from neorec.recall.merge import merge_rrf

    users = np.array([0, 1], dtype=np.int32)
    a = RecallResult(
        user_ids=users,
        item_ids=np.array([[10, 20, 30, 40], [10, 99, 88, 77]], dtype=np.int32),
        scores=np.array([[0.9, 0.8, 0.7, 0.6], [0.9, 0.5, 0.4, 0.3]], dtype=np.float32),
        channel="A",
    )
    b = RecallResult(
        user_ids=users,
        item_ids=np.array([[10, 30, 50, 60], [10, 11, 12, 13]], dtype=np.int32),
        scores=np.array([[0.95, 0.4, 0.3, 0.2], [0.7, 0.6, 0.5, 0.4]], dtype=np.float32),
        channel="B",
    )
    merged = merge_rrf([a, b], k_rrf=60, candidate_pool_size=5)
    # Item 10 ranks #1 in BOTH channels → must come out on top for both users.
    assert merged.item_ids[0, 0] == 10
    assert merged.item_ids[1, 0] == 10


def test_merge_norm_weighted_respects_weights() -> None:
    """Down-weighting a channel should change the merged ranking accordingly."""
    from neorec.recall.base import RecallResult
    from neorec.recall.merge import merge_norm_weighted

    users = np.array([0], dtype=np.int32)
    deep = RecallResult(
        user_ids=users,
        item_ids=np.array([[5, 6, 7]], dtype=np.int32),
        scores=np.array([[1.0, 0.5, 0.1]], dtype=np.float32),
        channel="deep",
    )
    heuristic = RecallResult(
        user_ids=users,
        item_ids=np.array([[7, 5, 6]], dtype=np.int32),
        scores=np.array([[1.0, 0.5, 0.1]], dtype=np.float32),
        channel="heur",
    )
    # Equal weights — item 5 and item 7 each get one #1 vote.
    merged_eq = merge_norm_weighted([deep, heuristic],
                                    weights={"deep": 1.0, "heur": 1.0},
                                    candidate_pool_size=3)
    # Strongly down-weight the heuristic — item 5 must now beat item 7.
    merged_skew = merge_norm_weighted([deep, heuristic],
                                      weights={"deep": 10.0, "heur": 0.1},
                                      candidate_pool_size=3)
    # Top-2 should contain 5 and 7 in either ordering for the equal-weights run,
    # but item 5 must be strictly above item 7 in the skewed run.
    rank5 = list(merged_skew.item_ids[0]).index(5)
    rank7 = list(merged_skew.item_ids[0]).index(7)
    assert rank5 < rank7
    assert set(merged_eq.item_ids[0, :2]) == {5, 7}
