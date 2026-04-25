"""Smoke tests for recall channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neorec.recall.als import ALSRecaller


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
