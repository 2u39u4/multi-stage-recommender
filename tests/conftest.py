"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Ensure every test starts from the same seed."""
    from neorec.utils.seed import set_seed

    set_seed(42)


@pytest.fixture
def tiny_interactions(tmp_path: Path) -> Iterator[Path]:
    """Synthetic 50-user × 30-item interaction table as parquet."""
    import pandas as pd

    rng = np.random.default_rng(42)
    users = rng.integers(0, 50, size=500)
    items = rng.integers(0, 30, size=500)
    ts = rng.integers(1_700_000_000, 1_700_100_000, size=500)
    df = pd.DataFrame({"user_id": users, "item_id": items, "ts": ts, "label": 1})
    df = df.drop_duplicates(subset=["user_id", "item_id"])
    path = tmp_path / "interactions.parquet"
    df.to_parquet(path)
    yield path


@pytest.fixture
def tiny_y_pred_y_true() -> tuple[list[list[int]], list[list[int]]]:
    """Hand-crafted predictions to verify metric implementations."""
    y_true = [[1, 2, 3], [4, 5], [6]]
    y_pred = [
        [1, 7, 2, 8, 3],       # recall@3 = 1/3, hit@3 = 1
        [9, 4, 8, 5, 10],      # recall@3 = 1/2, hit@3 = 1
        [10, 11, 12, 13, 6],   # recall@3 = 0
    ]
    return y_pred, y_true
