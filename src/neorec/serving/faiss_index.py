"""FAISS index build / load for online vector retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_hnsw(
    item_embeddings: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    metric: str = "inner_product",
) -> "faiss.Index":  # type: ignore[name-defined]  # noqa: F821
    """Build an HNSW index — good latency / recall trade-off for ~1M items."""
    raise NotImplementedError  # TODO(W5 Day 32)


def save_index(index: "faiss.Index", path: str | Path) -> None:  # type: ignore[name-defined]  # noqa: F821
    raise NotImplementedError  # TODO(W5)


def load_index(path: str | Path) -> "faiss.Index":  # type: ignore[name-defined]  # noqa: F821
    raise NotImplementedError  # TODO(W5)
