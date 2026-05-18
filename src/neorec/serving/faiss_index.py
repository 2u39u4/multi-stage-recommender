"""FAISS index build / load for online vector retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _import_faiss():
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional wheel
        raise RuntimeError(
            "faiss-cpu is required for vector serving. Install with "
            "`pip install faiss-cpu` or use the Docker serving image."
        ) from exc
    return faiss


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def build_hnsw(
    item_embeddings: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    metric: str = "inner_product",
) -> "faiss.Index":  # type: ignore[name-defined]  # noqa: F821
    """Build an HNSW index — good latency / recall trade-off for ~1M items."""
    faiss = _import_faiss()
    vecs = np.asarray(item_embeddings, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError(f"item_embeddings must be 2-D, got shape={vecs.shape}")
    if vecs.shape[0] == 0:
        raise ValueError("Cannot build a FAISS index over zero items.")

    metric = metric.lower()
    if metric in {"inner_product", "ip", "cosine"}:
        # Cosine similarity is inner product over L2-normalised vectors.
        vecs = _l2_normalize(vecs)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric in {"l2", "euclidean"}:
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError("metric must be one of inner_product, cosine, or l2")

    dim = int(vecs.shape[1])
    index = faiss.IndexHNSWFlat(dim, int(m), faiss_metric)
    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = int(ef_search)
    index.add(vecs)
    return index


def save_index(index: "faiss.Index", path: str | Path) -> None:  # type: ignore[name-defined]  # noqa: F821
    faiss = _import_faiss()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str | Path) -> "faiss.Index":  # type: ignore[name-defined]  # noqa: F821
    faiss = _import_faiss()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return faiss.read_index(str(path))
