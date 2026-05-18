from __future__ import annotations

import numpy as np
import pytest

from neorec.serving.faiss_index import build_hnsw, load_index, save_index
from neorec.serving.pipeline import OnlinePipeline


def test_faiss_hnsw_roundtrip(tmp_path) -> None:
    pytest.importorskip("faiss")
    vecs = np.eye(4, dtype=np.float32)
    index = build_hnsw(vecs, m=8, ef_construction=20, ef_search=16)
    scores, ids = index.search(vecs[:1], 2)
    assert ids.shape == (1, 2)
    assert ids[0, 0] == 0
    assert scores[0, 0] >= scores[0, 1]

    path = tmp_path / "hnsw.index"
    save_index(index, path)
    loaded = load_index(path)
    _, ids2 = loaded.search(vecs[:1], 1)
    assert ids2[0, 0] == 0


def test_online_pipeline_popularity_fallback_filters_seen() -> None:
    pipe = OnlinePipeline(
        cfg={
            "rerank": {
                "rules": {
                    "max_per_genre_ratio": 1.0,
                    "max_per_year_bucket": 10,
                    "filter_already_watched": True,
                }
            }
        },
        item_embeddings=np.eye(5, dtype=np.float32),
        item_meta={i: {"genres": [1], "year_bucket": 1} for i in range(5)},
        item_titles={i: f"item-{i}" for i in range(5)},
        user_seen={1: {0, 2}},
        fallback_items=[0, 1, 2, 3, 4],
    )

    response = pipe.recommend(user_id=1, k=3, diversity=1.0)
    assert [item.item_id for item in response.items] == [1, 3, 4]
    assert response.latency_ms["total"] >= 0
    assert response.items[0].title == "item-1"
