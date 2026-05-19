"""Lightweight end-to-end serving pipeline integration test.

This deliberately avoids training all models inside CI. The goal is to verify
the online funnel contract end-to-end: recall candidates enter the pipeline,
already-seen items are filtered, MMR/rules produce a top-K list, and latency
breakdown is returned.
"""

from __future__ import annotations

import numpy as np

from neorec.serving.pipeline import OnlinePipeline


def test_online_pipeline_contract_on_tiny_catalog() -> None:
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
        item_embeddings=np.eye(6, dtype=np.float32),
        item_meta={i: {"genres": [i % 2 + 1], "year_bucket": i // 2} for i in range(6)},
        item_titles={i: f"movie-{i}" for i in range(6)},
        user_seen={7: {0, 3}},
        fallback_items=[0, 1, 2, 3, 4, 5],
    )

    response = pipe.recommend(user_id=7, k=4, diversity=0.7)

    assert response.user_id == 7
    assert [item.item_id for item in response.items] == [1, 2, 4, 5]
    assert all(item.title is not None for item in response.items)
    assert all("MMR" in (item.explain or "") for item in response.items)
    assert {"recall", "pre_rank", "fine_rank", "rerank", "total"} <= set(response.latency_ms)
    assert response.latency_ms["total"] >= 0
