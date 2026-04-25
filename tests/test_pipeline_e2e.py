"""End-to-end pipeline integration test (W4+)."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skip(reason="E2E pipeline scheduled for W4–W5."),
]


def test_full_pipeline_on_tiny_dataset() -> None:
    """Train all channels + rankers on a 50×30 toy dataset and assert:
       * every stage runs without exceptions
       * Recall@10 > 0 on the held-out positive
       * end-to-end latency < 500 ms
    """
    ...
