"""Unit tests for ranking models (skipped until implementation lands)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Ranking models scheduled for W3.")


def test_deepfm_forward_shapes() -> None: ...
def test_din_attention_shape() -> None: ...
def test_din_ablation_falls_back_to_sum_pooling() -> None: ...
