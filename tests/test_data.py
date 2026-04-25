"""Smoke tests for the data pipeline (will be implemented in W1)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Data pipeline implementation scheduled for W1.")


def test_preprocess_creates_parquet_tables() -> None: ...
def test_leave_one_out_split() -> None: ...
def test_id_reindexing_is_dense() -> None: ...
