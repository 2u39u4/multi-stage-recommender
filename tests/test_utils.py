"""Tests for seed, logger, timer utilities."""

from __future__ import annotations

import random
import time

import numpy as np

from neorec.utils.seed import set_seed
from neorec.utils.timer import Timer


def test_seed_reproducibility() -> None:
    set_seed(123)
    a_py, a_np = random.random(), np.random.rand(3).tolist()
    set_seed(123)
    b_py, b_np = random.random(), np.random.rand(3).tolist()
    assert a_py == b_py
    assert a_np == b_np


def test_timer_measures_elapsed() -> None:
    with Timer("sleep") as t:
        time.sleep(0.02)
    assert t.elapsed_ms >= 15.0   # be generous with scheduler jitter
    assert t.elapsed_ms < 1000.0


def test_timer_as_decorator() -> None:
    timer = Timer("foo")

    @timer
    def work() -> int:
        time.sleep(0.005)
        return 7

    assert work() == 7
