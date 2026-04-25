"""Lightweight timing context manager for latency breakdowns."""

from __future__ import annotations

import logging
import time
from contextlib import ContextDecorator
from typing import Any

log = logging.getLogger(__name__)


class Timer(ContextDecorator):
    """Context manager / decorator that records wall-clock elapsed time.

    Examples
    --------
    >>> with Timer("recall") as t:
    ...     run_recall()
    >>> print(t.elapsed_ms)
    """

    def __init__(self, name: str = "block", log_on_exit: bool = False) -> None:
        self.name = name
        self.log_on_exit = log_on_exit
        self._start: float | None = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._start is not None
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self.log_on_exit:
            log.info("[%s] %.2f ms", self.name, self.elapsed_ms)
