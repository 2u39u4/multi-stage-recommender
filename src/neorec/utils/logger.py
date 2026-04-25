"""Centralized logging setup."""

from __future__ import annotations

import logging
import sys

_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    level: str | int = "INFO",
    fmt: str = _DEFAULT_FMT,
    stream: object = sys.stdout,
) -> None:
    """Configure the root logger idempotently."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)

    for noisy in ("matplotlib", "urllib3", "PIL", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
