"""Download MovieLens datasets.

The URL and target size are read from ``cfg.data.url`` and ``cfg.data.size``.
Files land in ``cfg.paths.data_raw / cfg.data.name``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from neorec.utils.io import download_file, ensure_dir, unzip

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> Path:
    """Download and extract the configured MovieLens release.

    Returns the directory containing the extracted files (e.g.
    ``data/raw/movielens_1m/ml-1m``).
    """
    raw_dir = ensure_dir(Path(cfg.paths.data_raw) / cfg.data.name)
    archive_name = Path(str(cfg.data.url)).name
    archive_path = raw_dir / archive_name

    download_file(str(cfg.data.url), archive_path)
    extracted = unzip(archive_path, raw_dir)

    inner = next((p for p in extracted.iterdir() if p.is_dir()), extracted)
    log.info("Dataset ready at: %s", inner)
    return inner
