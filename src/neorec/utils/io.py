"""Small filesystem and download helpers."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file(
    url: str,
    dest: str | Path,
    chunk_size: int = 1 << 20,
    overwrite: bool = False,
) -> Path:
    """Stream-download a URL to ``dest`` with a tqdm progress bar.

    Returns the local path. Skips download if file already exists and
    ``overwrite`` is False.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        log.info("File already present, skipping: %s", dest)
        return dest

    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    log.info("Downloading %s → %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "neorec/0.1"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0)) or None
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            out.write(buf)
            if bar is not None:
                bar.update(len(buf))
        if bar is not None:
            bar.close()

    tmp.rename(dest)
    return dest


def sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def unzip(zip_path: str | Path, dest_dir: str | Path, overwrite: bool = False) -> Path:
    """Extract a zip archive into ``dest_dir`` (creating it if needed).

    Skips extraction only if the zip's top-level directory already exists in
    ``dest_dir`` (so that having only the ``.zip`` itself in there does not
    short-circuit extraction).
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        top_dirs = {m.split("/", 1)[0] for m in members if "/" in m or not m.endswith("/")}
        already_extracted = bool(top_dirs) and all(
            (dest_dir / d).exists() for d in top_dirs
        )

    if already_extracted and not overwrite:
        log.info("Already extracted: %s", dest_dir)
        return dest_dir

    if overwrite:
        for d in top_dirs:
            target = dest_dir / d
            if target.exists():
                shutil.rmtree(target)

    ensure_dir(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    log.info("Extracted %s → %s", zip_path, dest_dir)
    return dest_dir


def write_json(obj: Any, path: str | Path) -> None:
    import json

    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    import json

    with open(path) as f:
        return json.load(f)
