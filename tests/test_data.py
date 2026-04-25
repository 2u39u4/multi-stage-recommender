"""Smoke tests for the data pipeline."""

from __future__ import annotations

import zipfile
from pathlib import Path

from omegaconf import OmegaConf

from neorec.data.preprocess import run as run_preprocess


def _write_tiny_ml1m(raw_dir: Path) -> None:
    """Synthesize a minimal ml-1m-shaped dataset (10 users × 8 movies)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    # ratings.dat: UserID::MovieID::Rating::Timestamp
    rows = []
    ts = 978_300_000
    for u in range(1, 11):
        for m in [(u % 8) + 1, ((u + 1) % 8) + 1, ((u + 2) % 8) + 1,
                  ((u + 3) % 8) + 1, ((u + 4) % 8) + 1]:
            rows.append(f"{u}::{m}::5::{ts}")
            ts += 1
    (raw_dir / "ratings.dat").write_text("\n".join(rows))

    # users.dat: UserID::Gender::Age::Occupation::Zip-code
    user_lines = [f"{u}::M::25::4::00000" for u in range(1, 11)]
    (raw_dir / "users.dat").write_text("\n".join(user_lines))

    # movies.dat: MovieID::Title::Genres
    genres = ["Action|Drama", "Comedy", "Drama", "Action", "Sci-Fi",
              "Romance", "Thriller", "Comedy|Drama"]
    movie_lines = [f"{i + 1}::Movie {i + 1} (199{i}) ::{g}" for i, g in enumerate(genres)]
    (raw_dir / "movies.dat").write_text("\n".join(movie_lines))


def test_preprocess_creates_all_outputs(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "movielens_1m" / "ml-1m"
    _write_tiny_ml1m(raw_root)

    cfg = OmegaConf.create(
        {
            "paths": {
                "data_raw": str(tmp_path / "data" / "raw"),
                "data_processed": str(tmp_path / "data" / "processed"),
            },
            "data": {
                "name": "movielens_1m",
                "size": "1m",
                "split": {"strategy": "leave_one_out", "min_interactions_per_user": 2,
                          "valid_ratio": 0.1, "test_ratio": 0.1},
                "feedback": {"type": "implicit", "rating_threshold": 4.0},
                "features": {"sequence": {"max_len": 50, "padding_value": 0}},
            },
        }
    )

    paths = run_preprocess(cfg)

    for key in ("interactions", "user_features", "item_features",
                "sequences", "split", "id_maps", "stats"):
        assert paths[key].exists(), f"{key} missing"


def test_zipfile_helper_round_trip(tmp_path: Path) -> None:
    """The unzip helper must skip extraction when top-level dir is already there."""
    from neorec.utils.io import unzip

    src = tmp_path / "src"
    (src / "ml-1m").mkdir(parents=True)
    (src / "ml-1m" / "hello.txt").write_text("hi")

    archive = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(src / "ml-1m" / "hello.txt", arcname="ml-1m/hello.txt")

    extracted_dir = tmp_path / "out"
    out = unzip(archive, extracted_dir)
    assert (out / "ml-1m" / "hello.txt").read_text() == "hi"

    # Second call must short-circuit (top-level dir present)
    out2 = unzip(archive, extracted_dir)
    assert out2 == extracted_dir
