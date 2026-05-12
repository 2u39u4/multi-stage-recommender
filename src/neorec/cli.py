"""Unified command-line interface for NeoRec.

Examples
--------
    neorec data download
    neorec data preprocess
    neorec train recall=als
    neorec train recall=sasrec recall.model.embedding_dim=128
    neorec train rank=deepfm
    neorec rerank rank=din rerank=mmr rerank.mmr.lambda=0.7
    neorec eval pipeline=full
    neorec serve

Anything after the subcommand and before flags is passed to Hydra as a list
of dotted overrides, e.g. ``train recall=als recall.model.factors=128``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from neorec.utils.logger import setup_logging
from neorec.utils.seed import set_seed

log = logging.getLogger(__name__)

CONFIG_DIR = (Path(__file__).resolve().parents[2] / "configs").as_posix()


# ---------------------------------------------------------------------------
# Helper that composes the config and dispatches to the right Python entry.
# ---------------------------------------------------------------------------
def _compose(overrides: list[str] | None = None) -> DictConfig:
    overrides = list(overrides or [])
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _bootstrap(cfg: DictConfig) -> None:
    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format)
    set_seed(int(cfg.seed))
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))


# ---------------------------------------------------------------------------
# Typer apps
# ---------------------------------------------------------------------------
app = typer.Typer(
    name="neorec",
    add_completion=False,
    help="NeoRec — production-grade multi-stage recommender system.",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
data_app = typer.Typer(
    add_completion=False,
    help="Dataset pipeline: download & preprocessing.",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
train_app = typer.Typer(
    add_completion=False,
    help="Train recall or ranking models.",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
eval_app = typer.Typer(
    add_completion=False,
    help="Offline evaluation and ablation studies.",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")


def _extras(ctx: typer.Context) -> list[str]:
    return list(ctx.args)


_HYDRA_CTX = {"allow_extra_args": True, "ignore_unknown_options": True}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@data_app.command("download", context_settings=_HYDRA_CTX)
def data_download(ctx: typer.Context) -> None:
    """Download the raw dataset declared in configs/data/*."""
    cfg = _compose(_extras(ctx))
    _bootstrap(cfg)
    from neorec.data.download import run

    run(cfg)


@data_app.command("preprocess", context_settings=_HYDRA_CTX)
def data_preprocess(ctx: typer.Context) -> None:
    """Clean, re-index, split, and featurize the raw data."""
    cfg = _compose(_extras(ctx))
    _bootstrap(cfg)
    from neorec.data.preprocess import run

    run(cfg)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
@train_app.command("recall", context_settings=_HYDRA_CTX)
def train_recall(ctx: typer.Context) -> None:
    """Train a single recall channel."""
    cfg = _compose(_extras(ctx))
    _bootstrap(cfg)
    from neorec.recall.train import run

    run(cfg)


@train_app.command("rank", context_settings=_HYDRA_CTX)
def train_rank(ctx: typer.Context) -> None:
    """Train a ranking model (pre-rank or fine-rank)."""
    cfg = _compose(_extras(ctx))
    _bootstrap(cfg)
    from neorec.ranking.train import run

    run(cfg)


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------
@app.command("rerank", context_settings=_HYDRA_CTX)
def rerank(ctx: typer.Context) -> None:
    """Run the rerank pipeline (recall → rank → MMR / debias / rules)."""
    cfg = _compose(_extras(ctx))
    _bootstrap(cfg)
    from neorec.rerank.pipeline import run

    run(cfg)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
@eval_app.command("pipeline", context_settings=_HYDRA_CTX)
def eval_pipeline(ctx: typer.Context) -> None:
    """Run the full end-to-end evaluation."""
    cfg = _compose(_extras(ctx))
    _bootstrap(cfg)
    from neorec.eval.pipeline import run

    run(cfg)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command("serve")
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Launch the FastAPI serving app."""
    import uvicorn

    uvicorn.run("neorec.serving.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    sys.exit(app())
