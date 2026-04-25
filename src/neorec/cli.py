"""Unified command-line interface for NeoRec.

Examples
--------
    neorec data download dataset=movielens_1m
    neorec data preprocess
    neorec train recall=als
    neorec train recall=sasrec recall.model.embedding_dim=128
    neorec train rank=deepfm
    neorec eval pipeline=full
    neorec serve
"""

from __future__ import annotations

import logging

import hydra
import typer
from omegaconf import DictConfig, OmegaConf

from neorec.utils.logger import setup_logging
from neorec.utils.seed import set_seed

log = logging.getLogger(__name__)

app = typer.Typer(
    name="neorec",
    add_completion=False,
    help="NeoRec — production-grade multi-stage recommender system.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Internal Hydra-driven entry point — called by each Typer subcommand.
# ---------------------------------------------------------------------------
def _run_with_hydra(
    task: str,
    overrides: list[str] | None = None,
) -> None:
    """Compose a Hydra config, then dispatch to the right task runner."""
    overrides = overrides or []

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
    def _main(cfg: DictConfig) -> None:
        setup_logging(level=cfg.logging.level, fmt=cfg.logging.format)
        set_seed(cfg.seed)
        log.info("Task: %s", task)
        log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

        if task == "data.download":
            from neorec.data.download import run as _r
        elif task == "data.preprocess":
            from neorec.data.preprocess import run as _r
        elif task == "train.recall":
            from neorec.recall.train import run as _r
        elif task == "train.rank":
            from neorec.ranking.train import run as _r
        elif task == "eval.pipeline":
            from neorec.eval.pipeline import run as _r
        else:
            raise ValueError(f"Unknown task: {task}")

        _r(cfg)

    import sys

    sys.argv = [sys.argv[0], *overrides]
    _main()


# ---------------------------------------------------------------------------
# Public subcommands
# ---------------------------------------------------------------------------
data_app = typer.Typer(help="Dataset pipeline: download & preprocessing.")
train_app = typer.Typer(help="Train recall or ranking models.")
eval_app = typer.Typer(help="Offline evaluation and ablation studies.")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")


@data_app.command("download")
def data_download(overrides: list[str] = typer.Argument(None)) -> None:
    """Download the raw dataset declared in configs/data/*."""
    _run_with_hydra("data.download", overrides)


@data_app.command("preprocess")
def data_preprocess(overrides: list[str] = typer.Argument(None)) -> None:
    """Clean, re-index, split, and featurize the raw data."""
    _run_with_hydra("data.preprocess", overrides)


@train_app.command("recall")
def train_recall(overrides: list[str] = typer.Argument(None)) -> None:
    """Train a single recall channel."""
    _run_with_hydra("train.recall", overrides)


@train_app.command("rank")
def train_rank(overrides: list[str] = typer.Argument(None)) -> None:
    """Train a ranking model (pre-rank or fine-rank)."""
    _run_with_hydra("train.rank", overrides)


@eval_app.command("pipeline")
def eval_pipeline(overrides: list[str] = typer.Argument(None)) -> None:
    """Run the full end-to-end evaluation and log to MLflow."""
    _run_with_hydra("eval.pipeline", overrides)


@app.command("serve")
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Launch the FastAPI serving app."""
    import uvicorn

    uvicorn.run(
        "neorec.serving.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
