"""FastAPI server — exposes /recommend, /health, /metrics."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from hydra import compose, initialize_config_dir
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from neorec.serving.pipeline import OnlinePipeline

log = logging.getLogger(__name__)
CONFIG_DIR = (Path(__file__).resolve().parents[3] / "configs").as_posix()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNTER = Counter(
    "neorec_requests_total",
    "Total number of recommendation requests",
    ["endpoint", "status"],
)
STAGE_LATENCY = Histogram(
    "neorec_stage_latency_ms",
    "Per-stage latency (ms)",
    ["stage"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RecItem(BaseModel):
    item_id: int
    title: str | None = None
    score: float
    channel: str
    explain: str | None = None


class RecResponse(BaseModel):
    user_id: int
    items: list[RecItem]
    latency_ms: dict[str, float]


def _compose_serving_config() -> Any:
    overrides = [
        "data.oof_split=true",
        "rank=din",
        "rerank=mmr",
    ]
    extra = os.environ.get("NEOREC_CONFIG_OVERRIDES", "")
    overrides.extend([x for x in extra.split() if x])
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


# ---------------------------------------------------------------------------
# Lifespan — load heavy artefacts once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index, ranking models, and feature cache connections."""
    log.info("NeoRec API starting up…")
    app.state.pipeline = None
    app.state.startup_error = None
    if os.environ.get("NEOREC_DISABLE_PIPELINE_LOAD", "0") == "1":
        log.warning("Pipeline loading disabled by NEOREC_DISABLE_PIPELINE_LOAD=1.")
    else:
        try:
            cfg = _compose_serving_config()
            app.state.pipeline = OnlinePipeline.from_config(cfg)
        except Exception as exc:
            # Keep /health and /metrics available even when local artefacts have not
            # been built yet; /recommend will return 503 with this diagnostic.
            app.state.startup_error = str(exc)
            log.exception("Failed to hydrate OnlinePipeline.")
    yield
    log.info("NeoRec API shutting down.")


app = FastAPI(
    title="NeoRec",
    description="Multi-stage recommender serving API.",
    version=os.environ.get("NEOREC_VERSION", "0.1.0"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "pipeline_ready": app.state.pipeline is not None,
        "startup_error": app.state.startup_error,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recommend/{user_id}", response_model=RecResponse)
def recommend(
    user_id: int,
    k: int = Query(10, ge=1, le=100),
    diversity: float = Query(0.5, ge=0.0, le=1.0),
) -> RecResponse:
    t0 = time.perf_counter()
    pipeline = app.state.pipeline
    if pipeline is None:
        REQUEST_COUNTER.labels("recommend", "503").inc()
        detail = "Pipeline not ready"
        if getattr(app.state, "startup_error", None):
            detail += f": {app.state.startup_error}"
        raise HTTPException(status_code=503, detail=detail)

    try:
        response = pipeline.recommend(user_id=user_id, k=k, diversity=diversity)
    except KeyError as e:
        REQUEST_COUNTER.labels("recommend", "404").inc()
        raise HTTPException(status_code=404, detail=str(e)) from e

    for stage, ms in response.latency_ms.items():
        STAGE_LATENCY.labels(stage).observe(ms)
    REQUEST_COUNTER.labels("recommend", "200").inc()

    total = (time.perf_counter() - t0) * 1000.0
    response.latency_ms.setdefault("total", total)
    return RecResponse(
        user_id=response.user_id,
        items=[RecItem(**i.__dict__) for i in response.items],
        latency_ms=response.latency_ms,
    )
