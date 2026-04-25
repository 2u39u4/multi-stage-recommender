"""FastAPI server — exposes /recommend, /health, /metrics."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Lifespan — load heavy artefacts once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index, ranking models, and feature cache connections."""
    log.info("NeoRec API starting up…")
    # TODO(W5 Day 29): instantiate OnlinePipeline.from_config(...) and attach to app.state
    app.state.pipeline = None
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
    return {"status": "ok", "version": app.version}


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
        raise HTTPException(status_code=503, detail="Pipeline not ready")

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
