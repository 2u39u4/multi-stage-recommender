"""Smoke tests for the FastAPI app — verifies wiring, not model outputs."""

from __future__ import annotations

from fastapi.testclient import TestClient

from neorec.serving.api import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_metrics_endpoint_exposes_prometheus() -> None:
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        # Prometheus exposition format uses text/plain
        assert "neorec_requests_total" in r.text


def test_recommend_returns_503_without_pipeline() -> None:
    with TestClient(app) as client:
        r = client.get("/recommend/1")
        assert r.status_code == 503
