"""Smoke tests for the FastAPI app — verifies wiring, not model outputs."""

from __future__ import annotations

from fastapi.testclient import TestClient

from neorec.serving.api import app
from neorec.serving.pipeline import RecommendationItem, RecommendationResponse


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
        app.state.pipeline = None
        app.state.startup_error = None
        r = client.get("/recommend/1")
        assert r.status_code == 503


class _DummyPipeline:
    def recommend(self, user_id: int, k: int = 10, diversity: float = 0.5):
        assert user_id == 1
        assert k == 2
        assert diversity == 0.7
        return RecommendationResponse(
            user_id=user_id,
            items=[
                RecommendationItem(
                    item_id=42,
                    title="Toy Story (1995)",
                    score=0.9,
                    channel="din",
                    explain="test",
                )
            ],
            latency_ms={"recall": 1.0, "total": 2.0},
        )


def test_recommend_uses_pipeline_and_records_latency() -> None:
    with TestClient(app) as client:
        app.state.pipeline = _DummyPipeline()
        r = client.get("/recommend/1?k=2&diversity=0.7")
        assert r.status_code == 200
        payload = r.json()
        assert payload["user_id"] == 1
        assert payload["items"][0]["item_id"] == 42
        assert payload["items"][0]["title"] == "Toy Story (1995)"
        assert payload["latency_ms"]["recall"] == 1.0
