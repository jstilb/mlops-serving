"""Integration tests for the FastAPI application.

Uses httpx.AsyncClient for async testing of all API endpoints.
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    """Create an async test client with the full app.

    Uses a temp directory for the model registry to avoid polluting
    the project directory with test artifacts.
    """
    monkeypatch.setenv("MLOPS_MODEL_REGISTRY_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("MLOPS_LOG_FORMAT", "console")

    app = create_app()

    # Manually trigger lifespan
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


class TestHealthEndpoints:
    """Tests for health and readiness checks."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """Health endpoint returns 200 with status info."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_readiness_check(self, client: AsyncClient):
        """Readiness endpoint returns check details."""
        response = await client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert "ready" in data
        assert "checks" in data


class TestPredictionEndpoints:
    """Tests for prediction API."""

    @pytest.mark.asyncio
    async def test_single_prediction(self, client: AsyncClient):
        """Single sample prediction returns valid response."""
        response = await client.post(
            "/api/v1/predict",
            json={
                "features": [[13.0, 1.5, 2.3, 15.0, 110.0, 2.5, 2.8, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0]],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["model_id"] == "default"
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_batch_prediction(self, client: AsyncClient):
        """Batch prediction returns results for all samples."""
        features = [
            [13.0, 1.5, 2.3, 15.0, 110.0, 2.5, 2.8, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0],
            [12.0, 2.0, 2.0, 20.0, 90.0, 2.0, 2.0, 0.4, 1.0, 4.0, 0.8, 2.5, 800.0],
            [14.0, 1.0, 2.5, 10.0, 120.0, 3.0, 3.0, 0.2, 2.0, 6.0, 1.2, 3.5, 1200.0],
        ]
        response = await client.post("/api/v1/predict", json={"features": features})
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 3

    @pytest.mark.asyncio
    async def test_prediction_with_probabilities(self, client: AsyncClient):
        """Predictions include probability distributions when requested."""
        response = await client.post(
            "/api/v1/predict",
            json={
                "features": [[13.0, 1.5, 2.3, 15.0, 110.0, 2.5, 2.8, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0]],
                "include_probabilities": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["probabilities"] is not None
        assert len(data["probabilities"]) == 1

    @pytest.mark.asyncio
    async def test_prediction_invalid_features(self, client: AsyncClient):
        """Invalid features return 422."""
        response = await client.post(
            "/api/v1/predict",
            json={"features": []},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_request_tracing_headers(self, client: AsyncClient):
        """Response includes tracing headers."""
        response = await client.post(
            "/api/v1/predict",
            json={
                "features": [[13.0, 1.5, 2.3, 15.0, 110.0, 2.5, 2.8, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0]],
            },
        )
        assert "x-request-id" in response.headers
        assert "x-trace-id" in response.headers
        assert "x-response-time-ms" in response.headers


class TestModelEndpoints:
    """Tests for model management API."""

    @pytest.mark.asyncio
    async def test_list_models(self, client: AsyncClient):
        """List models returns registered models."""
        response = await client.get("/api/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_get_model_info(self, client: AsyncClient):
        """Get specific model version info."""
        response = await client.get("/api/v1/models/default/v1.0")
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == "default"
        assert data["version"] == "v1.0"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_nonexistent_model(self, client: AsyncClient):
        """Getting non-existent model returns 404."""
        response = await client.get("/api/v1/models/nonexistent/v1.0")
        assert response.status_code == 404


class TestABTestEndpoints:
    """Tests for A/B testing API."""

    @pytest.mark.asyncio
    async def test_list_ab_tests_empty(self, client: AsyncClient):
        """Initially no A/B tests exist."""
        response = await client.get("/api/v1/models/ab-tests")
        assert response.status_code == 200
        assert response.json() == []


class TestDriftEndpoint:
    """Tests for drift detection API."""

    @pytest.mark.asyncio
    async def test_check_drift(self, client: AsyncClient):
        """Drift check returns report."""
        # First make some predictions to populate drift windows
        for _ in range(5):
            await client.post(
                "/api/v1/predict",
                json={
                    "features": [[13.0, 1.5, 2.3, 15.0, 110.0, 2.5, 2.8, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0]],
                },
            )

        response = await client.get("/api/v1/models/default/drift")
        assert response.status_code == 200

        data = response.json()
        assert "overall_drift_detected" in data
        assert "features" in data
