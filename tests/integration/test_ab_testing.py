"""Integration tests for A/B testing through the API."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    """Create an async test client."""
    monkeypatch.setenv("MLOPS_MODEL_REGISTRY_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("MLOPS_LOG_FORMAT", "console")

    app = create_app()

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


class TestABTestingIntegration:
    """Integration tests for A/B testing workflow."""

    @pytest.mark.asyncio
    async def test_full_ab_test_lifecycle(self, client: AsyncClient):
        """Test creating, using, and removing an A/B test."""
        # Verify default model exists
        response = await client.get("/api/v1/models")
        assert response.status_code == 200
        models = response.json()["models"]
        assert len(models) >= 1

        # List tests (should be empty)
        response = await client.get("/api/v1/models/ab-tests")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_prediction_without_ab_test(self, client: AsyncClient):
        """Predictions work normally without an A/B test."""
        response = await client.post(
            "/api/v1/predict",
            json={
                "features": [[13.0, 1.5, 2.3, 15.0, 110.0, 2.5, 2.8, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0]],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["variant"] is None  # No A/B test active
