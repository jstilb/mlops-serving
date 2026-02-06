"""Health and readiness check endpoints.

Provides /health for liveness probes and /ready for readiness probes.
These are critical for container orchestration (Docker, Kubernetes).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_model_loader, get_registry, get_settings_dep
from src.api.schemas import HealthResponse, ReadinessResponse
from src.config import Settings
from src.models.loader import ModelLoader
from src.models.registry import ModelRegistry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings_dep),
    loader: ModelLoader = Depends(get_model_loader),
) -> HealthResponse:
    """Liveness probe. Returns 200 if the service is running.

    Used by container orchestrators to determine if the container is alive.
    Does not check dependencies -- that is what /ready is for.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        models_loaded=loader.cache_info["size"],
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(
    registry: ModelRegistry = Depends(get_registry),
    loader: ModelLoader = Depends(get_model_loader),
) -> ReadinessResponse:
    """Readiness probe. Returns 200 only if the service can serve predictions.

    Checks:
    - Model registry is accessible
    - At least one model is registered
    - An active model version exists
    """
    checks: dict[str, bool] = {}

    # Check registry access
    try:
        models = registry.list_models()
        checks["registry_accessible"] = True
        checks["models_registered"] = len(models) > 0
    except Exception:
        checks["registry_accessible"] = False
        checks["models_registered"] = False

    # Check for active model
    if checks.get("models_registered"):
        try:
            for model_id in models:
                active = registry.get_active_version(model_id)
                if active is not None:
                    checks["active_model_exists"] = True
                    break
            else:
                checks["active_model_exists"] = False
        except Exception:
            checks["active_model_exists"] = False
    else:
        checks["active_model_exists"] = False

    ready = all(checks.values())

    return ReadinessResponse(ready=ready, checks=checks)
