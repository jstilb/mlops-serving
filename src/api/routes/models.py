"""Model management endpoints for registry operations.

Provides CRUD-like operations on the model registry: listing models,
viewing versions, promoting, and deprecating model versions.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_ab_manager, get_drift_detector, get_model_loader, get_registry
from src.api.schemas import (
    ABTestCreateRequest,
    ABTestResponse,
    DriftReportResponse,
    ModelInfo,
    ModelListResponse,
    ModelPromoteRequest,
)
from src.models.loader import ModelLoader
from src.models.registry import ModelRegistry, ModelStatus
from src.monitoring.drift import DriftDetector
from src.serving.ab_testing import ABTestConfig, ABTestManager

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models(
    registry: ModelRegistry = Depends(get_registry),
) -> ModelListResponse:
    """List all registered models and their versions."""
    all_models: list[ModelInfo] = []

    for model_id in registry.list_models():
        for meta in registry.list_versions(model_id):
            all_models.append(
                ModelInfo(
                    model_id=meta.model_id,
                    version=meta.version,
                    status=meta.status.value,
                    algorithm=meta.algorithm,
                    created_at=meta.created_at,
                    promoted_at=meta.promoted_at,
                    training_metrics=meta.training_metrics,
                    feature_names=meta.feature_names,
                    target_names=meta.target_names,
                )
            )

    return ModelListResponse(models=all_models, total=len(all_models))


# --- Drift Detection Endpoints ---
# NOTE: This route MUST be defined before /{model_id}/{version} to avoid
# the path parameter matching "drift" as a version string.


@router.get("/{model_id}/drift", response_model=DriftReportResponse)
async def check_drift(
    model_id: str,
    drift_detector: DriftDetector | None = Depends(get_drift_detector),
) -> DriftReportResponse:
    """Check data drift for a model's features."""
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detection not configured")

    report = drift_detector.check_drift()

    return DriftReportResponse(
        model_id=report.model_id,
        overall_drift_detected=report.overall_drift_detected,
        drifted_features=report.drifted_features,
        features=[
            {
                "feature_name": f.feature_name,
                "ks_statistic": f.ks_statistic,
                "p_value": f.p_value,
                "is_drifted": f.is_drifted,
                "sample_size": f.sample_size,
                "reference_mean": f.reference_mean,
                "current_mean": f.current_mean,
            }
            for f in report.features
        ],
    )


@router.get("/{model_id}/{version}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    version: str,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelInfo:
    """Get metadata for a specific model version."""
    try:
        meta = registry.get_metadata(model_id, version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return ModelInfo(
        model_id=meta.model_id,
        version=meta.version,
        status=meta.status.value,
        algorithm=meta.algorithm,
        created_at=meta.created_at,
        promoted_at=meta.promoted_at,
        training_metrics=meta.training_metrics,
        feature_names=meta.feature_names,
        target_names=meta.target_names,
    )


@router.post("/{model_id}/promote", response_model=ModelInfo)
async def promote_model(
    model_id: str,
    request: ModelPromoteRequest,
    registry: ModelRegistry = Depends(get_registry),
    loader: ModelLoader = Depends(get_model_loader),
) -> ModelInfo:
    """Promote a model version to a new status.

    Valid transitions:
    - shadow -> canary -> active
    - any -> deprecated

    When promoting to active, the current active version is automatically deprecated.
    """
    try:
        target_status = ModelStatus(request.to_status)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{request.to_status}'. Must be one of: {[s.value for s in ModelStatus]}",
        ) from e

    try:
        meta = registry.promote(model_id, request.version, target_status)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Invalidate cache for this model so new version is loaded
    loader.invalidate(model_id)

    return ModelInfo(
        model_id=meta.model_id,
        version=meta.version,
        status=meta.status.value,
        algorithm=meta.algorithm,
        created_at=meta.created_at,
        promoted_at=meta.promoted_at,
        training_metrics=meta.training_metrics,
        feature_names=meta.feature_names,
        target_names=meta.target_names,
    )


# --- A/B Testing Endpoints ---


@router.post("/ab-tests", response_model=ABTestResponse)
async def create_ab_test(
    request: ABTestCreateRequest,
    ab_manager: ABTestManager = Depends(get_ab_manager),
) -> ABTestResponse:
    """Create an A/B test between two model versions."""
    config = ABTestConfig(
        model_id=request.model_id,
        control_version=request.control_version,
        treatment_version=request.treatment_version,
        traffic_split=request.traffic_split,
        name=request.name,
    )
    try:
        ab_manager.create_test(config)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    return ABTestResponse(
        model_id=config.model_id,
        name=config.name,
        control_version=config.control_version,
        treatment_version=config.treatment_version,
        traffic_split=config.traffic_split,
    )


@router.get("/ab-tests", response_model=list[ABTestResponse])
async def list_ab_tests(
    ab_manager: ABTestManager = Depends(get_ab_manager),
) -> list[ABTestResponse]:
    """List all active A/B tests."""
    return [ABTestResponse(**t) for t in ab_manager.list_tests()]


@router.delete("/ab-tests/{model_id}")
async def delete_ab_test(
    model_id: str,
    ab_manager: ABTestManager = Depends(get_ab_manager),
) -> dict[str, str]:
    """Remove an A/B test."""
    ab_manager.remove_test(model_id)
    return {"status": "removed", "model_id": model_id}


