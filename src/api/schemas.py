"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Prediction Schemas ---


class PredictionRequest(BaseModel):
    """Single or batch prediction request."""

    features: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Feature vectors. Each inner list is one sample.",
        json_schema_extra={"examples": [[[5.1, 3.5, 1.4, 0.2]]]},
    )
    model_id: str = Field(default="default", description="Model identifier")
    model_version: str | None = Field(
        default=None, description="Specific model version. None = active version."
    )
    include_probabilities: bool = Field(
        default=True, description="Include class probabilities in response"
    )
    request_id: str | None = Field(
        default=None, description="Optional request ID for tracing and A/B test sticky sessions"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]],
                    "model_id": "default",
                    "include_probabilities": True,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prediction response with metadata."""

    predictions: list[int | float | str]
    probabilities: list[list[float]] | None = None
    model_id: str
    model_version: str
    latency_ms: float
    variant: str | None = Field(
        default=None, description="A/B test variant (control/treatment) if test is active"
    )


# --- Health Schemas ---


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    models_loaded: int


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    checks: dict[str, bool]


# --- Model Management Schemas ---


class ModelInfo(BaseModel):
    """Model version information."""

    model_id: str
    version: str
    status: str
    algorithm: str
    created_at: str
    promoted_at: str | None = None
    training_metrics: dict[str, float]
    feature_names: list[str]
    target_names: list[str]


class ModelPromoteRequest(BaseModel):
    """Request to promote a model version."""

    model_id: str
    version: str
    to_status: str = Field(
        description="Target status: active, shadow, canary, deprecated"
    )


class ModelListResponse(BaseModel):
    """Response listing registered models."""

    models: list[ModelInfo]
    total: int


# --- A/B Test Schemas ---


class ABTestCreateRequest(BaseModel):
    """Request to create an A/B test."""

    model_id: str
    control_version: str
    treatment_version: str
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0)
    name: str = ""


class ABTestResponse(BaseModel):
    """A/B test configuration response."""

    model_id: str
    name: str
    control_version: str
    treatment_version: str
    traffic_split: float


# --- Drift Schemas ---


class DriftFeatureResult(BaseModel):
    """Drift detection result for a single feature."""

    feature_name: str
    ks_statistic: float
    p_value: float
    is_drifted: bool
    sample_size: int
    reference_mean: float
    current_mean: float


class DriftReportResponse(BaseModel):
    """Drift detection report."""

    model_id: str
    overall_drift_detected: bool
    drifted_features: list[str]
    features: list[DriftFeatureResult]
