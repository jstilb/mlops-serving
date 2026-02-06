"""Prediction logic with pre/post processing and metrics instrumentation.

Handles single and batch predictions with automatic metric recording,
drift monitoring, and structured logging.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from src.models.loader import ModelLoader
from src.models.registry import ModelRegistry
from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import (
    BATCH_SIZE,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
)

logger = structlog.get_logger(__name__)


class PredictionResult:
    """Structured prediction output."""

    def __init__(
        self,
        predictions: list[Any],
        probabilities: list[list[float]] | None = None,
        model_id: str = "",
        model_version: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        self.predictions = predictions
        self.probabilities = probabilities
        self.model_id = model_id
        self.model_version = model_version
        self.latency_ms = latency_ms

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "predictions": self.predictions,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.probabilities is not None:
            result["probabilities"] = self.probabilities
        return result


class Predictor:
    """Production predictor with instrumentation and drift detection.

    Args:
        model_loader: Model loader with caching.
        registry: Model registry for metadata lookups.
        drift_detector: Optional drift detector for feature monitoring.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        registry: ModelRegistry,
        drift_detector: DriftDetector | None = None,
    ) -> None:
        self.model_loader = model_loader
        self.registry = registry
        self.drift_detector = drift_detector

    def predict(
        self,
        features: list[list[float]],
        model_id: str = "default",
        model_version: str | None = None,
        *,
        include_probabilities: bool = True,
    ) -> PredictionResult:
        """Run prediction with full instrumentation.

        Args:
            features: Input features as list of feature vectors.
            model_id: Model to use for prediction.
            model_version: Specific version (None = active version).
            include_probabilities: Whether to include class probabilities.

        Returns:
            PredictionResult with predictions and metadata.

        Raises:
            RuntimeError: If model cannot be loaded.
            ValueError: If input features are invalid.
        """
        start_time = time.perf_counter()

        # Resolve version
        if model_version is None:
            active = self.registry.get_active_version(model_id)
            if active is None:
                raise RuntimeError(f"No active version for model '{model_id}'")
            model_version = active.version

        try:
            # Load model
            model = self.model_loader.get(model_id, model_version)

            # Convert to numpy
            feature_array = np.array(features, dtype=np.float64)
            BATCH_SIZE.observe(len(feature_array))

            # Run prediction
            predictions = model.predict(feature_array).tolist()

            # Get probabilities if available
            probabilities = None
            if include_probabilities and hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_array)
                probabilities = [[round(p, 4) for p in row] for row in proba]

            # Record drift observation
            if self.drift_detector is not None:
                self.drift_detector.observe(feature_array)

            # Calculate latency
            latency_s = time.perf_counter() - start_time
            latency_ms = latency_s * 1000

            # Record metrics
            PREDICTION_LATENCY.labels(
                model_id=model_id,
                model_version=model_version,
                endpoint="predict",
            ).observe(latency_s)
            PREDICTION_COUNT.labels(
                model_id=model_id,
                model_version=model_version,
                status="success",
            ).inc(len(predictions))

            logger.info(
                "prediction_complete",
                model_id=model_id,
                model_version=model_version,
                batch_size=len(features),
                latency_ms=round(latency_ms, 2),
            )

            return PredictionResult(
                predictions=predictions,
                probabilities=probabilities,
                model_id=model_id,
                model_version=model_version,
                latency_ms=latency_ms,
            )

        except Exception as e:
            PREDICTION_ERRORS.labels(
                model_id=model_id,
                model_version=model_version or "unknown",
                error_type=type(e).__name__,
            ).inc()
            logger.error(
                "prediction_failed",
                model_id=model_id,
                model_version=model_version,
                error=str(e),
            )
            raise
