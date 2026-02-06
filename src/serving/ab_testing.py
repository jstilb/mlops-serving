"""A/B testing with traffic splitting between model versions.

Supports weighted random assignment and deterministic assignment
based on request ID for consistent user experiences.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.monitoring.metrics import AB_TEST_ASSIGNMENTS
from src.serving.predictor import PredictionResult, Predictor

logger = structlog.get_logger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test between model versions."""

    model_id: str
    control_version: str
    treatment_version: str
    traffic_split: float = 0.5  # fraction to treatment
    sticky_sessions: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.traffic_split <= 1.0:
            raise ValueError(f"traffic_split must be between 0 and 1, got {self.traffic_split}")


@dataclass
class ABTestResult:
    """Result of an A/B test prediction with variant info."""

    prediction: PredictionResult
    variant: str  # "control" or "treatment"
    test_name: str = ""


class ABTestManager:
    """Manages A/B tests between model versions.

    Supports:
    - Weighted random traffic splitting
    - Sticky sessions via request ID hashing
    - Metrics recording per variant

    Args:
        predictor: Predictor instance for running inference.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self._active_tests: dict[str, ABTestConfig] = {}

    def create_test(self, config: ABTestConfig) -> None:
        """Register a new A/B test.

        Args:
            config: Test configuration.

        Raises:
            ValueError: If a test already exists for this model.
        """
        if config.model_id in self._active_tests:
            raise ValueError(f"A/B test already exists for model '{config.model_id}'")

        self._active_tests[config.model_id] = config
        logger.info(
            "ab_test_created",
            model_id=config.model_id,
            control=config.control_version,
            treatment=config.treatment_version,
            split=config.traffic_split,
        )

    def remove_test(self, model_id: str) -> None:
        """Remove an A/B test."""
        self._active_tests.pop(model_id, None)

    def get_test(self, model_id: str) -> ABTestConfig | None:
        """Get active test config for a model."""
        return self._active_tests.get(model_id)

    def predict(
        self,
        features: list[list[float]],
        model_id: str = "default",
        request_id: str | None = None,
        *,
        include_probabilities: bool = True,
    ) -> ABTestResult:
        """Run prediction through A/B test if active, otherwise use active model.

        Args:
            features: Input feature vectors.
            model_id: Model identifier.
            request_id: Optional request ID for sticky sessions.
            include_probabilities: Include class probabilities.

        Returns:
            ABTestResult with prediction and variant assignment.
        """
        test = self._active_tests.get(model_id)

        if test is None:
            # No active test - use default active model
            result = self.predictor.predict(
                features, model_id, include_probabilities=include_probabilities
            )
            return ABTestResult(prediction=result, variant="default")

        # Assign variant
        variant = self._assign_variant(test, request_id)
        version = test.treatment_version if variant == "treatment" else test.control_version

        # Record assignment
        AB_TEST_ASSIGNMENTS.labels(model_id=model_id, variant=variant).inc()

        # Run prediction with assigned version
        result = self.predictor.predict(
            features, model_id, version, include_probabilities=include_probabilities
        )

        logger.info(
            "ab_test_prediction",
            model_id=model_id,
            variant=variant,
            version=version,
            request_id=request_id,
        )

        return ABTestResult(
            prediction=result,
            variant=variant,
            test_name=test.name,
        )

    def list_tests(self) -> list[dict[str, Any]]:
        """List all active A/B tests."""
        return [
            {
                "model_id": config.model_id,
                "name": config.name,
                "control_version": config.control_version,
                "treatment_version": config.treatment_version,
                "traffic_split": config.traffic_split,
            }
            for config in self._active_tests.values()
        ]

    @staticmethod
    def _assign_variant(config: ABTestConfig, request_id: str | None) -> str:
        """Assign a request to control or treatment.

        Uses deterministic hashing for sticky sessions, or random assignment.
        """
        if config.sticky_sessions and request_id:
            # Deterministic assignment based on request ID hash
            hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            fraction = (hash_val % 10000) / 10000.0
        else:
            fraction = random.random()

        return "treatment" if fraction < config.traffic_split else "control"
