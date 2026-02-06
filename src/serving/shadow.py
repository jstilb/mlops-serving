"""Shadow deployment for safe model validation.

Runs a shadow model alongside the primary model. The shadow model's
predictions are recorded for comparison but never returned to clients.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from src.monitoring.metrics import SHADOW_PREDICTION_DIVERGENCE
from src.serving.predictor import PredictionResult, Predictor

logger = structlog.get_logger(__name__)


class ShadowDeployment:
    """Shadow model deployment for offline comparison.

    The primary model serves traffic normally. The shadow model runs
    in parallel, and its predictions are logged for analysis without
    affecting the response to the client.

    Args:
        predictor: Predictor instance.
        model_id: Model identifier.
        primary_version: Version serving live traffic.
        shadow_version: Version running in shadow mode.
    """

    def __init__(
        self,
        predictor: Predictor,
        model_id: str,
        primary_version: str,
        shadow_version: str,
    ) -> None:
        self.predictor = predictor
        self.model_id = model_id
        self.primary_version = primary_version
        self.shadow_version = shadow_version
        self._comparison_log: list[dict[str, Any]] = []
        self._max_log_size = 10000

    def predict(
        self,
        features: list[list[float]],
        *,
        include_probabilities: bool = True,
    ) -> PredictionResult:
        """Run prediction on both primary and shadow models.

        Only the primary model's result is returned. Shadow results
        are logged for comparison.

        Args:
            features: Input feature vectors.
            include_probabilities: Include class probabilities.

        Returns:
            PredictionResult from the primary model only.
        """
        # Primary prediction (this is what the client gets)
        primary_result = self.predictor.predict(
            features,
            self.model_id,
            self.primary_version,
            include_probabilities=include_probabilities,
        )

        # Shadow prediction (logged, not returned)
        try:
            shadow_result = self.predictor.predict(
                features,
                self.model_id,
                self.shadow_version,
                include_probabilities=include_probabilities,
            )

            # Compare results
            divergence = self._compute_divergence(
                primary_result.predictions, shadow_result.predictions
            )

            SHADOW_PREDICTION_DIVERGENCE.labels(model_id=self.model_id).observe(divergence)

            comparison = {
                "primary_predictions": primary_result.predictions,
                "shadow_predictions": shadow_result.predictions,
                "divergence": divergence,
                "primary_latency_ms": primary_result.latency_ms,
                "shadow_latency_ms": shadow_result.latency_ms,
            }

            if len(self._comparison_log) >= self._max_log_size:
                self._comparison_log = self._comparison_log[-self._max_log_size // 2 :]

            self._comparison_log.append(comparison)

            logger.info(
                "shadow_comparison",
                model_id=self.model_id,
                divergence=round(divergence, 4),
                primary_version=self.primary_version,
                shadow_version=self.shadow_version,
            )

        except Exception:
            logger.exception(
                "shadow_prediction_failed",
                model_id=self.model_id,
                shadow_version=self.shadow_version,
            )

        return primary_result

    def get_comparison_summary(self) -> dict[str, Any]:
        """Get summary statistics of shadow vs primary comparisons."""
        if not self._comparison_log:
            return {"total_comparisons": 0, "message": "No comparisons recorded yet"}

        divergences = [c["divergence"] for c in self._comparison_log]
        primary_latencies = [c["primary_latency_ms"] for c in self._comparison_log]
        shadow_latencies = [c["shadow_latency_ms"] for c in self._comparison_log]

        agreement_count = sum(1 for d in divergences if d == 0.0)

        return {
            "total_comparisons": len(self._comparison_log),
            "agreement_rate": round(agreement_count / len(self._comparison_log), 4),
            "mean_divergence": round(float(np.mean(divergences)), 4),
            "p95_divergence": round(float(np.percentile(divergences, 95)), 4),
            "primary_mean_latency_ms": round(float(np.mean(primary_latencies)), 2),
            "shadow_mean_latency_ms": round(float(np.mean(shadow_latencies)), 2),
            "primary_version": self.primary_version,
            "shadow_version": self.shadow_version,
        }

    @staticmethod
    def _compute_divergence(primary: list[Any], shadow: list[Any]) -> float:
        """Compute divergence between primary and shadow predictions.

        For classification: fraction of disagreements.
        """
        if len(primary) != len(shadow):
            return 1.0

        if not primary:
            return 0.0

        disagreements = sum(1 for p, s in zip(primary, shadow) if p != s)
        return disagreements / len(primary)
