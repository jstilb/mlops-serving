"""Data drift detection using statistical tests on feature distributions.

Compares incoming feature distributions against a reference distribution
(captured at training time) using the Kolmogorov-Smirnov test.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog
from scipy import stats

from src.monitoring.metrics import DRIFT_DETECTED, FEATURE_DRIFT_SCORE

logger = structlog.get_logger(__name__)


@dataclass
class DriftResult:
    """Result of a drift detection check for a single feature."""

    feature_name: str
    ks_statistic: float
    p_value: float
    is_drifted: bool
    sample_size: int
    reference_mean: float
    current_mean: float


@dataclass
class DriftReport:
    """Aggregated drift report across all features."""

    model_id: str
    features: list[DriftResult]
    overall_drift_detected: bool
    timestamp: str = ""

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature_name for f in self.features if f.is_drifted]

    @property
    def drift_summary(self) -> dict[str, float]:
        return {f.feature_name: f.ks_statistic for f in self.features}


class DriftDetector:
    """Monitors feature distributions for data drift.

    Uses a sliding window of recent predictions compared against
    reference distributions from training data via KS test.

    Args:
        model_id: Model identifier for metric labels.
        reference_data: Training data feature matrix (n_samples, n_features).
        feature_names: Names for each feature column.
        window_size: Number of recent samples to keep for comparison.
        threshold: P-value threshold below which drift is flagged.
    """

    def __init__(
        self,
        model_id: str,
        reference_data: np.ndarray,
        feature_names: list[str],
        window_size: int = 1000,
        threshold: float = 0.05,
    ) -> None:
        self.model_id = model_id
        self.feature_names = feature_names
        self.window_size = window_size
        self.threshold = threshold
        self._lock = threading.Lock()

        # Store reference distributions per feature
        self._reference: dict[str, np.ndarray] = {}
        for i, name in enumerate(feature_names):
            self._reference[name] = reference_data[:, i].copy()

        # Sliding window for incoming data
        self._windows: dict[str, deque[float]] = {
            name: deque(maxlen=window_size) for name in feature_names
        }

    def observe(self, features: np.ndarray) -> None:
        """Add a new observation to the sliding windows.

        Args:
            features: Feature vector(s). Shape (n_features,) or (n_samples, n_features).
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        with self._lock:
            for i, name in enumerate(self.feature_names):
                for val in features[:, i]:
                    self._windows[name].append(float(val))

    def check_drift(self, min_samples: int = 50) -> DriftReport:
        """Run drift detection across all features.

        Args:
            min_samples: Minimum samples required before checking drift.

        Returns:
            DriftReport with per-feature results.
        """
        results: list[DriftResult] = []

        with self._lock:
            for name in self.feature_names:
                window = self._windows[name]
                sample_size = len(window)

                if sample_size < min_samples:
                    results.append(
                        DriftResult(
                            feature_name=name,
                            ks_statistic=0.0,
                            p_value=1.0,
                            is_drifted=False,
                            sample_size=sample_size,
                            reference_mean=float(np.mean(self._reference[name])),
                            current_mean=0.0,
                        )
                    )
                    continue

                current_data = np.array(list(window))
                reference_data = self._reference[name]

                ks_stat, p_value = stats.ks_2samp(reference_data, current_data)
                is_drifted = p_value < self.threshold

                # Update Prometheus metrics
                FEATURE_DRIFT_SCORE.labels(
                    model_id=self.model_id, feature_name=name
                ).set(ks_stat)

                if is_drifted:
                    DRIFT_DETECTED.labels(
                        model_id=self.model_id, feature_name=name
                    ).inc()
                    logger.warning(
                        "drift_detected",
                        model_id=self.model_id,
                        feature=name,
                        ks_statistic=round(ks_stat, 4),
                        p_value=round(p_value, 6),
                    )

                results.append(
                    DriftResult(
                        feature_name=name,
                        ks_statistic=round(ks_stat, 4),
                        p_value=round(p_value, 6),
                        is_drifted=is_drifted,
                        sample_size=sample_size,
                        reference_mean=round(float(np.mean(reference_data)), 4),
                        current_mean=round(float(np.mean(current_data)), 4),
                    )
                )

        overall = any(r.is_drifted for r in results)
        return DriftReport(
            model_id=self.model_id,
            features=results,
            overall_drift_detected=overall,
        )

    def reset(self) -> None:
        """Clear all sliding windows."""
        with self._lock:
            for window in self._windows.values():
                window.clear()
