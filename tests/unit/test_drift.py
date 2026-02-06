"""Unit tests for data drift detection."""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift import DriftDetector


class TestDriftDetection:
    """Tests for the DriftDetector."""

    def test_no_drift_on_same_distribution(self, wine_data):
        """No drift detected when serving data matches training data."""
        X, _, feature_names, _ = wine_data

        detector = DriftDetector(
            model_id="test",
            reference_data=X,
            feature_names=feature_names,
            window_size=500,
            threshold=0.01,  # Strict threshold but same data should pass
        )

        # Feed in the entire dataset (same distribution as reference)
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), size=200, replace=True)
        for idx in indices:
            detector.observe(X[idx])

        report = detector.check_drift()
        assert not report.overall_drift_detected

    def test_drift_on_shifted_distribution(self, wine_data):
        """Drift detected when serving data is significantly different."""
        X, _, feature_names, _ = wine_data

        detector = DriftDetector(
            model_id="test",
            reference_data=X,
            feature_names=feature_names,
            window_size=200,
            threshold=0.05,
        )

        # Feed in shifted data (add large offset)
        shifted = X + 100.0
        for i in range(200):
            detector.observe(shifted[i % len(shifted)])

        report = detector.check_drift()
        assert report.overall_drift_detected
        assert len(report.drifted_features) > 0

    def test_insufficient_samples_no_drift(self, drift_detector):
        """Not enough samples means no drift is reported."""
        # Only feed 5 samples (min is 50)
        for _ in range(5):
            drift_detector.observe(np.array([1.0] * 13))

        report = drift_detector.check_drift(min_samples=50)
        assert not report.overall_drift_detected

    def test_reset_clears_windows(self, drift_detector, wine_data):
        """Reset clears all observation windows."""
        X, _, _, _ = wine_data

        for i in range(100):
            drift_detector.observe(X[i])

        drift_detector.reset()

        report = drift_detector.check_drift(min_samples=50)
        # After reset, not enough samples
        for feature in report.features:
            assert feature.sample_size == 0

    def test_drift_report_summary(self, drift_detector, wine_data):
        """Drift report provides summary statistics."""
        X, _, _, _ = wine_data

        for i in range(100):
            drift_detector.observe(X[i])

        report = drift_detector.check_drift()
        summary = report.drift_summary

        assert len(summary) == 13  # 13 wine features
        for stat in summary.values():
            assert 0.0 <= stat <= 1.0
