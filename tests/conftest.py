"""Shared test fixtures for unit and integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

from src.models.loader import ModelLoader
from src.models.registry import ModelRegistry, ModelStatus
from src.monitoring.drift import DriftDetector
from src.serving.ab_testing import ABTestManager
from src.serving.predictor import Predictor


@pytest.fixture
def tmp_registry(tmp_path: Path) -> ModelRegistry:
    """Create a temporary model registry."""
    return ModelRegistry(tmp_path / "models")


@pytest.fixture
def wine_data():
    """Load Wine dataset for testing."""
    data = load_wine()
    return data.data, data.target, list(data.feature_names), list(data.target_names)


@pytest.fixture
def trained_model(wine_data):
    """Train a simple model for testing."""
    X, y, _, _ = wine_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def registered_model(tmp_registry: ModelRegistry, trained_model, wine_data):
    """Register and activate a model for testing."""
    X, y, feature_names, target_names = wine_data

    meta = tmp_registry.register(
        model=trained_model,
        model_id="test-model",
        version="v1.0",
        algorithm="RandomForest",
        training_metrics={"accuracy": 0.95},
        feature_names=feature_names,
        target_names=target_names,
        training_data=X,
        description="Test model",
    )

    tmp_registry.promote("test-model", "v1.0", ModelStatus.ACTIVE)
    return meta


@pytest.fixture
def model_loader(tmp_registry: ModelRegistry) -> ModelLoader:
    """Create a model loader."""
    return ModelLoader(tmp_registry)


@pytest.fixture
def predictor(model_loader: ModelLoader, tmp_registry: ModelRegistry) -> Predictor:
    """Create a predictor."""
    return Predictor(model_loader, tmp_registry)


@pytest.fixture
def ab_manager(predictor: Predictor) -> ABTestManager:
    """Create an A/B test manager."""
    return ABTestManager(predictor)


@pytest.fixture
def drift_detector(wine_data) -> DriftDetector:
    """Create a drift detector with reference data."""
    X, _, feature_names, _ = wine_data
    return DriftDetector(
        model_id="test-model",
        reference_data=X,
        feature_names=feature_names,
        window_size=100,
        threshold=0.05,
    )


@pytest.fixture
def sample_features(wine_data) -> list[list[float]]:
    """Get sample feature vectors for prediction."""
    X, _, _, _ = wine_data
    return X[:3].tolist()
