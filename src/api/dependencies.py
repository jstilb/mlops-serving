"""FastAPI dependency injection for shared resources.

Provides singleton instances of the model registry, loader, predictor,
and other services. Initialized at app startup via lifespan events.
"""

from __future__ import annotations

from src.config import Settings, get_settings
from src.models.loader import ModelLoader
from src.models.registry import ModelRegistry
from src.monitoring.drift import DriftDetector
from src.serving.ab_testing import ABTestManager
from src.serving.predictor import Predictor

# Module-level singletons (initialized in app lifespan)
_registry: ModelRegistry | None = None
_loader: ModelLoader | None = None
_predictor: Predictor | None = None
_ab_manager: ABTestManager | None = None
_drift_detector: DriftDetector | None = None
_settings: Settings | None = None


def init_dependencies(
    registry: ModelRegistry,
    loader: ModelLoader,
    predictor: Predictor,
    ab_manager: ABTestManager,
    drift_detector: DriftDetector | None = None,
    settings: Settings | None = None,
) -> None:
    """Initialize all dependency singletons. Called during app startup."""
    global _registry, _loader, _predictor, _ab_manager, _drift_detector, _settings
    _registry = registry
    _loader = loader
    _predictor = predictor
    _ab_manager = ab_manager
    _drift_detector = drift_detector
    _settings = settings or get_settings()


def get_settings_dep() -> Settings:
    """Get application settings."""
    if _settings is None:
        return get_settings()
    return _settings


def get_registry() -> ModelRegistry:
    """Get model registry instance."""
    if _registry is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _registry


def get_model_loader() -> ModelLoader:
    """Get model loader instance."""
    if _loader is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _loader


def get_predictor() -> Predictor:
    """Get predictor instance."""
    if _predictor is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _predictor


def get_ab_manager() -> ABTestManager:
    """Get A/B test manager instance."""
    if _ab_manager is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _ab_manager


def get_drift_detector() -> DriftDetector | None:
    """Get drift detector instance (may be None if not configured)."""
    return _drift_detector
