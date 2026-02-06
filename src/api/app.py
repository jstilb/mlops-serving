"""FastAPI application factory with lifespan management.

Creates the production-ready FastAPI app with all routes, middleware,
and background services configured.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import structlog
from fastapi import FastAPI

from src.api.dependencies import init_dependencies
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.metrics import setup_metrics
from src.api.routes import health, models, predict
from src.config import Settings, get_settings
from src.models.loader import ModelLoader
from src.models.registry import ModelRegistry, ModelStatus
from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import ACTIVE_MODEL_VERSION, MODELS_IN_MEMORY
from src.serving.ab_testing import ABTestManager
from src.serving.predictor import Predictor

logger = structlog.get_logger(__name__)


def _configure_logging(settings: Settings) -> None:
    """Configure structlog for structured JSON logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer() if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _auto_train_model(registry: ModelRegistry) -> None:
    """Train a default model if none exists in the registry.

    This ensures the service can start successfully even without
    pre-trained models, making demo/development easier.
    """
    if registry.list_models():
        return

    logger.info("no_models_found", action="auto_training_default_model")

    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Load dataset
    data = load_wine()
    X, y = data.data, data.target

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Evaluate
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    # Register and promote to active
    meta = registry.register(
        model=model,
        model_id="default",
        version="v1.0",
        algorithm="RandomForest",
        training_metrics={
            "accuracy_mean": round(float(scores.mean()), 4),
            "accuracy_std": round(float(scores.std()), 4),
        },
        feature_names=list(data.feature_names),
        target_names=list(data.target_names),
        training_data=X,
        description="Auto-trained Wine classification model (RandomForest, 100 trees)",
        tags={"dataset": "wine", "auto_trained": "true"},
    )

    registry.promote("default", "v1.0", ModelStatus.ACTIVE)

    # Save reference data for drift detection
    ref_path = registry.registry_path / "default" / "reference_data.npy"
    np.save(ref_path, X)

    logger.info(
        "auto_train_complete",
        model_id="default",
        version="v1.0",
        accuracy=round(float(scores.mean()), 4),
        features=len(data.feature_names),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown logic."""
    settings = get_settings()
    _configure_logging(settings)

    logger.info("startup", app=settings.app_name, version=settings.app_version)

    # Initialize model registry
    registry = ModelRegistry(settings.model_registry_path)

    # Auto-train if no models exist
    _auto_train_model(registry)

    # Initialize model loader
    loader = ModelLoader(registry, max_cache_size=10)

    # Initialize drift detector if reference data exists
    drift_detector: DriftDetector | None = None
    if settings.drift_detection_enabled:
        ref_path = registry.registry_path / "default" / "reference_data.npy"
        if ref_path.exists():
            ref_data = np.load(ref_path)
            active = registry.get_active_version("default")
            if active:
                drift_detector = DriftDetector(
                    model_id="default",
                    reference_data=ref_data,
                    feature_names=active.feature_names,
                    window_size=settings.drift_window_size,
                    threshold=settings.drift_threshold,
                )

    # Initialize predictor and A/B test manager
    predictor = Predictor(loader, registry, drift_detector)
    ab_manager = ABTestManager(predictor)

    # Set dependency injection singletons
    init_dependencies(
        registry=registry,
        loader=loader,
        predictor=predictor,
        ab_manager=ab_manager,
        drift_detector=drift_detector,
        settings=settings,
    )

    # Pre-load active models into cache
    for model_id in registry.list_models():
        active = registry.get_active_version(model_id)
        if active:
            loader.get(model_id, active.version)
            ACTIVE_MODEL_VERSION.info({
                "model_id": model_id,
                "version": active.version,
                "algorithm": active.algorithm,
            })

    MODELS_IN_MEMORY.set(loader.cache_info["size"])

    logger.info(
        "startup_complete",
        models_loaded=loader.cache_info["size"],
        drift_detection=drift_detector is not None,
    )

    yield

    # Shutdown
    logger.info("shutdown", app=settings.app_name)
    loader.clear_cache()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="MLOps Model Serving",
        description=(
            "Production model serving system with version management, "
            "A/B testing, shadow deployment, and monitoring."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware (order matters -- outermost first)
    app.add_middleware(RequestLoggingMiddleware)

    # Prometheus metrics
    if settings.metrics_enabled:
        setup_metrics(app)

    # Routes
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(models.router)

    return app


# Application instance for uvicorn
app = create_app()
