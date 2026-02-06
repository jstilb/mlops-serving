"""Application configuration with environment-based overrides."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    app_name: str = "mlops-serving"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Model Registry
    model_registry_path: Path = Field(
        default_factory=lambda: Path(os.getenv("MODEL_REGISTRY_PATH", "models"))
    )

    # A/B Testing
    ab_testing_enabled: bool = False
    shadow_mode_enabled: bool = False
    traffic_split_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Monitoring
    metrics_enabled: bool = True
    drift_detection_enabled: bool = True
    drift_window_size: int = 1000
    drift_threshold: float = 0.05

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    model_config = {"env_prefix": "MLOPS_", "env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
