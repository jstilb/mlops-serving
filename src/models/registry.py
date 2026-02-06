"""Model registry with version management and metadata tracking.

Manages model lifecycle: registration, promotion, deprecation, and retrieval.
Models are stored on disk with JSON metadata for portability.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    """Model lifecycle states."""

    ACTIVE = "active"
    SHADOW = "shadow"
    CANARY = "canary"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelMetadata(BaseModel):
    """Metadata for a registered model version."""

    model_id: str
    version: str
    status: ModelStatus = ModelStatus.SHADOW
    algorithm: str = ""
    framework: str = "scikit-learn"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: str | None = None
    deprecated_at: str | None = None
    training_metrics: dict[str, float] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    target_names: list[str] = Field(default_factory=list)
    data_hash: str = ""
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)


class ModelRegistry:
    """File-based model registry with version management.

    Directory structure:
        registry_path/
            model_id/
                v1/
                    model.joblib
                    metadata.json
                v2/
                    model.joblib
                    metadata.json
    """

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        model: Any,
        model_id: str,
        version: str,
        *,
        algorithm: str = "",
        training_metrics: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
        training_data: np.ndarray | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> ModelMetadata:
        """Register a new model version in the registry.

        Args:
            model: Trained model object (must be serializable with joblib).
            model_id: Unique identifier for the model.
            version: Semantic version string (e.g., "v1", "v2.1").
            algorithm: Algorithm name (e.g., "RandomForest").
            training_metrics: Dictionary of metric name -> value.
            feature_names: List of input feature names.
            target_names: List of target class names.
            training_data: Training data for computing data hash.
            description: Human-readable description.
            tags: Arbitrary key-value tags.

        Returns:
            ModelMetadata for the registered model.

        Raises:
            ValueError: If model version already exists.
        """
        version_path = self.registry_path / model_id / version
        if version_path.exists():
            raise ValueError(f"Model {model_id}/{version} already exists")

        version_path.mkdir(parents=True, exist_ok=True)

        # Serialize model
        model_path = version_path / "model.joblib"
        joblib.dump(model, model_path)

        # Compute data hash if training data provided
        data_hash = ""
        if training_data is not None:
            data_hash = hashlib.sha256(training_data.tobytes()).hexdigest()[:16]

        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            algorithm=algorithm,
            training_metrics=training_metrics or {},
            feature_names=feature_names or [],
            target_names=target_names or [],
            data_hash=data_hash,
            description=description,
            tags=tags or {},
        )

        metadata_path = version_path / "metadata.json"
        metadata_path.write_text(metadata.model_dump_json(indent=2))

        return metadata

    def get_metadata(self, model_id: str, version: str) -> ModelMetadata:
        """Get metadata for a specific model version.

        Raises:
            FileNotFoundError: If model version does not exist.
        """
        metadata_path = self.registry_path / model_id / version / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model {model_id}/{version} not found")

        return ModelMetadata.model_validate_json(metadata_path.read_text())

    def load_model(self, model_id: str, version: str) -> Any:
        """Load a model artifact from the registry.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        model_path = self.registry_path / model_id / version / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file for {model_id}/{version} not found")

        return joblib.load(model_path)

    def list_models(self) -> list[str]:
        """List all registered model IDs."""
        if not self.registry_path.exists():
            return []
        return sorted(
            d.name for d in self.registry_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    def list_versions(self, model_id: str) -> list[ModelMetadata]:
        """List all versions of a model with their metadata."""
        model_path = self.registry_path / model_id
        if not model_path.exists():
            return []

        versions = []
        for version_dir in sorted(model_path.iterdir()):
            if version_dir.is_dir():
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    versions.append(ModelMetadata.model_validate_json(metadata_file.read_text()))
        return versions

    def get_active_version(self, model_id: str) -> ModelMetadata | None:
        """Get the currently active version of a model."""
        for meta in self.list_versions(model_id):
            if meta.status == ModelStatus.ACTIVE:
                return meta
        return None

    def get_shadow_version(self, model_id: str) -> ModelMetadata | None:
        """Get the shadow version of a model (if any)."""
        for meta in self.list_versions(model_id):
            if meta.status == ModelStatus.SHADOW:
                return meta
        return None

    def promote(self, model_id: str, version: str, to_status: ModelStatus) -> ModelMetadata:
        """Promote a model version to a new status.

        When promoting to ACTIVE, the current active version is deprecated.

        Args:
            model_id: Model identifier.
            version: Version to promote.
            to_status: Target status.

        Returns:
            Updated ModelMetadata.

        Raises:
            FileNotFoundError: If model version does not exist.
            ValueError: If promotion transition is invalid.
        """
        metadata = self.get_metadata(model_id, version)

        # If promoting to active, deprecate current active
        if to_status == ModelStatus.ACTIVE:
            current_active = self.get_active_version(model_id)
            if current_active and current_active.version != version:
                self._update_status(model_id, current_active.version, ModelStatus.DEPRECATED)

        metadata.status = to_status
        now = datetime.now(timezone.utc).isoformat()

        if to_status == ModelStatus.ACTIVE:
            metadata.promoted_at = now
        elif to_status == ModelStatus.DEPRECATED:
            metadata.deprecated_at = now

        self._save_metadata(model_id, version, metadata)
        return metadata

    def delete_version(self, model_id: str, version: str) -> None:
        """Delete a model version from the registry.

        Raises:
            FileNotFoundError: If version does not exist.
            ValueError: If attempting to delete an active model.
        """
        metadata = self.get_metadata(model_id, version)
        if metadata.status == ModelStatus.ACTIVE:
            raise ValueError("Cannot delete active model version. Deprecate it first.")

        version_path = self.registry_path / model_id / version
        shutil.rmtree(version_path)

    def _update_status(self, model_id: str, version: str, status: ModelStatus) -> None:
        """Update the status of a model version."""
        metadata = self.get_metadata(model_id, version)
        metadata.status = status
        if status == ModelStatus.DEPRECATED:
            metadata.deprecated_at = datetime.now(timezone.utc).isoformat()
        self._save_metadata(model_id, version, metadata)

    def _save_metadata(self, model_id: str, version: str, metadata: ModelMetadata) -> None:
        """Persist metadata to disk."""
        metadata_path = self.registry_path / model_id / version / "metadata.json"
        metadata_path.write_text(metadata.model_dump_json(indent=2))
