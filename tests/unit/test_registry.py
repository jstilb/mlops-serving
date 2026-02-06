"""Unit tests for the model registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.registry import ModelMetadata, ModelRegistry, ModelStatus


class TestModelRegistry:
    """Tests for ModelRegistry operations."""

    def test_register_model(self, tmp_registry: ModelRegistry, trained_model, wine_data):
        """Registering a model creates files and returns metadata."""
        X, y, feature_names, target_names = wine_data

        meta = tmp_registry.register(
            model=trained_model,
            model_id="wine-clf",
            version="v1.0",
            algorithm="RandomForest",
            training_metrics={"accuracy": 0.95},
            feature_names=feature_names,
            target_names=target_names,
            training_data=X,
        )

        assert meta.model_id == "wine-clf"
        assert meta.version == "v1.0"
        assert meta.status == ModelStatus.SHADOW
        assert meta.training_metrics["accuracy"] == 0.95
        assert len(meta.feature_names) == 13
        assert meta.data_hash != ""

    def test_register_duplicate_version_raises(self, tmp_registry, trained_model):
        """Cannot register the same version twice."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")

        with pytest.raises(ValueError, match="already exists"):
            tmp_registry.register(model=trained_model, model_id="m", version="v1.0")

    def test_list_models(self, tmp_registry, trained_model):
        """List returns all registered model IDs."""
        tmp_registry.register(model=trained_model, model_id="model-a", version="v1.0")
        tmp_registry.register(model=trained_model, model_id="model-b", version="v1.0")

        models = tmp_registry.list_models()
        assert "model-a" in models
        assert "model-b" in models

    def test_list_versions(self, tmp_registry, trained_model):
        """List versions returns all versions for a model."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        tmp_registry.register(model=trained_model, model_id="m", version="v2.0")

        versions = tmp_registry.list_versions("m")
        assert len(versions) == 2
        version_strings = [v.version for v in versions]
        assert "v1.0" in version_strings
        assert "v2.0" in version_strings

    def test_get_metadata(self, tmp_registry, trained_model):
        """Get metadata returns correct model info."""
        tmp_registry.register(
            model=trained_model,
            model_id="m",
            version="v1.0",
            algorithm="RF",
            description="Test model",
        )

        meta = tmp_registry.get_metadata("m", "v1.0")
        assert meta.algorithm == "RF"
        assert meta.description == "Test model"

    def test_get_metadata_not_found(self, tmp_registry):
        """Get metadata raises for non-existent model."""
        with pytest.raises(FileNotFoundError):
            tmp_registry.get_metadata("nonexistent", "v1.0")

    def test_load_model(self, tmp_registry, trained_model, wine_data):
        """Load model returns a functional model."""
        X, _, _, _ = wine_data
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")

        loaded = tmp_registry.load_model("m", "v1.0")
        predictions = loaded.predict(X[:5])
        assert len(predictions) == 5

    def test_promote_to_active(self, tmp_registry, trained_model):
        """Promoting to active sets status correctly."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        meta = tmp_registry.promote("m", "v1.0", ModelStatus.ACTIVE)

        assert meta.status == ModelStatus.ACTIVE
        assert meta.promoted_at is not None

    def test_promote_deprecates_previous_active(self, tmp_registry, trained_model):
        """Promoting a new version to active deprecates the old one."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        tmp_registry.promote("m", "v1.0", ModelStatus.ACTIVE)

        tmp_registry.register(model=trained_model, model_id="m", version="v2.0")
        tmp_registry.promote("m", "v2.0", ModelStatus.ACTIVE)

        v1 = tmp_registry.get_metadata("m", "v1.0")
        v2 = tmp_registry.get_metadata("m", "v2.0")

        assert v1.status == ModelStatus.DEPRECATED
        assert v2.status == ModelStatus.ACTIVE

    def test_get_active_version(self, tmp_registry, trained_model):
        """Get active version returns the currently active model."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        tmp_registry.promote("m", "v1.0", ModelStatus.ACTIVE)

        active = tmp_registry.get_active_version("m")
        assert active is not None
        assert active.version == "v1.0"

    def test_get_active_version_none(self, tmp_registry, trained_model):
        """Get active version returns None when no version is active."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        assert tmp_registry.get_active_version("m") is None

    def test_delete_version(self, tmp_registry, trained_model):
        """Can delete a non-active model version."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        tmp_registry.delete_version("m", "v1.0")

        assert len(tmp_registry.list_versions("m")) == 0

    def test_delete_active_version_raises(self, tmp_registry, trained_model):
        """Cannot delete an active model version."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        tmp_registry.promote("m", "v1.0", ModelStatus.ACTIVE)

        with pytest.raises(ValueError, match="Cannot delete active"):
            tmp_registry.delete_version("m", "v1.0")
