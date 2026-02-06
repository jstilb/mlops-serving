"""Unit tests for the predictor and serving components."""

from __future__ import annotations

import pytest

from src.models.loader import ModelLoader
from src.models.registry import ModelRegistry, ModelStatus
from src.serving.ab_testing import ABTestConfig, ABTestManager
from src.serving.predictor import Predictor
from src.serving.shadow import ShadowDeployment


class TestPredictor:
    """Tests for the Predictor class."""

    def test_predict_single(self, tmp_registry, trained_model, wine_data, sample_features):
        """Single prediction returns correct structure."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.promote("test", "v1.0", ModelStatus.ACTIVE)

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)

        result = predictor.predict([sample_features[0]], "test")
        assert len(result.predictions) == 1
        assert result.model_id == "test"
        assert result.model_version == "v1.0"
        assert result.latency_ms > 0

    def test_predict_batch(self, tmp_registry, trained_model, wine_data, sample_features):
        """Batch prediction returns correct number of results."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.promote("test", "v1.0", ModelStatus.ACTIVE)

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)

        result = predictor.predict(sample_features, "test")
        assert len(result.predictions) == 3

    def test_predict_with_probabilities(self, tmp_registry, trained_model, wine_data, sample_features):
        """Predictions include probability distributions."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.promote("test", "v1.0", ModelStatus.ACTIVE)

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)

        result = predictor.predict(sample_features, "test", include_probabilities=True)
        assert result.probabilities is not None
        assert len(result.probabilities) == 3
        # Each probability row should sum to ~1.0
        for proba in result.probabilities:
            assert abs(sum(proba) - 1.0) < 0.01

    def test_predict_no_active_version_raises(self, tmp_registry, trained_model):
        """Prediction fails if no active version exists."""
        tmp_registry.register(model=trained_model, model_id="test", version="v1.0")

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)

        with pytest.raises(RuntimeError, match="No active version"):
            predictor.predict([[1.0, 2.0]], "test")

    def test_predict_specific_version(self, tmp_registry, trained_model, wine_data, sample_features):
        """Can predict with a specific model version."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)

        result = predictor.predict(sample_features, "test", "v1.0")
        assert result.model_version == "v1.0"


class TestModelLoader:
    """Tests for the ModelLoader caching."""

    def test_cache_hit(self, tmp_registry, trained_model):
        """Second load should come from cache."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        tmp_registry.promote("m", "v1.0", ModelStatus.ACTIVE)

        loader = ModelLoader(tmp_registry)
        loader.get("m", "v1.0")
        assert loader.cache_info["size"] == 1

        loader.get("m", "v1.0")  # Should be cache hit
        assert loader.cache_info["size"] == 1

    def test_cache_invalidation(self, tmp_registry, trained_model):
        """Invalidation removes model from cache."""
        tmp_registry.register(model=trained_model, model_id="m", version="v1.0")
        loader = ModelLoader(tmp_registry)
        loader.get("m", "v1.0")
        assert loader.cache_info["size"] == 1

        loader.invalidate("m", "v1.0")
        assert loader.cache_info["size"] == 0

    def test_cache_eviction(self, tmp_registry, trained_model):
        """Cache evicts oldest entry when full."""
        loader = ModelLoader(tmp_registry, max_cache_size=2)

        for i in range(3):
            v = f"v{i}.0"
            tmp_registry.register(model=trained_model, model_id="m", version=v)
            loader.get("m", v)

        assert loader.cache_info["size"] == 2


class TestABTesting:
    """Tests for A/B testing manager."""

    def test_create_ab_test(self, tmp_registry, trained_model, wine_data, sample_features):
        """Create and run an A/B test."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v2.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.promote("test", "v1.0", ModelStatus.ACTIVE)

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)
        ab_manager = ABTestManager(predictor)

        config = ABTestConfig(
            model_id="test",
            control_version="v1.0",
            treatment_version="v2.0",
            traffic_split=0.5,
            name="test-experiment",
        )
        ab_manager.create_test(config)

        result = ab_manager.predict(sample_features, "test")
        assert result.variant in ("control", "treatment")
        assert len(result.prediction.predictions) == 3

    def test_sticky_sessions(self, tmp_registry, trained_model, wine_data, sample_features):
        """Same request ID gets same variant assignment."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v2.0",
            feature_names=feature_names,
            target_names=target_names,
        )

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)
        ab_manager = ABTestManager(predictor)

        config = ABTestConfig(
            model_id="test",
            control_version="v1.0",
            treatment_version="v2.0",
            traffic_split=0.5,
            sticky_sessions=True,
        )
        ab_manager.create_test(config)

        # Same request ID should always get same variant
        results = [
            ab_manager.predict(sample_features, "test", request_id="user-123")
            for _ in range(10)
        ]
        variants = {r.variant for r in results}
        assert len(variants) == 1  # All same variant

    def test_duplicate_test_raises(self, predictor, ab_manager):
        """Cannot create two tests for same model."""
        config = ABTestConfig(
            model_id="m", control_version="v1", treatment_version="v2"
        )
        ab_manager.create_test(config)

        with pytest.raises(ValueError, match="already exists"):
            ab_manager.create_test(config)


class TestShadowDeployment:
    """Tests for shadow deployment."""

    def test_shadow_returns_primary_result(self, tmp_registry, trained_model, wine_data, sample_features):
        """Shadow deployment returns primary model result only."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v2.0",
            feature_names=feature_names,
            target_names=target_names,
        )

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)

        shadow = ShadowDeployment(
            predictor=predictor,
            model_id="test",
            primary_version="v1.0",
            shadow_version="v2.0",
        )

        result = shadow.predict(sample_features)
        assert result.model_version == "v1.0"

    def test_shadow_comparison_summary(self, tmp_registry, trained_model, wine_data, sample_features):
        """Shadow deployment tracks comparison statistics."""
        X, _, feature_names, target_names = wine_data
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v1.0",
            feature_names=feature_names,
            target_names=target_names,
        )
        tmp_registry.register(
            model=trained_model,
            model_id="test",
            version="v2.0",
            feature_names=feature_names,
            target_names=target_names,
        )

        loader = ModelLoader(tmp_registry)
        predictor = Predictor(loader, tmp_registry)

        shadow = ShadowDeployment(
            predictor=predictor,
            model_id="test",
            primary_version="v1.0",
            shadow_version="v2.0",
        )

        for _ in range(5):
            shadow.predict(sample_features)

        summary = shadow.get_comparison_summary()
        assert summary["total_comparisons"] == 5
        assert "agreement_rate" in summary
