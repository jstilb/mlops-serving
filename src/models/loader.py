"""Model loading with in-memory caching and lazy initialization.

Provides thread-safe model loading with LRU-style caching to avoid
repeated disk reads for frequently accessed models.
"""

from __future__ import annotations

import threading
from typing import Any

import structlog

from src.models.registry import ModelRegistry, ModelStatus

logger = structlog.get_logger(__name__)


class ModelLoader:
    """Thread-safe model loader with in-memory cache.

    Caches loaded models to avoid repeated deserialization overhead.
    Cache is invalidated on model promotion or explicit clear.
    """

    def __init__(self, registry: ModelRegistry, max_cache_size: int = 10) -> None:
        self.registry = registry
        self.max_cache_size = max_cache_size
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str, version: str | None = None) -> Any:
        """Load a model, using cache when available.

        Args:
            model_id: Model identifier.
            version: Specific version. If None, loads the active version.

        Returns:
            Loaded model object.

        Raises:
            FileNotFoundError: If model or version not found.
            RuntimeError: If no active version exists and no version specified.
        """
        if version is None:
            active = self.registry.get_active_version(model_id)
            if active is None:
                raise RuntimeError(f"No active version found for model '{model_id}'")
            version = active.version

        cache_key = f"{model_id}/{version}"

        with self._lock:
            if cache_key in self._cache:
                logger.debug("model_cache_hit", model_id=model_id, version=version)
                return self._cache[cache_key]

        # Load outside lock to avoid blocking other threads
        model = self.registry.load_model(model_id, version)
        logger.info("model_loaded", model_id=model_id, version=version)

        with self._lock:
            if len(self._cache) >= self.max_cache_size:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug("model_cache_eviction", evicted=oldest_key)

            self._cache[cache_key] = model

        return model

    def invalidate(self, model_id: str, version: str | None = None) -> None:
        """Invalidate cache entries for a model.

        Args:
            model_id: Model to invalidate.
            version: Specific version. If None, invalidates all versions.
        """
        with self._lock:
            if version:
                key = f"{model_id}/{version}"
                self._cache.pop(key, None)
            else:
                keys_to_remove = [k for k in self._cache if k.startswith(f"{model_id}/")]
                for key in keys_to_remove:
                    del self._cache[key]

    def clear_cache(self) -> None:
        """Clear the entire model cache."""
        with self._lock:
            self._cache.clear()

    @property
    def cache_info(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_cache_size,
                "cached_models": list(self._cache.keys()),
            }
