"""Custom Prometheus metrics for model serving observability.

Tracks prediction latency, throughput, errors, model versions,
and feature distributions for drift detection.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# --- Request Metrics ---

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests",
    labelnames=["model_id", "model_version", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5),
)

PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total number of predictions made",
    labelnames=["model_id", "model_version", "status"],
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total prediction errors",
    labelnames=["model_id", "model_version", "error_type"],
)

BATCH_SIZE = Histogram(
    "prediction_batch_size",
    "Number of samples in batch prediction requests",
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500),
)

# --- Model Metrics ---

ACTIVE_MODEL_VERSION = Info(
    "active_model",
    "Currently active model version information",
)

MODEL_LOAD_TIME = Histogram(
    "model_load_seconds",
    "Time to load model from registry",
    labelnames=["model_id", "model_version"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODELS_IN_MEMORY = Gauge(
    "models_in_memory",
    "Number of models currently cached in memory",
)

# --- A/B Testing Metrics ---

AB_TEST_ASSIGNMENTS = Counter(
    "ab_test_assignments_total",
    "Number of traffic assignments per variant",
    labelnames=["model_id", "variant"],
)

SHADOW_PREDICTION_DIVERGENCE = Histogram(
    "shadow_prediction_divergence",
    "Divergence between primary and shadow model predictions",
    labelnames=["model_id"],
    buckets=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# --- Drift Detection Metrics ---

FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "KS statistic measuring feature distribution drift",
    labelnames=["model_id", "feature_name"],
)

DRIFT_DETECTED = Counter(
    "drift_detected_total",
    "Number of times drift was detected",
    labelnames=["model_id", "feature_name"],
)

PREDICTION_DISTRIBUTION = Histogram(
    "prediction_distribution",
    "Distribution of prediction outputs",
    labelnames=["model_id", "class_label"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
