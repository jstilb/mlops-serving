# Architecture

## System Overview

mlops-serving is a production model serving system designed around three core principles:

1. **Reliability** - Health checks, graceful degradation, structured logging
2. **Observability** - Prometheus metrics, data drift detection, request tracing
3. **Safety** - Shadow deployment, A/B testing, evaluation gates before promotion

## Request Flow

```
Client Request
      |
      v
+------------------+
|  FastAPI App      |
|  +------------+  |
|  | Logging MW |  |  <-- Request ID, trace ID, timing
|  +------------+  |
|  | Metrics MW |  |  <-- Prometheus counters/histograms
|  +------------+  |
|       |          |
|  +------------+  |
|  | Router     |  |
|  +------------+  |
+--------|---------+
         |
    +----+----+
    |         |
    v         v
/predict   /models
    |
    v
+------------------+
| A/B Test Manager |  <-- Traffic splitting (if test active)
+--------|---------+
         |
    +----+----+
    |         |
    v         v
Control   Treatment
    |         |
    v         v
+------------------+
|   Predictor      |
|  +------------+  |
|  | Pre-process|  |
|  +------------+  |
|  | Model.pred |  |  <-- scikit-learn inference
|  +------------+  |
|  | Metrics    |  |  <-- Latency, count, errors
|  +------------+  |
|  | Drift Obs  |  |  <-- Feature distribution tracking
|  +------------+  |
+------------------+
         |
         v
+------------------+
|  Model Loader    |  <-- In-memory LRU cache
+--------|---------+
         |
         v
+------------------+
|  Model Registry  |  <-- File-based, versioned
|  models/         |
|    default/      |
|      v1.0/       |
|        model.job |
|        meta.json |
+------------------+
```

## Component Architecture

### Model Registry

File-based registry with JSON metadata. Each model version is stored as:
- `model.joblib` - Serialized scikit-learn model
- `metadata.json` - Version, status, metrics, feature names

Model lifecycle: `shadow` -> `canary` -> `active` -> `deprecated`

### Model Loader

Thread-safe loader with LRU caching. Avoids repeated disk I/O for
frequently accessed models. Cache is invalidated on promotion events.

### Predictor

Central prediction orchestrator. Handles:
- Feature validation and numpy conversion
- Model inference via the loader
- Prometheus metric recording (latency, count, errors)
- Drift detector observation

### A/B Testing

Traffic splitting between model versions:
- Weighted random assignment
- Sticky sessions via request ID hashing (MD5-based deterministic routing)
- Per-variant metrics for comparison

### Shadow Deployment

Runs a shadow model alongside the primary:
- Primary result is returned to client
- Shadow result is logged for comparison
- Divergence metrics track agreement rate

### Drift Detection

Kolmogorov-Smirnov test comparing serving data against training distribution:
- Sliding window of recent observations per feature
- Reference distributions captured at training time
- Per-feature drift scores exposed as Prometheus gauges

## Monitoring Stack

```
+----------+     +------------+     +---------+
|  API     | --> | Prometheus | --> | Grafana |
| /metrics |     | (scrape)   |     | (viz)   |
+----------+     +------------+     +---------+
```

Pre-built Grafana dashboard tracks:
- P50/P95/P99 prediction latency
- Prediction throughput by model version
- Error rate
- Feature drift scores
- A/B test traffic distribution
- Shadow model divergence

## Technology Choices

See ADRs in `docs/decisions/` for detailed rationale:
- [001: FastAPI over Flask](decisions/001-fastapi-over-flask.md)
- [002: Model Versioning Strategy](decisions/002-model-versioning-strategy.md)
