# Deployment Guide

## Local Development

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for full stack)

### Running Locally (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Start the server (auto-trains model on first run)
uvicorn src.api.app:app --reload --port 8000
```

### Running with Docker Compose

```bash
# Start all services (API + Prometheus + Grafana)
docker-compose up --build

# Services:
#   API:        http://localhost:8000
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3000 (admin/mlops-demo)
```

### Stopping Services

```bash
docker-compose down

# Remove volumes (clears model data and metrics)
docker-compose down -v
```

## Configuration

All settings are configurable via environment variables with the `MLOPS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MLOPS_HOST` | `0.0.0.0` | Server bind address |
| `MLOPS_PORT` | `8000` | Server port |
| `MLOPS_WORKERS` | `1` | Uvicorn worker count |
| `MLOPS_LOG_LEVEL` | `INFO` | Log level |
| `MLOPS_LOG_FORMAT` | `json` | Log format (`json` or `console`) |
| `MLOPS_METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `MLOPS_DRIFT_DETECTION_ENABLED` | `true` | Enable drift monitoring |
| `MLOPS_DRIFT_THRESHOLD` | `0.05` | P-value threshold for drift |
| `MLOPS_MODEL_REGISTRY_PATH` | `models` | Path to model registry |

## Model Management

### Training a New Model

```bash
# Train with defaults
python -m train.train_model

# Train with custom parameters
python -m train.train_model --n-estimators 200 --max-depth 15 --version v2.0

# Train and immediately promote to active
python -m train.train_model --promote
```

### Promoting a Model

```bash
# Via CLI
python -m src.cli promote default v2.0 active

# Via API
curl -X POST http://localhost:8000/api/v1/models/default/promote \
  -H "Content-Type: application/json" \
  -d '{"model_id": "default", "version": "v2.0", "to_status": "active"}'
```

### Listing Models

```bash
# Via CLI
python -m src.cli list

# Via API
curl http://localhost:8000/api/v1/models
```

## Monitoring

### Grafana Dashboard

1. Open http://localhost:3000
2. Log in with `admin` / `mlops-demo`
3. Navigate to the "MLOps Model Serving" dashboard

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_latency_seconds` | Histogram | Prediction latency |
| `prediction_total` | Counter | Total predictions |
| `prediction_errors_total` | Counter | Failed predictions |
| `feature_drift_score` | Gauge | KS statistic per feature |
| `models_in_memory` | Gauge | Cached model count |

## CI/CD Pipeline

### Test Pipeline (on push/PR)

1. **Lint** - ruff check and format verification
2. **Test** - pytest with coverage
3. **Docker** - Build and verify image size

### Deploy Pipeline (on tag)

1. **Evaluate** - Train candidate, compare against baseline
2. **Build** - Multi-platform Docker build and push to GHCR
