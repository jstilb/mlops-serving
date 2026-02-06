# Multi-stage build for production model serving
# Stage 1: Build dependencies
# Stage 2: Slim production image

# --- Build Stage ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency specification
COPY pyproject.toml .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir . 2>/dev/null || \
    pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.6.0" \
    "scikit-learn>=1.5.0" \
    "joblib>=1.4.0" \
    "numpy>=1.26.0" \
    "prometheus-client>=0.21.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "structlog>=24.4.0" \
    "httpx>=0.27.0" \
    "scipy>=1.14.0"

# --- Production Stage ---
FROM python:3.11-slim AS production

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY src/ src/
COPY train/ train/

# Create model directory
RUN mkdir -p models && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
