"""Prometheus metrics middleware for HTTP request instrumentation.

Uses prometheus-fastapi-instrumentator for automatic HTTP metrics
and exposes the /metrics endpoint.
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def setup_metrics(app: FastAPI) -> Instrumentator:
    """Configure Prometheus metrics instrumentation for the FastAPI app.

    Exposes:
    - http_requests_total
    - http_request_duration_seconds
    - http_request_size_bytes
    - http_response_size_bytes

    The /metrics endpoint is automatically added.

    Args:
        app: FastAPI application instance.

    Returns:
        Configured Instrumentator instance.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/health", "/ready"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )

    instrumentator.instrument(app).expose(app, include_in_schema=True, tags=["monitoring"])

    return instrumentator
