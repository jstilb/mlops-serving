"""Structured request logging middleware with request tracing.

Assigns a unique trace ID to each request and logs request/response
details using structlog for structured JSON output.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every request with timing and trace IDs.

    Adds X-Request-ID and X-Trace-ID headers to responses for
    distributed tracing support.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4())[:8])

        # Bind context for all log messages in this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                client_host=request.client.host if request.client else "unknown",
            )

            # Add tracing headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "request_failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
            )
            raise
