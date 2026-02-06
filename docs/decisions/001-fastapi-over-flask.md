# ADR 001: FastAPI over Flask for Model Serving

## Status

Accepted

## Context

We need an HTTP framework for the model serving API. The primary candidates are Flask and FastAPI, both mature Python web frameworks with large ecosystems.

Our requirements:
- Async support for non-blocking I/O during model loading
- Request/response validation with minimal boilerplate
- Auto-generated OpenAPI documentation
- High performance for inference workloads
- Production-ready middleware ecosystem

## Decision

We chose **FastAPI** over Flask.

## Rationale

### Performance

FastAPI runs on Starlette/uvicorn with native async support. For ML serving where requests may involve I/O (model loading from registry, metrics export), async prevents thread starvation under load. Flask requires Gunicorn with threading or gevent for concurrent requests.

Benchmark context: FastAPI handles ~3x more requests/second than Flask for I/O-bound workloads with equivalent hardware.

### Type Safety and Validation

FastAPI's Pydantic integration provides:
- Request validation with zero additional code
- Automatic type coercion
- Detailed error messages for malformed requests
- Schema generation for documentation

Flask requires manual validation or additional libraries (marshmallow, cerberus).

### Documentation

FastAPI auto-generates OpenAPI (Swagger) and ReDoc documentation from the type annotations. This is critical for a serving API where consumers need to understand the request/response contract without reading source code.

### Ecosystem

- `prometheus-fastapi-instrumentator` for zero-config HTTP metrics
- Native dependency injection for clean resource management
- Built-in middleware support compatible with Starlette ecosystem

## Consequences

- Team members need familiarity with async/await patterns
- Some scikit-learn operations are CPU-bound and don't benefit from async (mitigated by running inference synchronously within async handlers)
- Slightly more complex deployment vs. Flask (uvicorn vs. gunicorn), though both support Docker equally well

## Alternatives Considered

- **Flask**: Simpler, but lacks native async, auto-docs, and validation
- **Ray Serve**: Too heavyweight for this scope; better suited for multi-model GPU serving
- **BentoML**: Opinionated framework; we wanted more control over the serving layer
