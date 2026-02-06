# ADR 002: File-Based Model Versioning Strategy

## Status

Accepted

## Context

We need a model registry that supports:
- Multiple model versions running simultaneously
- Safe promotion workflow (shadow -> canary -> active)
- Metadata tracking (training metrics, data lineage)
- Rollback capability

Options range from managed services (MLflow, Weights & Biases) to custom solutions.

## Decision

We implemented a **file-based model registry** with JSON metadata and joblib serialization.

## Rationale

### Simplicity

A file-based registry requires zero external infrastructure. Models are stored as directories with two files:
- `model.joblib` - Serialized model artifact
- `metadata.json` - Version, status, metrics, feature names

This makes the system portable, debuggable, and easy to back up.

### Version Lifecycle

We defined five lifecycle states:

```
shadow --> canary --> active --> deprecated --> archived
```

- **Shadow**: Model registered but not serving traffic. Used for shadow deployment comparison.
- **Canary**: Receiving a small fraction of traffic for validation.
- **Active**: Primary model serving all traffic.
- **Deprecated**: Previously active, retained for rollback.
- **Archived**: No longer needed, can be garbage collected.

Promoting to `active` automatically deprecates the current active version, ensuring exactly one active version at all times.

### Metadata Tracking

Each version records:
- Training metrics (accuracy, F1, etc.)
- Feature names and target names (schema enforcement)
- Data hash (training data lineage)
- Timestamps for creation, promotion, deprecation

This enables automated evaluation gates: a new model must demonstrate equivalent or better metrics before promotion.

## Consequences

- No distributed locking; single-writer assumption (fine for single-instance serving)
- No model artifact deduplication (each version stores a full copy)
- Metadata queries require scanning directories (acceptable for < 100 versions)

## Alternatives Considered

- **MLflow**: Full-featured but requires a tracking server, database, and artifact store. Overkill for demonstrating the versioning concept.
- **DVC**: Good for data versioning but model serving is not its focus.
- **S3 + DynamoDB**: Cloud-native but adds infrastructure dependencies.

For production at scale, we would migrate to MLflow or a cloud-native registry. The file-based approach demonstrates the same concepts without infrastructure overhead.
