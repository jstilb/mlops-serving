"""Alert rule definitions for model serving monitoring.

Defines alert conditions and generates Prometheus alerting rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlertRule:
    """Prometheus alert rule definition."""

    name: str
    expression: str
    duration: str
    severity: str
    summary: str
    description: str


# Pre-defined alert rules for model serving
SERVING_ALERTS: list[AlertRule] = [
    AlertRule(
        name="HighPredictionLatency",
        expression='histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 0.5',
        duration="5m",
        severity="warning",
        summary="High prediction latency detected",
        description="P95 prediction latency is above 500ms for 5 minutes.",
    ),
    AlertRule(
        name="HighErrorRate",
        expression=(
            "rate(prediction_errors_total[5m]) / rate(prediction_total[5m]) > 0.05"
        ),
        duration="5m",
        severity="critical",
        summary="High prediction error rate",
        description="More than 5% of predictions are failing.",
    ),
    AlertRule(
        name="DataDriftDetected",
        expression="feature_drift_score > 0.3",
        duration="10m",
        severity="warning",
        summary="Data drift detected",
        description="Feature distribution drift exceeds threshold for 10 minutes.",
    ),
    AlertRule(
        name="ModelNotLoaded",
        expression="models_in_memory == 0",
        duration="1m",
        severity="critical",
        summary="No models loaded in memory",
        description="The serving system has zero models loaded.",
    ),
]


def generate_prometheus_rules() -> dict:
    """Generate Prometheus alerting rules YAML-compatible dict."""
    rules = []
    for alert in SERVING_ALERTS:
        rules.append(
            {
                "alert": alert.name,
                "expr": alert.expression,
                "for": alert.duration,
                "labels": {"severity": alert.severity},
                "annotations": {
                    "summary": alert.summary,
                    "description": alert.description,
                },
            }
        )

    return {"groups": [{"name": "mlops_serving", "rules": rules}]}
