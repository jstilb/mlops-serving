"""Model evaluation script for CI/CD deployment gates.

Compares a candidate model against the current active model.
The candidate must meet or exceed baseline metrics to pass the gate.

Exit codes:
    0: Candidate passes evaluation gate
    1: Candidate fails evaluation gate
    2: Error during evaluation

Usage:
    python -m train.evaluate --candidate-version v2.0
    python -m train.evaluate --candidate-version v2.0 --threshold 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.models.registry import ModelRegistry


def evaluate(
    registry: ModelRegistry,
    candidate_version: str,
    model_id: str = "default",
    threshold: float | None = None,
) -> bool:
    """Evaluate a candidate model against the current active model.

    Args:
        registry: Model registry.
        candidate_version: Version to evaluate.
        model_id: Model identifier.
        threshold: Minimum accuracy threshold. If None, compares against active model.

    Returns:
        True if candidate passes the evaluation gate.
    """
    # Load test data
    data = load_wine()
    X, y = data.data, data.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load candidate model
    try:
        candidate = registry.load_model(model_id, candidate_version)
        candidate_meta = registry.get_metadata(model_id, candidate_version)
    except FileNotFoundError:
        print(f"ERROR: Candidate model {model_id}/{candidate_version} not found")
        sys.exit(2)

    # Evaluate candidate
    y_pred_candidate = candidate.predict(X_test)
    candidate_accuracy = accuracy_score(y_test, y_pred_candidate)
    candidate_f1 = f1_score(y_test, y_pred_candidate, average="weighted")

    results = {
        "candidate": {
            "version": candidate_version,
            "accuracy": round(candidate_accuracy, 4),
            "f1_weighted": round(candidate_f1, 4),
        }
    }

    # Compare against baseline
    if threshold is not None:
        # Use fixed threshold
        passed = candidate_accuracy >= threshold
        results["threshold"] = threshold
        results["comparison"] = "fixed_threshold"
    else:
        # Compare against active model
        active = registry.get_active_version(model_id)
        if active is None:
            print("No active model to compare against. Candidate passes by default.")
            results["comparison"] = "no_baseline"
            passed = True
        else:
            baseline = registry.load_model(model_id, active.version)
            y_pred_baseline = baseline.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
            baseline_f1 = f1_score(y_test, y_pred_baseline, average="weighted")

            results["baseline"] = {
                "version": active.version,
                "accuracy": round(baseline_accuracy, 4),
                "f1_weighted": round(baseline_f1, 4),
            }
            results["comparison"] = "vs_active"

            # Candidate must be at least as good as baseline
            passed = candidate_accuracy >= baseline_accuracy * 0.99  # 1% tolerance

    results["passed"] = passed

    print(json.dumps(results, indent=2))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model for deployment gate")
    parser.add_argument("--candidate-version", required=True, help="Version to evaluate")
    parser.add_argument("--model-id", default="default", help="Model identifier")
    parser.add_argument("--threshold", type=float, default=None, help="Min accuracy threshold")
    parser.add_argument(
        "--registry-path", type=Path, default=Path("models"), help="Registry path"
    )

    args = parser.parse_args()
    registry = ModelRegistry(args.registry_path)

    passed = evaluate(
        registry,
        candidate_version=args.candidate_version,
        model_id=args.model_id,
        threshold=args.threshold,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
