"""Training script for the Wine classification model.

Trains a RandomForest classifier on the Wine dataset, evaluates it
with cross-validation, and registers it in the model registry.

Usage:
    python -m train.train_model
    python -m train.train_model --version v2.0 --n-estimators 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.registry import ModelRegistry, ModelStatus
from src.models.versioning import get_next_version


def train_and_register(
    registry: ModelRegistry,
    version: str | None = None,
    n_estimators: int = 100,
    max_depth: int = 10,
    promote: bool = False,
) -> None:
    """Train a model and register it in the registry.

    Args:
        registry: Model registry instance.
        version: Version string. Auto-increments if None.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth.
        promote: Whether to promote to active immediately.
    """
    print("Loading Wine dataset...")
    data = load_wine()
    X, y = data.data, data.target

    # Train/test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    # Test set evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    test_precision = precision_score(y_test, y_pred, average="weighted")
    test_recall = recall_score(y_test, y_pred, average="weighted")

    print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # Determine version
    if version is None:
        version = get_next_version(registry, "default")
    print(f"\nRegistering model as default/{version}...")

    # Register
    meta = registry.register(
        model=model,
        model_id="default",
        version=version,
        algorithm="RandomForest",
        training_metrics={
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(test_accuracy, 4),
            "test_f1_weighted": round(test_f1, 4),
            "test_precision_weighted": round(test_precision, 4),
            "test_recall_weighted": round(test_recall, 4),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        feature_names=list(data.feature_names),
        target_names=list(data.target_names),
        training_data=X,
        description=f"Wine classifier - RandomForest ({n_estimators} trees, depth {max_depth})",
        tags={"dataset": "wine", "framework": "scikit-learn"},
    )

    # Save reference data for drift detection
    ref_path = registry.registry_path / "default" / "reference_data.npy"
    np.save(ref_path, X)

    if promote:
        registry.promote("default", version, ModelStatus.ACTIVE)
        print(f"Model promoted to active.")

    print(f"Model registered: default/{version}")
    print(f"Metrics: accuracy={test_accuracy:.4f}, f1={test_f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register a model")
    parser.add_argument("--version", type=str, default=None, help="Model version")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--promote", action="store_true", help="Promote to active")
    parser.add_argument(
        "--registry-path", type=Path, default=Path("models"), help="Registry path"
    )

    args = parser.parse_args()
    registry = ModelRegistry(args.registry_path)

    train_and_register(
        registry,
        version=args.version,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        promote=args.promote,
    )


if __name__ == "__main__":
    main()
