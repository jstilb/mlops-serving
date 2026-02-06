"""CLI for model management operations.

Provides command-line access to the model registry for training,
listing, promoting, and inspecting models.

Usage:
    python -m src.cli list
    python -m src.cli info default v1.0
    python -m src.cli promote default v1.0 active
    python -m src.cli train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import get_settings
from src.models.registry import ModelRegistry, ModelStatus


def cmd_list(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """List all registered models."""
    models = registry.list_models()
    if not models:
        print("No models registered.")
        return

    for model_id in models:
        versions = registry.list_versions(model_id)
        print(f"\n{model_id}:")
        for v in versions:
            status_marker = " *" if v.status == ModelStatus.ACTIVE else ""
            print(f"  {v.version} [{v.status.value}]{status_marker} - {v.algorithm}")
            if v.training_metrics:
                for k, val in v.training_metrics.items():
                    print(f"    {k}: {val}")


def cmd_info(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Show detailed info for a model version."""
    try:
        meta = registry.get_metadata(args.model_id, args.version)
        print(json.dumps(meta.model_dump(), indent=2))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_promote(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Promote a model version."""
    try:
        status = ModelStatus(args.status)
        meta = registry.promote(args.model_id, args.version, status)
        print(f"Promoted {meta.model_id}/{meta.version} to {meta.status.value}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_train(registry: ModelRegistry, args: argparse.Namespace) -> None:
    """Train and register a new model version."""
    from train.train_model import train_and_register
    train_and_register(registry)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MLOps Model Serving CLI")
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Model registry path (default: from settings)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    subparsers.add_parser("list", help="List all models")

    # info
    info_parser = subparsers.add_parser("info", help="Show model info")
    info_parser.add_argument("model_id", help="Model identifier")
    info_parser.add_argument("version", help="Model version")

    # promote
    promote_parser = subparsers.add_parser("promote", help="Promote model version")
    promote_parser.add_argument("model_id", help="Model identifier")
    promote_parser.add_argument("version", help="Model version")
    promote_parser.add_argument("status", choices=[s.value for s in ModelStatus])

    # train
    subparsers.add_parser("train", help="Train and register new model")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()
    registry_path = args.registry_path or settings.model_registry_path
    registry = ModelRegistry(registry_path)

    commands = {
        "list": cmd_list,
        "info": cmd_info,
        "promote": cmd_promote,
        "train": cmd_train,
    }

    commands[args.command](registry, args)


if __name__ == "__main__":
    main()
