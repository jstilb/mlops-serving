"""Model version management utilities.

Handles version string parsing, comparison, and auto-incrementing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.registry import ModelRegistry


@dataclass(frozen=True)
class SemanticVersion:
    """Parsed semantic version for comparison."""

    major: int
    minor: int

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        """Parse a version string like 'v1', 'v1.2', or '1.2'.

        Args:
            version_str: Version string to parse.

        Returns:
            Parsed SemanticVersion.

        Raises:
            ValueError: If version string is invalid.
        """
        cleaned = version_str.lstrip("v")
        match = re.match(r"^(\d+)(?:\.(\d+))?$", cleaned)
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")

        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        return cls(major=major, minor=minor)

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}"

    def __lt__(self, other: SemanticVersion) -> bool:
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: SemanticVersion) -> bool:
        return (self.major, self.minor) <= (other.major, other.minor)

    def next_major(self) -> SemanticVersion:
        """Get next major version."""
        return SemanticVersion(major=self.major + 1, minor=0)

    def next_minor(self) -> SemanticVersion:
        """Get next minor version."""
        return SemanticVersion(major=self.major, minor=self.minor + 1)


def get_next_version(registry: ModelRegistry, model_id: str, *, bump: str = "minor") -> str:
    """Determine the next version string for a model.

    Args:
        registry: Model registry instance.
        model_id: Model identifier.
        bump: Version bump type ("major" or "minor").

    Returns:
        Next version string (e.g., "v2.0" or "v1.3").
    """
    versions = registry.list_versions(model_id)
    if not versions:
        return "v1.0"

    parsed = [SemanticVersion.parse(v.version) for v in versions]
    latest = max(parsed)

    if bump == "major":
        return str(latest.next_major())
    return str(latest.next_minor())
