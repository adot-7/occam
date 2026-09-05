"""Small, dependency-light settings helpers used by later work packages."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root when running from an editable checkout."""

    return Path(__file__).resolve().parents[2]


def environment(name: str, default: str | None = None) -> str | None:
    """Read a setting without ever providing a secret default."""

    return os.environ.get(name, default)
