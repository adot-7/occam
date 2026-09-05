"""Load and validate the repository's JSON Schema contracts."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load one schema from the checked-in schema directory."""

    path = SCHEMA_DIR / name
    try:
        with path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"schema file not found: {path}") from exc
    Draft202012Validator.check_schema(schema)
    return schema


def _validate(instance: Any, schema_name: str) -> None:
    validator = Draft202012Validator(load_schema(schema_name), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.path) or "$"
        raise ValueError(f"{schema_name} validation failed at {path}: {error.message}")


def validate_event(instance: dict[str, Any]) -> None:
    """Validate one serialized event against ``events.schema.json``."""

    _validate(instance, "events.schema.json")


def validate_state(instance: dict[str, Any]) -> None:
    """Validate one serialized state snapshot against ``state.schema.json``."""

    _validate(instance, "state.schema.json")


def validate_task(instance: dict[str, Any]) -> None:
    """Validate one task manifest against ``task.schema.json``."""

    _validate(instance, "task.schema.json")
