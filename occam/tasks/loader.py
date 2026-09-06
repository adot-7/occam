"""Load one committed task pack and its JSONL evaluation cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from occam.config.settings import project_root
from occam.core.models import Case, Task
from occam.store.schema import validate_task


class TaskPackError(ValueError):
    """A task pack is missing or does not satisfy the task contract."""


@dataclass(frozen=True)
class TaskPack:
    """The manifest and ordered cases selected from one task-pack directory."""

    directory: Path
    task: Task
    cases: tuple[Case, ...]

    def select(self, n_cases: int | None = None) -> list[Case]:
        """Return the pack's first ``n_cases`` in committed order."""

        if n_cases is None:
            return list(self.cases)
        if n_cases < 1:
            raise TaskPackError("cases must be at least 1")
        if n_cases > len(self.cases):
            raise TaskPackError(
                f"requested {n_cases} cases, but {self.task.name} contains only {len(self.cases)}"
            )
        return list(self.cases[:n_cases])


def resolve_task_directory(task: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve a pack directory or a committed pack name."""

    candidate = Path(task)
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.is_dir():
        return candidate
    repository_root = Path(root) if root is not None else project_root()
    named = repository_root / "tasks" / str(task)
    if named.is_dir():
        return named
    raise TaskPackError(f"task pack not found: {task}")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaskPackError(f"cannot read task manifest {path}") from exc
    except yaml.YAMLError as exc:
        raise TaskPackError(f"invalid YAML in task manifest {path}") from exc
    if not isinstance(raw, dict):
        raise TaskPackError(f"task manifest must be an object: {path}")
    return raw


def _load_cases(path: Path) -> tuple[Case, ...]:
    if not path.is_file():
        raise TaskPackError(f"case file not found: {path}")
    cases: list[Case] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TaskPackError(f"cannot read case file {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            case = Case.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise TaskPackError(f"invalid case in {path} line {line_number}: {exc}") from exc
        if case.id in seen:
            raise TaskPackError(f"duplicate case id {case.id!r} in {path}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise TaskPackError(f"case file is empty: {path}")
    return tuple(cases)


def load_task_pack(task: str | Path, *, root: str | Path | None = None) -> TaskPack:
    """Load and validate ``task.yaml`` plus its sibling ``cases.jsonl``."""

    directory = resolve_task_directory(task, root=root)
    manifest_path = directory / "task.yaml"
    raw = _load_yaml(manifest_path)
    try:
        validate_task(raw)
        manifest = Task.model_validate(raw)
    except ValueError as exc:
        raise TaskPackError(f"invalid task manifest {manifest_path}: {exc}") from exc
    cases = _load_cases(directory / "cases.jsonl")
    return TaskPack(directory=directory, task=manifest, cases=cases)


__all__ = ["TaskPack", "TaskPackError", "load_task_pack", "resolve_task_directory"]
