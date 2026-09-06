"""Read, validate, render, and reset the run-to-run lesson ledger.

The JSONL file is the durable contract.  ``lessons.md`` is a deterministic
human-readable projection used by the CLI and the demo; it is never the source
of truth.  Lesson diagnosis and writing belong to WP-10, so this module only
accepts already-formed :class:`~occam.core.models.Lesson` values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from occam.core.models import Lesson

MAX_LESSON_WORDS = 60
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_LONG_NUMBER = re.compile(r"(?<![\w])(?:\d{4,}|\d{1,3}(?:,\d{3})+)(?![\w])")
_INVOICE_ID = re.compile(r"\b(?:invoice|inv)[-_ ]?[a-z0-9]+\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")


class LessonStoreError(ValueError):
    """A lesson ledger is malformed or violates its write contract."""


def _as_path(namespace: str | Path) -> Path:
    """Resolve a memory namespace without widening it to a recursive target."""

    path = Path(namespace)
    if not str(path) or str(path) == ".":
        raise LessonStoreError("memory namespace must name a directory")
    if path.name == "lessons.jsonl":
        path = path.parent
    return path


def validate_lesson_text(
    text: str,
    *,
    case_values: Iterable[float | int | str] = (),
) -> str:
    """Apply the generic, case-independent part of the leak-guard contract.

    WP-10 owns diagnosis and the comparison against a task pack's expected
    values.  This reusable boundary handles what can be checked without a pack:
    ISO dates, long numbers, and invoice identifiers.  ``case_values`` is an
    optional hook for WP-10's near-expected-value check and is intentionally
    not inferred by the store.
    """

    if not isinstance(text, str):
        raise LessonStoreError("lesson text must be a string")
    normalized = text.strip()
    if not normalized:
        raise LessonStoreError("lesson text must not be blank")
    if len(normalized.split()) > MAX_LESSON_WORDS:
        raise LessonStoreError(f"lesson text must contain at most {MAX_LESSON_WORDS} words")
    if _ISO_DATE.search(normalized):
        raise LessonStoreError("lesson text must not contain an ISO date")
    if _INVOICE_ID.search(normalized):
        raise LessonStoreError("lesson text must not contain an invoice id")
    if _LONG_NUMBER.search(normalized):
        raise LessonStoreError("lesson text must not contain a four-digit-or-longer number")

    expected_numbers: list[float] = []
    for value in case_values:
        if isinstance(value, bool):
            continue
        try:
            number = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            expected_numbers.append(number)
    for token in _NUMBER.findall(normalized):
        try:
            candidate = float(token.replace(",", ""))
        except ValueError:
            continue
        for expected in expected_numbers:
            tolerance = max(1e-9, abs(expected) * 0.01)
            if abs(candidate - expected) <= tolerance:
                raise LessonStoreError("lesson text contains a case-specific value")
    return normalized


def validate_lesson(
    lesson: Lesson | dict[str, Any],
    *,
    case_values: Iterable[float | int | str] = (),
) -> Lesson:
    """Validate one typed lesson and its leak-guard-compatible text."""

    try:
        parsed = lesson if isinstance(lesson, Lesson) else Lesson.model_validate(lesson)
    except ValidationError as exc:
        raise LessonStoreError(f"invalid lesson: {exc}") from exc
    validate_lesson_text(parsed.text, case_values=case_values)
    return parsed


class LessonStore:
    """An append-only JSONL store for one memory namespace."""

    def __init__(self, namespace: str | Path):
        self.directory = _as_path(namespace)
        self.lessons_path = self.directory / "lessons.jsonl"
        self.markdown_path = self.directory / "lessons.md"

    def load(self, *, active_only: bool = True) -> list[Lesson]:
        """Read lessons in append order, validating every non-empty line."""

        if not self.lessons_path.exists():
            return []
        if not self.lessons_path.is_file():
            raise LessonStoreError(f"lesson path is not a file: {self.lessons_path}")

        lessons: list[Lesson] = []
        seen_ids: set[str] = set()
        try:
            lines = self.lessons_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LessonStoreError(f"cannot read {self.lessons_path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LessonStoreError(
                    f"invalid JSON in {self.lessons_path} line {line_number}: {exc.msg}"
                ) from exc
            try:
                lesson = validate_lesson(payload)
            except LessonStoreError as exc:
                raise LessonStoreError(
                    f"invalid lesson in {self.lessons_path} line {line_number}: {exc}"
                ) from exc
            if lesson.id in seen_ids:
                raise LessonStoreError(f"duplicate lesson id {lesson.id!r} on line {line_number}")
            seen_ids.add(lesson.id)
            if not active_only or lesson.status == "active":
                lessons.append(lesson)
        return lessons

    def append(self, lesson: Lesson | dict[str, Any]) -> Lesson:
        """Validate and append an immutable lesson, then refresh ``lessons.md``."""

        parsed = validate_lesson(lesson)
        existing = self.load(active_only=False)
        if any(item.id == parsed.id for item in existing):
            raise LessonStoreError(f"lesson id {parsed.id!r} already exists")
        self.directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            parsed.model_dump(mode="json", exclude_none=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with self.lessons_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            raise LessonStoreError(f"cannot append to {self.lessons_path}") from exc
        self._write_markdown([*existing, parsed])
        return parsed

    def reset(self) -> int:
        """Remove this namespace's lesson projections and return removed count."""

        count = len(self.load(active_only=False)) if self.lessons_path.exists() else 0
        for path in (self.lessons_path, self.markdown_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise LessonStoreError(f"cannot reset {self.directory}") from exc
        return count

    def markdown(self) -> str:
        """Return the deterministic Markdown projection, including retired rows."""

        return render_lessons_markdown(self.load(active_only=False))

    def show(self) -> str:
        """Materialize and return ``lessons.md`` for CLI/demo consumers."""

        lessons = self.load(active_only=False)
        markdown = render_lessons_markdown(lessons)
        if lessons or self.markdown_path.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            self._write_markdown(lessons)
        return markdown

    def _write_markdown(self, lessons: Sequence[Lesson]) -> None:
        markdown = render_lessons_markdown(lessons)
        temporary = self.markdown_path.with_name(f".{self.markdown_path.name}.tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(markdown, encoding="utf-8", newline="\n")
            temporary.replace(self.markdown_path)
        except OSError as exc:
            raise LessonStoreError(f"cannot write {self.markdown_path}") from exc


def render_lessons_markdown(lessons: Iterable[Lesson | dict[str, Any]]) -> str:
    """Render lessons without exposing a second mutable source of truth."""

    parsed = [validate_lesson(lesson) for lesson in lessons]
    lines = ["# Occam lessons", ""]
    if not parsed:
        lines.extend(["No lessons recorded.", ""])
        return "\n".join(lines)
    for lesson in parsed:
        label = "tool" if lesson.kind == "tool_note" else "rule"
        target = f" · {lesson.tool}" if lesson.tool else ""
        lines.extend(
            [
                f"## [{label}] {lesson.id}{target}",
                "",
                lesson.text,
                "",
                f"- status: `{lesson.status}`",
                f"- born: `{lesson.born['run_id']}` · generation `{lesson.born['generation']}`",
                f"- evidence: `{json.dumps(lesson.evidence, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    return "\n".join(lines)


def load_lessons(namespace: str | Path, *, active_only: bool = True) -> list[Lesson]:
    """Convenience wrapper used by the architect and callers outside the store."""

    return LessonStore(namespace).load(active_only=active_only)


def append_lesson(namespace: str | Path, lesson: Lesson | dict[str, Any]) -> Lesson:
    """Append one lesson to a namespace."""

    return LessonStore(namespace).append(lesson)


def reset_lessons(namespace: str | Path) -> int:
    """Reset one exact namespace and return the number of removed lessons."""

    return LessonStore(namespace).reset()


def render_lessons_markdown_file(namespace: str | Path) -> str:
    """Compatibility alias for callers that want the CLI projection."""

    return LessonStore(namespace).show()


__all__ = [
    "LessonStore",
    "LessonStoreError",
    "MAX_LESSON_WORDS",
    "append_lesson",
    "load_lessons",
    "render_lessons_markdown",
    "render_lessons_markdown_file",
    "reset_lessons",
    "validate_lesson",
    "validate_lesson_text",
]
