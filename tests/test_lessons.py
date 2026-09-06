"""WP-06 lesson ledger, provenance, and CLI contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from occam.cli import app
from occam.core import Lesson
from occam.memory.lessons import LessonStore, LessonStoreError, validate_lesson_text

RUNNER = CliRunner()


def lesson_payload(
    lesson_id: str = "lesson_rate_date",
    *,
    kind: str = "tool_note",
    tool: str | None = "fx_rate",
    text: str = "Trust the date returned by the rate tool.",
) -> dict:
    return {
        "id": lesson_id,
        "kind": kind,
        "text": text,
        "tool": tool,
        "evidence": {
            "run_id": "run1",
            "generation": 0,
            "case_ids": ["case_a"],
            "trace_refs": ["g000/case_a/r_rates"],
        },
        "born": {"run_id": "run1", "generation": 0},
        "status": "active",
    }


def test_lesson_model_enforces_kind_and_provenance() -> None:
    valid = Lesson.model_validate(lesson_payload())
    assert valid.tool == "fx_rate"

    with pytest.raises(ValidationError, match="require a non-empty tool"):
        Lesson.model_validate(lesson_payload(tool=None))
    with pytest.raises(ValidationError, match="must not name a tool"):
        Lesson.model_validate(lesson_payload(kind="domain_rule", tool="fx_rate"))
    with pytest.raises(ValidationError, match="include case_ids"):
        Lesson.model_validate({**lesson_payload(), "evidence": {"run_id": "run1", "generation": 0}})
    with pytest.raises(ValidationError, match="at most 60 words"):
        Lesson.model_validate({**lesson_payload(), "text": "word " * 61})


def test_lesson_store_round_trips_jsonl_and_markdown_and_resets(tmp_path: Path) -> None:
    store = LessonStore(tmp_path / "memory" / "fx_recon")
    first = store.append(lesson_payload())
    second = store.append(
        lesson_payload(
            "lesson_fee",
            kind="domain_rule",
            tool=None,
            text="A bank fee is a charge, not an FX movement; add it back to the received value.",
        )
    )

    assert [lesson.id for lesson in store.load()] == [first.id, second.id]
    rendered = store.show()
    assert "lesson_rate_date" in rendered
    assert "lesson_fee" in rendered
    assert store.markdown_path.read_text(encoding="utf-8") == rendered

    with pytest.raises(LessonStoreError, match="already exists"):
        store.append(first)
    assert store.reset() == 2
    assert store.load() == []
    assert not store.lessons_path.exists()
    assert not store.markdown_path.exists()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Use the 2026-04-03 response date.", "ISO date"),
        ("Do not copy invoice INV-2291 into a lesson.", "invoice id"),
        ("The case total was 41872.35.", "four-digit"),
    ],
)
def test_generic_lesson_leak_boundaries(text: str, message: str) -> None:
    with pytest.raises(LessonStoreError, match=message):
        validate_lesson_text(text)

    assert validate_lesson_text("Trust the returned rate date and do not adjust it.")
    with pytest.raises(LessonStoreError, match="case-specific"):
        validate_lesson_text("Use the observed total 100.00.", case_values=[100.0])


def test_lessons_cli_show_and_reset(tmp_path: Path) -> None:
    memory = tmp_path / "memory" / "fx_recon"
    LessonStore(memory).append(lesson_payload())

    shown = RUNNER.invoke(app, ["lessons", "show", "--memory", str(memory)])
    assert shown.exit_code == 0
    assert "# Occam lessons" in shown.output
    assert "Trust the date returned" in shown.output

    reset = RUNNER.invoke(app, ["lessons", "reset", "--memory", str(memory)])
    assert reset.exit_code == 0
    assert "removed 1 lesson(s)" in reset.output
    assert not (memory / "lessons.jsonl").exists()
