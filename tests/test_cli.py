"""Failure-path and derived-state coverage for the validation CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from occam.cli import app

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (ROOT / "fixtures" / "demo_run1", ROOT / "fixtures" / "demo_run2")
RUNNER = CliRunner()


def test_validate_validates_derived_state_when_snapshot_is_absent() -> None:
    fixture = FIXTURES[0]
    result = RUNNER.invoke(app, ["validate", str(fixture)])

    assert result.exit_code == 0
    assert "valid: 281 events" in result.output
    assert "derived state schema validated" in result.output
    assert "state.json absent" in result.output


def test_validate_accepts_both_canonical_replay_fixtures() -> None:
    for fixture in FIXTURES:
        result = RUNNER.invoke(app, ["validate", str(fixture)])
        assert result.exit_code == 0
        assert "deterministic: yes" in result.output


def test_validate_rejects_an_empty_event_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    (run_dir / "events.jsonl").touch()

    result = RUNNER.invoke(app, ["validate", str(run_dir)])

    assert result.exit_code != 0
    assert "event log is empty" in result.output


def test_validate_rejects_invalid_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "corrupt"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text('{"ts":\n', encoding="utf-8")

    result = RUNNER.invoke(app, ["validate", str(run_dir)])

    assert result.exit_code != 0
    assert "invalid JSON" in result.output


def test_validate_rejects_an_invalid_state_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "invalid_state"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        '{"data":{"level":"info","message":"ok"},'
        '"run_id":"cli_test","seq":0,"ts":"2026-09-05T00:00:00Z","type":"log"}\n',
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text("{}\n", encoding="utf-8")

    result = RUNNER.invoke(app, ["validate", str(run_dir)])

    assert result.exit_code != 0
    assert "state.schema.json validation failed" in result.output
