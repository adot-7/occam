"""Failure-path and derived-state coverage for the validation CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from occam.cli import app
from occam.llm import CompletionError, MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (ROOT / "fixtures" / "demo_run1", ROOT / "fixtures" / "demo_run2")
RUNNER = CliRunner()


def test_validate_validates_derived_state_when_snapshot_is_absent() -> None:
    fixture = FIXTURES[0]
    result = RUNNER.invoke(app, ["validate", str(fixture)])

    assert result.exit_code == 0
    assert "valid: 231 events" in result.output
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


def test_llm_ping_no_cache_bypasses_the_shared_completion_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, configs: Any) -> None:
            del configs

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            del args
            calls.append(kwargs)
            return SimpleNamespace(
                tokens_in=1,
                tokens_out=1,
                cost_usd=0.0,
                cost_label="metered",
                cached=False,
                tool_calls=[{"id": "call_1"}],
            )

    monkeypatch.setattr("occam.llm.LLMClient", FakeClient)

    result = RUNNER.invoke(app, ["llm", "ping", "worker_fast", "--no-cache"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["use_cache"] is False


def test_llm_ping_reports_only_sanitized_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, configs: Any) -> None:
            del configs

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise CompletionError(
                "LLM completion failed after 1 attempt(s): BadRequestError; "
                "raw provider payload; provider_error[status=400; "
                "type=invalid_request_error; code=unsupported_parameter; "
                "parameter=max_tokens]"
            )

    monkeypatch.setattr("occam.llm.LLMClient", FakeClient)

    result = RUNNER.invoke(app, ["llm", "ping", "worker_fast"])

    assert result.exit_code == 2
    assert (
        "LLM ping failed: CompletionError; provider_error[status=400; "
        "type=invalid_request_error; code=unsupported_parameter; "
        "parameter=max_tokens]"
    ) in result.output
    assert "raw provider payload" not in result.output


def test_llm_ping_keeps_generic_llm_error_output_without_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, configs: Any) -> None:
            del configs

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise CompletionError("provider returned empty content without native tool calls")

    monkeypatch.setattr("occam.llm.LLMClient", FakeClient)

    result = RUNNER.invoke(app, ["llm", "ping", "worker_fast"])

    assert result.exit_code == 2
    assert result.output.strip() == "LLM ping failed: CompletionError"


def test_llm_ping_keeps_missing_credentials_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, configs: Any) -> None:
            del configs

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise MissingCredentialsError(
                "credentials unavailable for worker_fast; set TENSORMUX_API_KEY"
            )

    monkeypatch.setattr("occam.llm.LLMClient", FakeClient)

    result = RUNNER.invoke(app, ["llm", "ping", "worker_fast"])

    assert result.exit_code == 2
    assert result.output.strip() == (
        "credentials unavailable for worker_fast; set TENSORMUX_API_KEY"
    )
