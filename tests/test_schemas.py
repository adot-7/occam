"""Contract tests for the checked-in schemas and replay fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from occam.core import (
    Architecture,
    CaseResult,
    Event,
    GenerationState,
    Lesson,
    MetricsSnapshot,
    State,
)
from occam.core.models import EventType
from occam.store.reader import EventReader
from occam.store.reducer import reduce, state_json_bytes
from occam.store.schema import load_schema, validate_event, validate_state, validate_task
from occam.store.writer import EventWriter

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (ROOT / "fixtures" / "demo_run1", ROOT / "fixtures" / "demo_run2")
RUN1 = FIXTURES[0]
RUN2 = FIXTURES[1]


def test_schemas_are_valid_draft_2020_documents() -> None:
    for name in ("events.schema.json", "state.schema.json", "task.schema.json"):
        Draft202012Validator.check_schema(load_schema(name))


def test_every_fixture_event_validates_against_event_schema() -> None:
    required_types = {
        "run.started",
        "lesson.written",
        "reliability.completed",
        "architecture.proposed",
        "execution.started",
        "execution.case",
        "execution.completed",
        "ablation.started",
        "ablation.role",
        "ablation.completed",
        "baseline.completed",
        "diagnosis.emitted",
        "mutation.applied",
        "metrics.snapshot",
        "run.completed",
    }
    run1_types: set[str] | None = None
    for fixture in FIXTURES:
        events_path = fixture / "events.jsonl"
        with events_path.open(encoding="utf-8") as handle:
            payloads = [json.loads(line) for line in handle if line.strip()]

        assert len(payloads) >= 100
        assert [payload["seq"] for payload in payloads] == list(range(len(payloads)))
        event_types = {payload["type"] for payload in payloads}
        expected_types = required_types - ({"lesson.written"} if fixture == RUN2 else set())
        assert expected_types <= event_types
        assert event_types <= set(get_args(EventType))
        if fixture == RUN1:
            run1_types = event_types
        for payload in payloads:
            validate_event(payload)
            Event.model_validate(payload)
    assert run1_types is not None
    assert "log" in run1_types


def test_fixture_reduces_to_valid_byte_identical_state() -> None:
    for fixture in FIXTURES:
        events = EventReader(fixture).read()
        first = reduce(events)
        second = reduce(events)

        first_bytes = state_json_bytes(first)
        assert first_bytes == state_json_bytes(second)
        validate_state(json.loads(first_bytes))
        assert first.last_seq == len(events) - 1
        assert first.completed is True
        assert first.run_name in {"run1", "run2"}
        assert first.memory_ns == "memory/fx_recon"
        assert len(first.lessons) >= len(first.lessons_loaded)

    run1_state = reduce(EventReader(RUN1).read())
    run2_state = reduce(EventReader(RUN2).read())
    assert run1_state.best_generation == 3
    assert len(run1_state.generations) == 5
    assert len(run1_state.lessons_loaded) == 0
    assert len(run1_state.lessons) == 3
    assert run2_state.best_generation == 1
    assert len(run2_state.generations) == 2
    assert len(run2_state.lessons_loaded) == 3
    assert len(run2_state.lessons) == 3
    assert run2_state.generations["g001"].reliability is not None


def test_core_architecture_model_matches_fixture_contract() -> None:
    architecture_event = next(
        event for event in EventReader(RUN1) if event.type == "architecture.proposed"
    )
    architecture = Architecture.model_validate(architecture_event.data["architecture"])
    assert len(architecture.roles) == 5
    assert architecture.final_role == "r_report"
    assert architecture.control == "deterministic"


def test_metrics_snapshot_has_the_complete_metrics_contract() -> None:
    metrics_event = next(event for event in EventReader(RUN1) if event.type == "metrics.snapshot")
    metrics = MetricsSnapshot.model_validate(metrics_event.data)
    assert metrics.ci.lo <= metrics.pass_rate <= metrics.ci.hi
    assert metrics.tokens > 0
    assert metrics.reliability > 0
    assert metrics.tool_calls_per_case > 0
    assert metrics.reliability_pass3 is None
    assert metrics.speed > 0
    assert metrics.latency_s_p50 <= metrics.latency_s_p90


def test_metrics_snapshot_rejects_missing_required_fields() -> None:
    event = next(event for event in EventReader(RUN1) if event.type == "metrics.snapshot")
    payload = event.model_dump(mode="json")
    payload["data"].pop("tool_calls_per_case")
    with pytest.raises(ValueError, match="events.schema.json"):
        validate_event(payload)
    with pytest.raises(ValidationError):
        MetricsSnapshot.model_validate(payload["data"])


def test_event_and_state_models_reject_schema_invalid_values() -> None:
    event_data = {"level": "info", "message": "model validation"}
    with pytest.raises(ValidationError, match="ISO 8601"):
        Event(ts=0, run_id="model_test", seq=0, type="log", data=event_data)
    with pytest.raises(ValidationError, match="timezone"):
        Event(
            ts="2026-09-05T00:00:00",
            run_id="model_test",
            seq=0,
            type="log",
            data=event_data,
        )
    with pytest.raises(ValidationError):
        Event(
            ts="2026-09-05T00:00:00Z",
            run_id="",
            seq=0,
            type="log",
            data=event_data,
        )
    with pytest.raises(ValidationError):
        State(run_id="")
    with pytest.raises(ValidationError):
        State(run_id="model_test", current_generation=-1)
    with pytest.raises(ValidationError):
        GenerationState(generation=-1)
    with pytest.raises(ValueError, match="empty event stream"):
        reduce([])


def test_reducer_rejects_an_event_that_is_only_envelope_valid() -> None:
    invalid_log = Event(
        ts="2026-09-05T00:00:00Z",
        run_id="model_test",
        seq=0,
        type="log",
        data={},
    )

    with pytest.raises(ValueError, match="events.schema.json"):
        reduce([invalid_log])


def test_diagnoses_follow_highest_cost_witness_rule() -> None:
    events = EventReader(RUN1).read()
    rows_by_generation: dict[int, list[dict[str, object]]] = {}
    diagnoses: dict[int, dict[str, object]] = {}
    for event in events:
        generation = event.data.get("generation")
        if not isinstance(generation, int):
            continue
        if event.type == "ablation.role":
            rows_by_generation.setdefault(generation, []).append(event.data)
        elif event.type == "diagnosis.emitted":
            diagnoses[generation] = event.data

    for generation, diagnosis in diagnoses.items():
        witnesses = [row for row in rows_by_generation[generation] if row["verdict"] == "witness"]
        mutation = diagnosis["chosen_mutation"]
        if witnesses:
            highest_cost = max(float(row["cost_share"]) for row in witnesses)
            assert mutation["type"] == "prune"
            assert any(
                row["role_id"] == mutation["target_role"]
                and float(row["cost_share"]) == highest_cost
                for row in witnesses
            )
        else:
            assert mutation["type"] != "prune"


def test_full_repeat_is_a_fresh_nonzero_accounted_run() -> None:
    for fixture in FIXTURES:
        state = reduce(EventReader(fixture).read())
        for generation in state.generations.values():
            repeat = generation.executions["full_repeat"]
            assert repeat["cost_usd"] > 0
            assert repeat["latency_s_mean"] > 0
            assert repeat["tokens"] > 0
            assert all(case["cost_usd"] > 0 for case in repeat["cases"])
            assert all(case["latency_s"] > 0 for case in repeat["cases"])
            assert sum(case["cost_usd"] for case in repeat["cases"]) == repeat["cost_usd"]


def test_new_strict_models_capture_lessons_and_invoice_sub_results() -> None:
    lesson = Lesson(
        id="lesson_1",
        kind="tool_note",
        text="Use the returned rate date.",
        tool="fx_rate",
        evidence={"run_id": "run1", "generation": 0, "case_ids": []},
        born={"run_id": "run1", "generation": 0},
    )
    assert lesson.status == "active"
    with pytest.raises(ValidationError):
        Lesson.model_validate({**lesson.model_dump(), "unexpected": True})

    result = CaseResult(case_id="fxa_001", passed=False, sub_results={"INV-1": False})
    assert result.sub_results == {"INV-1": False}
    with pytest.raises(ValidationError):
        CaseResult.model_validate({**result.model_dump(), "unexpected": True})


def test_task_manifest_shape_is_schema_compatible() -> None:
    task = {
        "name": "fx_recon_a",
        "domain": "finance_ops",
        "goal": "Compute FX gain or loss in INR for a receivables ledger.",
        "answer_format": "A JSON object with total_inr and per_invoice.",
        "tools": ["fx_rate", "fx_series", "python_exec"],
        "checker": "numeric_exact",
        "examples": 3,
        "source": {
            "kind": "generated",
            "repo": "occam",
            "file": "tasks/fx_recon_a/cases.jsonl",
            "license": "MIT",
        },
    }
    validate_task(task)


def test_writer_appends_valid_events_and_refreshes_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with EventWriter(run_dir, run_id="writer_test") as writer:
        writer.append(
            {
                "ts": "2026-09-05T00:00:00Z",
                "type": "run.started",
                "data": {
                    "task": {"name": "demo", "domain": "test", "n_cases": 0, "goal": "test"},
                    "config": {},
                    "run_name": "writer-test",
                    "memory_ns": "memory/test",
                    "lessons_loaded": [],
                },
            }
        )

    events = EventReader(run_dir).read()
    state = EventReader(run_dir).load_state()
    assert len(events) == 1
    assert state.last_seq == 0
    assert (run_dir / "state.json").exists()
