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
    RoleTrace,
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


SCHEMA_FILES = ("events.schema.json", "state.schema.json", "task.schema.json")


def test_schemas_are_valid_draft_2020_documents() -> None:
    for name in SCHEMA_FILES:
        Draft202012Validator.check_schema(load_schema(name))


def test_the_two_schema_copies_stay_synchronised() -> None:
    """``schemas/`` is what the PRD points at; ``occam/schemas/`` is what ships.

    ``occam.store.schema`` loads the packaged copy, so a contract change applied
    to only one of them would pass every other test in this file while shipping
    a wheel that disagrees with the repo.
    """

    for name in SCHEMA_FILES:
        repo_copy = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        packaged_copy = json.loads((ROOT / "occam" / "schemas" / name).read_text(encoding="utf-8"))
        assert repo_copy == packaged_copy, f"schemas/{name} and occam/schemas/{name} have diverged"


def test_ablation_started_requires_a_bounded_noise_rate() -> None:
    event = next(event for event in EventReader(RUN1) if event.type == "ablation.started")

    missing = event.model_dump(mode="json")
    missing["data"].pop("noise_rate")
    with pytest.raises(ValueError, match="events.schema.json"):
        validate_event(missing)

    out_of_range = event.model_dump(mode="json")
    out_of_range["data"]["noise_rate"] = 1.01
    with pytest.raises(ValueError, match="events.schema.json"):
        validate_event(out_of_range)


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

        assert len(payloads) >= 90
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
            if payload["type"] == "execution.case":
                data = payload["data"]
                assert {"answer_prefix", "grade_error", "role_error"} <= data.keys()
                assert len(data["answer_prefix"]) <= 120
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


def test_run2_acceptance_evidence_records_three_loaded_lessons() -> None:
    compare = json.loads((RUN2 / "compare.json").read_text(encoding="utf-8"))
    assert compare["run2"]["lessons_loaded"] == 3
    state = reduce(EventReader(RUN2).read())
    assert [lesson.id for lesson in state.lessons_loaded] == [
        "lesson_fx_rate_date",
        "lesson_bank_fee",
        "lesson_fx_series",
    ]

    work_packages = (ROOT / "prd" / "05-WORK-PACKAGES.md").read_text(encoding="utf-8")
    assert (
        "fixtures/demo_run1` (5 gens, holiday failures at g0, witness prune, "
        "3 lessons written, pass³)" in work_packages
    )
    assert "With 2 lessons present" in work_packages


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
            # Binary floats: the fixture's total and this sum differ only in the
            # last ulp, so compare within tolerance rather than bit-for-bit.
            assert sum(case["cost_usd"] for case in repeat["cases"]) == pytest.approx(
                repeat["cost_usd"]
            )


def test_full_repeat_matches_the_configured_ablation_subset() -> None:
    for fixture in FIXTURES:
        events = EventReader(fixture).read()
        ablation_case_ids = {
            event.data["generation"]: event.data["case_ids"]
            for event in events
            if event.type == "ablation.started"
        }
        assert ablation_case_ids
        for generation, case_ids in ablation_case_ids.items():
            assert len(case_ids) == 10
            repeat_started = next(
                event
                for event in events
                if event.type == "execution.started"
                and event.data["generation"] == generation
                and event.data["variant"] == "full_repeat"
            )
            repeat_cases = [
                event.data
                for event in events
                if event.type == "execution.case"
                and event.data["generation"] == generation
                and event.data["variant"] == "full_repeat"
            ]
            assert repeat_started.data["n_cases"] == len(case_ids) == len(repeat_cases)
            assert [case["case_id"] for case in repeat_cases] == case_ids

        for generation in ablation_case_ids:
            full_cases = {
                event.data["case_id"]: event.data["passed"]
                for event in events
                if event.type == "execution.case"
                and event.data["generation"] == generation
                and event.data["variant"] == "full"
            }
            repeat_cases = {
                event.data["case_id"]: event.data["passed"]
                for event in events
                if event.type == "execution.case"
                and event.data["generation"] == generation
                and event.data["variant"] == "full_repeat"
            }
            assert set(repeat_cases) == set(ablation_case_ids[generation])
            if generation == 0:
                disagreements = sum(
                    full_cases[case_id] != repeat_cases[case_id] for case_id in repeat_cases
                )
                assert disagreements == 1
                assert disagreements / len(repeat_cases) == 0.10


def _event_position(events, generation: int, event_type: str, **fields: object) -> int:
    return next(
        index
        for index, event in enumerate(events)
        if event.type == event_type
        and event.data.get("generation") == generation
        and all(event.data.get(key) == value for key, value in fields.items())
    )


def test_fixture_generation_events_follow_the_canonical_ablation_order() -> None:
    """Noise is measured before the optional baseline and ablation table."""

    for fixture in FIXTURES:
        events = EventReader(fixture).read()
        generations = sorted(
            {event.data["generation"] for event in events if "generation" in event.data}
        )
        for generation in generations:
            full_completed = _event_position(
                events, generation, "execution.completed", variant="full"
            )
            repeat_started = _event_position(
                events, generation, "execution.started", variant="full_repeat"
            )
            repeat_completed = _event_position(
                events, generation, "execution.completed", variant="full_repeat"
            )
            baseline = _event_position(events, generation, "baseline.completed")
            ablation_started = _event_position(events, generation, "ablation.started")
            ablation_completed = _event_position(events, generation, "ablation.completed")
            metrics = _event_position(events, generation, "metrics.snapshot")
            rows = [
                index
                for index, event in enumerate(events)
                if event.type == "ablation.role" and event.data.get("generation") == generation
            ]

            assert full_completed < repeat_started < repeat_completed < baseline
            assert baseline < ablation_started < ablation_completed < metrics
            assert all(ablation_started < row < ablation_completed for row in rows)


def test_noise_floor_is_consistent_across_ablation_started_and_metrics() -> None:
    """One noise floor, three places it shows up, all of which must agree.

    `03 §4.1` measures it from the two full runs; `ablation.started` carries it
    so the TUI can print the `03 §8` footer while the table is still filling;
    and `metrics.snapshot.reliability` is its complement (`03 §7`). The fixtures
    only record `passed` per case, so pass-disagreement stands in for the
    answer-disagreement the engine actually compares.
    """

    for fixture in FIXTURES:
        events = EventReader(fixture).read()
        state = reduce(events)
        started = {
            event.data["generation"]: event.data
            for event in events
            if event.type == "ablation.started"
        }
        generations = set(started)
        assert generations, f"{fixture.name} has no ablation.started events"
        for generation in generations:
            full_cases = {
                event.data["case_id"]: event.data["passed"]
                for event in events
                if event.type == "execution.case"
                and event.data["generation"] == generation
                and event.data["variant"] == "full"
            }
            repeat_cases = {
                event.data["case_id"]: event.data["passed"]
                for event in events
                if event.type == "execution.case"
                and event.data["generation"] == generation
                and event.data["variant"] == "full_repeat"
            }
            disagreements = sum(
                full_cases[case_id] != repeat_cases[case_id] for case_id in repeat_cases
            )
            noise_rate = disagreements / len(repeat_cases)
            metrics = next(
                event.data
                for event in events
                if event.type == "metrics.snapshot" and event.data["generation"] == generation
            )
            assert metrics["reliability"] == pytest.approx(1 - noise_rate)
            assert started[generation]["noise_rate"] == pytest.approx(noise_rate)
            assert state.generations[f"g{generation:03d}"].ablation["noise_rate"] == pytest.approx(
                noise_rate
            )


def test_best_generation_reliability_and_summary_are_cross_event_consistent() -> None:
    events = EventReader(RUN1).read()
    state = reduce(events)
    run_completed = next(event for event in events if event.type == "run.completed")
    reliability_events = [event for event in events if event.type == "reliability.completed"]
    assert state.best_generation == run_completed.data["best_generation"] == 3
    assert [event.data["generation"] for event in reliability_events] == [3]
    assert state.generations["g003"].reliability == reliability_events[0].data
    assert state.generations["g004"].reliability is None
    assert state.generations["g003"].metrics is not None
    assert state.generations["g003"].metrics.reliability_pass3 == 0.85
    assert state.generations["g004"].metrics is not None
    assert state.generations["g004"].metrics.reliability_pass3 is None

    summary = run_completed.data["summary"]
    best_metrics = state.generations["g003"].metrics
    assert summary["final_pass_rate"] == best_metrics.pass_rate
    assert summary["final_cost_usd"] == best_metrics.cost_usd
    assert summary["calls_per_case_final"] == best_metrics.tool_calls_per_case

    ranked = sorted(
        (
            generation.metrics.pass_rate,
            -generation.metrics.cost_usd,
            generation.generation,
        )
        for generation in state.generations.values()
        if generation.metrics is not None
    )
    assert ranked[-1][2] == state.best_generation
    reliability_index = next(
        index for index, event in enumerate(events) if event.type == "reliability.completed"
    )
    first_g4_index = next(
        index
        for index, event in enumerate(events)
        if event.type == "architecture.proposed" and event.data["generation"] == 4
    )
    assert reliability_index < first_g4_index


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
    assert result.grade_error is None
    with pytest.raises(ValidationError):
        CaseResult.model_validate({**result.model_dump(), "unexpected": True})


def test_execution_case_diagnostics_are_required_and_bounded() -> None:
    event = next(event for event in EventReader(RUN1) if event.type == "execution.case")

    missing = event.model_dump(mode="json")
    missing["data"].pop("answer_prefix")
    with pytest.raises(ValueError, match="events.schema.json"):
        validate_event(missing)

    too_long = event.model_dump(mode="json")
    too_long["data"]["answer_prefix"] = "x" * 121
    with pytest.raises(ValueError, match="events.schema.json"):
        validate_event(too_long)


def test_reducer_retains_execution_case_diagnostics() -> None:
    event = next(event for event in EventReader(RUN1) if event.type == "execution.case")
    state = reduce(EventReader(RUN1).read())
    execution = state.generations[f"g{event.data['generation']:03d}"].executions[
        event.data["variant"]
    ]
    retained = next(case for case in execution["cases"] if case["case_id"] == event.data["case_id"])
    assert retained["answer_prefix"] == event.data["answer_prefix"]
    assert retained["grade_error"] is None
    assert retained["role_error"] is None


def test_role_trace_carries_the_displayed_cost_and_its_label() -> None:
    """A granted role costs $0 to bill and list-rate-equivalent to show.

    `00 §7` and `01 §5` require the equivalent to be shown *and* labelled, and
    ablation's `cost_share` divides by the displayed number — so a trace that
    only carried the $0 bill would make structural fidelity meaningless.
    """

    granted = RoleTrace(
        tokens_in=1000,
        tokens_out=500,
        cost_usd=0.00026,
        billed_cost_usd=0.0,
        cost_label="list-rate-equivalent",
    )
    assert granted.cost_usd > granted.billed_cost_usd
    assert granted.cost_label == "list-rate-equivalent"

    # Metered is the default, so an executor that forgets to label cannot
    # silently claim a granted rate.
    assert RoleTrace().cost_label == "metered"
    assert RoleTrace().billed_cost_usd == 0.0
    with pytest.raises(ValidationError):
        RoleTrace(cost_label="")
    with pytest.raises(ValidationError):
        RoleTrace(cost_usd=-1.0)
    with pytest.raises(ValidationError):
        RoleTrace.model_validate({"unexpected": True})


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
