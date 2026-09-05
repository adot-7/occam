"""Contract tests for the checked-in schemas and replay fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from occam.core import Architecture, Event, MetricsSnapshot
from occam.store.reader import EventReader
from occam.store.reducer import reduce, state_json_bytes
from occam.store.schema import load_schema, validate_event, validate_state, validate_task
from occam.store.writer import EventWriter

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "demo_run"


def test_schemas_are_valid_draft_2020_documents() -> None:
    for name in ("events.schema.json", "state.schema.json", "task.schema.json"):
        Draft202012Validator.check_schema(load_schema(name))


def test_every_fixture_event_validates_against_event_schema() -> None:
    events_path = FIXTURE / "events.jsonl"
    with events_path.open(encoding="utf-8") as handle:
        payloads = [json.loads(line) for line in handle if line.strip()]

    assert len(payloads) >= 100
    assert [payload["seq"] for payload in payloads] == list(range(len(payloads)))
    for payload in payloads:
        validate_event(payload)
        Event.model_validate(payload)


def test_fixture_reduces_to_valid_byte_identical_state() -> None:
    events = EventReader(FIXTURE).read()
    first = reduce(events)
    second = reduce(events)

    first_bytes = state_json_bytes(first)
    assert first_bytes == state_json_bytes(second)
    validate_state(json.loads(first_bytes))
    assert first.last_seq == len(events) - 1
    assert first.best_generation == 3
    assert len(first.generations) == 4
    reverted = first.generations["g002"]
    assert reverted.reverted is True
    assert reverted.mutation is not None
    assert reverted.mutation["type"] == "split"
    assert reverted.mutation["diff"]
    assert reverted.mutation["reverted"] is True
    assert reverted.mutation["revert"]["reason"]
    assert reverted.revert == reverted.mutation["revert"]


def test_core_architecture_model_matches_fixture_contract() -> None:
    architecture_event = next(
        event for event in EventReader(FIXTURE) if event.type == "architecture.proposed"
    )
    architecture = Architecture.model_validate(architecture_event.data["architecture"])
    assert len(architecture.roles) == 5
    assert architecture.final_role == "r_synth"
    assert architecture.control == "deterministic"


def test_metrics_snapshot_has_the_complete_metrics_contract() -> None:
    metrics_event = next(
        event for event in EventReader(FIXTURE) if event.type == "metrics.snapshot"
    )
    metrics = MetricsSnapshot.model_validate(metrics_event.data)
    assert metrics.ci.lo <= metrics.pass_rate <= metrics.ci.hi
    assert metrics.tokens > 0
    assert metrics.reliability > 0
    assert metrics.speed > 0
    assert metrics.latency_s_p50 <= metrics.latency_s_p90


def test_metrics_snapshot_rejects_missing_required_fields() -> None:
    event = next(event for event in EventReader(FIXTURE) if event.type == "metrics.snapshot")
    payload = event.model_dump(mode="json")
    payload["data"].pop("reliability")
    with pytest.raises(ValueError, match="events.schema.json"):
        validate_event(payload)
    with pytest.raises(ValidationError):
        MetricsSnapshot.model_validate(payload["data"])


def test_diagnoses_follow_highest_cost_witness_rule() -> None:
    events = EventReader(FIXTURE).read()
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
    state = reduce(EventReader(FIXTURE).read())
    for generation in state.generations.values():
        repeat = generation.executions["full_repeat"]
        assert repeat["cost_usd"] > 0
        assert repeat["latency_s_mean"] > 0
        assert repeat["tokens"] > 0
        assert all(case["cost_usd"] > 0 for case in repeat["cases"])
        assert all(case["latency_s"] > 0 for case in repeat["cases"])
        assert sum(case["cost_usd"] for case in repeat["cases"]) == repeat["cost_usd"]


def test_task_manifest_shape_is_schema_compatible() -> None:
    task = {
        "name": "smfr_2inv",
        "domain": "financial_reasoning",
        "goal": "Solve the financial reasoning cases.",
        "answer_format": "A JSON list of investor names.",
        "tools": ["python_exec", "lookup_price", "list_transactions"],
        "checker": "json_set_equal",
        "examples": 3,
        "source": {
            "kind": "huggingface",
            "repo": "the-illusion-of-multi-agent-advantages/smfr-dataset",
            "file": "balanced_dataset_single_2_fixed.jsonl",
            "license": "CC-BY-4.0",
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
                },
            }
        )

    events = EventReader(run_dir).read()
    state = EventReader(run_dir).load_state()
    assert len(events) == 1
    assert state.last_seq == 0
    assert (run_dir / "state.json").exists()
