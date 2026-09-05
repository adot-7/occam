"""Contract tests for the checked-in schemas and replay fixture."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from occam.core import Architecture, Event
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


def test_core_architecture_model_matches_fixture_contract() -> None:
    architecture_event = next(
        event for event in EventReader(FIXTURE) if event.type == "architecture.proposed"
    )
    architecture = Architecture.model_validate(architecture_event.data["architecture"])
    assert len(architecture.roles) == 5
    assert architecture.final_role == "r_synth"
    assert architecture.control == "deterministic"


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
