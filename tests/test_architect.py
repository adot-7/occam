"""WP-06 architect context, strict output, and event tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from occam.core import Architecture
from occam.engine.architect import DOMAIN_RULE_HEADING, Architect, ArchitectError
from occam.llm import LLMClient, ModelConfig, ProviderResponse, RetryPolicy
from occam.memory.lessons import LessonStore
from occam.store.reader import EventReader
from occam.store.schema import validate_event
from occam.store.writer import EventWriter

TASK = {
    "name": "fx_recon_a",
    "domain": "finance_ops",
    "goal": "Compute FX gain or loss in INR.",
    "answer_format": "Return a fenced JSON object.",
    "tools": ["fx_rate", "fx_series", "python_exec"],
    "checker": "fx_total",
    "examples": 3,
    "memory": "memory/fx_recon",
}


def lesson_payload(
    lesson_id: str,
    kind: str,
    text: str,
    tool: str | None,
) -> dict[str, Any]:
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


def architecture_payload() -> dict[str, Any]:
    return {
        "id": "g000",
        "parent_id": None,
        "roles": [
            {
                "id": "r_parse",
                "name": "Ledger Parser",
                "justification": "context_isolation",
                "model": "worker_fast",
                "system_prompt": "Parse the ledger into structured records.",
                "tools": [],
                "inputs": ["task"],
                "output_key": "parsed",
            },
            {
                "id": "r_rates",
                "name": "Rate Fetcher",
                "justification": "parallel",
                "model": "worker_fast",
                "system_prompt": "Fetch the rates needed by the calculator.",
                "tools": ["fx_rate", "fx_series"],
                "inputs": ["r_parse"],
                "output_key": "rates",
            },
            {
                "id": "r_calc",
                "name": "FX Calculator",
                "justification": "control",
                "model": "worker_fast",
                "system_prompt": "Compute the answer using the parsed ledger and rates.",
                "tools": ["python_exec"],
                "inputs": ["r_parse", "r_rates"],
                "output_key": "answer",
            },
        ],
        "final_role": "r_calc",
        "control": "deterministic",
        "notes": "",
    }


class StubArchitectLLM:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        response_text: str | None = None,
        configs: dict[str, Any] | None = None,
    ) -> None:
        self.payload = payload or architecture_payload()
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []
        self.configs = configs or {"architect": object(), "worker_fast": object()}

    def complete(self, model_key: str, messages: Any, **kwargs: Any) -> Any:
        self.calls.append({"model_key": model_key, "messages": messages, **kwargs})
        text = self.response_text
        if text is None:
            text = json.dumps(self.payload)
        return SimpleNamespace(text=text, tool_calls=[])


class FailingArchitectLLM(StubArchitectLLM):
    def complete(self, model_key: str, messages: Any, **kwargs: Any) -> Any:
        raise RuntimeError("sensitive provider payload")


class SequenceArchitectLLM(StubArchitectLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)

    def complete(self, model_key: str, messages: Any, **kwargs: Any) -> Any:
        self.calls.append({"model_key": model_key, "messages": messages, **kwargs})
        return SimpleNamespace(text=self.responses.pop(0), tool_calls=[])


class SequenceProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, config: ModelConfig, messages: Any, **kwargs: Any) -> ProviderResponse:
        self.calls.append({"config": config, "messages": messages, **kwargs})
        return ProviderResponse(
            text=self.responses.pop(0),
            tokens_in=100,
            tokens_out=10,
        )


def make_lesson_store(path: Path) -> LessonStore:
    store = LessonStore(path)
    store.append(
        lesson_payload(
            "lesson_fx_rate_date",
            "tool_note",
            "Trust the response rate_date and returned rate as-is.",
            "fx_rate",
        )
    )
    store.append(
        lesson_payload(
            "lesson_bank_fee",
            "domain_rule",
            "A bank fee is a charge, not FX; add it back to received value.",
            None,
        )
    )
    return store


def test_architect_reads_two_lessons_and_injects_them_without_hardcoding(tmp_path: Path) -> None:
    memory = tmp_path / "memory" / "fx_recon"
    make_lesson_store(memory)
    llm = StubArchitectLLM()
    architect = Architect(llm=llm, memory=memory)

    architecture = architect.propose(TASK)
    request_text = json.dumps(llm.calls[0]["messages"], ensure_ascii=False)

    assert isinstance(architecture, Architecture)
    assert "Trust the response rate_date" in request_text
    assert "A bank fee is a charge" in request_text
    assert DOMAIN_RULE_HEADING in request_text
    assert architect.last_context is not None
    assert "Trust the response rate_date" in architect.last_context.tools[0].description
    assert "A bank fee is a charge" in request_text
    assert [lesson.id for lesson in architect.last_lessons] == [
        lesson.id for lesson in LessonStore(memory).load()
    ]


def test_architect_with_zero_lessons_keeps_plain_tool_context(tmp_path: Path) -> None:
    memory = tmp_path / "memory" / "fx_recon"
    llm = StubArchitectLLM()
    architect = Architect(llm=llm, memory=memory)

    architect.propose(TASK)
    request_text = json.dumps(llm.calls[0]["messages"], ensure_ascii=False)

    assert "rate_date" not in request_text
    assert "A bank fee is a charge" not in request_text
    assert architect.last_context is not None
    assert architect.last_context.domain_rules == []


def test_architect_guides_tool_roles_to_allow_enough_turns(tmp_path: Path) -> None:
    llm = StubArchitectLLM()
    Architect(llm=llm, memory=tmp_path / "memory").propose(TASK)

    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "tool calls are expected" in system_prompt
    assert "max_turns >= 16" in system_prompt


def test_architect_emits_schema_compatible_event(tmp_path: Path) -> None:
    memory = tmp_path / "memory" / "fx_recon"
    writer_dir = tmp_path / "run"
    with EventWriter(writer_dir, run_id="run1") as writer:
        writer.append(
            {
                "ts": "2026-09-06T00:00:00Z",
                "type": "run.started",
                "data": {
                    "task": {"name": "fx", "domain": "test", "n_cases": 0, "goal": "test"},
                    "config": {},
                    "run_name": "run1",
                    "memory_ns": str(memory),
                    "lessons_loaded": [],
                },
            }
        )
        Architect(llm=StubArchitectLLM(), memory=memory, writer=writer).propose(TASK)

    events = EventReader(writer_dir).read()
    proposal = next(event for event in events if event.type == "architecture.proposed")
    validate_event(proposal.model_dump(mode="json"))
    assert proposal.data["generation"] == 0
    assert proposal.data["architecture"]["id"] == "g000"


@pytest.mark.parametrize(
    "response_text",
    [
        lambda payload: f"Here is the architecture:\n{json.dumps(payload)}\n",
        lambda payload: f"```json\n{json.dumps(payload)}\n```\n",
    ],
)
def test_architect_accepts_one_json_object_with_minimal_wrapping(
    tmp_path: Path,
    response_text: Any,
) -> None:
    payload = architecture_payload()
    llm = StubArchitectLLM(response_text=response_text(payload))

    architecture = Architect(llm=llm, memory=tmp_path / "memory").propose(TASK)

    assert architecture.id == "g000"
    assert architecture.final_role == "r_calc"


@pytest.mark.parametrize(
    "response_text",
    [
        "No architecture was produced.",
        '{"roles":',
        lambda payload: f"{json.dumps(payload)}\n{json.dumps(payload)}",
        lambda payload: f"```json\n{json.dumps(payload)}\n```\n```json\n{json.dumps(payload)}\n```",
        lambda payload: f"```json\n[{json.dumps(payload)}]\n```",
    ],
)
def test_architect_rejects_missing_malformed_or_ambiguous_json(
    tmp_path: Path,
    response_text: Any,
) -> None:
    payload = architecture_payload()
    text = response_text(payload) if callable(response_text) else response_text

    with pytest.raises(ArchitectError, match="output-shape failure"):
        Architect(
            llm=StubArchitectLLM(response_text=text),
            memory=tmp_path / "memory",
        ).propose(TASK)


def test_architect_keeps_strict_validation_after_wrapping_is_removed(tmp_path: Path) -> None:
    payload = {**architecture_payload(), "unexpected": True}
    text = f"Here is the architecture:\n```json\n{json.dumps(payload)}\n```"

    with pytest.raises(ArchitectError, match="proposal validation failure.*strict validation"):
        Architect(
            llm=StubArchitectLLM(response_text=text),
            memory=tmp_path / "memory",
        ).propose(TASK)


def test_architect_provider_failure_is_distinct_and_redacted(tmp_path: Path) -> None:
    with pytest.raises(ArchitectError, match="provider failure: RuntimeError") as error:
        Architect(llm=FailingArchitectLLM(), memory=tmp_path / "memory").propose(TASK)

    assert "sensitive provider payload" not in str(error.value)


def test_architect_schema_and_prompt_advertise_only_configured_role_keys(tmp_path: Path) -> None:
    llm = StubArchitectLLM(
        configs={
            "architect": object(),
            "worker_alt": object(),
            "worker_fast": object(),
        }
    )

    Architect(llm=llm, memory=tmp_path / "memory").propose(TASK)

    call = llm.calls[0]
    role_schema = call["response_schema"]["$defs"]["Role"]
    assert role_schema["properties"]["model"]["enum"] == ["worker_alt", "worker_fast"]
    request_text = json.dumps(call["messages"], ensure_ascii=False)
    assert "CONFIGURED ROLE MODEL KEYS" in request_text
    assert "provider model IDs" in request_text
    assert "worker_fast" in request_text
    assert "gpt-4.1" not in request_text


def test_architect_accepts_a_configured_worker_model_key(tmp_path: Path) -> None:
    payload = architecture_payload()
    llm = StubArchitectLLM(
        payload=payload,
        response_text=f"Here is the plan:\n{json.dumps(payload)}",
    )

    architecture = Architect(llm=llm, memory=tmp_path / "memory").propose(TASK)

    assert {role.model for role in architecture.roles} == {"worker_fast"}


def test_architect_rejects_raw_provider_model_ids_before_execution(tmp_path: Path) -> None:
    payload = architecture_payload()
    payload["roles"][0]["model"] = "gpt-4.1"

    with pytest.raises(ArchitectError, match=r"role model key\(s\) are not configured") as error:
        Architect(llm=StubArchitectLLM(payload), memory=tmp_path / "memory").propose(TASK)

    assert "gpt-4.1" in str(error.value)
    assert "worker_fast" in str(error.value)


def test_architect_repairs_unknown_dag_input_once_and_emits_one_event(tmp_path: Path) -> None:
    invalid = architecture_payload()
    invalid["roles"][2]["inputs"] = ["r_parse", "parsed_case"]
    valid = architecture_payload()
    llm = SequenceArchitectLLM([json.dumps(invalid), json.dumps(valid)])
    emitted: list[str] = []

    architecture = Architect(
        llm=llm,
        memory=tmp_path / "memory",
        event_sink=lambda event_type, _data: emitted.append(event_type),
    ).propose(TASK)

    assert architecture.final_role == "r_calc"
    assert len(llm.calls) == 2
    repair_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "Validation category: DAG input contract" in repair_prompt
    assert "parsed_case" in repair_prompt
    assert '"worker_fast"' in repair_prompt
    assert '"task"' in repair_prompt
    assert '"r_parse", "parsed"' in repair_prompt
    assert emitted == ["architecture.proposed"]


def test_architect_repair_exhaustion_preserves_strict_failure(tmp_path: Path) -> None:
    invalid = architecture_payload()
    invalid["roles"][2]["inputs"] = ["r_parse", "parsed_case"]
    llm = SequenceArchitectLLM([json.dumps(invalid), json.dumps(invalid)])
    emitted: list[str] = []

    with pytest.raises(ArchitectError, match="invalid DAG"):
        Architect(
            llm=llm,
            memory=tmp_path / "memory",
            event_sink=lambda event_type, _data: emitted.append(event_type),
        ).propose(TASK)

    assert len(llm.calls) == 2
    assert emitted == []


def test_architect_repair_calls_are_truthfully_accounted(tmp_path: Path) -> None:
    invalid = architecture_payload()
    invalid["roles"][2]["inputs"] = ["r_parse", "parsed_case"]
    provider = SequenceProvider([json.dumps(invalid), json.dumps(architecture_payload())])
    architect_config = ModelConfig(
        key="architect",
        provider="test",
        model="claude-sonnet-5",
        in_per_m=2.0,
        out_per_m=10.0,
    )
    worker_config = ModelConfig(
        key="worker_fast",
        provider="test",
        model="worker-test",
    )
    llm = LLMClient(
        configs={"architect": architect_config, "worker_fast": worker_config},
        providers={"architect": provider},
        cache_dir=tmp_path / "cache",
        retry_policy=RetryPolicy(max_attempts=1),
    )

    architecture = Architect(llm=llm, memory=tmp_path / "memory").propose(TASK)

    assert architecture.final_role == "r_calc"
    assert len(provider.calls) == 2
    assert llm.provider_call_count == 2
    expected_cost = 2 * (100 * 2.0 + 10 * 10.0) / 1_000_000
    assert llm.displayed_cost_usd == pytest.approx(expected_cost)
    assert llm.billed_cost_usd == pytest.approx(expected_cost)


def test_architect_overwrites_model_guessed_id_instead_of_raising(tmp_path: Path) -> None:
    # A model that invents a plausible-looking id ("fx_recon_a_v1" instead of
    # "g000") must not kill the run - id/parent_id are engine bookkeeping,
    # assigned after the model responds, not something it can get "wrong".
    payload = {**architecture_payload(), "id": "fx_recon_a_v1", "parent_id": "not-a-real-parent"}
    llm = StubArchitectLLM(payload)

    architecture = Architect(llm=llm, memory=tmp_path / "memory").propose(TASK)

    assert architecture.id == "g000"
    assert architecture.parent_id is None

    schema = llm.calls[0]["response_schema"]
    assert "id" not in schema["properties"]
    assert "parent_id" not in schema["properties"]
    assert "id" not in schema["required"]
    assert "parent_id" not in schema["required"]


def test_architect_assigns_parent_id_from_the_loop_for_later_generations(tmp_path: Path) -> None:
    payload = {**architecture_payload(), "id": "whatever-the-model-felt-like", "parent_id": "g000"}
    llm = StubArchitectLLM(payload)

    architecture = Architect(llm=llm, memory=tmp_path / "memory").propose(
        TASK, generation=1, parent_id="g000-actual-parent"
    )

    assert architecture.id == "g001"
    assert architecture.parent_id == "g000-actual-parent"


@pytest.mark.parametrize("field", ("id", "name", "model", "output_key"))
def test_architect_rejects_empty_schema_constrained_role_fields_without_writer(
    tmp_path: Path, field: str
) -> None:
    payload = architecture_payload()
    payload["roles"][0][field] = ""

    with pytest.raises(ArchitectError, match="strict validation"):
        Architect(llm=StubArchitectLLM(payload), memory=tmp_path / "memory").propose(TASK)


def test_architect_event_compatibility_guard_rejects_invalid_payload() -> None:
    payload = architecture_payload()
    payload["roles"][0]["output_key"] = ""

    with pytest.raises(ArchitectError, match="event schema"):
        Architect._validate_event_compatibility({"architecture": payload, "generation": 0})


def test_architect_rejects_non_strict_or_ambiguous_output(tmp_path: Path) -> None:
    bad_payload = {**architecture_payload(), "unexpected": True}
    with pytest.raises(ArchitectError, match="strict validation"):
        Architect(llm=StubArchitectLLM(bad_payload), memory=tmp_path / "memory").propose(TASK)

    llm = StubArchitectLLM({**architecture_payload(), "control": "llm"})
    with pytest.raises(ArchitectError, match="router convention"):
        Architect(llm=llm, memory=tmp_path / "memory").propose(TASK)
