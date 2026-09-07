"""Stubbed end-to-end gates for the engine/store boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from occam.core import Case
from occam.engine.loop import RunConfig, RunEngine
from occam.llm.client import Completion
from occam.llm.config import ModelConfig
from occam.store.reader import EventReader
from occam.store.reducer import reduce, state_json_bytes
from occam.store.schema import validate_event, validate_state
from occam.tasks.loader import load_task_pack
from occam.tools.registry import ToolRegistry


def _architecture_payload() -> dict[str, Any]:
    return {
        "roles": [
            {
                "id": "ledger_parser",
                "name": "Ledger Parser",
                "justification": "context_isolation",
                "model": "worker_fast",
                "system_prompt": "Parse the ledger.",
                "tools": [],
                "inputs": ["task"],
                "output_key": "parsed_ledger",
            },
            {
                "id": "rate_fetcher",
                "name": "Rate Fetcher",
                "justification": "parallel",
                "model": "worker_fast",
                "system_prompt": "Fetch the required rates.",
                "tools": [],
                "inputs": ["ledger_parser"],
                "output_key": "fx_rates",
            },
            {
                "id": "calculator",
                "name": "FX Calculator",
                "justification": "control",
                "model": "worker_fast",
                "system_prompt": "Compute the requested result.",
                "tools": [],
                "inputs": ["task", "rate_fetcher"],
                "output_key": "answer",
            },
        ],
        "final_role": "calculator",
        "control": "deterministic",
        "notes": "stubbed end-to-end architecture",
    }


class StubRunLLM:
    """A deterministic architect/worker boundary with real executor accounting."""

    def __init__(self, cases: list[Case], *, correct: bool) -> None:
        self.correct = correct
        self.expected = {case.input: case.expected for case in cases}
        worker_config = ModelConfig(
            key="worker_fast",
            provider="stub",
            model="stub-worker",
            api_key="test-key",
            grant_equiv_in_per_m=0.06,
            grant_equiv_out_per_m=0.40,
            rpm=60,
        )
        self.configs = {
            "worker_fast": worker_config,
            "architect": replace(worker_config, key="architect", model="stub-architect"),
        }
        self.provider_call_count = 0
        self._displayed_cost_usd = 0.0
        self._billed_cost_usd = 0.0

    @property
    def displayed_cost_usd(self) -> float:
        return self._displayed_cost_usd

    @property
    def billed_cost_usd(self) -> float:
        return self._billed_cost_usd

    def complete(
        self,
        model_key: str,
        messages: list[dict[str, Any]],
        *_args: Any,
        **_kwargs: Any,
    ) -> Completion:
        self.provider_call_count += 1
        system = str(messages[0].get("content", ""))
        if model_key == "architect" and "failure diagnostician" not in system:
            text = json.dumps(_architecture_payload(), sort_keys=True)
        elif model_key == "architect":
            text = json.dumps(
                {
                    "text": "Review the failed answer.",
                    "failure_summary": "The stubbed worker answer is wrong.",
                    "chosen_mutation": {
                        "type": "rewrite_prompt",
                        "target_role": "calculator",
                        "rationale": "Preserve the required answer format.",
                    },
                    "lessons": [],
                },
                sort_keys=True,
            )
        else:
            user = str(messages[1].get("content", ""))
            expected = next(
                (value for case_input, value in self.expected.items() if case_input in user),
                None,
            )
            if self.correct and expected is not None:
                answer = expected
            else:
                answer = {"total_inr": 0, "per_invoice": {}}
            text = f"```json\n{json.dumps(answer, sort_keys=True)}\n```"

        cost = 0.000001
        self._displayed_cost_usd += cost
        return Completion(
            text=text,
            tool_calls=[],
            tokens_in=10,
            tokens_out=10,
            cost_usd=cost,
            billed_cost_usd=0.0,
            latency_s=0.0,
            cached=False,
            cost_label="stub",
        )


def _run_stubbed(tmp_path: Path, *, correct: bool):
    pack = load_task_pack("fx_recon_a")
    cases = pack.select(3)
    llm = StubRunLLM(cases, correct=correct)
    outcome = RunEngine(
        pack,
        RunConfig(
            run_name="stubbed-correct" if correct else "stubbed-wrong",
            out=tmp_path / "runs",
            memory=tmp_path / "memory",
            max_generations=1,
            n_cases=3,
            ablate_cases=3,
        ),
        llm=llm,
        registry=ToolRegistry(),
    ).run(max_generations=3)
    return outcome


def _assert_event_contracts(outcome: Any, *, expected_pass_rate: float) -> None:
    events = EventReader(outcome.run_dir).read()
    serialized = [event.model_dump(mode="json", exclude_none=False) for event in events]
    for event in serialized:
        validate_event(event)
    assert [event.seq for event in events] == list(range(len(events)))
    started = next(event for event in events if event.type == "run.started")
    assert started.data["config"]["max_generations"] == 3

    full_completed = [
        event
        for event in events
        if event.type == "execution.completed" and event.data["variant"] == "full"
    ]
    assert [event.data["generation"] for event in full_completed] == [0, 1, 2]
    assert [event.data["pass_rate"] for event in full_completed] == [expected_pass_rate] * 3
    assert sum(event.type == "run.completed" for event in events) == 1

    first_state = reduce(events)
    second_state = reduce(events)
    first_bytes = state_json_bytes(first_state)
    assert first_bytes == state_json_bytes(second_state)
    validate_state(json.loads(first_bytes))
    assert first_state.completed is True
    assert first_state.last_seq == len(events) - 1
    assert (outcome.run_dir / "state.json").read_bytes() == first_bytes


def test_stubbed_correct_run_completes_three_generations(tmp_path: Path) -> None:
    outcome = _run_stubbed(tmp_path, correct=True)

    assert outcome.summary["final_pass_rate"] == pytest.approx(1.0)
    assert outcome.summary["displayed_cost_usd"] > 0.0
    _assert_event_contracts(outcome, expected_pass_rate=1.0)


def test_stubbed_wrong_run_completes_three_generations_without_raising(tmp_path: Path) -> None:
    outcome = _run_stubbed(tmp_path, correct=False)

    assert outcome.summary["final_pass_rate"] == pytest.approx(0.0)
    assert outcome.summary["displayed_cost_usd"] > 0.0
    _assert_event_contracts(outcome, expected_pass_rate=0.0)
