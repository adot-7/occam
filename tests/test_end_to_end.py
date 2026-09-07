"""Stubbed end-to-end gates for the engine/store boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from occam.core import Case
from occam.engine import loop as loop_module
from occam.engine.diagnose import DiagnosisResult
from occam.engine.loop import RunConfig, RunEngine
from occam.engine.mutate import Mutation
from occam.engine.mutate import apply_mutation as real_apply_mutation
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

    def __init__(
        self,
        cases: list[Case],
        *,
        correct: bool,
    ) -> None:
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
                        "type": "prune",
                        "target_role": "rate_fetcher",
                        "rationale": "The failing run should not prune without passing evidence.",
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
    completed = next(event for event in events if event.type == "run.completed")
    assert completed.data["summary"]["failed_completions"] == 0

    first_state = reduce(events)
    second_state = reduce(events)
    first_bytes = state_json_bytes(first_state)
    assert first_bytes == state_json_bytes(second_state)
    validate_state(json.loads(first_bytes))
    assert first_state.completed is True
    assert first_state.summary["failed_completions"] == 0
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

    events = EventReader(outcome.run_dir).read()
    ablation_rows = [event for event in events if event.type == "ablation.role"]
    assert ablation_rows
    assert all(event.data["verdict"] == "uncertain" for event in ablation_rows)
    assert all(
        not event.data["witnesses"] for event in events if event.type == "ablation.completed"
    )
    mutations = [event for event in events if event.type == "mutation.applied"]
    assert [event.data["type"] for event in mutations] == ["rewrite_prompt", "rewrite_prompt"]


def test_single_role_prune_failure_is_caught_inside_run_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_task_pack("fx_recon_a")
    cases = pack.select(2)
    llm = StubRunLLM(
        cases,
        correct=False,
    )

    diagnosis_architectures: list[int] = []

    def unsafe_diagnosis(*_args: Any, **kwargs: Any) -> DiagnosisResult:
        architecture = kwargs["architecture"]
        diagnosis_architectures.append(len(architecture.roles))
        if len(diagnosis_architectures) == 1:
            mutation = Mutation(
                type="collapse",
                target_role=architecture.final_role,
                rationale="keep one direct solver for the next generation",
            )
        else:
            mutation = Mutation(
                type="prune",
                target_role=architecture.final_role,
                rationale="remove the redundant solver",
            )
        return DiagnosisResult(
            text="force the structural failure",
            failure_summary="the single role is not removable",
            mutation=mutation,
        )

    calls: list[tuple[int, str]] = []

    def apply_prune(*args: Any, **kwargs: Any) -> Any:
        architecture = args[0]
        mutation = args[1]
        calls.append((len(architecture.roles), mutation.type))
        return real_apply_mutation(*args, **kwargs)

    monkeypatch.setattr(loop_module, "diagnose", unsafe_diagnosis)
    monkeypatch.setattr(loop_module, "apply_mutation", apply_prune)

    outcome = RunEngine(
        pack,
        RunConfig(
            run_name="single-role-prune-failure",
            out=tmp_path / "runs",
            memory=tmp_path / "memory",
            max_generations=3,
            n_cases=2,
            ablate_cases=2,
        ),
        llm=llm,
        registry=ToolRegistry(),
    ).run()

    events = EventReader(outcome.run_dir).read()
    assert diagnosis_architectures == [3, 1]
    assert calls == [(3, "collapse"), (1, "prune")]
    assert outcome.best_generation == 1
    assert outcome.summary["generations"] == 2
    assert events[-1].type == "run.completed"
    assert events[-1].data["best_generation"] == 1
    logs = [event for event in events if event.type == "log"]
    assert logs[-1].data == {
        "level": "warning",
        "message": (
            "Diagnosis or mutation failed; terminating safely at the best completed generation."
        ),
    }
