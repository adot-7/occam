"""Focused contract tests for WP-10 orchestration boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from occam.core.models import (
    Architecture,
    Case,
    CaseResult,
    ConfidenceInterval,
    Role,
    RoleTrace,
    RunResult,
)
from occam.engine.ablation import AblationRow, AblationTable
from occam.engine.compare import compare_runs
from occam.engine.diagnose import diagnose, lesson_passes_leak_guard
from occam.engine.pass3 import run_pass3
from occam.memory.lessons import LessonStore


def _architecture() -> Architecture:
    return Architecture(
        id="g000",
        parent_id=None,
        roles=[
            Role(
                id="cheap",
                name="Cheap",
                justification="parallel",
                model="worker_fast",
                system_prompt="cheap",
                tools=[],
                inputs=["task"],
                output_key="cheap_out",
            ),
            Role(
                id="witness_a",
                name="Witness A",
                justification="verification",
                model="worker_fast",
                system_prompt="witness a",
                tools=[],
                inputs=["cheap"],
                output_key="witness_a_out",
            ),
            Role(
                id="witness_b",
                name="Witness B",
                justification="verification",
                model="worker_fast",
                system_prompt="witness b",
                tools=[],
                inputs=["cheap"],
                output_key="witness_b_out",
            ),
        ],
        final_role="witness_b",
    )


def test_wp10_leak_guard_accepts_canonical_lessons_and_rejects_case_values() -> None:
    assert lesson_passes_leak_guard(
        "The response's rate_date is the actual ECB business day used and may be earlier "
        "than the requested date; use the returned rate as-is."
    )
    assert lesson_passes_leak_guard(
        "When a settlement is net of a bank fee in the reporting currency, add the fee "
        "back before computing FX gain or loss."
    )
    for text in (
        "Use the 2026-04-03 response date.",
        "Do not mention INV-2291 in a lesson.",
        "The observed total was 41872.35.",
    ):
        assert not lesson_passes_leak_guard(text, case_values=[41872.35])


def test_wp10_witness_rule_forces_highest_cost_prune_when_some_cases_pass(
    tmp_path: Path,
) -> None:
    rows = [
        AblationRow(
            role_id="witness_a",
            role_name="Witness A",
            justification="verification",
            influence=0.0,
            influence_ci=ConfidenceInterval(lo=-0.1, hi=0.1),
            divergence=0.0,
            cost_share=0.2,
            verdict="witness",
            n_cases=2,
        ),
        AblationRow(
            role_id="witness_b",
            role_name="Witness B",
            justification="verification",
            influence=0.0,
            influence_ci=ConfidenceInterval(lo=-0.1, hi=0.1),
            divergence=0.0,
            cost_share=0.6,
            verdict="witness",
            n_cases=2,
        ),
    ]
    table = AblationTable(generation=0, case_ids=["c1", "c2"], noise_rate=0.0, rows=rows)
    full = RunResult(
        architecture_id="g000",
        variant="full",
        results=[
            CaseResult(
                case_id="c1",
                answer="bad",
                passed=False,
                per_role={"cheap": RoleTrace(cost_usd=0.1)},
            ),
            CaseResult(
                case_id="c2",
                answer="good",
                passed=True,
                per_role={"cheap": RoleTrace(cost_usd=0.1)},
            ),
        ],
        pass_rate=0.5,
    )
    cases = [
        Case(id="c1", input="input 1", expected={"total_inr": 100.0}),
        Case(id="c2", input="input 2", expected={"total_inr": 100.0}),
    ]

    class StubLLM:
        def complete(self, *_args: object, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "text": "remove the redundant checker",
                    "failure_summary": "the final answer failed",
                    "chosen_mutation": {
                        "type": "rewrite_prompt",
                        "target_role": "cheap",
                        "rationale": "model suggestion",
                    },
                    "lessons": [],
                }
            )

    events: list[tuple[str, dict]] = []
    store = LessonStore(tmp_path / "memory")
    result = diagnose(
        {"goal": "test"},
        generation=0,
        full=full,
        table=table,
        architecture=_architecture(),
        cases=cases,
        llm=StubLLM(),
        lesson_store=store,
        run_id="run1",
        event_sink=lambda event_type, data: events.append((event_type, dict(data))),
    )

    assert result.mutation.type == "prune"
    assert result.mutation.target_role == "witness_b"
    assert [event_type for event_type, _ in events] == ["diagnosis.emitted"]


def test_wp10_all_failed_diagnosis_falls_back_from_prune(tmp_path: Path) -> None:
    row = AblationRow(
        role_id="witness_b",
        role_name="Witness B",
        justification="verification",
        influence=0.0,
        influence_ci=ConfidenceInterval(lo=-0.1, hi=0.1),
        divergence=0.0,
        cost_share=0.6,
        verdict="witness",
        n_cases=1,
    )
    table = AblationTable(generation=0, case_ids=["c1"], noise_rate=0.0, rows=[row])
    full = RunResult(
        architecture_id="g000",
        variant="full",
        results=[CaseResult(case_id="c1", answer="bad", passed=False)],
        pass_rate=0.0,
    )
    case = Case(id="c1", input="input", expected={"total_inr": 100.0})

    class StubLLM:
        def complete(self, *_args: object, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "text": "the answer needs a safer prompt",
                    "failure_summary": "all cases failed",
                    "chosen_mutation": {
                        "type": "prune",
                        "target_role": "witness_b",
                        "rationale": "the witness appears redundant",
                    },
                    "lessons": [],
                }
            )

    result = diagnose(
        {"goal": "test"},
        generation=0,
        full=full,
        table=table,
        architecture=_architecture(),
        cases=[case],
        llm=StubLLM(),
        lesson_store=LessonStore(tmp_path / "memory"),
        run_id="run1",
    )

    assert result.mutation.type == "rewrite_prompt"
    assert result.mutation.target_role == "witness_b"


def test_wp10_diagnosis_does_not_prune_to_one_role(tmp_path: Path) -> None:
    full_architecture = _architecture()
    architecture = full_architecture.model_copy(
        update={
            "roles": full_architecture.roles[:2],
            "final_role": "witness_a",
        }
    )
    row = AblationRow(
        role_id="witness_a",
        role_name="Witness A",
        justification="verification",
        influence=0.0,
        influence_ci=ConfidenceInterval(lo=-0.1, hi=0.1),
        divergence=0.0,
        cost_share=0.6,
        verdict="witness",
        n_cases=2,
    )
    table = AblationTable(generation=0, case_ids=["c1", "c2"], noise_rate=0.0, rows=[row])
    full = RunResult(
        architecture_id="g000",
        variant="full",
        results=[
            CaseResult(case_id="c1", answer="bad", passed=False),
            CaseResult(case_id="c2", answer="good", passed=True),
        ],
        pass_rate=0.5,
    )
    cases = [
        Case(id="c1", input="input 1", expected={"total_inr": 100.0}),
        Case(id="c2", input="input 2", expected={"total_inr": 100.0}),
    ]

    class StubLLM:
        def complete(self, *_args: object, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "text": "the answer needs a safer prompt",
                    "failure_summary": "the final answer failed",
                    "chosen_mutation": {
                        "type": "prune",
                        "target_role": "witness_a",
                        "rationale": "the witness appears redundant",
                    },
                    "lessons": [],
                }
            )

    result = diagnose(
        {"goal": "test"},
        generation=0,
        full=full,
        table=table,
        architecture=architecture,
        cases=cases,
        llm=StubLLM(),
        lesson_store=LessonStore(tmp_path / "memory"),
        run_id="run1",
    )

    assert result.mutation.type == "rewrite_prompt"
    assert result.mutation.target_role == "witness_a"


def test_wp10_compare_writes_second_run_artifact(tmp_path: Path) -> None:
    payload = compare_runs(
        Path("fixtures/demo_run1"),
        Path("fixtures/demo_run2"),
        write=False,
    )
    assert payload["run2"]["lessons_loaded"] == 3
    assert set(payload["delta"]) == {
        "g0_pass_rate",
        "final_pass_rate",
        "g0_tool_calls_per_case",
        "g0_cost_per_case",
        "generations_to_plateau",
        "reliability_pass3",
    }


def test_wp10_pass3_runs_three_uncached_passes(tmp_path: Path) -> None:
    def always_passes(_answer: str, _expected: object) -> bool:
        return True

    class FakeExecutor:
        llm = object()
        tools = {}
        grader = staticmethod(always_passes)
        run_name = "test"
        max_tokens = None
        case_concurrency = 1
        _model_concurrency = 1

        calls: list[tuple[str, bool]] = []

        def __init__(self, **_kwargs: object) -> None:
            pass

        def run_variant(self, architecture, cases, *, variant, use_cache, generation, grader):
            FakeExecutor.calls.append((variant, use_cache))
            return RunResult(
                architecture_id=architecture.id,
                variant=variant,
                results=[CaseResult(case_id=case.id, answer="ok", passed=True) for case in cases],
                pass_rate=1.0,
                cost_usd=0.1,
                latency_s_mean=0.1,
                tokens=1,
            )

    executor = FakeExecutor()
    result = run_pass3(
        _architecture(),
        [Case(id="c1", input="x", expected={})],
        executor=executor,  # type: ignore[arg-type]
        generation=0,
        run_dir=tmp_path,
        grader=lambda _answer, _expected: True,
    )
    assert result.reliable_cases == 1
    assert len(result.runs) == 3
    assert FakeExecutor.calls == [
        ("pass3:1", False),
        ("pass3:2", False),
        ("pass3:3", False),
    ]
