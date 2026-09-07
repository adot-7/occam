"""Cost-matched chain-of-thought self-consistency baseline."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from occam.core.models import Architecture, Case, CaseResult, Role, RunResult, Task
from occam.engine.executor import Executor

MAX_SAMPLES = 9
BASELINE_MODEL = "worker_fast"


@dataclass(frozen=True)
class BaselineResult:
    """The voted baseline result and the event fields shown by the TUI."""

    run: RunResult
    k: int
    matched_to_cost_usd: float
    samples: tuple[RunResult, ...] = ()

    @property
    def pass_rate(self) -> float:
        return self.run.pass_rate

    @property
    def cost_usd(self) -> float:
        return self.run.cost_usd

    @property
    def latency_s_mean(self) -> float:
        return self.run.latency_s_mean

    @property
    def billed_cost_usd(self) -> float:
        return sum(
            trace.billed_cost_usd
            for sample in self.samples
            for case in sample.results
            for trace in case.per_role.values()
        )

    def event_data(self, generation: int) -> dict[str, Any]:
        return {
            "generation": generation,
            "method": "cot_sc",
            "k": self.k,
            "pass_rate": self.pass_rate,
            "cost_usd": self.cost_usd,
            "latency_s_mean": self.latency_s_mean,
            "matched_to_cost_usd": self.matched_to_cost_usd,
        }


def _normalise(answer: str) -> str:
    """Normalize only incidental whitespace for majority voting."""

    return " ".join(answer.split())


def _baseline_architecture(task: Task, *, model_key: str) -> Architecture:
    role = Role(
        id="cot_sc_solver",
        name="Cost-matched CoT-SC solver",
        justification="control",
        model=model_key,
        system_prompt=(
            f"Solve this task carefully.\n\n{task.goal}\n\n"
            f"Required answer format:\n{task.answer_format}\n\n"
            "Think step by step, use the available tools when needed, and end with the "
            "required answer."
        ),
        tools=list(task.tools),
        inputs=["task"],
        output_key="answer",
        max_turns=6,
    )
    return Architecture(
        id="baseline-cot-sc",
        parent_id=None,
        roles=[role],
        final_role=role.id,
        control="deterministic",
        notes="Single-role chain-of-thought self-consistency baseline.",
    )


def _k_for_cost(full_cost: float, sample_cost: float) -> int:
    """Apply the PRD's floor-and-clamp cost matching rule."""

    if not math.isfinite(full_cost) or full_cost <= 0 or sample_cost <= 0:
        return 1
    return max(1, min(MAX_SAMPLES, math.floor(full_cost / sample_cost)))


def _aggregate_case(
    case: Case,
    samples: list[CaseResult],
    *,
    grader: Any,
) -> CaseResult:
    answers = [_normalise(sample.answer) for sample in samples]
    counts = Counter(answers)
    # Counter preserves first-seen order on ties, matching the PRD's tie rule.
    winner = max(counts, key=counts.get)
    chosen = next(
        sample for sample, answer in zip(samples, answers, strict=True) if answer == winner
    )
    passed, sub_results = chosen.passed, dict(chosen.sub_results)
    grade_error = chosen.grade_error
    role_error = chosen.role_error
    if role_error:
        passed = False
    elif grader is not None:
        try:
            outcome = grader(winner, case.expected)
            if isinstance(outcome, bool):
                passed, raw_sub_results = outcome, {}
            else:
                passed = bool(
                    outcome.get("passed", False)
                    if isinstance(outcome, Mapping)
                    else getattr(outcome, "passed", False)
                )
                raw_sub_results = (
                    outcome.get("sub_results", {})
                    if isinstance(outcome, Mapping)
                    else getattr(outcome, "sub_results", {})
                )
            sub_results = {str(key): bool(value) for key, value in dict(raw_sub_results).items()}
        except Exception:  # noqa: BLE001 - baseline failure is a normal failed case
            passed, sub_results = False, {}
    return CaseResult(
        case_id=case.id,
        answer=chosen.answer,
        passed=passed,
        grade_error=grade_error,
        role_error=role_error,
        sub_results=sub_results,
        tokens_in=sum(sample.tokens_in for sample in samples),
        tokens_out=sum(sample.tokens_out for sample in samples),
        cost_usd=sum(sample.cost_usd for sample in samples),
        latency_s=sum(sample.latency_s for sample in samples),
        per_role=chosen.per_role,
    )


def _write_results(run_dir: str | Path, results: list[CaseResult]) -> None:
    destination = Path(run_dir) / "baseline" / "results.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")


def run_baseline(
    task: Task,
    cases: list[Case],
    *,
    full_cost_usd: float,
    executor: Executor,
    run_dir: str | Path | None = None,
    generation: int = 0,
    model_key: str = BASELINE_MODEL,
    grader: Any = None,
) -> BaselineResult:
    """Run CoT-SC with ``k`` samples matched to the current full-run cost.

    Sampling deliberately bypasses the content cache: cache hits would turn
    self-consistency into repeated copies of one answer and would under-report
    the baseline's spend.
    """

    if not cases:
        raise ValueError("baseline needs at least one case")
    architecture = _baseline_architecture(task, model_key=model_key)
    samples: list[RunResult] = []
    first = executor.__class__(
        llm=executor.llm,
        tools=executor.tools,
        grader=grader if grader is not None else executor.grader,
        writer=None,
        run_dir=None,
        run_name=executor.run_name,
        temperature=0.7,
        max_tokens=executor.max_tokens,
        case_concurrency=executor.case_concurrency,
        model_concurrency=executor._model_concurrency,
    )
    sample = first.run_variant(
        architecture,
        cases,
        variant="baseline:cot_sc:1",
        use_cache=False,
        generation=generation,
        grader=grader,
    )
    samples.append(sample)
    k = _k_for_cost(full_cost_usd, sample.cost_usd)
    for sample_number in range(2, k + 1):
        samples.append(
            first.run_variant(
                architecture,
                cases,
                variant=f"baseline:cot_sc:{sample_number}",
                use_cache=False,
                generation=generation,
                grader=grader,
            )
        )

    by_case: dict[str, list[CaseResult]] = {case.id: [] for case in cases}
    for sample in samples:
        for result in sample.results:
            by_case[result.case_id].append(result)
    voted = [_aggregate_case(case, by_case[case.id], grader=grader) for case in cases]
    cost = sum(result.cost_usd for result in voted)
    latency = sum(result.latency_s for result in voted) / len(voted)
    result = BaselineResult(
        run=RunResult(
            architecture_id=architecture.id,
            variant="baseline:cot_sc",
            results=voted,
            pass_rate=sum(result.passed for result in voted) / len(voted),
            cost_usd=cost,
            latency_s_mean=latency,
            tokens=sum(result.tokens_in + result.tokens_out for result in voted),
        ),
        k=k,
        matched_to_cost_usd=full_cost_usd,
        samples=tuple(samples),
    )
    if run_dir is not None:
        _write_results(run_dir, voted)
    return result


__all__ = [
    "BASELINE_MODEL",
    "BaselineResult",
    "MAX_SAMPLES",
    "run_baseline",
]
