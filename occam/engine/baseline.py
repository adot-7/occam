"""Cost-matched chain-of-thought self-consistency baseline."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from occam.core.models import Architecture, Case, CaseResult, Role, RunResult, Task
from occam.engine.executor import Executor

MAX_SAMPLES = 9
BASELINE_MODEL = "worker_fast"
# The baseline is a comparator, not a reason to leave the run unbounded.  Its
# lane gets one provider attempt with an SDK request timeout; a case and the
# whole cost-matching phase have independent upper bounds as well.
BASELINE_PHASE_TIMEOUT_S = 10 * 60.0
BASELINE_CASE_TIMEOUT_S = 3 * 60.0
BASELINE_COMPLETION_TIMEOUT_S = 30.0
BASELINE_COMPLETION_MAX_ATTEMPTS = 1
_CASE_TIMEOUT_PREFIX = "TimeoutError: execution case exceeded "
_PHASE_TIMEOUT_PREFIX = "TimeoutError: execution case exceeded phase deadline"


@dataclass(frozen=True)
class BaselineResult:
    """The voted baseline result and the event fields shown by the TUI."""

    run: RunResult
    k: int
    matched_to_cost_usd: float
    samples: tuple[RunResult, ...] = ()
    complete: bool = True
    status: str = "completed"
    reason: str | None = None
    completed_case_count: int | None = None
    total_case_count: int | None = None

    @property
    def pass_rate(self) -> float:
        return self.run.pass_rate

    @property
    def cost_usd(self) -> float:
        if self.complete:
            return self.run.cost_usd
        return sum(sample.cost_usd for sample in self.samples)

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
        if not self.complete:
            raise ValueError("an incomplete baseline has no completion event")
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
        truncated=chosen.truncated,
        per_role=chosen.per_role,
    )


def _write_results(run_dir: str | Path, results: list[CaseResult]) -> None:
    destination = Path(run_dir) / "baseline" / "results.jsonl"
    _write_jsonl(destination, results)


def _write_jsonl(destination: Path, results: Sequence[CaseResult]) -> None:
    """Atomically persist a baseline result snapshot."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for result in results:
                handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_progress_results(
    run_dir: str | Path,
    sample_number: int,
    results: Mapping[str, CaseResult],
) -> None:
    """Persist the cases observed so far for one baseline sample."""

    destination = (
        Path(run_dir) / "baseline" / "progress" / f"sample-{sample_number:03d}.results.jsonl"
    )
    _write_jsonl(destination, sorted(results.values(), key=lambda result: result.case_id))


def _is_case_timeout(result: CaseResult) -> bool:
    error = result.role_error or ""
    return error.startswith(_CASE_TIMEOUT_PREFIX) and not error.startswith(_PHASE_TIMEOUT_PREFIX)


def _is_phase_timeout(result: CaseResult) -> bool:
    return (result.role_error or "").startswith(_PHASE_TIMEOUT_PREFIX)


def _is_timeout(result: CaseResult) -> bool:
    return _is_case_timeout(result) or _is_phase_timeout(result)


def _billed_cost(result: CaseResult) -> float:
    return sum(trace.billed_cost_usd for trace in result.per_role.values())


def _positive_limit(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


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
    phase_timeout_s: float = BASELINE_PHASE_TIMEOUT_S,
    case_timeout_s: float = BASELINE_CASE_TIMEOUT_S,
    completion_timeout_s: float = BASELINE_COMPLETION_TIMEOUT_S,
    completion_max_attempts: int = BASELINE_COMPLETION_MAX_ATTEMPTS,
    event_sink: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> BaselineResult:
    """Run CoT-SC with ``k`` samples matched to the current full-run cost.

    Sampling deliberately bypasses the content cache: cache hits would turn
    self-consistency into repeated copies of one answer and would under-report
    the baseline's spend.
    """

    if not cases:
        raise ValueError("baseline needs at least one case")
    phase_timeout_s = _positive_limit("phase_timeout_s", phase_timeout_s)
    case_timeout_s = _positive_limit("case_timeout_s", case_timeout_s)
    completion_timeout_s = _positive_limit("completion_timeout_s", completion_timeout_s)
    if isinstance(completion_max_attempts, bool) or completion_max_attempts < 1:
        raise ValueError("completion_max_attempts must be at least 1")
    architecture = _baseline_architecture(task, model_key=model_key)
    samples: list[RunResult] = []
    phase_started = time.monotonic()
    phase_deadline = phase_started + phase_timeout_s
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
        case_timeout_s=case_timeout_s,
        completion_timeout_s=completion_timeout_s,
        completion_max_attempts=completion_max_attempts,
        phase_deadline_s=phase_deadline,
    )

    def incomplete(reason: str, observed: Mapping[str, CaseResult]) -> BaselineResult:
        sample = (
            samples[-1]
            if samples
            else RunResult(
                architecture_id=architecture.id,
                variant="baseline:cot_sc",
                results=[],
            )
        )
        completed = sum(not _is_timeout(result) for result in observed.values())
        return BaselineResult(
            run=sample,
            k=len(samples),
            matched_to_cost_usd=full_cost_usd,
            samples=tuple(samples),
            complete=False,
            status="incomplete",
            reason=reason,
            completed_case_count=completed,
            total_case_count=len(cases),
        )

    k = 1
    sample_number = 1
    observed: dict[str, CaseResult] = {}
    while sample_number <= k:
        remaining = phase_deadline - time.monotonic()
        if remaining <= 0:
            return incomplete("phase_timeout", observed)
        # The case timeout is also the sample deadline because all cases are
        # scheduled together; queued model-lane work must not keep the phase
        # alive indefinitely.
        first.case_timeout_s = min(case_timeout_s, remaining)
        observed = {}

        def on_case(
            result: CaseResult,
            number: int = sample_number,
            sample_observed: dict[str, CaseResult] = observed,
        ) -> None:
            sample_observed[result.case_id] = result
            if run_dir is not None:
                _write_progress_results(run_dir, number, sample_observed)
            if event_sink is not None:
                completed = sum(not _is_timeout(item) for item in sample_observed.values())
                timed_out = sum(_is_timeout(item) for item in sample_observed.values())
                displayed = sum(sample.cost_usd for sample in samples) + sum(
                    item.cost_usd for item in sample_observed.values()
                )
                billed = sum(
                    _billed_cost(sample_result)
                    for sample in samples
                    for sample_result in sample.results
                ) + sum(_billed_cost(item) for item in sample_observed.values())
                event_sink(
                    "log",
                    {
                        "level": "info",
                        "message": (
                            f"Baseline progress: sample={number}; "
                            f"completed_cases={completed}/{len(cases)}; "
                            f"timed_out_cases={timed_out}; "
                            f"displayed_cost_usd={displayed:.8f}; "
                            f"billed_cost_usd={billed:.8f}"
                        ),
                    },
                )

        sample = first.run_variant(
            architecture,
            cases,
            variant=f"baseline:cot_sc:{sample_number}",
            use_cache=False,
            generation=generation,
            grader=grader,
            case_callback=on_case,
        )
        samples.append(sample)
        if any(_is_phase_timeout(result) for result in sample.results):
            return incomplete("phase_timeout", observed)
        if any(_is_case_timeout(result) for result in sample.results):
            return incomplete("case_timeout", observed)
        # The executor bounds queued and active cases, but it can return after
        # the deadline while finalising a sample.  Such a sample is observed
        # progress, not a completed baseline sample.
        if time.monotonic() >= phase_deadline:
            return incomplete("phase_timeout", observed)
        if sample_number == 1:
            k = _k_for_cost(full_cost_usd, sample.cost_usd)
        sample_number += 1

    by_case: dict[str, list[CaseResult]] = {case.id: [] for case in cases}
    for sample in samples:
        for result in sample.results:
            by_case[result.case_id].append(result)
    voted = [_aggregate_case(case, by_case[case.id], grader=grader) for case in cases]
    if time.monotonic() >= phase_deadline:
        return incomplete("phase_timeout", observed)
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
    "BASELINE_CASE_TIMEOUT_S",
    "BASELINE_COMPLETION_MAX_ATTEMPTS",
    "BASELINE_COMPLETION_TIMEOUT_S",
    "BASELINE_PHASE_TIMEOUT_S",
    "BaselineResult",
    "MAX_SAMPLES",
    "run_baseline",
]
