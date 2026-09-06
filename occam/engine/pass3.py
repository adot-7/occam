"""Final-generation three-pass reliability measurement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from occam.core.models import Architecture, Case, RunResult
from occam.engine.executor import Executor


@dataclass(frozen=True)
class Pass3Result:
    """Three independent result sets and the reported reliability."""

    runs: tuple[RunResult, ...]
    reliable_cases: int
    n_cases: int

    @property
    def reliability_pass3(self) -> float:
        return self.reliable_cases / self.n_cases if self.n_cases else 0.0

    @property
    def displayed_cost_usd(self) -> float:
        return sum(run.cost_usd for run in self.runs)

    @property
    def billed_cost_usd(self) -> float:
        return sum(
            trace.billed_cost_usd
            for run in self.runs
            for case in run.results
            for trace in case.per_role.values()
        )

    def event_data(self, generation: int) -> dict[str, Any]:
        return {
            "generation": generation,
            "method": "pass3",
            "reliable_cases": self.reliable_cases,
            "n_cases": self.n_cases,
            "reliability_pass3": self.reliability_pass3,
        }


def run_pass3(
    architecture: Architecture,
    cases: list[Case],
    *,
    executor: Executor,
    generation: int,
    run_dir: str | Path | None = None,
    grader: Any = None,
) -> Pass3Result:
    """Run every final-generation case three times with the cache bypassed."""

    if not cases:
        return Pass3Result(runs=(), reliable_cases=0, n_cases=0)
    runner = executor.__class__(
        llm=executor.llm,
        tools=executor.tools,
        grader=grader if grader is not None else executor.grader,
        writer=None,
        run_dir=run_dir,
        run_name=executor.run_name,
        temperature=0.0,
        max_tokens=executor.max_tokens,
        case_concurrency=executor.case_concurrency,
        model_concurrency=executor._model_concurrency,
    )
    runs: list[RunResult] = []
    for pass_number in range(1, 4):
        runs.append(
            runner.run_variant(
                architecture,
                cases,
                variant=f"pass3:{pass_number}",
                use_cache=False,
                generation=generation,
                grader=grader,
            )
        )
    by_case = {case.id: [run.results[index] for run in runs] for index, case in enumerate(cases)}
    reliable = sum(all(result.passed for result in results) for results in by_case.values())
    return Pass3Result(
        runs=tuple(runs),
        reliable_cases=reliable,
        n_cases=len(cases),
    )


__all__ = ["Pass3Result", "run_pass3"]
