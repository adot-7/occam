"""Synthetic architectures and result sets.

A deterministic stand-in for the WP-05 executor: it implements exactly the
:class:`occam.engine.ablation.VariantRunner` interface, so the ablation math is
testable — and falsifiable — without any LLM calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from occam.core.models import Architecture, Case, CaseResult, Role, RoleTrace, RunResult
from occam.engine.ablation import FULL, FULL_REPEAT, PairedOutcome


def make_case(case_id: str, **meta: object) -> Case:
    """One evaluation case; ``meta`` feeds the stratified ablation subset."""

    return Case(id=case_id, input=f"ledger for {case_id}", expected={"total": 0.0}, meta=dict(meta))


def make_cases(n: int, prefix: str = "c") -> list[Case]:
    """``n`` plain cases named ``c01``, ``c02``, ..."""

    return [make_case(f"{prefix}{index:02d}") for index in range(1, n + 1)]


def make_role(
    role_id: str,
    *,
    name: str | None = None,
    inputs: Sequence[str] = ("task",),
    justification: str = "control",
) -> Role:
    """A minimal role; only id/name/inputs/justification matter to ablation."""

    return Role(
        id=role_id,
        name=name or role_id.replace("r_", "").replace("_", " ").title(),
        justification=justification,  # type: ignore[arg-type]
        model="worker_fast",
        system_prompt=f"You are {role_id}.",
        tools=[],
        inputs=list(inputs),
        output_key=f"{role_id}_out",
    )


def make_architecture(roles: Sequence[Role], final_role: str | None = None) -> Architecture:
    """Wrap roles into a generation-0 architecture."""

    return Architecture(
        id="g000",
        parent_id=None,
        roles=list(roles),
        final_role=final_role or roles[-1].id,
    )


def make_run_result(
    variant: str,
    outcomes: Iterable[tuple[str, str, bool]],
    *,
    role_costs: Mapping[str, float] | None = None,
    role_tool_calls: Mapping[str, int] | None = None,
    latencies: Mapping[str, float] | None = None,
    architecture_id: str = "g000",
) -> RunResult:
    """Build a ``RunResult`` from ``(case_id, answer, passed)`` triples."""

    costs = dict(role_costs or {})
    calls = dict(role_tool_calls or {})
    results: list[CaseResult] = []
    for case_id, answer, passed in outcomes:
        per_role = {
            role_id: RoleTrace(
                tokens_in=100,
                tokens_out=50,
                cost_usd=cost,
                latency_s=1.0,
                output=f"{role_id}:{case_id}",
                tool_calls=[{"name": "fx_rate"} for _ in range(calls.get(role_id, 0))],
            )
            for role_id, cost in costs.items()
        }
        results.append(
            CaseResult(
                case_id=case_id,
                answer=answer,
                passed=passed,
                tokens_in=100 * len(per_role),
                tokens_out=50 * len(per_role),
                cost_usd=sum(costs.values()),
                latency_s=(latencies or {}).get(case_id, 10.0),
                per_role=per_role,
            )
        )
    passed_n = sum(1 for case in results if case.passed)
    return RunResult(
        architecture_id=architecture_id,
        variant=variant,
        results=results,
        pass_rate=passed_n / len(results) if results else 0.0,
        cost_usd=sum(case.cost_usd for case in results),
        latency_s_mean=(sum(case.latency_s for case in results) / len(results)) if results else 0.0,
        tokens=sum(case.tokens_in + case.tokens_out for case in results),
    )


def pairs_from(full_passed: Sequence[bool], variant_passed: Sequence[bool]) -> list[PairedOutcome]:
    """Paired outcomes with answers that differ exactly where correctness does."""

    return [
        PairedOutcome(
            case_id=f"c{index:02d}",
            full_answer="a",
            variant_answer="a" if left == right else "b",
            full_passed=left,
            variant_passed=right,
        )
        for index, (left, right) in enumerate(zip(full_passed, variant_passed, strict=True))
    ]


@dataclass
class RoleEffect:
    """What knocking out one role does, case by case.

    ``changed`` drives divergence; ``broken`` turns a passing case into a
    failure; ``fixed`` does the reverse (that is how a harmful role looks).
    """

    changed: set[str] = field(default_factory=set)
    broken: set[str] = field(default_factory=set)
    fixed: set[str] = field(default_factory=set)


@dataclass
class SyntheticRunner:
    """A ``VariantRunner`` whose every answer is decided by the test.

    ``passing`` are the cases the full architecture gets right; ``effects`` maps
    role id to its :class:`RoleEffect` (a role with no entry is inert, so its
    knockout is byte-identical to the full run); ``drifting`` are the cases where
    ``full_repeat`` disagrees with ``full`` — the noise floor.
    """

    cases: Sequence[Case]
    passing: set[str]
    effects: Mapping[str, RoleEffect] = field(default_factory=dict)
    drifting: set[str] = field(default_factory=set)
    role_costs: Mapping[str, float] = field(default_factory=dict)
    role_tool_calls: Mapping[str, int] = field(default_factory=dict)
    calls: list[tuple[str, str | None, bool]] = field(default_factory=list)

    def run_variant(
        self,
        architecture: Architecture,
        cases: Sequence[Case],
        *,
        variant: str,
        ablate_role: str | None = None,
        use_cache: bool = True,
    ) -> RunResult:
        """Return the recorded answers for one variant over ``cases``."""

        self.calls.append((variant, ablate_role, use_cache))
        outcomes: list[tuple[str, str, bool]] = []
        for case in cases:
            answer = f"{case.id}:answer"
            passed = case.id in self.passing
            if variant == FULL_REPEAT and case.id in self.drifting:
                answer = f"{case.id}:drift"
                passed = not passed
            elif ablate_role is not None:
                effect = self.effects.get(ablate_role, RoleEffect())
                if case.id in effect.changed:
                    answer = f"{case.id}:without-{ablate_role}"
                if case.id in effect.broken:
                    passed = False
                if case.id in effect.fixed:
                    passed = True
            outcomes.append((case.id, answer, passed))
        return make_run_result(
            variant,
            outcomes,
            role_costs=self.role_costs,
            role_tool_calls=self.role_tool_calls,
            architecture_id=architecture.id,
        )

    def full_result(self, architecture: Architecture) -> RunResult:
        """The generation's full run, as the loop would already have it."""

        return self.run_variant(architecture, self.cases, variant=FULL)
