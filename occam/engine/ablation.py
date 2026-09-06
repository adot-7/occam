"""Role ablation: noise floor, divergence, paired influence, verdicts, SF.

Implements `prd/03-ABLATION-AND-METRICS.md` §§1-5. The math lives in pure
functions over :class:`RunResult` payloads; :func:`ablate` only orchestrates a
:class:`VariantRunner` (the executor, WP-05) and emits events.

The verdict rule is code, never a prompt (`03 §4.3`), and ``uncertain`` is never
prunable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from occam.core.models import (
    Architecture,
    Case,
    ConfidenceInterval,
    Justification,
    Role,
    RunResult,
    Verdict,
)
from occam.engine.emit import EventSink, null_sink
from occam.metrics.aggregate import role_cost_shares
from occam.metrics.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_mean_ci, clamp, mean

DEFAULT_EPS = 0.05
PRUNABLE_VERDICTS: tuple[Verdict, ...] = ("witness", "harmful")

FULL = "full"
FULL_REPEAT = "full_repeat"

_ROUND = 6


def ablate_variant(role_id: str) -> str:
    """The ``RunResult.variant`` string for role ``role_id``'s knockout."""

    return f"ablate:{role_id}"


def sentinel_for(role: Role) -> str:
    """The marker a knocked-out role's ``output_key`` renders as (`01 §4.3`)."""

    return f"[no input from {role.name}]"


def descendants(architecture: Architecture, role_id: str) -> tuple[str, ...]:
    """Role ids that must recompute when ``role_id`` is knocked out.

    Everything else is a byte-identical cache hit, which is what makes ablation
    affordable (`03 §3.1`). Returned in the architecture's own role order so the
    executor's work is deterministic.
    """

    consumers: dict[str, list[str]] = {role.id: [] for role in architecture.roles}
    for role in architecture.roles:
        for upstream in role.inputs:
            if upstream in consumers:
                consumers[upstream].append(role.id)
    seen: set[str] = set()
    frontier = list(consumers.get(role_id, ()))
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(consumers.get(current, ()))
    return tuple(role.id for role in architecture.roles if role.id in seen)


class VariantRunner(Protocol):
    """The narrow interface ablation needs from the executor (WP-05).

    ``variant`` is ``"full"``, ``"full_repeat"`` or ``f"ablate:{role_id}"`` and
    is echoed back on the :class:`RunResult`. The returned result must carry one
    :class:`~occam.core.models.CaseResult` per requested case, with a normalised
    ``answer``, the grader's ``passed``, and ``per_role`` traces for the roles
    that ran.

    ``ablate_role`` names the knocked-out role: it is removed from the DAG, its
    ``output_key`` renders as :func:`sentinel_for`, its ancestors are cache hits
    and its :func:`descendants` recompute.

    ``use_cache=False`` must bypass the LLM response cache. The noise floor is
    measured by re-running ``full`` as ``full_repeat``; served from cache it
    would report zero noise by construction.
    """

    def run_variant(
        self,
        architecture: Architecture,
        cases: Sequence[Case],
        *,
        variant: str,
        ablate_role: str | None = None,
        use_cache: bool = True,
    ) -> RunResult: ...


@dataclass(frozen=True)
class PairedOutcome:
    """One case seen twice: under the full architecture and under a variant."""

    case_id: str
    full_answer: str
    variant_answer: str
    full_passed: bool
    variant_passed: bool

    @property
    def changed(self) -> bool:
        """Did the answer change at all? (the divergence numerator)"""

        return self.full_answer != self.variant_answer

    @property
    def delta(self) -> float:
        """Paired correctness difference: full minus variant, in {-1, 0, 1}."""

        return float(self.full_passed) - float(self.variant_passed)


class AblationRow(BaseModel):
    """One row of the ablation table — one role's three numbers and verdict."""

    model_config = ConfigDict(extra="forbid")

    role_id: str = Field(min_length=1)
    role_name: str = ""
    justification: Justification = "unspecified"
    influence: float = Field(ge=-1.0, le=1.0)
    influence_ci: ConfidenceInterval
    divergence: float = Field(ge=0.0, le=1.0)
    cost_share: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    n_cases: int = Field(ge=0)

    def event_data(self, generation: int) -> dict[str, Any]:
        """The ``ablation.role`` payload — schema fields only."""

        return {
            "generation": generation,
            "role_id": self.role_id,
            "influence": self.influence,
            "influence_ci": self.influence_ci.model_dump(mode="json"),
            "divergence": self.divergence,
            "cost_share": self.cost_share,
            "verdict": self.verdict,
        }


class AblationTable(BaseModel):
    """The whole table for one generation, plus the noise floor it was judged against."""

    model_config = ConfigDict(extra="forbid")

    generation: int = Field(ge=0)
    case_ids: list[str] = Field(default_factory=list)
    noise_rate: float = Field(ge=0.0, le=1.0)
    eps: float = Field(default=DEFAULT_EPS, ge=0.0, le=1.0)
    rows: list[AblationRow] = Field(default_factory=list)

    @property
    def n_cases(self) -> int:
        """Size of the ablation subset these verdicts rest on."""

        return len(self.case_ids)

    @property
    def structural_fidelity(self) -> float:
        """Fraction of total spend going to load-bearing roles (`03 §5`)."""

        return structural_fidelity(self.rows)

    @property
    def witnesses(self) -> list[str]:
        """Role ids whose removal changed nothing beyond the noise floor."""

        return [row.role_id for row in self.rows if row.verdict == "witness"]

    @property
    def prunable(self) -> list[str]:
        """Role ids the mutation menu may prune — never ``uncertain`` (`03 §4.3`)."""

        return [row.role_id for row in self.rows if row.verdict in PRUNABLE_VERDICTS]

    def row(self, role_id: str) -> AblationRow:
        """Look up one role's row."""

        for row in self.rows:
            if row.role_id == role_id:
                return row
        raise KeyError(role_id)

    def started_event_data(self) -> dict[str, Any]:
        """The ``ablation.started`` payload."""

        return {
            "generation": self.generation,
            "roles": [row.role_id for row in self.rows],
            "n_cases": self.n_cases,
            "case_ids": list(self.case_ids),
        }

    def completed_event_data(self) -> dict[str, Any]:
        """The ``ablation.completed`` payload."""

        return {
            "generation": self.generation,
            "structural_fidelity": round(clamp(self.structural_fidelity), _ROUND),
            "witnesses": self.witnesses,
        }


def _normalise(answer: str) -> str:
    """Collapse incidental whitespace; the executor supplies normalised answers."""

    return " ".join(answer.split())


def _by_case(result: RunResult) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for case in result.results:
        if case.case_id in index:
            raise ValueError(f"duplicate case_id {case.case_id!r} in variant {result.variant!r}")
        index[case.case_id] = case
    return index


def pair_outcomes(
    full: RunResult,
    variant: RunResult,
    case_ids: Sequence[str] | None = None,
) -> list[PairedOutcome]:
    """Align two runs case by case — the pairing the whole method rests on."""

    full_index = _by_case(full)
    variant_index = _by_case(variant)
    ids = list(case_ids) if case_ids is not None else [case.case_id for case in variant.results]
    pairs: list[PairedOutcome] = []
    for case_id in ids:
        if case_id not in full_index:
            raise ValueError(f"case {case_id!r} missing from the full run")
        if case_id not in variant_index:
            raise ValueError(f"case {case_id!r} missing from variant {variant.variant!r}")
        left = full_index[case_id]
        right = variant_index[case_id]
        pairs.append(
            PairedOutcome(
                case_id=case_id,
                full_answer=_normalise(left.answer),
                variant_answer=_normalise(right.answer),
                full_passed=left.passed,
                variant_passed=right.passed,
            )
        )
    return pairs


def divergence(pairs: Sequence[PairedOutcome]) -> float:
    """Fraction of cases where the answer changed at all (`03 §2`)."""

    if not pairs:
        raise ValueError("divergence over an empty case set")
    return sum(1 for pair in pairs if pair.changed) / len(pairs)


def influence(pairs: Sequence[PairedOutcome]) -> float:
    """``pass_rate_full - pass_rate_variant`` as a paired difference (`03 §2`).

    Equivalently ``(#(full passed, variant failed) - #(full failed, variant
    passed)) / n``. Can be negative: that is the point.
    """

    if not pairs:
        raise ValueError("influence over an empty case set")
    return mean([pair.delta for pair in pairs])


def influence_ci(
    pairs: Sequence[PairedOutcome],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    label: str = "",
) -> ConfidenceInterval:
    """95% bootstrap CI over cases for :func:`influence` (`03 §4.2`)."""

    return bootstrap_mean_ci(
        [pair.delta for pair in pairs],
        resamples=resamples,
        seed=seed,
        label=label,
    )


def noise_rate(
    full: RunResult,
    full_repeat: RunResult,
    case_ids: Sequence[str] | None = None,
) -> float:
    """Fraction of cases where two runs of the *same* architecture disagree.

    Measured before ablating (`03 §4.1`), this is the band every divergence is
    judged against, and ``1 - noise_rate`` is the reported reliability.
    """

    pairs = pair_outcomes(full, full_repeat, case_ids)
    if not pairs:
        raise ValueError("noise floor over an empty case set")
    return divergence(pairs)


def verdict(
    role_divergence: float,
    ci: ConfidenceInterval,
    measured_noise_rate: float,
    eps: float = DEFAULT_EPS,
) -> Verdict:
    """The verdict rule from `03 §4.3`, in code.

    Order matters: a role that does not move the answer beyond the noise floor
    is a witness whatever its CI happens to say.
    """

    if role_divergence <= measured_noise_rate + eps:
        return "witness"
    if ci.lo > 0:
        return "load_bearing"
    if ci.hi < 0:
        return "harmful"
    return "uncertain"


def structural_fidelity(rows: Iterable[AblationRow]) -> float:
    """Σ ``cost_share`` over load-bearing roles (`03 §5`)."""

    return clamp(sum(row.cost_share for row in rows if row.verdict == "load_bearing"))


def build_row(
    role: Role,
    pairs: Sequence[PairedOutcome],
    *,
    cost_share: float,
    measured_noise_rate: float,
    eps: float = DEFAULT_EPS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> AblationRow:
    """Compute one role's divergence, influence, CI and verdict."""

    ci = influence_ci(pairs, resamples=resamples, seed=seed, label=role.id)
    # Round before judging so the verdict can never contradict the CI the TUI
    # prints next to it.
    rounded_ci = ConfidenceInterval(
        lo=round(clamp(ci.lo, -1.0, 1.0), _ROUND),
        hi=round(clamp(ci.hi, -1.0, 1.0), _ROUND),
    )
    role_divergence = round(clamp(divergence(pairs)), _ROUND)
    return AblationRow(
        role_id=role.id,
        role_name=role.name,
        justification=role.justification,
        influence=round(clamp(influence(pairs), -1.0, 1.0), _ROUND),
        influence_ci=rounded_ci,
        divergence=role_divergence,
        cost_share=round(clamp(cost_share), _ROUND),
        verdict=verdict(role_divergence, rounded_ci, measured_noise_rate, eps),
        n_cases=len(pairs),
    )


def build_table(
    architecture: Architecture,
    *,
    generation: int,
    case_ids: Sequence[str],
    full: RunResult,
    knockouts: Mapping[str, RunResult],
    measured_noise_rate: float,
    eps: float = DEFAULT_EPS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> AblationTable:
    """Assemble the ablation table from already-executed variants.

    Pure: no runner, no I/O. ``full`` is the generation's full run (its per-role
    traces set ``cost_share``); ``knockouts`` maps role id to that role's
    knockout run over ``case_ids``. Roles absent from ``knockouts`` are skipped,
    which is how "skip unchanged roles" (`03 §3.4`) reaches this layer.
    """

    ids = list(case_ids)
    if not ids:
        raise ValueError("ablation needs at least one case")
    roles = [role for role in architecture.roles if role.id in knockouts]
    shares = role_cost_shares(full, [role.id for role in architecture.roles])
    rows = [
        build_row(
            role,
            pair_outcomes(full, knockouts[role.id], ids),
            cost_share=shares.get(role.id, 0.0),
            measured_noise_rate=measured_noise_rate,
            eps=eps,
            resamples=resamples,
            seed=seed,
        )
        for role in roles
    ]
    return AblationTable(
        generation=generation,
        case_ids=ids,
        noise_rate=round(clamp(measured_noise_rate), _ROUND),
        eps=eps,
        rows=rows,
    )


def emit_table(sink: EventSink, table: AblationTable) -> AblationTable:
    """Emit ``ablation.started``, one ``ablation.role`` per row, then ``ablation.completed``."""

    sink("ablation.started", table.started_event_data())
    for row in table.rows:
        sink("ablation.role", row.event_data(table.generation))
    sink("ablation.completed", table.completed_event_data())
    return table


def ablate(
    architecture: Architecture,
    cases: Sequence[Case],
    *,
    runner: VariantRunner,
    generation: int,
    full: RunResult | None = None,
    roles: Sequence[str] | None = None,
    eps: float = DEFAULT_EPS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    sink: EventSink | None = None,
) -> AblationTable:
    """Measure the noise floor, knock out each role, emit the table.

    ``cases`` is the ablation subset; ``full`` is the generation's full run over
    all eval cases (re-run here only if the caller has none). Events are emitted
    as the run progresses, so the TUI fills the table row by row.
    """

    if not cases:
        raise ValueError("ablation needs at least one case")
    emit = sink or null_sink
    case_ids = [case.id for case in cases]

    requested = None if roles is None else set(roles)
    if requested is not None:
        unknown = sorted(requested - {role.id for role in architecture.roles})
        if unknown:
            raise ValueError(f"unknown role ids: {', '.join(unknown)}")
    targets = [role.id for role in architecture.roles if requested is None or role.id in requested]

    # Noise floor first (`03 §4.1`): every divergence below is judged against it.
    if full is None:
        full = runner.run_variant(architecture, cases, variant=FULL)
    repeat = runner.run_variant(architecture, cases, variant=FULL_REPEAT, use_cache=False)
    measured_noise_rate = noise_rate(full, repeat, case_ids)

    emit(
        "ablation.started",
        {
            "generation": generation,
            "roles": targets,
            "n_cases": len(case_ids),
            "case_ids": list(case_ids),
        },
    )

    shares = role_cost_shares(full, [role.id for role in architecture.roles])
    rows: list[AblationRow] = []
    for role in architecture.roles:
        if role.id not in targets:
            continue
        knockout = runner.run_variant(
            architecture,
            cases,
            variant=ablate_variant(role.id),
            ablate_role=role.id,
        )
        row = build_row(
            role,
            pair_outcomes(full, knockout, case_ids),
            cost_share=shares.get(role.id, 0.0),
            measured_noise_rate=measured_noise_rate,
            eps=eps,
            resamples=resamples,
            seed=seed,
        )
        rows.append(row)
        emit("ablation.role", row.event_data(generation))

    table = AblationTable(
        generation=generation,
        case_ids=case_ids,
        noise_rate=round(clamp(measured_noise_rate), _ROUND),
        eps=eps,
        rows=rows,
    )
    emit("ablation.completed", table.completed_event_data())
    return table


def case_stratum(case: Case) -> str:
    """Stratum label for the ablation subset: the case's truthy boolean meta flags."""

    flags = sorted(key for key, value in case.meta.items() if isinstance(value, bool) and value)
    return "+".join(flags) if flags else "plain"


def select_ablation_subset(cases: Sequence[Case], n: int) -> list[Case]:
    """Pick ``n`` of the eval cases, stratified by ``meta`` (`03 §3.2`).

    Deterministic by construction — strata in first-appearance order, cases in
    pack order within a stratum, round-robin across strata — so the subset is
    reproducible without a seed and the TUI's "ablated on n/M" is stable.
    """

    if n < 0:
        raise ValueError("n must be >= 0")
    if n >= len(cases):
        return list(cases)
    strata: dict[str, list[Case]] = {}
    for case in cases:
        strata.setdefault(case_stratum(case), []).append(case)
    picked: list[Case] = []
    depth = 0
    while len(picked) < n:
        added = False
        for bucket in strata.values():
            if depth < len(bucket):
                picked.append(bucket[depth])
                added = True
                if len(picked) == n:
                    break
        if not added:
            break
        depth += 1
    order = {case.id: index for index, case in enumerate(cases)}
    return sorted(picked, key=lambda case: order[case.id])


__all__ = [
    "DEFAULT_EPS",
    "FULL",
    "FULL_REPEAT",
    "PRUNABLE_VERDICTS",
    "AblationRow",
    "AblationTable",
    "PairedOutcome",
    "VariantRunner",
    "ablate",
    "ablate_variant",
    "build_row",
    "build_table",
    "case_stratum",
    "descendants",
    "divergence",
    "emit_table",
    "influence",
    "influence_ci",
    "noise_rate",
    "pair_outcomes",
    "select_ablation_subset",
    "sentinel_for",
    "structural_fidelity",
    "verdict",
]
