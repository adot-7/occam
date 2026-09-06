"""Run-level aggregation over the executor's :class:`RunResult` payloads.

Everything here is a pure function of a ``RunResult``: the engine computes,
the TUI reads events. Definitions follow `03 §7`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from occam.core.models import CaseResult, ConfidenceInterval, RunResult
from occam.metrics.stats import percentile, wilson_interval


class LatencyStats(BaseModel):
    """Wall-clock per case: mean plus the two reported quantiles."""

    model_config = ConfigDict(extra="forbid")

    mean: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p90: float = Field(ge=0.0)


def passed_count(result: RunResult) -> int:
    """Number of cases the grader marked as passing."""

    return sum(1 for case in result.results if case.passed)


def pass_rate(result: RunResult) -> float:
    """passed / n over the cases actually present in ``result``."""

    if not result.results:
        return 0.0
    return passed_count(result) / len(result.results)


def pass_rate_ci(result: RunResult) -> ConfidenceInterval:
    """Wilson 95% interval around :func:`pass_rate`."""

    return wilson_interval(passed_count(result), len(result.results))


def total_cost_usd(result: RunResult) -> float:
    """Total spend; cache hits are recorded at $0 by the executor."""

    return math.fsum(case.cost_usd for case in result.results)


def total_tokens(result: RunResult) -> int:
    """Tokens in + out across every case."""

    return sum(case.tokens_in + case.tokens_out for case in result.results)


def latency_stats(result: RunResult) -> LatencyStats:
    """Mean, p50 and p90 of per-case wall-clock latency."""

    latencies = [case.latency_s for case in result.results]
    if not latencies:
        return LatencyStats(mean=0.0, p50=0.0, p90=0.0)
    return LatencyStats(
        mean=math.fsum(latencies) / len(latencies),
        p50=percentile(latencies, 0.5),
        p90=percentile(latencies, 0.9),
    )


def tool_calls_per_case(result: RunResult) -> float:
    """Mean number of tool calls across all roles of a case."""

    if not result.results:
        return 0.0
    calls = sum(
        len(trace.tool_calls) for case in result.results for trace in case.per_role.values()
    )
    return calls / len(result.results)


def _case_results(result: RunResult, case_ids: Sequence[str] | None) -> Sequence[CaseResult]:
    """Return the requested cases, rejecting an accidentally incomplete pairing."""

    by_id = {case.case_id: case for case in result.results}
    if len(by_id) != len(result.results):
        raise ValueError(f"duplicate case_id in run variant {result.variant!r}")
    if case_ids is None:
        return result.results
    requested = list(case_ids)
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(
            f"case(s) missing from run variant {result.variant!r}: {', '.join(missing)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate case_id in requested cost-share subset")
    return [by_id[case_id] for case_id in requested]


def role_cost_totals(
    result: RunResult,
    case_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-role cost summed over cases, from the run's per-role traces.

    ``case_ids`` is the exact ablation subset when computing structural
    fidelity; omitting it retains the all-cases aggregation for callers that
    want a generation-wide accounting summary.
    """

    totals: dict[str, float] = {}
    for case in _case_results(result, case_ids):
        for role_id, trace in case.per_role.items():
            totals[role_id] = totals.get(role_id, 0.0) + trace.cost_usd
    return totals


def role_token_totals(result: RunResult) -> dict[str, int]:
    """Per-role tokens (in + out) summed over cases."""

    totals: dict[str, int] = {}
    for case in result.results:
        for role_id, trace in case.per_role.items():
            totals[role_id] = totals.get(role_id, 0) + trace.tokens_in + trace.tokens_out
    return totals


def role_cost_shares(
    result: RunResult,
    role_ids: Sequence[str] | None = None,
    *,
    case_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """``cost_share(r)`` for every role: that role's share of total spend.

    Read from the FULL run's per-role ``RoleTrace.cost_usd`` values (`03 §2`).
    The executor populates that field with the displayed cost, including the
    list-rate equivalent for granted models.  A zero denominator is an
    accounting failure, not permission to invent token or uniform shares: an
    ablation table with fabricated spend would make structural fidelity
    meaningless.
    """

    keys = (
        list(role_ids)
        if role_ids is not None
        else sorted(role_cost_totals(result, case_ids=case_ids))
    )
    if not keys:
        return {}
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate role_id in requested cost-share roles")
    costs = role_cost_totals(result, case_ids=case_ids)
    weights: dict[str, float] = {key: float(costs.get(key, 0.0)) for key in keys}
    total = math.fsum(weights.values())
    if total <= 0.0:
        raise ValueError(
            "cannot compute role cost shares: the full run has no positive "
            "displayed RoleTrace.cost_usd"
        )
    return {key: weights[key] / total for key in keys}


__all__ = [
    "LatencyStats",
    "latency_stats",
    "pass_rate",
    "pass_rate_ci",
    "passed_count",
    "role_cost_shares",
    "role_cost_totals",
    "role_token_totals",
    "tool_calls_per_case",
    "total_cost_usd",
    "total_tokens",
]
