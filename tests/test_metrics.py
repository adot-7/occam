"""Tests for the metrics strip (`prd/03-ABLATION-AND-METRICS.md §7`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from occam.core.models import CaseResult, RoleTrace, RunResult
from occam.engine.emit import collecting_sink, null_sink
from occam.metrics import (
    BaselineSummary,
    build_snapshot,
    compare_to_baseline,
    emit_snapshot,
    latency_stats,
    pass_rate,
    pass_rate_ci,
    percentile,
    role_cost_shares,
    snapshot_event_data,
    tool_calls_per_case,
    total_cost_usd,
    total_tokens,
    wilson_interval,
)
from tests._synthetic import make_run_result

ROOT = Path(__file__).resolve().parents[1]
EVENT_SCHEMA = json.loads((ROOT / "schemas" / "events.schema.json").read_text(encoding="utf-8"))
EVENT_VALIDATOR = Draft202012Validator(EVENT_SCHEMA, format_checker=FormatChecker())


def _run(
    outcomes: list[tuple[str, str, bool]],
    *,
    latencies: dict[str, float] | None = None,
) -> RunResult:
    return make_run_result(
        "full",
        outcomes,
        role_costs={"r_a": 0.02, "r_b": 0.08},
        role_tool_calls={"r_a": 3, "r_b": 1},
        latencies=latencies,
    )


def test_wilson_interval_brackets_the_point_estimate() -> None:
    interval = wilson_interval(11, 20)
    assert interval.lo < 0.55 < interval.hi
    assert (round(interval.lo, 3), round(interval.hi, 3)) == (0.342, 0.742)
    # Degenerate ends stay inside [0, 1] instead of running off the scale.
    assert wilson_interval(0, 10).lo == 0.0
    assert wilson_interval(10, 10).hi == pytest.approx(1.0)
    assert wilson_interval(0, 0) == wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_percentile_interpolates_between_order_statistics() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile([7.0], 0.9) == 7.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_run_level_aggregates_match_the_definitions() -> None:
    run = _run(
        [("c1", "a", True), ("c2", "b", False), ("c3", "c", True), ("c4", "d", True)],
        latencies={"c1": 10.0, "c2": 20.0, "c3": 30.0, "c4": 40.0},
    )
    assert pass_rate(run) == pytest.approx(0.75)
    assert pass_rate_ci(run).lo < 0.75 < pass_rate_ci(run).hi
    assert total_cost_usd(run) == pytest.approx(0.4)  # 4 cases x $0.10
    assert total_tokens(run) == 4 * (200 + 100)
    assert tool_calls_per_case(run) == pytest.approx(4.0)
    latency = latency_stats(run)
    assert (latency.mean, latency.p50, latency.p90) == pytest.approx((25.0, 25.0, 37.0))


def test_cost_share_is_the_role_share_of_total_spend() -> None:
    run = _run([("c1", "a", True), ("c2", "b", True)])
    shares = role_cost_shares(run, ["r_a", "r_b"])
    assert shares == {"r_a": pytest.approx(0.2), "r_b": pytest.approx(0.8)}
    assert sum(shares.values()) == pytest.approx(1.0)
    # A role the executor never ran still appears, at zero.
    assert role_cost_shares(run, ["r_a", "r_b", "r_c"])["r_c"] == 0.0


def test_cost_share_can_be_restricted_to_the_exact_ablation_subset() -> None:
    run = make_run_result(
        "full",
        [("c1", "a", True), ("c2", "b", True), ("c3", "c", True)],
        role_costs={"r_a": 1.0, "r_b": 1.0},
    )
    for trace in run.results[0].per_role.values():
        trace.cost_usd = 1.0
    for trace in run.results[1].per_role.values():
        trace.cost_usd = 3.0
    for trace in run.results[2].per_role.values():
        trace.cost_usd = 100.0

    assert role_cost_shares(run, ["r_a", "r_b"], case_ids=["c1", "c2"]) == {
        "r_a": pytest.approx(0.5),
        "r_b": pytest.approx(0.5),
    }
    with pytest.raises(ValueError, match="missing"):
        role_cost_shares(run, ["r_a", "r_b"], case_ids=["missing"])


def test_cost_share_requires_positive_displayed_role_trace_costs() -> None:
    granted = RunResult(
        architecture_id="g000",
        variant="full",
        results=[
            CaseResult(
                case_id="c1",
                passed=True,
                per_role={
                    "r_a": RoleTrace(
                        tokens_in=300,
                        tokens_out=100,
                        cost_usd=0.02,
                        billed_cost_usd=0.0,
                        cost_label="list-rate-equivalent",
                    ),
                    "r_b": RoleTrace(
                        tokens_in=100,
                        tokens_out=100,
                        cost_usd=0.01,
                        billed_cost_usd=0.0,
                        cost_label="list-rate-equivalent",
                    ),
                },
            )
        ],
    )
    assert role_cost_shares(granted, ["r_a", "r_b"]) == {
        "r_a": pytest.approx(2 / 3),
        "r_b": pytest.approx(1 / 3),
    }
    unpriced = RunResult(
        architecture_id="g000",
        variant="full",
        results=[
            CaseResult(
                case_id="c1",
                passed=True,
                per_role={
                    "r_a": RoleTrace(tokens_in=300, tokens_out=100),
                    "r_b": RoleTrace(tokens_in=100, tokens_out=100),
                },
            )
        ],
    )
    with pytest.raises(ValueError, match="undefined.*invariant violation"):
        role_cost_shares(unpriced, ["r_a", "r_b"])
    assert role_cost_shares(unpriced, []) == {}


def test_cost_shares_reject_duplicate_roles_or_result_case_ids() -> None:
    run = _run([("c1", "a", True)])
    with pytest.raises(ValueError, match="duplicate role_id"):
        role_cost_shares(run, ["r_a", "r_a"])

    duplicate_cases = make_run_result(
        "full",
        [("c1", "a", True), ("c1", "b", False)],
        role_costs={"r_a": 1.0},
    )
    with pytest.raises(ValueError, match="duplicate case_id"):
        role_cost_shares(duplicate_cases, ["r_a"])


def test_snapshot_reports_every_field_in_the_spec() -> None:
    run = _run(
        [("c1", "a", True), ("c2", "b", False), ("c3", "c", True), ("c4", "d", True)],
        latencies={"c1": 10.0, "c2": 20.0, "c3": 30.0, "c4": 40.0},
    )
    snapshot = build_snapshot(
        generation=3,
        full=run,
        noise_rate=0.1,
        structural_fidelity=0.87,
        baseline=BaselineSummary(pass_rate=0.5, cost_usd=0.5),
        reliability_pass3=0.6,
    )

    assert snapshot.generation == 3
    assert snapshot.pass_rate == pytest.approx(0.75)
    assert snapshot.ci == pass_rate_ci(run)
    assert snapshot.cost_usd == pytest.approx(0.4)
    assert snapshot.latency_s_mean == pytest.approx(25.0)
    assert (snapshot.latency_s_p50, snapshot.latency_s_p90) == pytest.approx((25.0, 37.0))
    assert snapshot.tokens == 1200
    assert snapshot.tool_calls_per_case == pytest.approx(4.0)
    assert snapshot.reliability == pytest.approx(0.9)  # 1 - noise_rate
    assert snapshot.reliability_pass3 == pytest.approx(0.6)
    assert snapshot.speed == pytest.approx(1 / 25.0)
    assert snapshot.structural_fidelity == pytest.approx(0.87)
    assert snapshot.vs_baseline.pass_delta == pytest.approx(0.25)
    assert snapshot.vs_baseline.cost_ratio == pytest.approx(0.8)


def test_baseline_comparison_handles_a_missing_or_free_baseline() -> None:
    assert compare_to_baseline(0.6, 0.4, None).pass_delta == 0.0
    assert compare_to_baseline(0.6, 0.4, None).cost_ratio == 1.0
    free = compare_to_baseline(0.6, 0.4, BaselineSummary(pass_rate=0.2, cost_usd=0.0))
    assert (free.pass_delta, free.cost_ratio) == pytest.approx((0.4, 0.0))


def test_snapshot_of_an_empty_run_is_defined_and_schema_valid() -> None:
    empty = RunResult(architecture_id="g000", variant="full")
    snapshot = build_snapshot(generation=0, full=empty, noise_rate=0.0, structural_fidelity=0.0)
    assert snapshot.pass_rate == 0.0
    assert snapshot.speed == 0.0
    assert snapshot.ci == wilson_interval(0, 0)
    EVENT_VALIDATOR.validate(
        {
            "ts": "2026-09-06T09:00:00Z",
            "run_id": "t",
            "seq": 0,
            "type": "metrics.snapshot",
            "data": snapshot_event_data(snapshot),
        }
    )


def test_emit_snapshot_produces_a_schema_valid_event() -> None:
    collected: list[dict] = []
    run = _run([("c1", "a", True), ("c2", "b", False)])
    snapshot = build_snapshot(
        generation=1,
        full=run,
        noise_rate=0.2,
        structural_fidelity=0.5,
        baseline=BaselineSummary(pass_rate=0.4, cost_usd=0.25),
    )

    emit_snapshot(collecting_sink(collected), snapshot)
    emit_snapshot(null_sink, snapshot)

    assert [event["type"] for event in collected] == ["metrics.snapshot"]
    EVENT_VALIDATOR.validate(
        {"ts": "2026-09-06T09:00:00Z", "run_id": "t", "seq": 0, **collected[0]}
    )
    assert collected[0]["data"]["reliability_pass3"] is None
