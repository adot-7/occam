"""WP-05 acceptance: a hand-written 3-role FX team over 5 cases, end to end.

The tool is a local stub with WP-04's response shape (`02 §2`); WP-04's real
registry was not on ``main`` when this was written, so the executor is wired
through :func:`occam.engine.executor.normalize_registry`, which accepts the
``name -> (ToolSpec, callable)`` mapping WP-04 will expose.

Two live variants are opt-in because a default test run must stay hermetic:

* ``OCCAM_LIVE_HTTP=1`` swaps the stub for real Frankfurter HTTP (no key).
* ``OCCAM_LIVE_LLM=1`` (plus ``TENSORMUX_API_KEY``) swaps the scripted agent
  for the real ``worker_fast`` lane.
"""

from __future__ import annotations

import json
import os

import pytest

from occam.core.models import Architecture, Case, Role
from occam.engine.executor import Executor
from occam.store.reader import EventReader
from occam.store.writer import EventWriter
from tests.executor_doubles import (
    CALCULATOR_MARK,
    FETCHER_MARK,
    FX_CASES,
    FX_RATE_SPEC,
    PARSER_MARK,
    WORKER,
    RecordingTool,
    ScriptedProvider,
    build_client,
    fx_agent_script,
    grade_fx_total,
    offline_fx_rate,
    sections,
)

PARSER = Role(
    id="r_parse",
    name="Ledger Parser",
    justification="context_isolation",
    model=WORKER,
    system_prompt=(
        f"{PARSER_MARK}\n"
        "Read the ledger and emit JSON: "
        '{"valuation_date": "...", "invoices": [...]}. Emit JSON only.'
    ),
    tools=[],
    inputs=["task"],
    output_key="parsed_ledger",
    max_turns=2,
)

FETCHER = Role(
    id="r_rates",
    name="Rate Fetcher",
    justification="parallel",
    model=WORKER,
    system_prompt=(
        f"{FETCHER_MARK}\n"
        "Look up every rate the ledger needs with fx_rate, then emit a JSON "
        'object mapping "CCY@date" to the tool response. Emit JSON only.'
    ),
    tools=["fx_rate"],
    inputs=["r_parse"],
    output_key="fx_rates",
    max_turns=4,
)

CALCULATOR = Role(
    id="r_calc",
    name="FX Calculator",
    justification="control",
    model=WORKER,
    system_prompt=(
        f"{CALCULATOR_MARK}\n"
        "Compute the FX gain/(loss) per invoice and the total in INR. End with "
        'a fenced JSON block {"total_inr": <number>, "per_invoice": {...}}.'
    ),
    tools=[],
    inputs=["r_parse", "r_rates"],
    output_key="revaluation",
    max_turns=2,
)

FX_ARCHITECTURE = Architecture(
    id="g000",
    parent_id=None,
    roles=[PARSER, FETCHER, CALCULATOR],
    final_role="r_calc",
    control="deterministic",
    notes="Hand-written WP-05 fixture: parse, fetch in parallel, then compute.",
)


def fx_cases(tool: RecordingTool) -> list[Case]:
    """Build the 5 cases, referencing them against the live rate source.

    The reference answer is computed with the same tool the team calls, so the
    live-HTTP variant grades against real ECB rates rather than stub ones.
    """

    def rate(currency: str, requested: str) -> float:
        return float(tool(date=requested, base=currency, symbol="INR")["rate"])

    cases = [
        Case(
            id=item.case_id,
            input=item.input,
            expected=item.expected_for(rate),
            meta=dict(item.meta),
        )
        for item in FX_CASES
    ]
    tool.reset()
    return cases


def live_fx_rate(date: str, base: str, symbol: str) -> dict[str, object]:
    """Real Frankfurter HTTP, in the shape WP-04's tool returns (02 §2)."""

    import httpx

    response = httpx.get(
        f"https://api.frankfurter.dev/v1/{date}",
        params={"base": base, "symbols": symbol},
        timeout=30.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "requested_date": date,
        "rate_date": payload["date"],
        "base": payload["base"],
        "symbol": symbol,
        "rate": payload["rates"][symbol],
    }


@pytest.fixture
def fx_tool() -> RecordingTool:
    implementation = live_fx_rate if os.environ.get("OCCAM_LIVE_HTTP") == "1" else offline_fx_rate
    return RecordingTool(FX_RATE_SPEC, implementation)


@pytest.fixture
def fx_executor(tmp_path, fx_tool):
    """A fresh executor whose LLM cache survives across runs within one test."""

    if os.environ.get("OCCAM_LIVE_LLM") == "1":
        from occam.llm.client import LLMClient

        client = LLMClient(cache_dir=tmp_path / "cache")
        provider = None
    else:
        provider = ScriptedProvider(fx_agent_script)
        client = build_client(provider, tmp_path / "cache")
    run_dir = tmp_path / "runs" / "wp05"
    writer = EventWriter(run_dir, run_id="wp05")
    executor = Executor(
        llm=client,
        tools={"fx_rate": (FX_RATE_SPEC, fx_tool)},
        grader=grade_fx_total,
        writer=writer,
        run_dir=run_dir,
    )
    yield executor, provider, run_dir
    writer.close()


def test_three_role_team_runs_five_cases_then_serves_the_second_run_from_cache(
    fx_executor, fx_tool
):
    executor, provider, run_dir = fx_executor
    cases = fx_cases(fx_tool)

    first = executor.execute(FX_ARCHITECTURE, cases)

    assert first.variant == "full"
    assert [item.case_id for item in first.results] == [item.id for item in cases]
    assert first.pass_rate == 1.0, [item.answer for item in first.results if not item.passed]
    assert first.cost_usd > 0.0
    assert fx_tool.count > 0
    if provider is not None:
        assert provider.count > 0
    # Every role ran on every case, and every role has its own accounting.
    for result in first.results:
        assert set(result.per_role) == {"r_parse", "r_rates", "r_calc"}
        assert result.cost_usd == pytest.approx(
            sum(trace.cost_usd for trace in result.per_role.values())
        )
        assert all(trace.cached is False for trace in result.per_role.values())
        assert result.sub_results and all(result.sub_results.values())

    # The raw tool responses survive: this is what lets diagnose see that a
    # requested date resolved to an earlier rate date (02 §1.3, L1).
    holiday = next(item for item in first.results if item.case_id == "fxs_002")
    resolutions = [
        (call["arguments"]["date"], call["response"]["rate_date"])
        for call in holiday.per_role["r_rates"].tool_calls
    ]
    assert any(requested != resolved for requested, resolved in resolutions), resolutions

    if provider is not None:
        provider.reset()
    fx_tool.reset()

    second = executor.execute(FX_ARCHITECTURE, cases, generation=1)

    assert second.pass_rate == 1.0
    assert second.cost_usd == 0.0
    for result in second.results:
        assert all(trace.cached is True for trace in result.per_role.values())
    if provider is not None:
        assert provider.count == 0, "second run must be served entirely from the LLM cache"

    events = EventReader(run_dir).read()
    assert [event.type for event in events].count("execution.case") == 10
    completed = [event for event in events if event.type == "execution.completed"]
    assert [event.data["variant"] for event in completed] == ["full", "full"]
    assert completed[0].data["cost_usd"] > 0.0
    assert completed[1].data["cost_usd"] == 0.0
    assert (run_dir / "generations" / "g001" / "results.jsonl").exists()


def test_knocking_out_role_two_caches_role_one_and_recomputes_role_three(fx_executor, fx_tool):
    executor, provider, run_dir = fx_executor
    cases = fx_cases(fx_tool)

    full = executor.execute(FX_ARCHITECTURE, cases)
    assert full.pass_rate == 1.0
    if provider is not None:
        provider.reset()

    knocked = executor.execute(FX_ARCHITECTURE, cases, ablate_role="r_rates")

    assert knocked.variant == "ablate:r_rates"
    for result in knocked.results:
        traces = result.per_role
        assert "r_rates" not in traces, "the ablated role is removed from the DAG"
        # Role 1 is upstream: byte-identical prompt, so a cache hit at $0.
        assert traces["r_parse"].cached is True
        assert traces["r_parse"].cost_usd == 0.0
        # Role 3 is a descendant: it sees the sentinel and recomputes.
        assert traces["r_calc"].cached is False
        assert traces["r_calc"].cost_usd > 0.0

    assert knocked.cost_usd > 0.0
    # Divergence: without rates, the answers change and the cases fail.
    assert knocked.pass_rate == 0.0
    full_answers = {item.case_id: item.answer for item in full.results}
    assert all(item.answer != full_answers[item.case_id] for item in knocked.results)

    if provider is not None:
        calculator_calls = [call for call in provider.calls if CALCULATOR_MARK in call.system]
        assert len(calculator_calls) == len(cases)
        assert all(
            sections(call.user)["fx_rates"] == "[no input from Rate Fetcher]"
            for call in calculator_calls
        )
        assert not [call for call in provider.calls if FETCHER_MARK in call.system]

    ablation_results = run_dir / "generations" / "g000" / "results.ablate_r_rates.jsonl"
    payload = [
        json.loads(line) for line in ablation_results.read_text(encoding="utf-8").splitlines()
    ]
    assert len(payload) == len(cases)
    assert "r_rates" not in payload[0]["per_role"]

    events = EventReader(run_dir).read()
    variants = [event.data["variant"] for event in events if event.type == "execution.started"]
    assert variants == ["full", "ablate:r_rates"]
