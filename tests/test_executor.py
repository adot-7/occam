"""Unit tests for the DAG executor (`prd/01-ARCHITECTURE.md` §4.2)."""

from __future__ import annotations

import json
import threading

import pytest

from occam.core.models import Architecture, Case, Role, ToolSpec
from occam.engine.executor import (
    ArchitectureError,
    CycleError,
    Executor,
    ExecutorError,
    descendants,
    normalize_registry,
    sentinel_for,
    topological_levels,
    validate_architecture,
    wilson_ci,
)
from occam.llm.client import CompletionError
from occam.store.reader import EventReader
from occam.store.writer import EventWriter
from tests.executor_doubles import (
    WORKER,
    ProviderCall,
    RecordingTool,
    ScriptedProvider,
    build_client,
    sections,
    text_response,
    tool_call_response,
)

ECHO_SPEC = ToolSpec(
    name="echo",
    description="Return the value you pass in.",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    },
)


def role(role_id: str, **overrides) -> Role:
    payload = {
        "id": role_id,
        "name": role_id.replace("_", " ").title(),
        "justification": "unspecified",
        "model": WORKER,
        "system_prompt": f"ROLE: {role_id}",
        "tools": [],
        "inputs": ["task"],
        "output_key": f"{role_id}_out",
    }
    payload.update(overrides)
    return Role.model_validate(payload)


def architecture(*roles: Role, final: str | None = None, control: str = "deterministic"):
    return Architecture(
        id="g000",
        parent_id=None,
        roles=list(roles),
        final_role=final or roles[-1].id,
        control=control,
    )


def case(case_id: str = "c1", text: str = "Compute the thing.", expected=None) -> Case:
    return Case(id=case_id, input=text, expected=expected)


# -- graph -----------------------------------------------------------------


def test_topological_levels_group_independent_roles():
    arch = architecture(
        role("a"),
        role("b"),
        role("c", inputs=["a", "b"]),
        final="c",
    )
    levels = [[member.id for member in level] for level in topological_levels(arch)]
    assert levels == [["a", "b"], ["c"]]


def test_topological_levels_drops_edges_into_an_excluded_role():
    arch = architecture(role("a"), role("b", inputs=["a"]), role("c", inputs=["b"]), final="c")
    levels = [[member.id for member in level] for level in topological_levels(arch, exclude=["b"])]
    assert levels == [["a", "c"]]


def test_cycle_is_rejected():
    arch = architecture(role("a", inputs=["b"]), role("b", inputs=["a"]), final="b")
    with pytest.raises(CycleError):
        topological_levels(arch)


def test_validate_architecture_rejects_unknown_input():
    arch = architecture(role("a", inputs=["ghost"]))
    with pytest.raises(ArchitectureError, match="unknown input"):
        validate_architecture(arch)


def test_validate_architecture_rejects_unbound_tool():
    arch = architecture(role("a", tools=["fx_rate"]))
    with pytest.raises(ArchitectureError, match="not in the registry"):
        validate_architecture(arch, tools={})


def test_validate_architecture_rejects_duplicate_ids():
    arch = architecture(role("a"), role("a"))
    with pytest.raises(ArchitectureError, match="duplicate role id"):
        validate_architecture(arch)


def test_descendants_are_transitive():
    arch = architecture(
        role("a"),
        role("b", inputs=["a"]),
        role("c", inputs=["b"]),
        role("d", inputs=["a"]),
        final="c",
    )
    assert descendants(arch, "a") == {"b", "c", "d"}
    assert descendants(arch, "b") == {"c"}
    assert descendants(arch, "c") == set()


def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = wilson_ci(8, 10)
    assert lo < 0.8 < hi
    assert wilson_ci(0, 0) == (0.0, 0.0)
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)


def test_normalize_registry_accepts_tuple_and_spec_carrier():
    tool = RecordingTool(ECHO_SPEC, lambda value: value)
    bindings = normalize_registry({"echo": (ECHO_SPEC, tool), "echo2": tool})
    assert bindings["echo"].spec.name == "echo"
    assert bindings["echo2"].spec.name == "echo"


# -- deterministic routing --------------------------------------------------


def test_deterministic_routing_isolates_context(tmp_path):
    """A role sees exactly the output_keys it declared, and nothing else."""

    def handler(call: ProviderCall):
        return text_response(f"output of {call.system}")

    provider = ScriptedProvider(handler)
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    arch = architecture(
        role("upstream"),
        role("sibling"),
        role("downstream", inputs=["upstream"]),
        final="downstream",
    )
    executor.execute(arch, [case()])

    downstream = next(call for call in provider.calls if call.system == "ROLE: downstream")
    body = sections(downstream.user)
    assert set(body) == {"upstream_out"}
    assert body["upstream_out"] == "output of ROLE: upstream"
    # No task text and no sibling output leaked in.
    assert "Compute the thing." not in downstream.user
    assert "sibling" not in downstream.user


def test_system_prompt_placeholder_is_substituted_not_duplicated(tmp_path):
    def handler(call: ProviderCall):
        return text_response("PARSED") if "upstream" in call.system else text_response("done")

    provider = ScriptedProvider(handler)
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    arch = architecture(
        role("upstream"),
        role(
            "downstream",
            inputs=["upstream"],
            system_prompt="ROLE: downstream\nLedger to use: {upstream_out}",
        ),
        final="downstream",
    )
    executor.execute(arch, [case()])

    downstream = next(call for call in provider.calls if "ROLE: downstream" in call.system)
    assert "Ledger to use: PARSED" in downstream.system
    assert downstream.user == "Produce your output now."


def test_control_llm_exposes_every_output(tmp_path):
    def handler(call: ProviderCall):
        return text_response(f"out-{call.system[-1]}")

    provider = ScriptedProvider(handler)
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    arch = architecture(
        role("a"),
        role("b"),
        role("c", inputs=["a"]),
        final="c",
        control="llm",
    )
    executor.execute(arch, [case()])

    router = next(call for call in provider.calls if call.system == "ROLE: c")
    body = sections(router.user)
    assert set(body) == {"task", "a_out", "b_out"}


# -- concurrency ------------------------------------------------------------


def test_independent_roles_run_concurrently(tmp_path):
    barrier = threading.Barrier(2, timeout=10)

    def handler(call: ProviderCall):
        if call.system in {"ROLE: a", "ROLE: b"}:
            barrier.wait()
        return text_response("ok")

    provider = ScriptedProvider(handler)
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    arch = architecture(role("a"), role("b"), role("c", inputs=["a", "b"]), final="c")

    # The barrier only releases if both independent roles are in flight at once.
    executor.execute(arch, [case()])
    assert provider.max_in_flight >= 2


def test_model_concurrency_is_bounded_by_the_lane(tmp_path):
    import time

    def handler(_call: ProviderCall):
        # Long enough that 48 unbounded role runs would pile up together.
        time.sleep(0.01)
        return text_response("ok")

    provider = ScriptedProvider(handler)
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        model_concurrency=2,
        case_concurrency=8,
    )
    arch = architecture(*[role(f"r{index}") for index in range(6)], final="r5")
    cases = [case(f"c{index}", f"case {index}") for index in range(8)]
    executor.execute(arch, cases)
    assert provider.max_in_flight == 2


def test_model_concurrency_derives_from_rpm(tmp_path):
    provider = ScriptedProvider(lambda _call: text_response("ok"))
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    # worker_fast is rpm 60 => one request per second => 4 seconds in flight.
    assert executor.model_concurrency(WORKER) == 4


# -- tool loop --------------------------------------------------------------


def test_tool_loop_records_raw_responses_and_offers_only_bound_tools(tmp_path):
    tool = RecordingTool(ECHO_SPEC, lambda value: {"echoed": value, "extra": "raw-detail"})

    def handler(call: ProviderCall):
        if any(message["role"] == "tool" for message in call.messages):
            return text_response("finished")
        return tool_call_response([("echo", {"value": "hello"})])

    provider = ScriptedProvider(handler)
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        tools={"echo": (ECHO_SPEC, tool)},
    )
    arch = architecture(role("a", tools=["echo"]), role("b", inputs=["a"]), final="b")
    result = executor.execute(arch, [case()])

    trace = result.results[0].per_role["a"]
    assert trace.output == "finished"
    assert len(trace.tool_calls) == 1
    recorded = trace.tool_calls[0]
    assert recorded["name"] == "echo"
    assert recorded["arguments"] == {"value": "hello"}
    # The RAW response is retained verbatim; diagnose reads this.
    assert recorded["response"] == {"echoed": "hello", "extra": "raw-detail"}
    assert recorded["error"] is None

    # Role "b" binds no tools, so none are offered to it.
    assert next(call for call in provider.calls if call.system == "ROLE: b").tools is None
    assert next(call for call in provider.calls if call.system == "ROLE: a").tool_names == ["echo"]


def test_tool_loop_stops_at_max_turns(tmp_path):
    tool = RecordingTool(ECHO_SPEC, lambda value: {"echoed": value})
    provider = ScriptedProvider(
        lambda _call: tool_call_response([("echo", {"value": "again"})], text="still working")
    )
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        tools={"echo": (ECHO_SPEC, tool)},
    )
    arch = architecture(role("a", tools=["echo"], max_turns=3))
    result = executor.execute(arch, [case()])

    trace = result.results[0].per_role["a"]
    assert len(trace.tool_calls) == 3
    assert trace.error is not None and "max_turns (3)" in trace.error
    assert result.results[0].passed is False


def test_unbound_tool_call_is_refused_and_fed_back(tmp_path):
    tool = RecordingTool(ECHO_SPEC, lambda value: {"echoed": value})

    def handler(call: ProviderCall):
        if any(message["role"] == "tool" for message in call.messages):
            return text_response("recovered")
        return tool_call_response([("echo", {"value": "x"})])

    provider = ScriptedProvider(handler)
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        tools={"echo": (ECHO_SPEC, tool)},
    )
    # The role does not bind "echo", so the call must be refused.
    arch = architecture(role("a", tools=[]))
    result = executor.execute(arch, [case()])

    recorded = result.results[0].per_role["a"].tool_calls[0]
    assert "is not bound to role" in recorded["error"]
    assert tool.count == 0
    assert result.results[0].per_role["a"].output == "recovered"


def test_tool_error_is_returned_to_the_model_not_raised(tmp_path):
    def explode(value: str):
        raise ValueError(f"bad value {value}")

    tool = RecordingTool(ECHO_SPEC, explode)

    def handler(call: ProviderCall):
        if any(message["role"] == "tool" for message in call.messages):
            return text_response("apologies, recovered")
        return tool_call_response([("echo", {"value": "boom"})])

    provider = ScriptedProvider(handler)
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        tools={"echo": (ECHO_SPEC, tool)},
    )
    result = executor.execute(architecture(role("a", tools=["echo"])), [case()])

    trace = result.results[0].per_role["a"]
    assert trace.error is None
    assert trace.tool_calls[0]["error"] == "ValueError: bad value boom"
    assert trace.output == "apologies, recovered"


# -- accounting, grading, failure -------------------------------------------


def test_cost_is_accounted_per_role(tmp_path):
    def handler(call: ProviderCall):
        tokens = 100 if call.system == "ROLE: a" else 400
        return text_response("ok", tokens_in=tokens, tokens_out=tokens)

    provider = ScriptedProvider(handler)
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    result = executor.execute(architecture(role("a"), role("b", inputs=["a"]), final="b"), [case()])

    case_result = result.results[0]
    per_role = case_result.per_role
    assert per_role["a"].cost_usd > 0
    assert per_role["b"].cost_usd == pytest.approx(4 * per_role["a"].cost_usd)
    assert case_result.cost_usd == pytest.approx(per_role["a"].cost_usd + per_role["b"].cost_usd)
    assert case_result.tokens_in == 500
    assert result.cost_usd == pytest.approx(case_result.cost_usd)


def test_sub_results_come_from_the_grader(tmp_path):
    provider = ScriptedProvider(lambda _call: text_response("42"))
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        grader=lambda answer, expected: {
            "passed": answer == expected["answer"],
            "sub_results": {"INV-1": True, "INV-2": False},
        },
    )
    result = executor.execute(architecture(role("a")), [case(expected={"answer": "42"})])
    assert result.results[0].passed is True
    assert result.results[0].sub_results == {"INV-1": True, "INV-2": False}
    assert result.pass_rate == 1.0


def test_llm_failure_fails_one_case_and_the_run_continues(tmp_path):
    def handler(call: ProviderCall):
        if "case 1" in call.user:
            raise CompletionError("provider exploded")
        return text_response("fine")

    provider = ScriptedProvider(handler)
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        grader=lambda answer, _expected: {"passed": answer == "fine", "sub_results": {}},
    )
    arch = architecture(role("a"), role("b", inputs=["a"]), final="b")
    cases = [case("c0", "case 0"), case("c1", "case 1"), case("c2", "case 2")]
    result = executor.execute(arch, cases)

    assert [item.case_id for item in result.results] == ["c0", "c1", "c2"]
    failed = result.results[1]
    assert failed.passed is False
    assert "CompletionError" in failed.per_role["a"].error
    # The downstream role never ran: no point spending tokens on garbage.
    assert "b" not in failed.per_role
    assert result.pass_rate == pytest.approx(2 / 3)


def test_a_grader_that_raises_does_not_crash_the_run(tmp_path):
    def grader(_answer, _expected):
        raise RuntimeError("checker bug")

    provider = ScriptedProvider(lambda _call: text_response("ok"))
    executor = Executor(llm=build_client(provider, tmp_path / "cache"), grader=grader)
    result = executor.execute(architecture(role("a")), [case()])
    assert result.results[0].passed is False


def test_execute_refuses_to_nest_in_a_running_loop(tmp_path):
    import asyncio

    provider = ScriptedProvider(lambda _call: text_response("ok"))
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))

    async def main():
        with pytest.raises(ExecutorError, match="running event loop"):
            executor.execute(architecture(role("a")), [case()])

    asyncio.run(main())


# -- knockouts --------------------------------------------------------------


def test_knockout_renders_a_sentinel_and_only_descendants_recompute(tmp_path):
    def handler(call: ProviderCall):
        return text_response(f"out::{call.system}")

    provider = ScriptedProvider(handler)
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    arch = architecture(
        role("a"),
        role("b", inputs=["a"]),
        role("c", inputs=["b"]),
        final="c",
    )
    full = executor.execute(arch, [case()])
    assert full.variant == "full"
    provider.reset()

    knocked = executor.execute(arch, [case()], ablate_role="b")
    assert knocked.variant == "ablate:b"

    traces = knocked.results[0].per_role
    assert "b" not in traces
    assert traces["a"].cached is True and traces["a"].cost_usd == 0.0
    assert traces["c"].cached is False and traces["c"].cost_usd > 0.0

    downstream = next(call for call in provider.calls if call.system == "ROLE: c")
    assert sections(downstream.user)["b_out"] == "[no input from B]"
    assert sentinel_for(next(item for item in arch.roles if item.id == "b")) == "[no input from B]"


def test_ablating_an_unknown_role_is_an_error(tmp_path):
    provider = ScriptedProvider(lambda _call: text_response("ok"))
    executor = Executor(llm=build_client(provider, tmp_path / "cache"))
    with pytest.raises(ArchitectureError, match="cannot ablate unknown role"):
        executor.execute(architecture(role("a")), [case()], ablate_role="zzz")


# -- events and persistence -------------------------------------------------


def test_events_are_emitted_in_case_order_and_validate(tmp_path):
    import time

    def handler(call: ProviderCall):
        # Finish the last case first so completion order != case order.
        if "case 0" in call.user:
            time.sleep(0.05)
        return text_response("ok")

    provider = ScriptedProvider(handler)
    run_dir = tmp_path / "runs" / "r1"
    writer = EventWriter(run_dir, run_id="r1")
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        writer=writer,
        run_dir=run_dir,
        grader=lambda _answer, _expected: {"passed": True, "sub_results": {}},
    )
    cases = [case(f"c{index}", f"case {index}") for index in range(3)]
    executor.execute(architecture(role("a")), cases, generation=2)
    writer.close()

    # EventReader validates every line against schemas/events.schema.json.
    events = EventReader(run_dir).read()
    types = [event.type for event in events]
    assert types == [
        "execution.started",
        "execution.case",
        "execution.case",
        "execution.case",
        "execution.completed",
    ]
    assert [event.seq for event in events] == [0, 1, 2, 3, 4]
    case_events = [event for event in events if event.type == "execution.case"]
    assert [event.data["case_id"] for event in case_events] == ["c0", "c1", "c2"]
    assert events[0].data == {"generation": 2, "variant": "full", "n_cases": 3}
    completed = events[-1].data
    assert completed["pass_rate"] == 1.0
    assert completed["ci"]["lo"] <= 1.0 <= completed["ci"]["hi"] + 1e-9


def test_results_jsonl_keeps_per_role_traces(tmp_path):
    tool = RecordingTool(ECHO_SPEC, lambda value: {"echoed": value, "rate_date": "2026-04-02"})

    def handler(call: ProviderCall):
        if any(message["role"] == "tool" for message in call.messages):
            return text_response("done")
        return tool_call_response([("echo", {"value": "v"})])

    provider = ScriptedProvider(handler)
    run_dir = tmp_path / "runs" / "r1"
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        tools={"echo": (ECHO_SPEC, tool)},
        run_dir=run_dir,
    )
    arch = architecture(role("a", tools=["echo"]))
    executor.execute(arch, [case()], generation=1)
    executor.execute(arch, [case()], generation=1, ablate_role="a")

    full = run_dir / "generations" / "g001" / "results.jsonl"
    payload = json.loads(full.read_text(encoding="utf-8").splitlines()[0])
    assert payload["per_role"]["a"]["tool_calls"][0]["response"]["rate_date"] == "2026-04-02"
    assert (run_dir / "generations" / "g001" / "results.ablate_a.jsonl").exists()
