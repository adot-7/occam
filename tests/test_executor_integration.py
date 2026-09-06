"""Integration coverage for the executor and the real WP-04 tool registry."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from occam.core.models import Architecture, Case, Role
from occam.engine.executor import ArchitectureError, Executor
from occam.store.reader import EventReader
from occam.store.writer import EventWriter
from occam.tools.fx import FXClient
from occam.tools.registry import ToolRegistry
from tests.executor_doubles import (
    WORKER,
    ProviderCall,
    ScriptedProvider,
    build_client,
    offline_fx_rate,
    text_response,
    tool_call_response,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CACHE = ROOT / "data" / "fx_cache"


def role(role_id: str, **overrides: Any) -> Role:
    payload: dict[str, Any] = {
        "id": role_id,
        "name": role_id.replace("_", " ").title(),
        "justification": "unspecified",
        "model": WORKER,
        "system_prompt": f"ROLE: {role_id}",
        "tools": [],
        "inputs": ["task"],
        "output_key": f"{role_id}_out",
        "max_turns": 2,
    }
    payload.update(overrides)
    return Role.model_validate(payload)


def architecture(*roles: Role, final: str | None = None) -> Architecture:
    return Architecture(
        id="g000",
        parent_id=None,
        roles=list(roles),
        final_role=final or roles[-1].id,
    )


def case(text: str = "lookup") -> Case:
    return Case(id="c1", input=text, expected=None)


class FXTransport(httpx.BaseTransport):
    """A real FXClient transport with deterministic ECB-shaped responses."""

    def __init__(self, handler: Any = None) -> None:
        self.requests: list[str] = []
        self.handler = handler or self._response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        return self.handler(request)

    @staticmethod
    def _response(request: httpx.Request) -> httpx.Response:
        requested_date = request.url.path.rsplit("/", maxsplit=1)[-1]
        base = request.url.params["base"]
        symbol = request.url.params["symbols"]
        rate = offline_fx_rate(requested_date, base, symbol)
        payload = {
            "amount": 1.0,
            "base": base,
            "date": rate["rate_date"],
            "rates": {symbol: rate["rate"]},
        }
        return httpx.Response(200, json=payload, request=request)


def fx_role_architecture() -> Architecture:
    return architecture(role("fx", tools=["fx_rate"]))


def test_executor_accepts_real_bindings_and_keeps_authoritative_accounting(tmp_path: Path) -> None:
    transport = FXTransport()
    client = FXClient(cache_dir=tmp_path / "fx_cache", transport=transport)
    registry = ToolRegistry(fx_client=client)

    def provider_handler(call: ProviderCall):
        if any(message["role"] == "tool" for message in call.messages):
            return text_response("done")
        return tool_call_response(
            [("fx_rate", {"date": "2026-04-04", "base": "EUR", "symbol": "INR"})]
        )

    provider = ScriptedProvider(provider_handler)
    executor = Executor(
        llm=build_client(provider, tmp_path / "llm_cache"),
        # Exercise the exact object returned by ToolRegistry.bindings(), not a
        # tuple-shaped test double.
        tools=registry.bindings(),
    )

    first = executor.execute(fx_role_architecture(), [case()])
    record = first.results[0].per_role["fx"].tool_calls[0]

    assert record == {
        "name": "fx_rate",
        "arguments": {"date": "2026-04-04", "base": "EUR", "symbol": "INR"},
        "status": "ok",
        "latency_s": record["latency_s"],
        "bytes": record["bytes"],
        "cached": False,
        "response": {
            "requested_date": "2026-04-04",
            "rate_date": "2026-04-02",
            "base": "EUR",
            "symbol": "INR",
            "rate": record["response"]["rate"],
        },
        "http_status": 200,
        "error": None,
    }
    assert record["latency_s"] >= 0.0
    assert record["bytes"] > 0
    assert len(transport.requests) == 1

    # A fresh LLM cache makes the role call the real FX binding again, while
    # the shared FXClient must answer from its now-warm disk cache.
    second_provider = ScriptedProvider(provider_handler)
    second = Executor(
        llm=build_client(second_provider, tmp_path / "llm_cache_2"),
        tools=registry,
    ).execute(fx_role_architecture(), [case()])
    cached_record = second.results[0].per_role["fx"].tool_calls[0]
    assert cached_record["cached"] is True
    assert cached_record["status"] == "ok"
    assert cached_record["http_status"] == 200
    assert cached_record["response"]["rate_date"] == "2026-04-02"
    assert len(transport.requests) == 1


def test_executor_reads_the_committed_fx_cache_through_the_real_registry(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(fx_client=FXClient(cache_dir=COMMITTED_CACHE))

    def provider_handler(call: ProviderCall):
        if any(message["role"] == "tool" for message in call.messages):
            return text_response("done")
        return tool_call_response(
            [("fx_rate", {"date": "2026-04-04", "base": "EUR", "symbol": "INR"})]
        )

    result = Executor(
        llm=build_client(
            ScriptedProvider(provider_handler),
            tmp_path / "llm_cache",
        ),
        tools=registry,
    ).execute(fx_role_architecture(), [case()])
    record = result.results[0].per_role["fx"].tool_calls[0]

    assert record["cached"] is True
    assert record["status"] == "ok"
    assert record["http_status"] == 200
    assert record["response"]["rate_date"] == "2026-04-02"


def test_executor_persists_registry_failure_fields_and_continues(tmp_path: Path) -> None:
    def failure_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "temporary"}, request=request)

    registry = ToolRegistry(
        fx_client=FXClient(
            cache_dir=tmp_path / "fx_cache",
            transport=FXTransport(failure_response),
        )
    )

    def provider_handler(call: ProviderCall):
        if any(message["role"] == "tool" for message in call.messages):
            return text_response("recovered")
        return tool_call_response(
            [("fx_rate", {"date": "2026-04-04", "base": "EUR", "symbol": "INR"})]
        )

    result = Executor(
        llm=build_client(ScriptedProvider(provider_handler), tmp_path / "llm_cache"),
        tools=registry,
    ).execute(fx_role_architecture(), [case()])
    trace = result.results[0].per_role["fx"]
    record = trace.tool_calls[0]

    assert trace.output == "recovered"
    assert trace.error is None
    assert record["status"] == "error"
    assert record["arguments"] == {
        "date": "2026-04-04",
        "base": "EUR",
        "symbol": "INR",
    }
    assert record["latency_s"] >= 0.0
    assert record["bytes"] == 0
    assert record["cached"] is False
    assert record["http_status"] == 503
    assert record["response"] is None
    assert "HTTPStatusError" in record["error"]


def test_independent_roles_offload_sync_tools_and_reach_transport_together(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2, timeout=5.0)

    def concurrent_response(request: httpx.Request) -> httpx.Response:
        barrier.wait()
        requested_date = request.url.path.rsplit("/", maxsplit=1)[-1]
        base = request.url.params["base"]
        symbol = request.url.params["symbols"]
        rate = offline_fx_rate(requested_date, base, symbol)
        return httpx.Response(
            200,
            json={
                "amount": 1.0,
                "base": base,
                "date": rate["rate_date"],
                "rates": {symbol: rate["rate"]},
            },
            request=request,
        )

    registry = ToolRegistry(
        fx_client=FXClient(
            cache_dir=tmp_path / "fx_cache",
            transport=FXTransport(concurrent_response),
            max_concurrency=2,
        )
    )

    def provider_handler(call: ProviderCall):
        if call.system == "ROLE: a":
            request = {"date": "2026-04-01", "base": "EUR", "symbol": "INR"}
        elif call.system == "ROLE: b":
            request = {"date": "2026-04-02", "base": "USD", "symbol": "INR"}
        else:  # pragma: no cover - catches an unexpected prompt
            raise AssertionError(call.system)
        if any(message["role"] == "tool" for message in call.messages):
            return text_response(f"{call.system} done")
        return tool_call_response([("fx_rate", request)])

    arch = architecture(
        role("a", tools=["fx_rate"]),
        role("b", tools=["fx_rate"]),
        final="b",
    )
    result = Executor(
        llm=build_client(ScriptedProvider(provider_handler), tmp_path / "llm_cache"),
        tools=registry,
        case_concurrency=1,
        model_concurrency=4,
    ).execute(arch, [case()])

    assert set(result.results[0].per_role) == {"a", "b"}
    assert len(registry.fx_client.calls) == 2
    assert all(
        trace.tool_calls[0]["status"] == "ok" for trace in result.results[0].per_role.values()
    )


def test_fan_out_uses_the_calling_role_runner_and_scope(tmp_path: Path) -> None:
    registry = ToolRegistry(fx_client=FXClient(cache_dir=tmp_path / "fx_cache"))

    def provider_handler(call: ProviderCall):
        if "### task\nparent" in call.user:
            if any(message["role"] == "tool" for message in call.messages):
                return text_response("parent complete")
            return tool_call_response([("fan_out", {"subtasks": ["one", "two"]})])
        return text_response(f"branch {call.user.splitlines()[-1]}")

    provider = ScriptedProvider(provider_handler)
    result = Executor(
        llm=build_client(provider, tmp_path / "llm_cache"),
        tools=registry.bindings(),
        case_concurrency=1,
        model_concurrency=4,
    ).execute(architecture(role("parent", tools=["fan_out"])), [case("parent")])

    trace = result.results[0].per_role["parent"]
    record = trace.tool_calls[0]
    assert trace.output == "parent complete"
    assert record["name"] == "fan_out"
    assert record["status"] == "ok"
    assert record["response"] == ["branch one", "branch two"]
    assert record["arguments"] == {"subtasks": ["one", "two"]}
    assert {call.user for call in provider.calls if "### task\n" in call.user} >= {
        "### task\none",
        "### task\ntwo",
    }


def test_fan_out_aggregates_nested_accounting_once_and_isolates_siblings(
    tmp_path: Path,
) -> None:
    transport = FXTransport()
    registry = ToolRegistry(
        fx_client=FXClient(cache_dir=tmp_path / "fx_cache", transport=transport)
    )
    branch_requests = {
        "one": {"date": "2026-04-01", "base": "EUR", "symbol": "INR"},
        "two": {"date": "2026-04-02", "base": "USD", "symbol": "INR"},
    }

    def provider_handler(call: ProviderCall):
        task = call.user.removeprefix("### task\n")
        has_tool_result = any(message["role"] == "tool" for message in call.messages)
        if call.system == "ROLE: sibling":
            return text_response("sibling complete", tokens_in=23, tokens_out=24)
        if task == "parent":
            if has_tool_result:
                return text_response("parent complete", tokens_in=11, tokens_out=12)
            return tool_call_response(
                [("fan_out", {"subtasks": ["one", "two"]})],
                tokens_in=7,
                tokens_out=8,
            )
        if task in branch_requests:
            if has_tool_result:
                return text_response(f"branch {task}", tokens_in=15, tokens_out=16)
            return tool_call_response(
                [("fx_rate", branch_requests[task])],
                tokens_in=13,
                tokens_out=14,
            )
        raise AssertionError(f"unexpected task: {task!r}")

    provider = ScriptedProvider(provider_handler)
    result = Executor(
        llm=build_client(provider, tmp_path / "llm_cache"),
        # Use the real registry publication, including the real fan_out runner.
        tools=registry.bindings(),
        case_concurrency=1,
        model_concurrency=4,
    ).execute(
        architecture(
            role("parent", tools=["fan_out", "fx_rate"]),
            role("sibling"),
            final="parent",
        ),
        [case("parent")],
    )

    parent = result.results[0].per_role["parent"]
    sibling = result.results[0].per_role["sibling"]
    # Parent: two own completions plus two completions per nested branch.
    assert parent.tokens_in == 74
    assert parent.tokens_out == 80
    assert parent.cost_usd == pytest.approx((74 * 0.06 + 80 * 0.40) / 1_000_000)
    assert parent.billed_cost_usd == 0.0
    assert parent.cost_label == "list-rate-equivalent"
    assert parent.cached is False
    assert parent.latency_s > 0.0

    # The outer call is retained once and each branch's authoritative FX call
    # is flattened into the same RoleTrace once, without a lossy duplicate log.
    assert len(parent.tool_calls) == 3
    assert sum(call["name"] == "fan_out" for call in parent.tool_calls) == 1
    nested_fx_calls = [call for call in parent.tool_calls if call["name"] == "fx_rate"]
    assert len(nested_fx_calls) == 2
    assert sorted(call["arguments"]["date"] for call in nested_fx_calls) == [
        "2026-04-01",
        "2026-04-02",
    ]
    assert all(call["status"] == "ok" for call in nested_fx_calls)
    assert all(
        call["response"]["requested_date"] == call["arguments"]["date"] for call in nested_fx_calls
    )
    assert len(transport.requests) == 2
    assert sum("### task\none" in call.user for call in provider.calls) == 2
    assert sum("### task\ntwo" in call.user for call in provider.calls) == 2

    # A same-level sibling retains only its own completion and accounting.
    assert sibling.tokens_in == 23
    assert sibling.tokens_out == 24
    assert sibling.tool_calls == []
    assert sibling.cost_usd == pytest.approx((23 * 0.06 + 24 * 0.40) / 1_000_000)
    assert sibling.billed_cost_usd == 0.0
    assert sibling.cost_label == "list-rate-equivalent"


def test_duplicate_output_keys_fail_before_provider_or_completion_event(tmp_path: Path) -> None:
    provider = ScriptedProvider(lambda _call: text_response("unreachable"))
    run_dir = tmp_path / "run"
    writer = EventWriter(run_dir, run_id="run")
    executor = Executor(
        llm=build_client(provider, tmp_path / "llm_cache"),
        tools=ToolRegistry(),
        writer=writer,
        run_dir=run_dir,
    )
    arch = architecture(
        role("a", output_key="shared"),
        role("b", output_key="shared"),
    )

    with pytest.raises(ArchitectureError, match="duplicate output_key"):
        executor.execute(arch, [case()])
    writer.close()

    assert provider.count == 0
    events_path = run_dir / "events.jsonl"
    assert not events_path.exists() or not events_path.read_text(encoding="utf-8")


class ObservingWriter:
    """Observe the result file at the instant completion is requested."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.inner = EventWriter(run_dir, run_id="run")
        self.completed_result: str | None = None

    def append(self, event: Any) -> Any:
        if event.get("type") == "execution.completed":
            path = self.run_dir / "generations" / "g000" / "results.jsonl"
            self.completed_result = path.read_text(encoding="utf-8") if path.exists() else None
        return self.inner.append(event)

    def close(self) -> None:
        self.inner.close()


def test_variant_results_are_durable_before_execution_completed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = ObservingWriter(run_dir)
    executor = Executor(
        llm=build_client(
            ScriptedProvider(lambda _call: text_response("done")),
            tmp_path / "llm_cache",
        ),
        tools=ToolRegistry(),
        writer=writer,
        run_dir=run_dir,
    )

    executor.execute(architecture(role("a")), [case()])
    writer.close()

    assert writer.completed_result is not None
    payload = json.loads(writer.completed_result.splitlines()[0])
    assert payload["case_id"] == "c1"
    events = EventReader(run_dir).read()
    assert events[-1].type == "execution.completed"
