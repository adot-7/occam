"""WP-15 tracing, import-order, and cache honesty checks."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from occam.llm import tracing
from occam.llm.client import LLMClient
from occam.llm.config import ModelConfig
from occam.llm.providers import ProviderResponse
from occam.tools.fan_out import fan_out
from occam.tools.fx import FXClient


class FakeSpan:
    def __init__(self, name: str, kind: str | None) -> None:
        self.name = name
        self.kind = kind
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value


class FakeTrace:
    def __init__(self, sdk: FakeSDK, name: str, kind: str | None) -> None:
        self.sdk = sdk
        self.span = FakeSpan(name, kind)

    def __enter__(self) -> FakeSpan:
        self.sdk.spans.append(self.span)
        return self.span

    def __exit__(self, *_: Any) -> bool:
        return False


class FakeSDK:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []
        self.lifecycle: list[str] = []

    def trace(self, name: str, *, kind: str | None = None) -> FakeTrace:
        return FakeTrace(self, name, kind)

    def flush(self) -> bool:
        self.lifecycle.append("flush")
        return True

    def shutdown(self) -> None:
        self.lifecycle.append("shutdown")


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    sdk = FakeSDK()
    monkeypatch.setattr(tracing, "neatlogs", sdk)
    monkeypatch.setattr(tracing, "ENABLED", True)
    monkeypatch.setattr(tracing, "_shutdown_complete", False)
    return sdk


def _span(sdk: FakeSDK, name: str) -> FakeSpan:
    return next(item for item in sdk.spans if item.name == name)


def test_no_key_is_a_real_noop_and_does_not_import_neatlogs(tmp_path: Path) -> None:
    script = """
import sys
from occam.llm import tracing

assert not tracing.is_enabled()
assert 'neatlogs' not in sys.modules
assert 'openai' not in sys.modules
with tracing.span('ignored', kind='TOOL') as span:
    assert span is None
from contextvars import ContextVar
from occam.tools.fan_out import fan_out
marker = ContextVar('marker', default=None)
marker.set('parent')
assert fan_out(['one', 'two'], runner=lambda _: marker.get()) == ['parent', 'parent']
assert tracing.flush() is False
assert tracing.shutdown() is False
print('no-key-ok')
"""
    environment = os.environ.copy()
    environment.pop("NEATLOGS_API_KEY", None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "no-key-ok"


def test_no_key_preserves_the_executor_event_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from occam.core.models import Architecture, Case, Role
    from occam.engine.executor import Executor
    from occam.store.reader import EventReader
    from occam.store.writer import EventWriter
    from tests.executor_doubles import build_client, text_response

    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "ENABLED", False)
    monkeypatch.setattr(tracing, "neatlogs", None)
    monkeypatch.setattr(tracing, "_shutdown_complete", False)

    class Provider:
        def complete(self, *_: Any, **__: Any) -> ProviderResponse:
            return text_response("done")

    role = Role(
        id="r_answer",
        name="Answer",
        justification="control",
        model="worker_fast",
        system_prompt="answer",
        tools=[],
        inputs=["task"],
        output_key="answer",
    )
    run_dir = tmp_path / "run"
    writer = EventWriter(run_dir, run_id="run-no-key")
    try:
        Executor(
            llm=build_client(Provider(), tmp_path / "cache"),
            writer=writer,
            run_dir=run_dir,
        ).execute(
            Architecture(id="g000", parent_id=None, roles=[role], final_role=role.id),
            [Case(id="case-1", input="answer", expected=None)],
        )
    finally:
        writer.close()

    events = EventReader(run_dir).read()
    assert [event.type for event in events] == [
        "execution.started",
        "execution.case",
        "execution.completed",
    ]
    assert all("trace" not in event.data for event in events)


def test_neatlogs_init_precedes_the_lazy_openai_import(tmp_path: Path) -> None:
    script = """
import importlib.util
import os
import sys
import types

events = []
sdk = types.ModuleType('neatlogs')
def init(**kwargs):
    events.append(('init', kwargs))
sdk.init = init
sdk.trace = lambda *args, **kwargs: None
sdk.flush = lambda: True
sdk.shutdown = lambda: None
sys.modules['neatlogs'] = sdk

class OpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class OpenAILoader:
    def create_module(self, spec):
        events.append(('openai', None))
        module = types.ModuleType(spec.name)
        module.OpenAI = OpenAI
        return module

    def exec_module(self, module):
        pass


class OpenAIFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'openai':
            return importlib.util.spec_from_loader(fullname, OpenAILoader())
        return None


sys.meta_path.insert(0, OpenAIFinder())

from occam.llm import ModelConfig, OpenAICompatibleProvider

config = ModelConfig(key='worker', provider='openai_compat', model='test-model', api_key='key')
OpenAICompatibleProvider()._client_for(config)
assert events[0][0] == 'init', events
openai_index = next(i for i, item in enumerate(events) if item[0] == 'openai')
assert events.index(('init', events[0][1])) < openai_index
assert events[0][1] == {
    'api_key': os.environ['NEATLOGS_API_KEY'],
    'workflow_name': 'occam',
    'instrumentations': ['openai'],
}
print('order-ok')
"""
    environment = os.environ.copy()
    environment["NEATLOGS_API_KEY"] = "test-neatlogs-key"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "order-ok"


def test_completion_and_uncached_fx_spans_carry_context_and_truthful_cache(
    fake_sdk: FakeSDK, tmp_path: Path
) -> None:
    class Provider:
        def complete(self, *_: Any, **__: Any) -> ProviderResponse:
            return ProviderResponse(text="ok", tokens_in=2, tokens_out=3)

    config = ModelConfig(key="worker", provider="fake", model="fake-model", api_key="key")
    client = LLMClient(
        {"worker": config},
        providers={"fake": Provider()},
        cache_dir=tmp_path / "llm-cache",
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json=(
                {"base": "EUR", "date": "2026-04-02", "rates": {"USD": 1.1}}
                if ".." not in request.url.path
                else {
                    "base": "EUR",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-02",
                    "rates": {
                        "2026-04-01": {"USD": 1.08},
                        "2026-04-02": {"USD": 1.1},
                    },
                }
            ),
            request=request,
        )
    )
    fx = FXClient(cache_dir=tmp_path / "fx-cache", transport=transport)

    context = {
        "run_id": "run-test",
        "generation": 2,
        "variant": "full_repeat",
        "case_id": "case-test",
        "role_id": "r_rates",
        "role_name": "Rate Fetcher",
        "justification": "parallel",
    }
    with tracing.trace_context(context):
        first = client.complete("worker", [{"role": "user", "content": "answer"}])
        second = client.complete("worker", [{"role": "user", "content": "answer"}])
        first_rate = fx.fx_rate("2026-04-04", "EUR", "USD")
        second_rate = fx.fx_rate("2026-04-04", "EUR", "USD")
        series = fx.fx_series("2026-04-01", "2026-04-04", "EUR", "USD")

    assert first.cached is False
    assert second.cached is True
    assert first_rate["rate_date"] == "2026-04-02"
    assert second_rate == first_rate
    assert series["rates"]["2026-04-02"] == 1.1

    completion_spans = [item for item in fake_sdk.spans if item.name == "llm.complete"]
    assert len(completion_spans) == 2
    assert all(item.kind == "CHAIN" for item in completion_spans)
    assert completion_spans[0].attributes["occam.role_id"] == "r_rates"
    assert completion_spans[0].attributes["occam.generation"] == 2
    assert completion_spans[0].attributes["occam.variant"] == "full_repeat"
    assert completion_spans[0].attributes["occam.cached"] is False
    assert completion_spans[1].attributes["occam.cached"] is True

    tool_spans = [item for item in fake_sdk.spans if item.name == "tool.fx_rate"]
    assert len(tool_spans) == 1
    assert tool_spans[0].kind == "TOOL"
    assert tool_spans[0].attributes["occam.role_id"] == "r_rates"
    assert tool_spans[0].attributes["occam.requested_date"] == "2026-04-04"
    assert tool_spans[0].attributes["occam.rate_date"] == "2026-04-02"
    assert tool_spans[0].attributes["occam.cached"] is False
    series_spans = [item for item in fake_sdk.spans if item.name == "tool.fx_series"]
    assert len(series_spans) == 1
    assert series_spans[0].attributes["occam.role_id"] == "r_rates"
    assert series_spans[0].attributes["occam.requested_date"] == "2026-04-01..2026-04-04"
    assert series_spans[0].attributes["occam.rate_date"] == "2026-04-01..2026-04-02"
    assert series_spans[0].attributes["occam.cached"] is False
    assert fx.calls[1].cached is True


def test_fan_out_propagates_context_to_fx_spans_and_restores_parent(
    fake_sdk: FakeSDK, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requested_date = request.url.path.rsplit("/", maxsplit=1)[-1]
        return httpx.Response(
            200,
            json={
                "base": request.url.params["base"],
                "date": requested_date,
                "rates": {request.url.params["symbols"]: 1.1},
            },
            request=request,
        )

    fx = FXClient(
        cache_dir=tmp_path / "fx-cache",
        transport=httpx.MockTransport(handler),
    )
    context = {
        "run_id": "run-fanout",
        "run_name": "fanout-test",
        "generation": 3,
        "variant": "full_repeat",
        "case_id": "case-fanout",
        "role_id": "r_rates",
        "role_name": "Rate Fetcher",
        "justification": "parallel",
        "model": "worker_fast",
    }
    dates = ["2026-04-01", "2026-04-02", "2026-04-03"]

    def runner(requested_date: str) -> str:
        with tracing.trace_context(sibling=requested_date):
            return str(fx.fx_rate(requested_date, "EUR", "USD")["rate_date"])

    with tracing.trace_context(context):
        assert fan_out(dates, runner=runner, max_workers=3) == dates
        fx.fx_rate("2026-04-04", "EUR", "USD")

    tool_spans = [item for item in fake_sdk.spans if item.name == "tool.fx_rate"]
    assert len(tool_spans) == 4
    branch_spans = {
        item.attributes["occam.sibling"]: item
        for item in tool_spans
        if "occam.sibling" in item.attributes
    }
    assert set(branch_spans) == set(dates)
    for requested_date, item in branch_spans.items():
        assert item.attributes["occam.run_id"] == "run-fanout"
        assert item.attributes["occam.run_name"] == "fanout-test"
        assert item.attributes["occam.generation"] == 3
        assert item.attributes["occam.variant"] == "full_repeat"
        assert item.attributes["occam.case_id"] == "case-fanout"
        assert item.attributes["occam.role_id"] == "r_rates"
        assert item.attributes["occam.role_name"] == "Rate Fetcher"
        assert item.attributes["occam.justification"] == "parallel"
        assert item.attributes["occam.model"] == "worker_fast"
        assert item.attributes["occam.requested_date"] == requested_date
        assert item.attributes["occam.rate_date"] == requested_date
        assert item.attributes["occam.cached"] is False

    parent_span = next(item for item in tool_spans if "occam.sibling" not in item.attributes)
    assert parent_span.attributes["occam.requested_date"] == "2026-04-04"
    assert "occam.sibling" not in parent_span.attributes


def test_fan_out_propagates_context_to_complete_spans_and_isolates_errors(
    fake_sdk: FakeSDK, tmp_path: Path
) -> None:
    class Provider:
        def complete(self, *_: Any, **__: Any) -> ProviderResponse:
            return ProviderResponse(text="done", tokens_in=2, tokens_out=3)

    config = ModelConfig(key="worker", provider="fake", model="fake-model", api_key="key")
    client = LLMClient(
        {"worker": config},
        providers={"fake": Provider()},
        cache_dir=tmp_path / "llm-cache",
    )
    context = {
        "run_id": "run-fanout",
        "run_name": "fanout-test",
        "generation": 5,
        "variant": "ablate:r_rates",
        "case_id": "case-fanout",
        "role_id": "r_calc",
        "role_name": "Calculator",
        "justification": "deterministic",
        "model": "worker_fast",
    }
    barrier = threading.Barrier(3, timeout=5.0)

    def runner(sibling: str) -> str:
        with tracing.trace_context(sibling=sibling):
            barrier.wait()
            result = client.complete(
                "worker",
                [{"role": "user", "content": f"branch {sibling}"}],
            )
            if sibling == "bad":
                raise RuntimeError("branch boom")
            return result.text

    with tracing.trace_context(context):
        results = fan_out(["one", "bad", "two"], runner=runner, max_workers=3)
        parent = client.complete("worker", [{"role": "user", "content": "parent"}])

    assert results == ["done", "[fan_out error on subtask 2: branch boom]", "done"]
    assert parent.text == "done"
    completion_spans = [item for item in fake_sdk.spans if item.name == "llm.complete"]
    assert len(completion_spans) == 4
    branches = {item.attributes["occam.sibling"]: item for item in completion_spans[:-1]}
    assert set(branches) == {"one", "bad", "two"}
    for sibling, item in branches.items():
        assert item.attributes["occam.run_id"] == "run-fanout"
        assert item.attributes["occam.run_name"] == "fanout-test"
        assert item.attributes["occam.generation"] == 5
        assert item.attributes["occam.variant"] == "ablate:r_rates"
        assert item.attributes["occam.case_id"] == "case-fanout"
        assert item.attributes["occam.role_id"] == "r_calc"
        assert item.attributes["occam.role_name"] == "Calculator"
        assert item.attributes["occam.justification"] == "deterministic"
        assert item.attributes["occam.model"] == "fake-model"
        assert item.attributes["occam.cached"] is False
        assert item.attributes["occam.sibling"] == sibling

    parent_span = completion_spans[-1]
    assert parent_span.attributes["occam.run_id"] == "run-fanout"
    assert parent_span.attributes["occam.generation"] == 5
    assert "occam.sibling" not in parent_span.attributes


def test_executor_fan_out_keeps_context_on_nested_roles(fake_sdk: FakeSDK, tmp_path: Path) -> None:
    from occam.core.models import Architecture, Case, Role
    from occam.engine.executor import Executor
    from occam.store.writer import EventWriter
    from occam.tools.registry import ToolRegistry
    from tests.executor_doubles import (
        ProviderCall,
        ScriptedProvider,
        build_client,
        text_response,
        tool_call_response,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requested_date = request.url.path.rsplit("/", maxsplit=1)[-1]
        return httpx.Response(
            200,
            json={
                "base": request.url.params["base"],
                "date": requested_date,
                "rates": {request.url.params["symbols"]: 1.1},
            },
            request=request,
        )

    dates = {"one": "2026-04-01", "two": "2026-04-02"}
    branch_barrier = threading.Barrier(2, timeout=5.0)

    def provider_handler(call: ProviderCall) -> ProviderResponse:
        task = call.user.removeprefix("### task\n")
        has_tool_result = any(message["role"] == "tool" for message in call.messages)
        if task == "parent":
            if has_tool_result:
                return ProviderResponse(text="parent done", tokens_in=2, tokens_out=3)
            return tool_call_response([("fan_out", {"subtasks": list(dates)})])
        if task in dates:
            if not has_tool_result:
                branch_barrier.wait()
                return tool_call_response(
                    [
                        (
                            "fx_rate",
                            {"date": dates[task], "base": "EUR", "symbol": "USD"},
                        )
                    ]
                )
            return text_response(f"{task} done")
        raise AssertionError(f"unexpected task: {task!r}")

    registry = ToolRegistry(
        fx_client=FXClient(
            cache_dir=tmp_path / "fx-cache",
            transport=httpx.MockTransport(handler),
        )
    )
    role = Role(
        id="r_parent",
        name="Parent",
        justification="parallel",
        model="worker_fast",
        system_prompt="parent",
        tools=["fan_out", "fx_rate"],
        inputs=["task"],
        output_key="answer",
        max_turns=2,
    )
    run_dir = tmp_path / "run"
    writer = EventWriter(run_dir, run_id="run-nested")
    try:
        result = Executor(
            llm=build_client(ScriptedProvider(provider_handler), tmp_path / "llm-cache"),
            tools=registry,
            writer=writer,
            run_dir=run_dir,
            run_name="nested-test",
            case_concurrency=1,
            model_concurrency=4,
        ).execute(
            Architecture(id="g000", parent_id=None, roles=[role], final_role=role.id),
            [Case(id="case-nested", input="parent", expected=None)],
            generation=7,
            variant="full_repeat",
        )
    finally:
        writer.close()

    assert result.results[0].answer == "parent done"
    spans = [item for item in fake_sdk.spans if item.name in {"llm.complete", "tool.fx_rate"}]
    completion_spans = [item for item in spans if item.name == "llm.complete"]
    tool_spans = [item for item in spans if item.name == "tool.fx_rate"]
    assert len(completion_spans) == 6
    assert len(tool_spans) == 2

    for item in spans:
        assert item.attributes["occam.run_id"] == "run-nested"
        assert item.attributes["occam.run_name"] == "nested-test"
        assert item.attributes["occam.generation"] == 7
        assert item.attributes["occam.variant"] == "full_repeat"
        assert item.attributes["occam.case_id"] == "case-nested"
        assert item.attributes["occam.role_id"] == "r_parent"
        assert item.attributes["occam.role_name"] == "Parent"
        assert item.attributes["occam.justification"] == "parallel"

    assert {item.attributes["occam.requested_date"] for item in tool_spans} == set(dates.values())
    assert all(item.attributes["occam.cached"] is False for item in spans)


def test_executor_propagates_generation_variant_and_role_to_spans(
    fake_sdk: FakeSDK, tmp_path: Path
) -> None:
    from occam.core.models import Architecture, Case, Role
    from occam.engine.executor import Executor
    from tests.executor_doubles import build_client, text_response

    class Provider:
        def complete(self, *_: Any, **__: Any) -> ProviderResponse:
            return text_response("done")

    role = Role(
        id="r_answer",
        name="Answer",
        justification="control",
        model="worker_fast",
        system_prompt="answer",
        tools=[],
        inputs=["task"],
        output_key="answer",
    )
    architecture = Architecture(id="g000", parent_id=None, roles=[role], final_role=role.id)
    executor = Executor(
        llm=build_client(Provider(), tmp_path / "cache"),
        run_name="trace-test",
    )
    result = executor.execute(
        architecture,
        [Case(id="case-1", input="answer", expected=None)],
        generation=4,
        variant="full_repeat",
    )

    assert result.results[0].answer == "done"
    case_span = _span(fake_sdk, "case.case-1")
    assert case_span.kind == "WORKFLOW"
    assert case_span.attributes["occam.generation"] == 4
    assert case_span.attributes["occam.variant"] == "full_repeat"
    assert case_span.attributes["occam.run_name"] == "trace-test"
    completion_span = _span(fake_sdk, "llm.complete")
    assert completion_span.attributes["occam.role_id"] == "r_answer"
    assert completion_span.attributes["occam.role_name"] == "Answer"
    assert completion_span.attributes["occam.generation"] == 4
    assert completion_span.attributes["occam.variant"] == "full_repeat"


def test_shutdown_flushes_before_stopping_exporter(fake_sdk: FakeSDK) -> None:
    with tracing.span("last.case", kind="WORKFLOW"):
        pass

    assert tracing.shutdown() is True
    assert fake_sdk.lifecycle == ["flush", "shutdown"]
    assert tracing.shutdown() is False
    assert fake_sdk.lifecycle == ["flush", "shutdown"]
