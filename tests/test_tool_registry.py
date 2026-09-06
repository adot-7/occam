"""The tool registry: plain descriptions, the note hook, and per-call accounting."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from occam.core.models import ToolSpec
from occam.tools.accounting import STATUS_ERROR, STATUS_OK, ToolCallLog
from occam.tools.fan_out import FanOutUnavailableError
from occam.tools.fx import FXClient
from occam.tools.registry import (
    BASE_DESCRIPTIONS,
    TOOL_NAMES,
    TOOL_NOTE_HEADER,
    ToolRegistry,
    UnknownToolError,
    append_tool_note,
    build_spec,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CACHE = ROOT / "data" / "fx_cache"

RESPONSES = {
    "/v1/2026-04-04": {
        "amount": 1.0,
        "base": "EUR",
        "date": "2026-04-02",
        "rates": {"USD": 1.1525},
    },
    "/v1/2026-03-02..2026-03-06": {
        "amount": 1.0,
        "base": "EUR",
        "start_date": "2026-03-02",
        "end_date": "2026-03-06",
        "rates": {
            "2026-03-02": {"USD": 1.1500},
            "2026-03-03": {"USD": 1.1510},
            "2026-03-04": {"USD": 1.1520},
            "2026-03-05": {"USD": 1.1530},
            "2026-03-06": {"USD": 1.1540},
        },
    },
}


class CountingTransport(httpx.BaseTransport):
    """Serves the recorded Frankfurter payloads and counts network round trips."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        payload = RESPONSES[request.url.path]
        return httpx.Response(200, json=payload, request=request)


@pytest.fixture
def offline_registry(tmp_path: Path) -> ToolRegistry:
    transport = CountingTransport()
    client = FXClient(cache_dir=tmp_path / "fx_cache", transport=transport)
    registry = ToolRegistry(fx_client=client)
    registry.transport = transport  # type: ignore[attr-defined]
    return registry


# -- registry shape ----------------------------------------------------


def test_registry_publishes_the_four_built_in_tools() -> None:
    registry = ToolRegistry()

    assert registry.names == TOOL_NAMES
    assert set(registry.names) == {"fx_rate", "fx_series", "python_exec", "fan_out"}


def test_registry_maps_each_name_to_a_spec_and_a_callable() -> None:
    bindings = ToolRegistry().bindings()

    for name, binding in bindings.items():
        assert isinstance(binding.spec, ToolSpec)
        assert binding.spec.name == name
        assert callable(binding.call)


def test_parameter_schemas_match_the_tool_signatures() -> None:
    required = {name: build_spec(name).parameters["required"] for name in TOOL_NAMES}

    assert required["fx_rate"] == ["date", "base", "symbol"]
    assert required["fx_series"] == ["start", "end", "base", "symbol"]
    assert required["python_exec"] == ["code"]
    assert required["fan_out"] == ["subtasks"]


def test_specs_are_independent_copies() -> None:
    first = build_spec("fx_rate")
    first.parameters["properties"]["date"]["description"] = "tampered"

    assert build_spec("fx_rate").parameters["properties"]["date"]["description"] != "tampered"


def test_asking_for_an_unpublished_tool_is_an_error() -> None:
    with pytest.raises(UnknownToolError):
        ToolRegistry().spec("web_search")


# -- plain base descriptions -------------------------------------------

FORBIDDEN_IN_BASE_DESCRIPTIONS = (
    "weekend",
    "saturday",
    "sunday",
    "holiday",
    "good friday",
    "easter",
    "business day",
    "working day",
    "ecb",
    "one call",
    "single call",
    "range",
    "batch",
    "last available",
    "previous day",
    "nearest",
    "fall back",
    "fallback",
)


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_base_descriptions_teach_nothing_the_system_must_learn(name: str) -> None:
    # The lessons in 02 §1.3 are the product. Hardcoding them here defeats it.
    spec = build_spec(name)
    lowered = f"{spec.description} {json.dumps(spec.parameters)}".lower()

    for banned in FORBIDDEN_IN_BASE_DESCRIPTIONS:
        assert banned not in lowered, f"{name} description leaks {banned!r}"


def test_base_descriptions_stay_one_plain_sentence() -> None:
    for description in BASE_DESCRIPTIONS.values():
        assert description
        assert TOOL_NOTE_HEADER not in description
        assert "\n" not in description


def test_python_exec_description_names_its_restricted_surface() -> None:
    description = build_spec("python_exec").description

    assert "restricted calculation subset" in description
    assert "not arbitrary Python" in description
    assert "unsupported constructs" in description


# -- the tool_note append hook -----------------------------------------


def test_append_tool_note_adds_a_note_under_a_header() -> None:
    updated = append_tool_note("Get the exchange rate.", "The response date is authoritative.")

    assert updated.startswith("Get the exchange rate.")
    assert TOOL_NOTE_HEADER in updated
    assert "- The response date is authoritative." in updated


def test_appending_the_same_note_twice_is_a_no_op() -> None:
    once = append_tool_note("Get the exchange rate.", "Trust the response date.")

    assert append_tool_note(once, "Trust the response date.") == once


def test_a_second_note_joins_the_existing_header() -> None:
    text = append_tool_note("Get the exchange rate.", "First.")
    text = append_tool_note(text, "Second.")

    assert text.count(TOOL_NOTE_HEADER) == 1
    assert text.endswith("- First.\n- Second.")


def test_notes_appear_in_the_spec_the_architect_hands_to_a_role() -> None:
    registry = ToolRegistry()
    assert TOOL_NOTE_HEADER not in registry.spec("fx_rate").description

    registry.add_tool_note("fx_rate", "The response date is the rate date.")

    assert "- The response date is the rate date." in registry.spec("fx_rate").description
    assert registry.tool_notes("fx_rate") == ("The response date is the rate date.",)
    # Other tools and the module-level base description are untouched.
    assert TOOL_NOTE_HEADER not in registry.spec("fx_series").description
    assert TOOL_NOTE_HEADER not in build_spec("fx_rate").description


def test_clearing_notes_restores_the_plain_description() -> None:
    registry = ToolRegistry()
    registry.add_tool_note("fx_series", "One request covers an interval.")
    registry.clear_tool_notes()

    assert registry.spec("fx_series").description == BASE_DESCRIPTIONS["fx_series"]


def test_notes_can_be_supplied_at_construction_time() -> None:
    registry = ToolRegistry(tool_notes={"fx_rate": ["Trust the response date."]})

    assert "- Trust the response date." in registry.spec("fx_rate").description


# -- fx tools and accounting -------------------------------------------


def test_fx_rate_returns_the_api_resolved_rate_date(offline_registry: ToolRegistry) -> None:
    result = offline_registry.call("fx_rate", date="2026-04-04", base="EUR", symbol="USD")

    assert result == {
        "requested_date": "2026-04-04",
        "rate_date": "2026-04-02",
        "base": "EUR",
        "symbol": "USD",
        "rate": 1.1525,
    }


def test_the_second_identical_fx_call_is_served_from_cache(
    offline_registry: ToolRegistry,
) -> None:
    first = offline_registry.call("fx_rate", date="2026-04-04", base="EUR", symbol="USD")
    second = offline_registry.call("fx_rate", date="2026-04-04", base="EUR", symbol="USD")

    assert first == second
    calls = offline_registry.calls
    assert [call.cached for call in calls] == [False, True]
    assert len(offline_registry.transport.requests) == 1  # type: ignore[attr-defined]


def test_every_fx_call_records_latency_bytes_status_and_cached(
    offline_registry: ToolRegistry,
) -> None:
    offline_registry.call("fx_rate", date="2026-04-04", base="EUR", symbol="USD")
    offline_registry.call("fx_rate", date="2026-04-04", base="EUR", symbol="USD")

    for call in offline_registry.calls:
        assert call.name == "fx_rate"
        assert call.status == STATUS_OK
        assert call.latency_s >= 0.0
        assert call.bytes > 0
        assert call.http_status == 200
        assert call.arguments == {"date": "2026-04-04", "base": "EUR", "symbol": "USD"}


def test_fx_series_returns_every_returned_day_and_is_accounted_for(
    offline_registry: ToolRegistry,
) -> None:
    result = offline_registry.call(
        "fx_series", start="2026-03-02", end="2026-03-06", base="EUR", symbol="USD"
    )

    assert len(result["rates"]) == 5
    assert result["rates"]["2026-03-02"] == 1.15
    (call,) = offline_registry.calls
    assert call.name == "fx_series"
    assert call.cached is False
    assert call.bytes > 0


def test_a_failing_tool_call_is_still_accounted_for(offline_registry: ToolRegistry) -> None:
    with pytest.raises(KeyError):
        offline_registry.call("fx_rate", date="1999-01-01", base="EUR", symbol="USD")

    (call,) = offline_registry.calls
    assert call.status == STATUS_ERROR
    assert not call.ok
    assert call.error
    assert offline_registry.log.failed_calls == 1


def test_accounting_records_render_for_a_role_trace(offline_registry: ToolRegistry) -> None:
    offline_registry.call("fx_rate", date="2026-04-04", base="EUR", symbol="USD")

    (record,) = offline_registry.log.as_dicts()

    assert set(record) == {
        "name",
        "arguments",
        "status",
        "latency_s",
        "bytes",
        "cached",
        "response",
        "http_status",
        "error",
    }


def test_the_committed_cache_answers_the_good_friday_lookup_offline() -> None:
    # No transport: any network attempt would fail. data/fx_cache is committed.
    registry = ToolRegistry(fx_client=FXClient(cache_dir=COMMITTED_CACHE))

    result = registry.call("fx_rate", date="2026-04-04", base="EUR", symbol="USD")

    assert result["requested_date"] == "2026-04-04"
    assert result["rate_date"] == "2026-04-02"
    assert registry.calls[0].cached is True


# -- python_exec and fan_out through the registry ----------------------


def test_python_exec_runs_through_the_registry_with_accounting() -> None:
    registry = ToolRegistry()

    assert registry.call("python_exec", code="print(6 * 7)") == "42\n"

    (call,) = registry.calls
    assert call.name == "python_exec"
    assert call.cached is False
    assert call.http_status is None
    assert call.bytes == 3


def test_python_exec_timeout_is_configurable_per_registry() -> None:
    registry = ToolRegistry(python_timeout_s=1.0)

    text = registry.call("python_exec", code="while True:\n    pass")

    assert "timed out after 1s" in text


def test_fan_out_runs_the_bound_role_runner() -> None:
    registry = ToolRegistry().for_role(lambda subtask: f"answer:{subtask}")

    results = registry.call("fan_out", subtasks=["one", "two"])

    assert results == ["answer:one", "answer:two"]
    (call,) = registry.calls
    assert call.name == "fan_out"
    assert call.bytes == len("answer:one") + len("answer:two")


def test_fan_out_without_a_bound_role_is_an_accounted_error() -> None:
    registry = ToolRegistry()

    with pytest.raises(FanOutUnavailableError):
        registry.call("fan_out", subtasks=["one"])

    assert registry.log.failed_calls == 1


# -- per-role registries ------------------------------------------------


def test_for_role_shares_the_fx_client_but_separates_accounting() -> None:
    parent = ToolRegistry()
    fetcher = parent.for_role(lambda subtask: subtask)
    solver = parent.for_role(lambda subtask: subtask)

    fetcher.call("python_exec", code="print(1)")

    assert fetcher.fx_client is parent.fx_client is solver.fx_client
    assert fetcher.log.n_tool_calls == 1
    assert solver.log.n_tool_calls == 0
    assert parent.log.n_tool_calls == 0


def test_for_role_can_share_one_log_when_the_caller_asks() -> None:
    shared = ToolCallLog()
    parent = ToolRegistry()

    parent.for_role(None, log=shared).call("python_exec", code="print(1)")
    parent.for_role(None, log=shared).call("python_exec", code="print(2)")

    assert shared.n_tool_calls == 2


def test_for_role_carries_the_learned_notes() -> None:
    parent = ToolRegistry()
    parent.add_tool_note("fx_rate", "Trust the response date.")

    assert "- Trust the response date." in parent.for_role(None).spec("fx_rate").description
