"""Test doubles for the DAG executor.

Nothing here is production code.  The scripted provider stands in for a hosted
model so the executor's own behaviour — routing, concurrency, the tool loop,
caching, knockouts — is what a test observes.  The FX agent double is a
deliberately literal reimplementation of what a competent three-role team
would do on the task in ``prd/02-DATA-AND-TASKS.md``.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from occam.core.models import ToolSpec
from occam.llm.cache import DiskCache
from occam.llm.client import LLMClient
from occam.llm.config import ModelConfig
from occam.llm.providers import ProviderResponse

WORKER = "worker_fast"

WORKER_CONFIG = ModelConfig(
    key=WORKER,
    provider="openai_compat",
    model="glm-4-7-flash",
    base_url="https://api.tensormux.example/v1",
    api_key="test-key",
    in_per_m=0.0,
    out_per_m=0.0,
    grant_equiv_in_per_m=0.06,
    grant_equiv_out_per_m=0.40,
    rpm=60,
    supports_json_schema=False,
    tool_choice_modes=("auto",),
)


def text_response(text: str, *, tokens_in: int = 40, tokens_out: int = 60) -> ProviderResponse:
    """A plain assistant answer with no tool calls."""

    return ProviderResponse(
        text=text,
        tool_calls=[],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        finish_reason="stop",
    )


def tool_call_response(
    calls: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    text: str = "",
    tokens_in: int = 40,
    tokens_out: int = 30,
    prefix: str = "call",
) -> ProviderResponse:
    """A native tool-call turn, shaped exactly like TensorMux returns one."""

    return ProviderResponse(
        text=text,
        tool_calls=[
            {
                "id": f"{prefix}-{position}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(dict(arguments), sort_keys=True),
                },
            }
            for position, (name, arguments) in enumerate(calls)
        ],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        finish_reason="tool_calls",
    )


@dataclass
class ProviderCall:
    """One recorded request, enough to assert on prompts and tools."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    max_tokens: int
    temperature: float

    @property
    def system(self) -> str:
        return str(self.messages[0].get("content", ""))

    @property
    def user(self) -> str:
        return str(self.messages[1].get("content", ""))

    @property
    def tool_names(self) -> list[str]:
        return [str(tool.get("name", "")) for tool in (self.tools or [])]


class ScriptedProvider:
    """Answers with ``handler(call) -> ProviderResponse`` and records requests.

    ``max_in_flight`` is tracked so a test can assert the executor really does
    run independent roles concurrently and really does respect the per-model
    bound.
    """

    def __init__(self, handler: Callable[[ProviderCall], ProviderResponse]) -> None:
        self.handler = handler
        self.calls: list[ProviderCall] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def complete(
        self,
        config: ModelConfig,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Any] | None = None,
        response_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        call = ProviderCall(
            model=config.model,
            messages=[dict(message) for message in messages],
            tools=None if tools is None else [dict(tool) for tool in tools],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        with self._lock:
            self.calls.append(call)
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            return self.handler(call)
        finally:
            with self._lock:
                self._in_flight -= 1

    @property
    def count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()
        self.max_in_flight = 0


class NeverEndingToolProvider(ScriptedProvider):
    """Keep requesting one tool, omitting text after the first turn."""

    def __init__(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        first_text: str,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = dict(arguments)
        self.first_text = first_text
        super().__init__(self._respond)

    def _respond(self, _call: ProviderCall) -> ProviderResponse:
        text = self.first_text if self.count == 1 else ""
        return tool_call_response([(self.tool_name, self.arguments)], text=text)


def build_client(
    provider: Any,
    cache_dir: Any,
    *,
    configs: Mapping[str, ModelConfig] | None = None,
) -> LLMClient:
    """An ``LLMClient`` wired to a double, with a real content-addressed cache."""

    return LLMClient(
        configs=dict(configs or {WORKER: WORKER_CONFIG}),
        providers={"openai_compat": provider},
        cache=DiskCache(cache_dir),
        sleeper=lambda _seconds: None,
    )


class RecordingTool:
    """Wraps a tool implementation and records every invocation."""

    def __init__(self, spec: ToolSpec, implementation: Callable[..., Any]) -> None:
        self.spec = spec
        self._implementation = implementation
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._implementation(**kwargs)

    @property
    def count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()


# --------------------------------------------------------------------------
# FX task double: rates, tool, cases, and a three-role agent script.
# --------------------------------------------------------------------------

FX_RATE_SPEC = ToolSpec(
    name="fx_rate",
    # Deliberately plain: holidays and the range endpoint are lessons the
    # system must learn, never tool documentation we hand it (AGENTS.md).
    description="Get the exchange rate between two currencies on a date.",
    parameters={
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "base": {"type": "string"},
            "symbol": {"type": "string"},
        },
        "required": ["date", "base", "symbol"],
        "additionalProperties": False,
    },
)

BASE_RATES = {"EUR": 96.0, "USD": 88.0, "GBP": 112.0, "JPY": 0.58}

# Good Friday 2026-04-03 through Easter Monday 2026-04-06: the window that
# makes `requested_date != rate_date` observable in a trace (02 §1.3, L1).
CLOSED_DAYS = frozenset({"2026-04-03", "2026-04-06"})


def resolve_rate_date(requested: str) -> str:
    """Walk back to the last business day, as the ECB feed does."""

    current = date.fromisoformat(requested)
    while current.weekday() >= 5 or current.isoformat() in CLOSED_DAYS:
        current -= timedelta(days=1)
    return current.isoformat()


def offline_fx_rate(date: str, base: str, symbol: str) -> dict[str, Any]:
    """Stub for WP-04's ``fx_rate``, with the same response shape (02 §2).

    The parameter names match the real tool, so ``date`` shadows the module's
    ``datetime.date`` here on purpose.
    """

    if symbol != "INR":
        raise ValueError(f"unsupported symbol {symbol!r}")
    if base not in BASE_RATES:
        raise ValueError(f"unsupported base {base!r}")
    rate_date = resolve_rate_date(date)
    return {
        "requested_date": date,
        "rate_date": rate_date,
        "base": base,
        "symbol": symbol,
        "rate": round(BASE_RATES[base] * (1 + _ordinal_offset(rate_date) / 10000), 6),
    }


def _ordinal_offset(rate_date: str) -> int:
    return (date.fromisoformat(rate_date) - date(2026, 1, 1)).days


#: ``(currency, requested_date) -> rate``, the shape a reference answer needs.
RateLookup = Callable[[str, str], float]


def offline_rate(currency: str, requested: str) -> float:
    """The rate the reference implementation of the case expects."""

    return float(offline_fx_rate(requested, currency, "INR")["rate"])


@dataclass(frozen=True)
class Invoice:
    """One ledger line, in the subset of shapes the doubles cover."""

    invoice_id: str
    counterparty: str
    currency: str
    amount: float
    issued: str
    settled: str | None = None
    received_currency: str | None = None
    received_amount: float | None = None

    def line(self, position: int) -> str:
        head = (
            f"{position}. {self.invoice_id} · {self.counterparty} · "
            f"{self.currency} {self.amount:.2f} · issued {self.issued} · "
        )
        if self.settled is None:
            return head + "OPEN"
        return (
            head + f"SETTLED {self.settled}, received "
            f"{self.received_currency} {self.received_amount:.2f}"
        )

    def gain(self, valuation_date: str, rate: RateLookup = offline_rate) -> float:
        """The formula in ``02 §1.2``; bank fees are out of the doubles' scope."""

        booked = self.amount * rate(self.currency, self.issued)
        if self.settled is None:
            value = self.amount * rate(self.currency, valuation_date)
        else:
            assert self.received_currency is not None
            assert self.received_amount is not None
            value = self.received_amount * rate(self.received_currency, self.settled)
        return value - booked


@dataclass(frozen=True)
class LedgerCase:
    """One synthetic ledger snapshot plus its reference answer."""

    case_id: str
    valuation_date: str
    invoices: tuple[Invoice, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def input(self) -> str:
        lines = "\n".join(
            invoice.line(position) for position, invoice in enumerate(self.invoices, start=1)
        )
        return (
            f"Valuation date: {self.valuation_date}. Reporting currency: INR.\n"
            f"Ledger:\n{lines}\n"
            "Compute the total FX gain/(loss) in INR as of the valuation date, "
            "and the gain/(loss) per invoice."
        )

    def expected_for(self, rate: RateLookup = offline_rate) -> dict[str, Any]:
        """The reference answer, computed with whichever rate source is live."""

        per_invoice = {
            invoice.invoice_id: round(invoice.gain(self.valuation_date, rate), 2)
            for invoice in self.invoices
        }
        return {
            "total_inr": round(sum(per_invoice.values()), 2),
            "per_invoice": per_invoice,
        }

    @property
    def expected(self) -> dict[str, Any]:
        return self.expected_for()


FX_CASES: tuple[LedgerCase, ...] = (
    LedgerCase(
        "fxs_001",
        "2026-04-30",
        (
            Invoice("INV-1001", "ACME GmbH", "EUR", 12400.0, "2026-03-14"),
            Invoice(
                "INV-1002",
                "Kyoto Labs",
                "JPY",
                1850000.0,
                "2026-03-20",
                "2026-04-02",
                "JPY",
                1850000.0,
            ),
        ),
        {"has_weekend_or_holiday": True},
    ),
    LedgerCase(
        "fxs_002",
        "2026-04-30",
        (
            # Settled on Good Friday: requested_date != rate_date.
            Invoice(
                "INV-1003", "Harbor Ltd", "GBP", 8000.0, "2026-04-01", "2026-04-03", "GBP", 8000.0
            ),
            Invoice("INV-1004", "Northwind", "USD", 5200.0, "2026-02-10"),
        ),
        {"has_weekend_or_holiday": True},
    ),
    LedgerCase(
        "fxs_003",
        "2026-03-31",
        (
            Invoice("INV-1005", "Pacific Co", "USD", 9100.0, "2026-01-19"),
            Invoice("INV-1006", "Alpine AG", "EUR", 3400.0, "2026-02-27"),
        ),
        {},
    ),
    LedgerCase(
        "fxs_004",
        "2026-04-30",
        (
            # Cross-currency settlement (02 §1.3, D2).
            Invoice(
                "INV-1007", "ACME GmbH", "EUR", 6250.0, "2026-03-05", "2026-04-20", "USD", 6900.0
            ),
            Invoice("INV-1008", "Kyoto Labs", "JPY", 940000.0, "2026-04-10"),
        ),
        {"has_cross_ccy": True},
    ),
    LedgerCase(
        "fxs_005",
        "2026-04-30",
        (
            # Settled on Easter Monday: another rate-date resolution.
            Invoice(
                "INV-1009", "Harbor Ltd", "GBP", 15500.0, "2026-02-02", "2026-04-06", "GBP", 15500.0
            ),
            Invoice("INV-1010", "Northwind", "USD", 2750.0, "2026-03-30"),
        ),
        {"has_weekend_or_holiday": True},
    ),
)


_LEDGER_LINE = re.compile(
    r"^\s*\d+\.\s+(?P<id>INV-\d+)\s+·\s+(?P<party>[^·]+?)\s+·\s+"
    r"(?P<ccy>[A-Z]{3})\s+(?P<amount>[\d.]+)\s+·\s+issued\s+(?P<issued>\d{4}-\d{2}-\d{2})\s+·\s+"
    r"(?:OPEN|SETTLED\s+(?P<settled>\d{4}-\d{2}-\d{2}),\s+received\s+"
    r"(?P<rccy>[A-Z]{3})\s+(?P<ramount>[\d.]+))\s*$",
    re.MULTILINE,
)
_VALUATION = re.compile(r"Valuation date:\s*(\d{4}-\d{2}-\d{2})")
_SECTION = re.compile(r"^### (?P<key>\S+)\n(?P<body>.*?)(?=\n### |\Z)", re.MULTILINE | re.DOTALL)

PARSER_MARK = "ROLE: ledger parser"
FETCHER_MARK = "ROLE: rate fetcher"
CALCULATOR_MARK = "ROLE: fx calculator"


def sections(user_message: str) -> dict[str, str]:
    """Split the executor's ``### key`` blocks back out of a user message."""

    return {
        match.group("key"): match.group("body").strip() for match in _SECTION.finditer(user_message)
    }


def _parse_ledger(case_text: str) -> dict[str, Any]:
    valuation = _VALUATION.search(case_text)
    invoices = []
    for match in _LEDGER_LINE.finditer(case_text):
        invoices.append(
            {
                "id": match.group("id"),
                "currency": match.group("ccy"),
                "amount": float(match.group("amount")),
                "issued": match.group("issued"),
                "settled": match.group("settled"),
                "received_currency": match.group("rccy"),
                "received_amount": (
                    float(match.group("ramount")) if match.group("ramount") else None
                ),
            }
        )
    return {
        "valuation_date": valuation.group(1) if valuation else "",
        "invoices": invoices,
    }


def _needed_rates(ledger: Mapping[str, Any]) -> list[tuple[str, str]]:
    needed: list[tuple[str, str]] = []
    for invoice in ledger["invoices"]:
        needed.append((invoice["currency"], invoice["issued"]))
        if invoice["settled"] is None:
            needed.append((invoice["currency"], ledger["valuation_date"]))
        else:
            needed.append((invoice["received_currency"], invoice["settled"]))
    ordered: list[tuple[str, str]] = []
    for pair in needed:
        if pair not in ordered:
            ordered.append(pair)
    return ordered


def fx_agent_script(call: ProviderCall) -> ProviderResponse:
    """A competent three-role FX team, scripted.

    Dispatch is on the role marker in the system prompt, so the double exercises
    the executor's real prompt assembly rather than a private side channel.
    """

    system = call.system
    if PARSER_MARK in system:
        ledger = _parse_ledger(sections(call.user).get("task", ""))
        return text_response(json.dumps(ledger, sort_keys=True))

    if FETCHER_MARK in system:
        block = sections(call.user).get("parsed_ledger", "")
        if block.startswith("[no input"):
            return text_response(json.dumps({"error": block}))
        ledger = json.loads(block)
        results = _tool_results(call.messages)
        pending = [pair for pair in _needed_rates(ledger) if f"{pair[0]}@{pair[1]}" not in results]
        if pending:
            return tool_call_response(
                [
                    ("fx_rate", {"date": requested, "base": currency, "symbol": "INR"})
                    for currency, requested in pending
                ],
                prefix="fx",
            )
        return text_response(json.dumps(results, sort_keys=True))

    if CALCULATOR_MARK in system:
        blocks = sections(call.user)
        ledger_block = blocks.get("parsed_ledger", "")
        rates_block = blocks.get("fx_rates", "")
        if ledger_block.startswith("[no input") or rates_block.startswith("[no input"):
            # No rates: the honest degraded answer, and a failing one.
            return text_response(
                "Missing an upstream input, so no revaluation is possible.\n"
                '```json\n{"total_inr": 0, "per_invoice": {}}\n```'
            )
        ledger = json.loads(ledger_block)
        rates = json.loads(rates_block)
        per_invoice: dict[str, float] = {}
        for invoice in ledger["invoices"]:
            booked = invoice["amount"] * rates[f"{invoice['currency']}@{invoice['issued']}"]["rate"]
            if invoice["settled"] is None:
                key = f"{invoice['currency']}@{ledger['valuation_date']}"
                value = invoice["amount"] * rates[key]["rate"]
            else:
                key = f"{invoice['received_currency']}@{invoice['settled']}"
                value = invoice["received_amount"] * rates[key]["rate"]
            per_invoice[invoice["id"]] = round(value - booked, 2)
        answer = {
            "total_inr": round(sum(per_invoice.values()), 2),
            "per_invoice": per_invoice,
        }
        return text_response(
            "Revaluation complete.\n```json\n" + json.dumps(answer, sort_keys=True) + "\n```"
        )

    raise AssertionError(f"no script for system prompt: {system[:80]!r}")


def _tool_results(messages: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rebuild the fetcher's rate table from the tool messages so far."""

    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and "requested_date" in payload:
            key = f"{payload['base']}@{payload['requested_date']}"
            results[key] = {
                "rate_date": payload["rate_date"],
                "rate": payload["rate"],
            }
    return results


_FENCED = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL)


def grade_fx_total(answer: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    """The ``fx_total`` tolerances of ``02 §1.2``, for tests only.

    WP-03's ``occam.tasks.checkers.fx_total`` is the real grader; the executor
    reaches it through ``resolve_grader`` once that module is on main.
    """

    expected_total = float(expected["total_inr"])
    expected_per_invoice = dict(expected.get("per_invoice", {}))
    blocks = _FENCED.findall(answer or "")
    sub_results = {invoice_id: False for invoice_id in expected_per_invoice}
    if not blocks:
        return {"passed": False, "sub_results": sub_results}
    try:
        candidate = json.loads(blocks[-1])
        candidate_total = float(candidate["total_inr"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"passed": False, "sub_results": sub_results}
    candidate_per_invoice = candidate.get("per_invoice", {}) or {}
    for invoice_id, value in expected_per_invoice.items():
        try:
            actual = float(candidate_per_invoice[invoice_id])
        except (KeyError, TypeError, ValueError):
            continue
        sub_results[invoice_id] = abs(actual - float(value)) <= max(1.0, 0.001 * abs(float(value)))
    passed = abs(candidate_total - expected_total) <= max(5.0, 0.001 * abs(expected_total))
    return {"passed": passed, "sub_results": sub_results}
