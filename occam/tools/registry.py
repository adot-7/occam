"""The tool registry: ``name -> (ToolSpec, callable)``.

Candidate agents see exactly what this module publishes.  Two rules govern it:

1. **Base descriptions stay plain.**  They say nothing about weekends, ECB
   holidays, or the fact that the series endpoint answers a whole month in one
   request.  Those are lessons the system has to *learn* — writing them here
   would hand the product its own result.  :func:`append_tool_note` is the hook
   the architect uses to append a learned ``tool_note`` at proposal time.
2. **Every call is accounted for.**  Latency, bytes, status and ``cached`` are
   recorded for successes and failures alike, feeding ``n_tool_calls`` and
   ``wasted_calls``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from occam.core.models import ToolSpec
from occam.tools.accounting import STATUS_ERROR, STATUS_OK, ToolCall, ToolCallLog
from occam.tools.fan_out import (
    DEFAULT_MAX_SUBTASKS,
    DEFAULT_MAX_WORKERS,
    RoleRunner,
    fan_out,
)
from occam.tools.fx import (
    FX_RATE_DESCRIPTION,
    FX_SERIES_DESCRIPTION,
    FXCall,
    FXClient,
    get_default_client,
)
from occam.tools.python_exec import DEFAULT_TIMEOUT_S, python_exec

FX_RATE = "fx_rate"
FX_SERIES = "fx_series"
PYTHON_EXEC = "python_exec"
FAN_OUT = "fan_out"

TOOL_NAMES: tuple[str, ...] = (FX_RATE, FX_SERIES, PYTHON_EXEC, FAN_OUT)

PYTHON_EXEC_DESCRIPTION = (
    "Evaluate a restricted calculation subset, not arbitrary Python; unsupported "
    "constructs return an error."
)
FAN_OUT_DESCRIPTION = (
    "Run your own instructions separately on each of several subtasks "
    "and return one result per subtask."
)

#: Header introducing learned ``tool_note`` lessons appended to a description.
TOOL_NOTE_HEADER = "Notes learned from previous runs:"

_CURRENCY = "Three-letter currency code, for example EUR."
_DATE = "Date in YYYY-MM-DD format."

BASE_DESCRIPTIONS: Mapping[str, str] = {
    FX_RATE: FX_RATE_DESCRIPTION,
    FX_SERIES: FX_SERIES_DESCRIPTION,
    PYTHON_EXEC: PYTHON_EXEC_DESCRIPTION,
    FAN_OUT: FAN_OUT_DESCRIPTION,
}

PARAMETERS: Mapping[str, dict[str, Any]] = {
    FX_RATE: {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": _DATE},
            "base": {"type": "string", "description": _CURRENCY},
            "symbol": {"type": "string", "description": _CURRENCY},
        },
        "required": ["date", "base", "symbol"],
        "additionalProperties": False,
    },
    FX_SERIES: {
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": _DATE},
            "end": {"type": "string", "description": _DATE},
            "base": {"type": "string", "description": _CURRENCY},
            "symbol": {"type": "string", "description": _CURRENCY},
        },
        "required": ["start", "end", "base", "symbol"],
        "additionalProperties": False,
    },
    PYTHON_EXEC: {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Restricted calculation source to evaluate.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    FAN_OUT: {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One instruction per subtask.",
            },
        },
        "required": ["subtasks"],
        "additionalProperties": False,
    },
}


class UnknownToolError(KeyError):
    """Raised when a role asks for a tool the registry does not publish."""


@dataclass(frozen=True)
class ToolBinding:
    """A published tool: its spec and the callable that runs it."""

    spec: ToolSpec
    call: Callable[..., Any]
    # Keep the owner available when a caller hands ``bindings()`` to another
    # component.  The public binding contract remains ``spec`` + ``call``;
    # this private provenance lets the executor create a role-scoped sibling
    # without requiring callers to pass the registry a second time.
    _registry: Any = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        """The registry key this binding is published under."""

        return self.spec.name


def append_tool_note(description: str, note: str) -> str:
    """Append one learned ``tool_note`` to a tool description.

    This is the architect-time hook from ``01 §6``.  Notes accumulate under a
    single header, and appending a note the description already carries is a
    no-op so that re-proposing an architecture stays idempotent.
    """

    text = note.strip()
    if not text:
        return description
    bullet = f"- {text}"
    if bullet in description:
        return description
    if TOOL_NOTE_HEADER in description:
        return f"{description.rstrip()}\n{bullet}"
    return f"{description.rstrip()}\n\n{TOOL_NOTE_HEADER}\n{bullet}"


def describe(name: str, notes: Sequence[str] = ()) -> str:
    """Return a tool's base description with ``notes`` appended."""

    description = BASE_DESCRIPTIONS[name]
    for note in notes:
        description = append_tool_note(description, note)
    return description


def build_spec(name: str, notes: Sequence[str] = ()) -> ToolSpec:
    """Build the ``ToolSpec`` a role is handed, with any learned notes applied."""

    if name not in BASE_DESCRIPTIONS:
        raise UnknownToolError(name)
    return ToolSpec(
        name=name,
        description=describe(name, notes),
        parameters=json.loads(json.dumps(PARAMETERS[name])),
    )


class ToolRegistry:
    """Publishes the built-in tools and accounts for every call made through it.

    One registry per role: :meth:`for_role` binds the role's own prompt runner
    for ``fan_out`` and gives the role a private accounting log, while sharing
    the FX client (and therefore its disk cache) across the whole run.
    """

    def __init__(
        self,
        *,
        fx_client: FXClient | None = None,
        role_runner: RoleRunner | None = None,
        log: ToolCallLog | None = None,
        tool_notes: Mapping[str, Sequence[str]] | None = None,
        python_timeout_s: float = DEFAULT_TIMEOUT_S,
        fan_out_max_workers: int = DEFAULT_MAX_WORKERS,
        fan_out_max_subtasks: int = DEFAULT_MAX_SUBTASKS,
    ) -> None:
        self._fx_client = fx_client
        self._role_runner = role_runner
        self._log = log if log is not None else ToolCallLog()
        self._notes: dict[str, list[str]] = {
            name: list(notes) for name, notes in (tool_notes or {}).items()
        }
        self.python_timeout_s = python_timeout_s
        self.fan_out_max_workers = fan_out_max_workers
        self.fan_out_max_subtasks = fan_out_max_subtasks
        self._handlers: dict[str, Callable[..., Any]] = {
            FX_RATE: self._fx_rate,
            FX_SERIES: self._fx_series,
            PYTHON_EXEC: self._python_exec,
            FAN_OUT: self._fan_out,
        }

    # -- publication ----------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Every published tool name, in a stable order."""

        return TOOL_NAMES

    def __contains__(self, name: object) -> bool:
        return name in self._handlers

    def spec(self, name: str) -> ToolSpec:
        """Return one tool's spec, with this registry's learned notes applied."""

        self._require(name)
        return build_spec(name, self._notes.get(name, ()))

    def specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        """Return specs for ``names`` (default: all), preserving the given order."""

        selected = TOOL_NAMES if names is None else tuple(names)
        return [self.spec(name) for name in selected]

    def binding(self, name: str) -> ToolBinding:
        """Return the ``(ToolSpec, callable)`` pair published under ``name``."""

        self._require(name)

        def invoke(**arguments: Any) -> Any:
            return self.call(name, **arguments)

        return ToolBinding(spec=self.spec(name), call=invoke, _registry=self)

    def bindings(self, names: Iterable[str] | None = None) -> dict[str, ToolBinding]:
        """Return the registry mapping ``name -> (ToolSpec, callable)``."""

        selected = TOOL_NAMES if names is None else tuple(names)
        return {name: self.binding(name) for name in selected}

    # -- learned notes --------------------------------------------------

    def add_tool_note(self, name: str, note: str) -> str:
        """Append a learned ``tool_note`` and return the resulting description."""

        self._require(name)
        text = note.strip()
        notes = self._notes.setdefault(name, [])
        if text and text not in notes:
            notes.append(text)
        return self.spec(name).description

    def tool_notes(self, name: str) -> tuple[str, ...]:
        """Return the notes currently appended to ``name``."""

        self._require(name)
        return tuple(self._notes.get(name, ()))

    def clear_tool_notes(self) -> None:
        """Drop every appended note, restoring the plain base descriptions."""

        self._notes.clear()

    # -- execution ------------------------------------------------------

    @property
    def log(self) -> ToolCallLog:
        """The accounting log this registry writes to."""

        return self._log

    @property
    def calls(self) -> tuple[ToolCall, ...]:
        """A snapshot of the calls made through this registry."""

        return self._log.calls

    @property
    def fx_client(self) -> FXClient:
        """The shared, cache-backed Frankfurter client."""

        if self._fx_client is None:
            self._fx_client = get_default_client()
        return self._fx_client

    def for_role(
        self,
        role_runner: RoleRunner | None,
        *,
        log: ToolCallLog | None = None,
    ) -> ToolRegistry:
        """Return a sibling registry bound to one role's prompt runner.

        The FX client and the learned notes are shared; the accounting log is
        private by default so each role's tool calls land in its own trace.
        """

        sibling = ToolRegistry(
            fx_client=self.fx_client,
            role_runner=role_runner,
            log=log,
            python_timeout_s=self.python_timeout_s,
            fan_out_max_workers=self.fan_out_max_workers,
            fan_out_max_subtasks=self.fan_out_max_subtasks,
        )
        sibling._notes = self._notes
        return sibling

    def call(self, name: str, **arguments: Any) -> Any:
        """Run a published tool, recording one accounting entry either way."""

        self._require(name)
        handler = self._handlers[name]
        started = time.perf_counter()
        try:
            result = handler(**arguments)
        except Exception as exc:
            self._log.record(
                ToolCall(
                    name=name,
                    arguments=dict(arguments),
                    status=STATUS_ERROR,
                    latency_s=time.perf_counter() - started,
                    bytes=0,
                    cached=False,
                    response=None,
                    http_status=_http_status(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        latency_s = time.perf_counter() - started
        fx_call = self.fx_client.thread_last_call if name in (FX_RATE, FX_SERIES) else None
        self._log.record(
            ToolCall(
                name=name,
                arguments=dict(arguments),
                status=STATUS_OK,
                latency_s=latency_s,
                bytes=_payload_bytes(result, fx_call),
                cached=bool(fx_call.cached) if fx_call is not None else False,
                response=_jsonable(result),
                http_status=fx_call.status if fx_call is not None else None,
            )
        )
        return result

    # -- handlers -------------------------------------------------------

    def _fx_rate(self, date: str, base: str, symbol: str) -> dict[str, Any]:
        return self.fx_client.fx_rate(date, base, symbol)

    def _fx_series(self, start: str, end: str, base: str, symbol: str) -> dict[str, Any]:
        return self.fx_client.fx_series(start, end, base, symbol)

    def _python_exec(self, code: str) -> str:
        return python_exec(code, timeout_s=self.python_timeout_s)

    def _fan_out(self, subtasks: Sequence[str]) -> list[str]:
        return fan_out(
            subtasks,
            runner=self._role_runner,
            max_workers=self.fan_out_max_workers,
            max_subtasks=self.fan_out_max_subtasks,
        )

    def _require(self, name: str) -> None:
        if name not in self._handlers:
            raise UnknownToolError(name)


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _payload_bytes(result: Any, fx_call: FXCall | None) -> int:
    """Size of what the tool handed back, in bytes.

    FX tools report the bytes that crossed the wire (or would have, on a cache
    hit) so that one series call and twenty rate calls are honestly comparable.
    """

    if fx_call is not None:
        return fx_call.response_bytes
    if isinstance(result, str):
        return len(result.encode("utf-8"))
    if isinstance(result, list) and all(isinstance(item, str) for item in result):
        return sum(len(item.encode("utf-8")) for item in result)
    return len(json.dumps(result, default=str).encode("utf-8"))


def _jsonable(value: Any) -> Any:
    """Return the JSON-safe response retained by the authoritative call log."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    return str(value)


def default_registry(**kwargs: Any) -> ToolRegistry:
    """Build a registry over the shared, disk-cached FX client."""

    return ToolRegistry(**kwargs)


__all__ = [
    "BASE_DESCRIPTIONS",
    "FAN_OUT",
    "FX_RATE",
    "FX_SERIES",
    "PARAMETERS",
    "PYTHON_EXEC",
    "TOOL_NAMES",
    "TOOL_NOTE_HEADER",
    "ToolBinding",
    "ToolRegistry",
    "UnknownToolError",
    "append_tool_note",
    "build_spec",
    "default_registry",
    "describe",
]
