"""Per-call accounting shared by every tool in the registry.

Every tool invocation, whether it reached the network or not, produces one
:class:`ToolCall`.  These records are what ``n_tool_calls`` and ``wasted_calls``
are derived from, so they are written for failed calls too.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

STATUS_OK = "ok"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation as the metrics and TUI layers see it.

    ``bytes`` measures the payload handed back to the caller, so a role that
    pulls a whole series and a role that pulls one rate stay comparable.
    ``http_status`` is set only for tools that speak HTTP.
    """

    name: str
    arguments: dict[str, Any]
    status: str
    latency_s: float
    bytes: int
    cached: bool
    http_status: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the call returned a result rather than raising."""

        return self.status == STATUS_OK

    def as_dict(self) -> dict[str, Any]:
        """Render the record for ``RoleTrace.tool_calls`` and the event log."""

        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "status": self.status,
            "latency_s": self.latency_s,
            "bytes": self.bytes,
            "cached": self.cached,
            "http_status": self.http_status,
            "error": self.error,
        }


class ToolCallLog:
    """Thread-safe, append-only list of :class:`ToolCall` records.

    The executor gives each role its own log so that per-role tool accounting
    stays separable, which is what ablation's cost share reads.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._calls: list[ToolCall] = []

    def record(self, call: ToolCall) -> ToolCall:
        """Append one record and return it unchanged."""

        with self._guard:
            self._calls.append(call)
        return call

    @property
    def calls(self) -> tuple[ToolCall, ...]:
        """Return an immutable snapshot of the recorded calls."""

        with self._guard:
            return tuple(self._calls)

    @property
    def n_tool_calls(self) -> int:
        """Total number of tool invocations, successful or not."""

        return len(self)

    @property
    def cached_calls(self) -> int:
        """Number of invocations answered from a cache rather than the network."""

        return sum(1 for call in self.calls if call.cached)

    @property
    def failed_calls(self) -> int:
        """Number of invocations that raised."""

        return sum(1 for call in self.calls if not call.ok)

    @property
    def latency_s(self) -> float:
        """Total wall-clock time spent inside tools."""

        return sum(call.latency_s for call in self.calls)

    @property
    def bytes(self) -> int:
        """Total payload bytes handed back by tools."""

        return sum(call.bytes for call in self.calls)

    def as_dicts(self) -> list[dict[str, Any]]:
        """Render every record for ``RoleTrace.tool_calls``."""

        return [call.as_dict() for call in self.calls]

    def reset(self) -> None:
        """Drop every record, e.g. between cases."""

        with self._guard:
            self._calls.clear()

    def __len__(self) -> int:
        with self._guard:
            return len(self._calls)

    def __iter__(self) -> Iterator[ToolCall]:
        return iter(self.calls)


__all__ = ["STATUS_ERROR", "STATUS_OK", "ToolCall", "ToolCallLog"]
