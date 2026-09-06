"""The engine's one-way door onto the event log.

Engine code takes an :data:`EventSink` — ``(type, data) -> None`` — so that
computation never depends on the store, and tests can collect events without a
run directory. :func:`writer_sink` is the production binding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from occam.store.writer import EventWriter

EventSink = Callable[[str, Mapping[str, Any]], None]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def null_sink(event_type: str, data: Mapping[str, Any]) -> None:
    """Discard events; used when a computation runs outside a run directory."""


def collecting_sink(collected: MutableSequence[dict[str, Any]]) -> EventSink:
    """Append ``{"type": ..., "data": ...}`` to ``collected`` for inspection."""

    def sink(event_type: str, data: Mapping[str, Any]) -> None:
        collected.append({"type": event_type, "data": dict(data)})

    return sink


def writer_sink(writer: EventWriter, run_id: str | None = None) -> EventSink:
    """Bind a sink to an :class:`occam.store.writer.EventWriter`.

    The writer stamps ``seq`` and validates against ``events.schema.json``, so a
    payload this sink cannot write is a bug that fails loudly at emission time.
    """

    def sink(event_type: str, data: Mapping[str, Any]) -> None:
        event: dict[str, Any] = {"ts": _now(), "type": event_type, "data": dict(data)}
        resolved = run_id if run_id is not None else getattr(writer, "run_id", None)
        if resolved is not None:
            event["run_id"] = resolved
        writer.append(event)

    return sink


__all__ = ["EventSink", "collecting_sink", "null_sink", "writer_sink"]
