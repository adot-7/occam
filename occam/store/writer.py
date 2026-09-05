"""Append-only event writer with a derived state snapshot."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from occam.core.models import Event
from occam.store.reader import EventReader
from occam.store.reducer import reduce, state_json_bytes
from occam.store.schema import validate_event, validate_state


class EventWriter:
    """Write validated events and refresh ``state.json`` after every append."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        existing = EventReader(self.run_dir).read() if self.events_path.exists() else []
        self._next_seq = existing[-1].seq + 1 if existing else 0
        self.run_id = run_id or (existing[0].run_id if existing else None)
        if existing and run_id is not None and run_id != existing[0].run_id:
            raise ValueError("run_id does not match the existing event log")

    def append(self, event: Event | Mapping[str, Any]) -> Event:
        """Append one event, assigning sequence/run id for a plain mapping."""

        if isinstance(event, Event):
            payload = event.model_dump(mode="json", exclude_none=False)
        else:
            payload = dict(event)
            payload.setdefault("seq", self._next_seq)
            if self.run_id is not None:
                payload.setdefault("run_id", self.run_id)
        if "run_id" not in payload:
            raise ValueError("run_id is required for the first event")
        if self.run_id is not None and payload["run_id"] != self.run_id:
            raise ValueError("event run_id does not match the existing event log")
        if payload.get("seq") != self._next_seq:
            raise ValueError(f"expected event seq {self._next_seq}, got {payload.get('seq')}")

        parsed = Event.model_validate(payload)
        serialized = parsed.model_dump(mode="json", exclude_none=False)
        validate_event(serialized)
        line = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.events_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        self.run_id = parsed.run_id
        self._next_seq += 1
        self._write_state()
        return parsed

    def write(self, event: Event | Mapping[str, Any]) -> Event:
        """Compatibility alias for callers that use ``write``."""

        return self.append(event)

    def append_event(self, event: Event | Mapping[str, Any]) -> Event:
        """Explicit alias used by engine code."""

        return self.append(event)

    def _write_state(self) -> None:
        state = reduce(EventReader(self.run_dir).read())
        payload = state.model_dump(mode="json", exclude_none=False)
        validate_state(payload)
        state_bytes = state_json_bytes(state)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.run_dir, prefix=".state.", delete=False
        ) as handle:
            temp_name = handle.name
            handle.write(state_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, self.run_dir / "state.json")

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        return None
