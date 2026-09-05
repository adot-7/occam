"""Readers for Occam's append-only event and snapshot files."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from occam.core.models import Event, State
from occam.store.schema import validate_event, validate_state


class EventReader:
    """Read and optionally follow ``events.jsonl`` in a run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / "events.jsonl"
        self.state_path = self.run_dir / "state.json"

    def read(self) -> list[Event]:
        """Read all events, validating JSON, the envelope, and event order."""

        if not self.events_path.exists():
            raise FileNotFoundError(f"event log not found: {self.events_path}")

        events: list[Event] = []
        expected_seq = 0
        run_id: str | None = None
        with self.events_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    message = f"invalid JSON on events.jsonl line {line_number}: {exc}"
                    raise ValueError(message) from exc
                validate_event(payload)
                try:
                    event = Event.model_validate(payload)
                except ValueError as exc:
                    message = f"invalid event on events.jsonl line {line_number}: {exc}"
                    raise ValueError(message) from exc
                if run_id is None:
                    run_id = event.run_id
                elif event.run_id != run_id:
                    raise ValueError(f"events.jsonl line {line_number} changes run_id")
                if event.seq != expected_seq:
                    raise ValueError(
                        f"events.jsonl line {line_number} has seq {event.seq}; "
                        f"expected {expected_seq}"
                    )
                expected_seq += 1
                events.append(event)
        return events

    def __iter__(self) -> Iterator[Event]:
        return iter(self.read())

    def tail(
        self,
        start_seq: int = 0,
        *,
        follow: bool = False,
        poll_interval: float = 0.1,
    ) -> Iterator[Event]:
        """Yield events at or after ``start_seq``; follow when requested."""

        next_seq = start_seq
        while True:
            events = self.read()
            for event in events:
                if event.seq >= next_seq:
                    yield event
                    next_seq = event.seq + 1
            if not follow:
                return
            time.sleep(poll_interval)

    def load_state(self) -> State:
        """Load and validate the latest derived state snapshot."""

        if not self.state_path.exists():
            raise FileNotFoundError(f"state snapshot not found: {self.state_path}")
        with self.state_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_state(payload)
        return State.model_validate(payload)
