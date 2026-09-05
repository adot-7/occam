"""Readers for Occam's append-only event and snapshot files."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from occam.core.models import Event, State
from occam.store.schema import validate_event, validate_state


class _IncompleteTrailingLine(Exception):
    """Signal that a live read saw a JSON line still being appended."""

    def __init__(self, events: list[Event], line_number: int, cause: json.JSONDecodeError):
        self.events = events
        self.line_number = line_number
        self.cause = cause
        super().__init__(f"incomplete JSON on events.jsonl trailing line {line_number}: {cause}")


class EventReader:
    """Read and optionally follow ``events.jsonl`` in a run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / "events.jsonl"
        self.state_path = self.run_dir / "state.json"

    def read(
        self,
        *,
        live: bool = False,
        retries: int = 5,
        retry_interval: float = 0.02,
    ) -> list[Event]:
        """Read all events, optionally tolerating a live partial trailing line.

        Normal reads are strict and reject empty logs, malformed JSON, and any
        invalid event.  A live read retries an incomplete final line because a
        writer may have flushed its JSON object before flushing the newline.  If
        that line remains incomplete after the retries, live mode returns the
        valid prefix so a follower can try again on its next poll; corruption in
        any non-trailing line is always rejected.
        """

        if retries < 0:
            raise ValueError("retries must be non-negative")
        if retry_interval < 0:
            raise ValueError("retry_interval must be non-negative")

        attempts = 0
        while True:
            try:
                return self._read_once(live=live)
            except _IncompleteTrailingLine as exc:
                if not live or attempts >= retries:
                    if live:
                        return exc.events
                    message = f"invalid JSON on events.jsonl line {exc.line_number}: {exc.cause}"
                    raise ValueError(message) from exc.cause
                attempts += 1
                time.sleep(retry_interval)

    def _read_once(self, *, live: bool) -> list[Event]:
        """Read one point-in-time view of the event log."""

        if not self.events_path.exists():
            raise FileNotFoundError(f"event log not found: {self.events_path}")

        events: list[Event] = []
        expected_seq = 0
        run_id: str | None = None
        with self.events_path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        last_nonempty_line = max(
            (line_number for line_number, line in enumerate(lines, start=1) if line.strip()),
            default=None,
        )
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                if live and line_number == last_nonempty_line and not line.endswith("\n"):
                    raise _IncompleteTrailingLine(events, line_number, exc) from exc
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
                    f"events.jsonl line {line_number} has seq {event.seq}; expected {expected_seq}"
                )
            expected_seq += 1
            events.append(event)
        if not events and not live:
            raise ValueError("event log is empty")
        return events

    def __iter__(self) -> Iterator[Event]:
        return iter(self.read())

    def tail(
        self,
        start_seq: int = 0,
        *,
        follow: bool = False,
        poll_interval: float = 0.1,
        retries: int = 5,
        retry_interval: float = 0.02,
    ) -> Iterator[Event]:
        """Yield events at or after ``start_seq``; follow when requested."""

        next_seq = start_seq
        while True:
            events = self.read(
                live=follow,
                retries=retries,
                retry_interval=retry_interval,
            )
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
