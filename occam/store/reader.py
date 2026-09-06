"""Readers for Occam's append-only event and snapshot files."""

from __future__ import annotations

import json
import os
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
        self._last_read_had_incomplete_trailing_line = False

    @property
    def last_read_had_incomplete_trailing_line(self) -> bool:
        """Whether the last live read returned before a partial final line."""

        return self._last_read_had_incomplete_trailing_line

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

        self._last_read_had_incomplete_trailing_line = False
        attempts = 0
        while True:
            try:
                return self._read_once(live=live)
            except _IncompleteTrailingLine as exc:
                if not live or attempts >= retries:
                    if live:
                        self._last_read_had_incomplete_trailing_line = True
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


class EventCursor:
    """Incrementally read an append-only event log from a byte offset.

    The first live read validates the complete available prefix.  Later reads
    validate only bytes appended after the cursor's offset while carrying the
    expected sequence, run id, and any incomplete trailing line forward.  A
    replacement or truncation is an error rather than an invitation to reset
    the cursor and silently replay a different log.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / "events.jsonl"
        self._offset = 0
        self._buffer = b""
        self._line_number = 0
        self._expected_seq = 0
        self._run_id: str | None = None
        self._identity: tuple[int, int] | None = None
        self._saw_event = False
        self._last_read_had_incomplete_trailing_line = False
        self._last_read_bytes = 0
        self._last_read_events = 0
        self._bytes_read = 0
        self._events_parsed = 0

    @property
    def offset(self) -> int:
        """Byte offset at the end of the data read from the file."""

        return self._offset

    @property
    def next_seq(self) -> int:
        """Sequence number expected from the next event line."""

        return self._expected_seq

    @property
    def run_id(self) -> str | None:
        """Run id established by the first validated event, if any."""

        return self._run_id

    @property
    def last_read_had_incomplete_trailing_line(self) -> bool:
        """Whether the last read ended with a buffered partial line."""

        return self._last_read_had_incomplete_trailing_line

    @property
    def last_read_bytes(self) -> int:
        """Number of newly appended bytes consumed by the last read call."""

        return self._last_read_bytes

    @property
    def last_read_events(self) -> int:
        """Number of events parsed by the last read call."""

        return self._last_read_events

    @property
    def bytes_read(self) -> int:
        """Cumulative bytes consumed from the event file."""

        return self._bytes_read

    @property
    def events_parsed(self) -> int:
        """Cumulative events parsed by the cursor."""

        return self._events_parsed

    def read(
        self,
        *,
        live: bool = False,
        retries: int = 5,
        retry_interval: float = 0.02,
    ) -> list[Event]:
        """Read newly appended events, retrying an incomplete final line."""

        if retries < 0:
            raise ValueError("retries must be non-negative")
        if retry_interval < 0:
            raise ValueError("retry_interval must be non-negative")

        self._last_read_had_incomplete_trailing_line = False
        self._last_read_bytes = 0
        self._last_read_events = 0
        collected: list[Event] = []
        attempts = 0
        while True:
            events, incomplete = self._read_once(live=live)
            collected.extend(events)
            if not incomplete:
                if not live and not self._saw_event:
                    raise ValueError("event log is empty")
                return collected
            if not live or attempts >= retries:
                self._last_read_had_incomplete_trailing_line = True
                if live:
                    return collected
                raise AssertionError("unreachable")  # pragma: no cover
            attempts += 1
            time.sleep(retry_interval)

    def _read_once(self, *, live: bool) -> tuple[list[Event], bool]:
        appended = self._read_appended_bytes()
        if not appended and not self._buffer:
            if not live and not self._saw_event:
                raise ValueError("event log is empty")
            return [], False
        if not appended and self._buffer and live:
            return [], True

        data = self._buffer + appended
        self._buffer = b""
        lines = data.split(b"\n")
        has_terminal_newline = data.endswith(b"\n")
        complete_lines = lines if has_terminal_newline else lines[:-1]
        remainder = b"" if has_terminal_newline else lines[-1]
        events: list[Event] = []

        for line in complete_lines:
            self._line_number += 1
            if line.strip():
                events.append(self._parse_line(line, self._line_number, allow_incomplete=False))

        if remainder.strip():
            try:
                self._line_number += 1
                events.append(self._parse_line(remainder, self._line_number, allow_incomplete=True))
            except _CursorIncompleteLine as exc:
                self._line_number -= 1
                self._buffer = remainder
                if live:
                    return events, True
                raise ValueError(
                    f"invalid JSON on events.jsonl line {exc.line_number}: {exc.cause}"
                ) from exc.cause

        return events, False

    def _read_appended_bytes(self) -> bytes:
        """Read from the previous offset and reject file identity changes."""

        try:
            with self.events_path.open("rb") as handle:
                metadata = os.fstat(handle.fileno())
                identity = (metadata.st_dev, metadata.st_ino)
                self._check_identity(identity, metadata.st_size)
                handle.seek(self._offset)
                appended = handle.read()
                new_offset = handle.tell()
                after_read = os.fstat(handle.fileno())
                if (after_read.st_dev, after_read.st_ino) != identity:
                    raise ValueError("events.jsonl was replaced while it was being read")
                if after_read.st_size < new_offset:
                    raise ValueError("events.jsonl was truncated while it was being read")
                try:
                    path_metadata = self.events_path.stat()
                except FileNotFoundError as exc:
                    raise ValueError("events.jsonl was replaced while it was being read") from exc
                if (path_metadata.st_dev, path_metadata.st_ino) != identity:
                    raise ValueError("events.jsonl was replaced while it was being read")
        except FileNotFoundError:
            raise

        self._offset = new_offset
        self._last_read_bytes += len(appended)
        self._bytes_read += len(appended)
        return appended

    def _check_identity(self, identity: tuple[int, int], size: int) -> None:
        if self._identity is None:
            self._identity = identity
        elif identity != self._identity:
            raise ValueError("events.jsonl was replaced while following it")
        if size < self._offset:
            raise ValueError("events.jsonl was truncated while following it")

    def _parse_line(self, line: bytes, line_number: int, *, allow_incomplete: bool) -> Event:
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            if allow_incomplete:
                raise _CursorIncompleteLine(line_number, exc) from exc
            raise ValueError(f"invalid UTF-8 on events.jsonl line {line_number}: {exc}") from exc
        try:
            payload: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            if allow_incomplete:
                raise _CursorIncompleteLine(line_number, exc) from exc
            raise ValueError(f"invalid JSON on events.jsonl line {line_number}: {exc}") from exc
        validate_event(payload)
        try:
            event = Event.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"invalid event on events.jsonl line {line_number}: {exc}") from exc
        if self._run_id is None:
            self._run_id = event.run_id
        elif event.run_id != self._run_id:
            raise ValueError(f"events.jsonl line {line_number} changes run_id")
        if event.seq != self._expected_seq:
            raise ValueError(
                f"events.jsonl line {line_number} has seq {event.seq}; "
                f"expected {self._expected_seq}"
            )
        self._expected_seq += 1
        self._saw_event = True
        self._events_parsed += 1
        self._last_read_events += 1
        return event


class _CursorIncompleteLine(Exception):
    """Signal that a cursor's final line is still being written."""

    def __init__(self, line_number: int, cause: UnicodeDecodeError | json.JSONDecodeError):
        self.line_number = line_number
        self.cause = cause
        super().__init__(str(cause))


__all__ = ["EventCursor", "EventReader"]
