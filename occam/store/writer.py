"""Append-only event writer with a derived state snapshot."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import IO, Any

from occam.core.models import Event
from occam.store.reader import EventReader
from occam.store.reducer import reduce, state_json_bytes
from occam.store.schema import validate_event, validate_state

if sys.platform == "win32":  # pragma: no cover - platform specific
    import msvcrt

    def _lock(handle: IO[str]) -> None:
        """Take an exclusive lock on the first byte of the lock file."""

        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                # LK_LOCK gives up after ~10s; keep waiting like flock does.
                time.sleep(0.05)

    def _unlock(handle: IO[str]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:  # pragma: no cover - platform specific
    import fcntl

    def _lock(handle: IO[str]) -> None:
        """Take an exclusive lock on the whole lock file."""

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def _unlock(handle: IO[str]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class EventWriter:
    """Write validated events and refresh ``state.json`` after every append."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        existing = self._read_existing()
        self._next_seq = existing[-1].seq + 1 if existing else 0
        if run_id is not None and not run_id:
            raise ValueError("run_id must be non-empty")
        self.run_id = run_id if run_id is not None else (existing[0].run_id if existing else None)
        if existing and run_id is not None and run_id != existing[0].run_id:
            raise ValueError("run_id does not match the existing event log")
        self.lock_path = self.run_dir / "events.jsonl.lock"
        self._lock_handle = self.lock_path.open("a+", encoding="utf-8")
        self._thread_lock = RLock()
        self._closed = False

    def append(self, event: Event | Mapping[str, Any]) -> Event:
        """Append one event, assigning sequence/run id for a plain mapping."""

        with self._append_lock():
            existing = self._read_existing()
            self._next_seq = existing[-1].seq + 1 if existing else 0
            if existing:
                existing_run_id = existing[0].run_id
                if self.run_id is None:
                    self.run_id = existing_run_id
                elif self.run_id != existing_run_id:
                    raise ValueError("run_id does not match the existing event log")

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

    def _read_existing(self) -> list[Event]:
        """Read an existing log, allowing a newly created zero-byte file."""

        if not self.events_path.exists() or self.events_path.stat().st_size == 0:
            return []
        return EventReader(self.run_dir).read()

    @contextmanager
    def _append_lock(self) -> Iterator[None]:
        """Coordinate sequence allocation and the following append."""

        with self._thread_lock:
            if self._closed:
                raise ValueError("event writer is closed")
            _lock(self._lock_handle)
            try:
                yield
            finally:
                _unlock(self._lock_handle)

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

    def close(self) -> None:
        """Close this writer's inter-process lock handle."""

        with self._thread_lock:
            if not self._closed:
                self._lock_handle.close()
                self._closed = True

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
