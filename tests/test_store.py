"""Focused event-store concurrency and live-tail regression tests."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from occam.store.reader import EventReader
from occam.store.writer import EventWriter


def _log_event(message: str) -> dict[str, object]:
    return {
        "ts": "2026-09-05T00:00:00Z",
        "type": "log",
        "data": {"level": "info", "message": message},
    }


def test_preopened_writers_allocate_distinct_sequences(tmp_path: Path) -> None:
    run_dir = tmp_path / "concurrent"
    writer_a = EventWriter(run_dir, run_id="concurrent_test")
    writer_b = EventWriter(run_dir, run_id="concurrent_test")
    barrier = threading.Barrier(3)

    def append_after_barrier(writer: EventWriter, message: str) -> None:
        barrier.wait()
        writer.append(_log_event(message))

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(append_after_barrier, writer_a, "writer-a"),
                executor.submit(append_after_barrier, writer_b, "writer-b"),
            ]
            barrier.wait()
            for future in futures:
                future.result()
    finally:
        writer_a.close()
        writer_b.close()

    events = EventReader(run_dir).read()
    assert [event.seq for event in events] == [0, 1]
    assert {event.data["message"] for event in events} == {"writer-a", "writer-b"}


def test_live_reader_retries_until_partial_tail_is_completed(tmp_path: Path) -> None:
    run_dir = tmp_path / "tail_retry"
    with EventWriter(run_dir, run_id="tail_test") as writer:
        writer.append(_log_event("first"))

    second = {
        "data": {"level": "info", "message": "second"},
        "run_id": "tail_test",
        "seq": 1,
        "ts": "2026-09-05T00:00:00Z",
        "type": "log",
    }
    second_line = json.dumps(second, separators=(",", ":")) + "\n"
    split_at = len(second_line) // 2
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(second_line[:split_at])

    def finish_append() -> None:
        time.sleep(0.03)
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(second_line[split_at:])

    finisher = threading.Thread(target=finish_append)
    finisher.start()
    reader = EventReader(run_dir)
    tail = reader.tail(
        follow=True,
        poll_interval=0.001,
        retries=20,
        retry_interval=0.01,
    )
    try:
        events = [next(tail), next(tail)]
    finally:
        tail.close()
        finisher.join()

    assert [event.seq for event in events] == [0, 1]


def test_live_reader_tolerates_persistent_partial_tail_but_strict_read_rejects(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "tail_partial"
    with EventWriter(run_dir, run_id="tail_test") as writer:
        writer.append(_log_event("first"))
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"run_id":"tail_test","seq":1')

    reader = EventReader(run_dir)
    assert [event.seq for event in reader.read(live=True, retries=0)] == [0]
    with pytest.raises(ValueError, match="invalid JSON"):
        reader.read()


def test_reader_rejects_corruption_in_the_middle_of_a_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "middle_corruption"
    with EventWriter(run_dir, run_id="tail_test") as writer:
        writer.append(_log_event("first"))
    second = {
        "data": {"level": "info", "message": "second"},
        "run_id": "tail_test",
        "seq": 1,
        "ts": "2026-09-05T00:00:00Z",
        "type": "log",
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
        handle.write(json.dumps(second, separators=(",", ":")) + "\n")

    with pytest.raises(ValueError, match="line 2"):
        EventReader(run_dir).read(live=True, retries=3, retry_interval=0)
