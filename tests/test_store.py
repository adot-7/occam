"""Focused event-store concurrency and live-tail regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from occam.store.reader import EventReader
from occam.store.writer import EventWriter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# One process's worth of appends, run in a real subprocess so the writer's file
# lock is exercised across processes rather than across threads of one process.
_APPENDER = """
import sys

from occam.store.writer import EventWriter

run_dir, run_id, worker, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
writer = EventWriter(run_dir, run_id=run_id)
try:
    for index in range(count):
        writer.append(
            {
                "ts": "2026-09-05T00:00:00Z",
                "type": "log",
                "data": {"level": "info", "message": f"{worker}-{index}"},
            }
        )
finally:
    writer.close()
"""


def _subprocess_env() -> dict[str, str]:
    """Make checkout imports explicit for scripts launched outside the repo."""

    package_init = _PROJECT_ROOT / "occam" / "__init__.py"
    if not package_init.is_file():
        raise RuntimeError(f"validated project root is missing {package_init}")

    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(_PROJECT_ROOT), existing_path) if path
    )
    return environment


def _log_event(message: str) -> dict[str, object]:
    return {
        "ts": "2026-09-05T00:00:00Z",
        "type": "log",
        "data": {"level": "info", "message": message},
    }


def test_occam_store_imports_in_a_fresh_interpreter_on_this_platform() -> None:
    """``occam.store`` must import everywhere the engine and TUI run.

    A fresh interpreter is the point: an already-imported module would hide a
    top-level platform-only import, which is exactly how a POSIX-only ``fcntl``
    import reached main and made three test modules uncollectable on Windows.
    """

    completed = subprocess.run(
        [sys.executable, "-c", "import occam.store; print(occam.store.EventWriter.__name__)"],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "EventWriter"


def test_subprocess_appender_imports_checkout_and_appends(tmp_path: Path) -> None:
    """The appender child must both import the checkout and write an event."""

    run_dir = tmp_path / "subprocess_append"
    run_dir.mkdir()
    script = tmp_path / "appender.py"
    script.write_text(textwrap.dedent(_APPENDER), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script), str(run_dir), "subprocess_test", "child", "1"],
        capture_output=True,
        env=_subprocess_env(),
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["data"]["message"] == "child-0"


def test_concurrent_processes_append_without_interleaving_or_corruption(tmp_path: Path) -> None:
    """Two processes appending at once must produce one clean, ordered log.

    Threads share the writer's lock handle; separate processes do not, so this
    is the case the file lock actually exists for.
    """

    run_dir = tmp_path / "cross_process"
    run_dir.mkdir()
    script = tmp_path / "appender.py"
    script.write_text(textwrap.dedent(_APPENDER), encoding="utf-8")
    per_worker = 8

    processes = [
        subprocess.Popen(
            [sys.executable, str(script), str(run_dir), "concurrent_test", worker, str(per_worker)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_subprocess_env(),
            text=True,
        )
        for worker in ("alpha", "beta")
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr

    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * per_worker
    # Every line is whole: no torn or interleaved writes.
    payloads = [json.loads(line) for line in lines]
    assert [payload["seq"] for payload in payloads] == list(range(2 * per_worker))
    assert {payload["data"]["message"] for payload in payloads} == {
        f"{worker}-{index}" for worker in ("alpha", "beta") for index in range(per_worker)
    }

    events = EventReader(run_dir).read()
    assert [event.seq for event in events] == list(range(2 * per_worker))
    assert json.loads((run_dir / "state.json").read_text(encoding="utf-8"))["last_seq"] == (
        2 * per_worker - 1
    )


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
