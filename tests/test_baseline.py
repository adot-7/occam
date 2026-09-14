"""Offline regression coverage for bounded baseline execution."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from occam.engine.baseline import run_baseline
from occam.engine.executor import Executor
from occam.llm.client import Completion
from occam.llm.config import ModelConfig
from occam.tasks.loader import load_task_pack
from occam.tools.registry import ToolRegistry


class FakeLLM:
    """A local completion double; it never crosses a provider boundary."""

    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        started_event: threading.Event | None = None,
        release_event: threading.Event | None = None,
    ) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.started_event = started_event
        self.release_event = release_event
        self._stats_lock = threading.Lock()
        self.configs = {
            "worker_fast": ModelConfig(
                key="worker_fast",
                provider="stub",
                model="offline-worker",
                api_key="offline",
                rpm=60,
            )
        }

    def complete(self, *_args: Any, **_kwargs: Any) -> Completion:
        with self._stats_lock:
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        if self.started_event is not None:
            self.started_event.set()
        try:
            if self.release_event is not None:
                if not self.release_event.wait(timeout=2.0):
                    raise AssertionError("fake release gate was not opened")
            elif self.delay_s:
                time.sleep(self.delay_s)
            return Completion(
                text="offline answer",
                tool_calls=[],
                tokens_in=2,
                tokens_out=3,
                cost_usd=0.01,
                billed_cost_usd=0.0,
                latency_s=self.delay_s,
                cached=False,
            )
        finally:
            with self._stats_lock:
                self.active_calls -= 1


class LateReturningExecutor(Executor):
    """Local executor double that crosses the deadline after case work."""

    return_gate: ClassVar[threading.Event | None] = None

    def run_variant(self, *args: Any, **kwargs: Any) -> Any:
        phase_deadline = self.phase_deadline_s
        self.phase_deadline_s = None
        try:
            result = super().run_variant(*args, **kwargs)
        finally:
            self.phase_deadline_s = phase_deadline
        gate = type(self).return_gate
        if gate is not None and not gate.wait(timeout=2.0):
            raise AssertionError("late-return gate was not opened")
        return result


def _run_in_thread(
    function: Callable[[], Any],
) -> tuple[threading.Thread, threading.Event, dict[str, Any]]:
    """Run a synchronous baseline probe while the test controls its provider gate."""

    done = threading.Event()
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = function()
        except BaseException as exc:  # noqa: BLE001 - surface worker failures in the test
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=target)
    thread.start()
    return thread, done, outcome


def _offline_task_and_cases(count: int = 1) -> tuple[Any, list[Any]]:
    pack = load_task_pack("fx_recon_a")
    # The baseline contract is what is under test; no tool binding is needed
    # for a local answer double.
    return pack.task.model_copy(update={"tools": []}), pack.select(count)


def test_baseline_timeout_persists_partial_case_and_truthful_cost(tmp_path: Path) -> None:
    task, cases = _offline_task_and_cases()
    llm = FakeLLM(delay_s=0.05)
    executor = Executor(
        llm=llm,
        tools=ToolRegistry(),
        case_concurrency=1,
        model_concurrency=1,
    )
    events: list[tuple[str, dict[str, Any]]] = []

    result = run_baseline(
        task,
        cases,
        full_cost_usd=1.0,
        executor=executor,
        run_dir=tmp_path / "run",
        phase_timeout_s=0.5,
        case_timeout_s=0.01,
        completion_timeout_s=0.2,
        completion_max_attempts=1,
        event_sink=lambda event_type, data: events.append((event_type, dict(data))),
    )

    assert result.complete is False
    assert result.status == "incomplete"
    assert result.reason == "case_timeout"
    assert result.completed_case_count == 0
    assert result.total_case_count == 1
    assert result.cost_usd == 0.0
    assert result.billed_cost_usd == 0.0
    progress = tmp_path / "run" / "baseline" / "progress" / "sample-001.results.jsonl"
    assert progress.exists()
    persisted = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert persisted[0]["case_id"] == cases[0].id
    assert persisted[0]["role_error"].startswith("TimeoutError: execution case exceeded")
    assert not (tmp_path / "run" / "baseline" / "results.jsonl").exists()
    assert events
    assert events[-1][0] == "log"
    assert "provider" not in events[-1][1]["message"].lower()


def test_baseline_phase_deadline_covers_queued_case_waves(tmp_path: Path) -> None:
    task, cases = _offline_task_and_cases(3)
    started_event = threading.Event()
    release_event = threading.Event()
    phase_progress_event = threading.Event()
    llm = FakeLLM(
        delay_s=0.2,
        started_event=started_event,
        release_event=release_event,
    )
    executor = Executor(
        llm=llm,
        tools=ToolRegistry(),
        case_concurrency=1,
        model_concurrency=1,
    )
    events: list[tuple[str, dict[str, Any]]] = []

    def event_sink(event_type: str, data: Any) -> None:
        event_data = dict(data)
        events.append((event_type, event_data))
        if "timed_out_cases=3" in event_data.get("message", ""):
            phase_progress_event.set()

    thread, done, outcome = _run_in_thread(
        lambda: run_baseline(
            task,
            cases,
            full_cost_usd=1.0,
            executor=executor,
            run_dir=tmp_path / "run",
            phase_timeout_s=0.15,
            case_timeout_s=0.1,
            completion_timeout_s=0.5,
            completion_max_attempts=1,
            event_sink=event_sink,
        )
    )

    assert started_event.wait(timeout=2.0)
    # All three cases have received their phase-deadline result, while the
    # first synchronous call is still gated.  A cancelled to_thread task must
    # not release either permit and start another provider call here.
    assert phase_progress_event.wait(timeout=2.0)
    assert llm.calls == 1
    assert llm.max_active_calls == 1
    assert llm.active_calls == 1
    assert not done.is_set()

    release_event.set()
    assert done.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert "error" not in outcome
    result = outcome["result"]
    assert llm.calls == 1
    assert llm.max_active_calls == 1
    assert llm.active_calls == 0
    assert result.complete is False
    assert result.status == "incomplete"
    assert result.reason == "phase_timeout"
    assert result.completed_case_count == 0
    assert result.total_case_count == 3
    progress = tmp_path / "run" / "baseline" / "progress" / "sample-001.results.jsonl"
    persisted = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert len(persisted) == 3
    assert all(
        row["role_error"].startswith("TimeoutError: execution case exceeded") for row in persisted
    )
    assert events[-1][1]["message"].startswith("Baseline progress: sample=1; completed_cases=0/3;")


def test_baseline_cancels_model_waiter_at_phase_deadline(tmp_path: Path) -> None:
    task, cases = _offline_task_and_cases(2)
    started_event = threading.Event()
    release_event = threading.Event()
    phase_progress_event = threading.Event()
    llm = FakeLLM(
        delay_s=0.2,
        started_event=started_event,
        release_event=release_event,
    )
    executor = Executor(
        llm=llm,
        tools=ToolRegistry(),
        case_concurrency=2,
        model_concurrency=1,
    )

    def event_sink(event_type: str, data: Any) -> None:
        if "timed_out_cases=2" in data.get("message", ""):
            phase_progress_event.set()

    thread, done, outcome = _run_in_thread(
        lambda: run_baseline(
            task,
            cases,
            full_cost_usd=1.0,
            executor=executor,
            run_dir=tmp_path / "run",
            phase_timeout_s=0.05,
            case_timeout_s=0.1,
            completion_timeout_s=0.5,
            completion_max_attempts=1,
            event_sink=event_sink,
        )
    )

    assert started_event.wait(timeout=2.0)
    # Case two has acquired its case permit but is waiting on the sole model
    # permit.  Both cases have received their phase result while case one is
    # still inside the gated synchronous call.
    assert phase_progress_event.wait(timeout=2.0)
    assert llm.calls == 1
    assert llm.max_active_calls == 1
    assert llm.active_calls == 1
    assert not done.is_set()

    release_event.set()
    assert done.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert "error" not in outcome
    result = outcome["result"]
    assert llm.calls == 1
    assert llm.max_active_calls == 1
    assert llm.active_calls == 0
    assert result.complete is False
    assert result.status == "incomplete"
    assert result.reason == "phase_timeout"
    assert result.completed_case_count == 0
    assert result.total_case_count == 2
    assert result.cost_usd == 0.0
    assert not (tmp_path / "run" / "baseline" / "results.jsonl").exists()


def test_baseline_two_case_probe_returns_by_phase_deadline(tmp_path: Path) -> None:
    task, cases = _offline_task_and_cases(2)
    started_event = threading.Event()
    release_event = threading.Event()
    phase_progress_event = threading.Event()
    llm = FakeLLM(
        delay_s=0.06,
        started_event=started_event,
        release_event=release_event,
    )
    executor = Executor(
        llm=llm,
        tools=ToolRegistry(),
        case_concurrency=1,
        model_concurrency=1,
    )

    def event_sink(event_type: str, data: Any) -> None:
        if "timed_out_cases=2" in data.get("message", ""):
            phase_progress_event.set()

    thread, done, outcome = _run_in_thread(
        lambda: run_baseline(
            task,
            cases,
            full_cost_usd=0.02,
            executor=executor,
            run_dir=tmp_path / "run",
            phase_timeout_s=0.10,
            case_timeout_s=0.10,
            completion_timeout_s=0.5,
            completion_max_attempts=1,
            event_sink=event_sink,
        )
    )

    assert started_event.wait(timeout=2.0)
    assert phase_progress_event.wait(timeout=2.0)
    assert llm.calls == 1
    assert llm.active_calls == 1
    assert not done.is_set()

    release_event.set()
    assert done.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert "error" not in outcome
    result = outcome["result"]
    assert llm.calls == 1
    assert result.complete is False
    assert result.status == "incomplete"
    assert result.reason == "phase_timeout"
    assert result.completed_case_count == 0
    assert result.total_case_count == 2
    assert llm.max_active_calls == 1
    assert llm.active_calls == 0
    assert not (tmp_path / "run" / "baseline" / "results.jsonl").exists()


def test_baseline_waits_for_slow_active_case_before_completion(tmp_path: Path) -> None:
    task, cases = _offline_task_and_cases()
    started_event = threading.Event()
    release_event = threading.Event()
    phase_progress_event = threading.Event()
    llm = FakeLLM(
        delay_s=0.5,
        started_event=started_event,
        release_event=release_event,
    )
    executor = Executor(
        llm=llm,
        tools=ToolRegistry(),
        case_concurrency=1,
        model_concurrency=1,
    )

    def event_sink(event_type: str, data: Any) -> None:
        if "timed_out_cases=1" in data.get("message", ""):
            phase_progress_event.set()

    thread, done, outcome = _run_in_thread(
        lambda: run_baseline(
            task,
            cases,
            full_cost_usd=1.0,
            executor=executor,
            run_dir=tmp_path / "run",
            phase_timeout_s=0.05,
            case_timeout_s=0.1,
            completion_timeout_s=0.5,
            completion_max_attempts=1,
            event_sink=event_sink,
        )
    )

    assert started_event.wait(timeout=2.0)
    assert phase_progress_event.wait(timeout=2.0)
    assert llm.calls == 1
    assert llm.max_active_calls == 1
    assert llm.active_calls == 1
    assert not done.is_set()

    release_event.set()
    assert done.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert "error" not in outcome
    result = outcome["result"]
    assert llm.calls == 1
    assert llm.active_calls == 0
    assert result.complete is False
    assert result.status == "incomplete"
    assert result.reason == "phase_timeout"
    assert result.completed_case_count == 0
    assert result.total_case_count == 1
    assert result.cost_usd == 0.0
    assert not (tmp_path / "run" / "baseline" / "results.jsonl").exists()


def test_baseline_rejects_sample_returned_after_phase_deadline(tmp_path: Path) -> None:
    task, cases = _offline_task_and_cases(2)
    llm = FakeLLM(delay_s=0.06)
    sample_done_event = threading.Event()
    return_gate = threading.Event()
    LateReturningExecutor.return_gate = return_gate
    executor = LateReturningExecutor(
        llm=llm,
        tools=ToolRegistry(),
        case_concurrency=1,
        model_concurrency=1,
    )

    def event_sink(event_type: str, data: Any) -> None:
        if "completed_cases=2/2" in data.get("message", ""):
            sample_done_event.set()

    thread: threading.Thread | None = None
    try:
        thread, done, outcome = _run_in_thread(
            lambda: run_baseline(
                task,
                cases,
                full_cost_usd=0.02,
                executor=executor,
                run_dir=tmp_path / "run",
                phase_timeout_s=0.10,
                case_timeout_s=0.10,
                completion_timeout_s=0.5,
                completion_max_attempts=1,
                event_sink=event_sink,
            )
        )

        # The sample has finished its cases, but the executor double holds its
        # return until the test opens the gate.  This gives the test a stable
        # point after the phase deadline without sleeping in the assertion.
        assert sample_done_event.wait(timeout=2.0)
        assert llm.calls == 2
        assert llm.active_calls == 0
        assert not done.is_set()

        return_gate.set()
        assert done.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert "error" not in outcome
        result = outcome["result"]
        assert result.complete is False
        assert result.status == "incomplete"
        assert result.reason == "phase_timeout"
        assert result.completed_case_count == 2
        assert result.total_case_count == 2
        assert result.cost_usd == 0.02
        assert result.billed_cost_usd == 0.0
        assert not (tmp_path / "run" / "baseline" / "results.jsonl").exists()
        progress = tmp_path / "run" / "baseline" / "progress" / "sample-001.results.jsonl"
        assert len(progress.read_text(encoding="utf-8").splitlines()) == 2
    finally:
        return_gate.set()
        if thread is not None:
            thread.join(timeout=2.0)
        LateReturningExecutor.return_gate = None


def test_completed_baseline_keeps_final_results_and_progress(tmp_path: Path) -> None:
    task, cases = _offline_task_and_cases()
    llm = FakeLLM()
    executor = Executor(
        llm=llm,
        tools=ToolRegistry(),
        case_concurrency=1,
        model_concurrency=1,
    )

    result = run_baseline(
        task,
        cases,
        full_cost_usd=0.01,
        executor=executor,
        run_dir=tmp_path / "run",
        phase_timeout_s=1.0,
        case_timeout_s=0.5,
        completion_timeout_s=0.2,
        completion_max_attempts=1,
    )

    assert result.complete is True
    assert result.k == 1
    assert result.cost_usd == 0.01
    assert (tmp_path / "run" / "baseline" / "results.jsonl").exists()
    assert (tmp_path / "run" / "baseline" / "progress" / "sample-001.results.jsonl").exists()
