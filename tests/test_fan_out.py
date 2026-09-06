"""``fan_out`` runs the calling role's prompt on each subtask, concurrently."""

from __future__ import annotations

import threading
import time

import pytest

from occam.tools.fan_out import FanOutUnavailableError, fan_out


def test_results_come_back_in_subtask_order_not_completion_order() -> None:
    def runner(subtask: str) -> str:
        time.sleep(0.05 if subtask == "a" else 0.0)
        return subtask.upper()

    assert fan_out(["a", "b", "c"], runner=runner) == ["A", "B", "C"]


def test_subtasks_actually_run_concurrently() -> None:
    barrier = threading.Barrier(3, timeout=5.0)

    def runner(subtask: str) -> str:
        barrier.wait()
        return subtask

    assert fan_out(["a", "b", "c"], runner=runner, max_workers=3) == ["a", "b", "c"]


def test_concurrency_is_bounded_by_max_workers() -> None:
    guard = threading.Lock()
    live = 0
    peak = 0

    def runner(subtask: str) -> str:
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        time.sleep(0.02)
        with guard:
            live -= 1
        return subtask

    fan_out([str(index) for index in range(8)], runner=runner, max_workers=2)

    assert peak <= 2


def test_one_failing_subtask_does_not_fail_the_batch() -> None:
    def runner(subtask: str) -> str:
        if subtask == "bad":
            raise RuntimeError("boom")
        return subtask

    results = fan_out(["ok", "bad", "fine"], runner=runner)

    assert results[0] == "ok"
    assert results[2] == "fine"
    assert "fan_out error on subtask 2" in results[1]
    assert "boom" in results[1]


def test_an_empty_batch_needs_no_workers() -> None:
    def runner(_: str) -> str:  # pragma: no cover - must not be called
        raise AssertionError("runner should not run for an empty batch")

    assert fan_out([], runner=runner) == []


def test_calling_fan_out_outside_a_role_is_an_error() -> None:
    with pytest.raises(FanOutUnavailableError):
        fan_out(["a"], runner=None)


def test_a_bare_string_is_not_a_list_of_subtasks() -> None:
    with pytest.raises(TypeError):
        fan_out("not a list", runner=str.upper)


def test_the_batch_size_is_capped() -> None:
    with pytest.raises(ValueError, match="at most 3 subtasks"):
        fan_out(["a", "b", "c", "d"], runner=str.upper, max_subtasks=3)
