"""Offline regression coverage for bounded baseline execution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from occam.engine.baseline import run_baseline
from occam.engine.executor import Executor
from occam.llm.client import Completion
from occam.llm.config import ModelConfig
from occam.tasks.loader import load_task_pack
from occam.tools.registry import ToolRegistry


class FakeLLM:
    """A local completion double; it never crosses a provider boundary."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0
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
        self.calls += 1
        if self.delay_s:
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


def _offline_task_and_cases() -> tuple[Any, list[Any]]:
    pack = load_task_pack("fx_recon_a")
    # The baseline contract is what is under test; no tool binding is needed
    # for a local answer double.
    return pack.task.model_copy(update={"tools": []}), pack.select(1)


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
