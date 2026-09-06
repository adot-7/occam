"""The minimal production run loop joining all v3 engine components."""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from occam.config.settings import project_root
from occam.core.models import Architecture, RunResult, Task
from occam.engine.ablation import FULL_REPEAT, ablate, noise_rate, select_ablation_subset
from occam.engine.architect import Architect
from occam.engine.baseline import BASELINE_MODEL, run_baseline
from occam.engine.diagnose import diagnose
from occam.engine.emit import writer_sink
from occam.engine.executor import Executor, resolve_grader
from occam.engine.mutate import apply_mutation
from occam.engine.pass3 import Pass3Result, run_pass3
from occam.llm.client import LLMClient
from occam.llm.config import ModelConfig, load_model_configs
from occam.memory.lessons import LessonStore
from occam.metrics.snapshot import BaselineSummary, build_snapshot, snapshot_event_data
from occam.store.writer import EventWriter
from occam.tasks.loader import TaskPack, load_task_pack
from occam.tools.fx import FXClient
from occam.tools.registry import ToolRegistry

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class RunConfig:
    """Bounded settings for one reproducible run."""

    run_name: str
    out: Path = Path("runs")
    memory: str | Path | None = None
    max_generations: int = 6
    n_cases: int | None = None
    ablate_cases: int = 10
    pass3: bool = False
    worker_model: str = BASELINE_MODEL
    architect_model: str = "architect"
    diagnose_model: str = "architect"

    def __post_init__(self) -> None:
        if not _RUN_NAME.fullmatch(self.run_name):
            raise ValueError("run-name must contain only letters, digits, '.', '_' or '-'")
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if self.n_cases is not None and self.n_cases < 1:
            raise ValueError("cases must be at least 1")
        if self.ablate_cases < 1:
            raise ValueError("ablate_cases must be at least 1")


@dataclass(frozen=True)
class RunOutcome:
    """Run artifact and judge-facing summary returned to the CLI."""

    run_dir: Path
    run_id: str
    best_generation: int
    summary: dict[str, Any]


def _public_models(configs: Mapping[str, ModelConfig]) -> dict[str, dict[str, Any]]:
    """Serialize model settings without ever putting a key in an event."""

    fields = (
        "provider",
        "model",
        "base_url",
        "in_per_m",
        "out_per_m",
        "grant_equiv_in_per_m",
        "grant_equiv_out_per_m",
        "rpm",
        "tpm",
        "supports_json_schema",
        "tool_choice_modes",
        "max_tokens",
    )
    result: dict[str, dict[str, Any]] = {}
    for key, config in configs.items():
        if isinstance(config, Mapping):
            result[key] = {field: config.get(field) for field in fields}
        else:
            result[key] = {field: getattr(config, field) for field in fields}
    return result


def _run_cost(run: RunResult) -> tuple[float, float]:
    displayed = sum(case.cost_usd for case in run.results)
    billed = sum(trace.billed_cost_usd for case in run.results for trace in case.per_role.values())
    return displayed, billed


def _score(result: RunResult) -> tuple[float, float]:
    """The PRD's lexicographic ``(pass_rate, -cost)`` score."""

    return result.pass_rate, -result.cost_usd


def _memory_label(task: Task, configured: str | Path | None) -> tuple[str, Path]:
    raw = configured if configured is not None else task.memory
    if not raw:
        raw = f"memory/{task.name}"
    path = Path(raw)
    absolute = path if path.is_absolute() else project_root() / path
    return str(raw), absolute


def _safe_run_dir(config: RunConfig) -> Path:
    destination = Path(config.out) / config.run_name
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run directory already exists and is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


class RunEngine:
    """Own one run directory and the engine/TUI event boundary."""

    def __init__(
        self,
        pack: TaskPack,
        config: RunConfig,
        *,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        grader: Any | None = None,
    ) -> None:
        self.pack = pack
        self.config = config
        self.llm = llm
        self.registry = registry
        self.grader = grader

    def run(self) -> RunOutcome:
        task = self.pack.task
        cases = self.pack.select(self.config.n_cases)
        run_dir = _safe_run_dir(self.config)
        run_id = self.config.run_name
        memory_ns, memory_path = _memory_label(task, self.config.memory)
        store = LessonStore(memory_path)
        lessons = store.load(active_only=True)

        llm = self.llm or LLMClient(cache_dir=run_dir / "cache")
        registry = self.registry or ToolRegistry(
            fx_client=FXClient(cache_dir=project_root() / "data" / "fx_cache")
        )
        grader = self.grader or resolve_grader(task.checker)
        initial_provider_calls = getattr(llm, "provider_call_count", None)
        initial_displayed_cost = getattr(llm, "displayed_cost_usd", None)
        initial_billed_cost = getattr(llm, "billed_cost_usd", None)
        writer = EventWriter(run_dir, run_id=run_id)
        sink = writer_sink(writer)
        shutil.copyfile(self.pack.directory / "task.yaml", run_dir / "task.yaml")

        config_event = {
            "models": _public_models(getattr(llm, "configs", None) or load_model_configs()),
            "budget_usd": None,
            "max_generations": self.config.max_generations,
            "cases": len(cases),
            "ablate_cases": min(self.config.ablate_cases, len(cases)),
            "pass3": self.config.pass3,
        }
        sink(
            "run.started",
            {
                "task": {
                    "name": task.name,
                    "domain": task.domain,
                    "n_cases": len(cases),
                    "goal": task.goal,
                },
                "config": config_event,
                "run_name": self.config.run_name,
                "memory_ns": memory_ns,
                "lessons_loaded": [lesson.model_dump(mode="json") for lesson in lessons],
            },
        )

        executor = Executor(
            llm=llm,
            tools=registry,
            grader=grader,
            writer=writer,
            run_dir=run_dir,
            run_name=self.config.run_name,
        )
        architectures: dict[int, Architecture] = {}
        full_runs: dict[int, RunResult] = {}
        metrics: dict[int, Any] = {}
        best_generation = 0
        best_architecture: Architecture | None = None
        best_score: tuple[float, float] | None = None
        non_improving = 0
        history: list[dict[str, Any]] = []
        all_displayed = 0.0
        all_billed = 0.0
        pass3_result: Pass3Result | None = None
        executor_results_seen = 0

        try:
            architect = Architect(
                llm=llm,
                registry=registry,
                memory=memory_path,
                model_key=self.config.architect_model,
                event_sink=sink,
            )
            architecture = architect.propose(
                task,
                generation=0,
                lessons=lessons,
                memory=memory_path,
            )
            for generation in range(self.config.max_generations):
                architectures[generation] = architecture
                full = executor.run_variant(
                    architecture,
                    cases,
                    variant="full",
                    generation=generation,
                    grader=grader,
                )
                full_runs[generation] = full

                subset = select_ablation_subset(cases, min(self.config.ablate_cases, len(cases)))
                repeat = executor.run_variant(
                    architecture,
                    subset,
                    variant=FULL_REPEAT,
                    use_cache=False,
                    generation=generation,
                    grader=grader,
                )
                measured_noise = noise_rate(full, repeat, [case.id for case in subset])

                baseline = run_baseline(
                    task,
                    cases,
                    full_cost_usd=full.cost_usd,
                    executor=executor,
                    run_dir=run_dir,
                    generation=generation,
                    model_key=self.config.worker_model,
                    grader=grader,
                )
                sink("baseline.completed", baseline.event_data(generation))
                all_displayed += baseline.cost_usd
                all_billed += baseline.billed_cost_usd

                table = ablate(
                    architecture,
                    subset,
                    runner=executor,
                    generation=generation,
                    full=full,
                    full_repeat=repeat,
                    measured_noise_rate=measured_noise,
                    sink=sink,
                )
                for result in executor.results_history[executor_results_seen:]:
                    displayed, billed = _run_cost(result)
                    all_displayed += displayed
                    all_billed += billed
                executor_results_seen = len(executor.results_history)
                score = _score(full)
                if best_score is None or score > best_score:
                    best_score = score
                    best_generation = generation
                    best_architecture = architecture
                    non_improving = 0
                else:
                    non_improving += 1
                history_entry = {
                    "generation": generation,
                    "pass_rate": full.pass_rate,
                    "cost_usd": full.cost_usd,
                    "tool_calls_per_case": sum(
                        len(trace.tool_calls)
                        for case in full.results
                        for trace in case.per_role.values()
                    )
                    / len(full.results)
                    if full.results
                    else 0.0,
                    "structural_fidelity": table.structural_fidelity,
                }
                history.append(history_entry)
                final = (
                    generation + 1 >= self.config.max_generations
                    or full.pass_rate >= 1.0
                    or non_improving >= 2
                )

                if final and self.config.pass3:
                    pass3_architecture = best_architecture or architecture
                    pass3_generation = best_generation
                    pass3_result = run_pass3(
                        pass3_architecture,
                        cases,
                        executor=executor,
                        generation=pass3_generation,
                        run_dir=run_dir,
                        grader=grader,
                    )
                    all_displayed += pass3_result.displayed_cost_usd
                    all_billed += pass3_result.billed_cost_usd

                baseline_summary = BaselineSummary(
                    pass_rate=baseline.pass_rate,
                    cost_usd=baseline.cost_usd,
                )
                snapshot = build_snapshot(
                    generation=generation,
                    full=full,
                    noise_rate=measured_noise,
                    structural_fidelity=table.structural_fidelity,
                    baseline=baseline_summary,
                    reliability_pass3=(
                        pass3_result.reliability_pass3
                        if pass3_result is not None and best_generation == generation
                        else None
                    ),
                )
                metrics[generation] = snapshot
                sink("metrics.snapshot", snapshot_event_data(snapshot))

                if final:
                    if pass3_result is not None:
                        sink("reliability.completed", pass3_result.event_data(best_generation))
                    break

                diagnosis = diagnose(
                    task,
                    generation=generation,
                    full=full,
                    table=table,
                    architecture=architecture,
                    cases=cases,
                    history=history,
                    lessons=store.load(active_only=True),
                    llm=llm,
                    lesson_store=store,
                    run_id=run_id,
                    event_sink=sink,
                    model_key=self.config.diagnose_model,
                )
                if diagnosis.rejected_lessons:
                    sink(
                        "log",
                        {
                            "level": "info",
                            "message": (
                                f"rejected {len(diagnosis.rejected_lessons)} lesson proposal(s) "
                                "by leak guard"
                            ),
                        },
                    )
                application = apply_mutation(
                    architecture,
                    diagnosis.mutation,
                    role_verdicts={row.role_id: row for row in table.rows},
                    available_tools=task.tools,
                    generation=generation + 1,
                    event_sink=sink,
                )
                architecture = application.architecture
                sink(
                    "architecture.proposed",
                    {
                        "generation": generation + 1,
                        "architecture": architecture.model_dump(mode="json"),
                    },
                )
        finally:
            writer.close()
            try:
                registry._fx_client.close() if getattr(registry, "_fx_client", None) else None
            except Exception:  # noqa: BLE001 - cleanup must not hide the run result
                pass

        final_generation = max(full_runs)
        final_metrics = metrics[final_generation]
        provider_calls = getattr(llm, "provider_call_count", None)
        if isinstance(provider_calls, int) and isinstance(initial_provider_calls, int):
            provider_calls -= initial_provider_calls
        displayed_cost = getattr(llm, "displayed_cost_usd", None)
        if isinstance(displayed_cost, (int, float)) and isinstance(
            initial_displayed_cost, (int, float)
        ):
            displayed_cost -= initial_displayed_cost
        else:
            displayed_cost = all_displayed
        billed_cost = getattr(llm, "billed_cost_usd", None)
        if isinstance(billed_cost, (int, float)) and isinstance(initial_billed_cost, (int, float)):
            billed_cost -= initial_billed_cost
        else:
            billed_cost = all_billed
        summary = {
            "final_pass_rate": final_metrics.pass_rate,
            "final_cost_usd": final_metrics.cost_usd,
            "calls_per_case_final": final_metrics.tool_calls_per_case,
            "generations": len(full_runs),
            "best_generation": best_generation,
            "provider_calls": provider_calls,
            "displayed_cost_usd": max(0.0, float(displayed_cost)),
            "billed_cost_usd": max(0.0, float(billed_cost)),
            "cost_label": "list-rate-equivalent where configured",
        }
        # The writer was closed in the cleanup block, so append completion with
        # a short-lived writer after all accounting is known.
        with EventWriter(run_dir, run_id=run_id) as completion_writer:
            completion_writer.append(
                {
                    "ts": _now_iso(),
                    "type": "run.completed",
                    "data": {"best_generation": best_generation, "summary": summary},
                }
            )
        return RunOutcome(
            run_dir=run_dir,
            run_id=run_id,
            best_generation=best_generation,
            summary=summary,
        )


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run_task(
    task: str | Path,
    config: RunConfig,
    *,
    root: str | Path | None = None,
    **kwargs: Any,
) -> RunOutcome:
    """Load a named/explicit pack and execute one full loop."""

    return RunEngine(load_task_pack(task, root=root), config, **kwargs).run()


def run(
    task: str | Path | TaskPack,
    *,
    run_name: str,
    **kwargs: Any,
) -> RunOutcome:
    """Convenience API matching the CLI's ``occam run`` vocabulary."""

    config = RunConfig(run_name=run_name, **kwargs)
    if isinstance(task, TaskPack):
        return RunEngine(task, config).run()
    return run_task(task, config)


__all__ = ["RunConfig", "RunEngine", "RunOutcome", "run", "run_task"]
