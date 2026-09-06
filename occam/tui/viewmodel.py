"""Read-only projections of a reduced :class:`State` for the TUI panels.

Nothing here touches the engine, an LLM, or the filesystem.  Panels receive a
:class:`RunView` and read derived values from it, so WP-09's detailed panels
never have to re-derive the same numbers out of raw event dictionaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from occam.core.models import Lesson, State

FULL_VARIANT = "full"
_RUN_INDEX = re.compile(r"(\d+)\s*$")


def _short_goal(goal: str) -> str:
    """Reduce a task goal to the short human label used in the header."""

    head = goal.split(":", 1)[0].split(".", 1)[0].strip()
    return head or goal.strip()


@dataclass(frozen=True)
class GenerationView:
    """Everything one generation contributes to the screen."""

    generation: int
    architecture: dict[str, Any] | None = None
    executions: dict[str, dict[str, Any]] = field(default_factory=dict)
    ablation: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    reliability: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    mutation: dict[str, Any] | None = None
    reverted: bool = False

    @property
    def label(self) -> str:
        return f"g{self.generation}"

    @property
    def roles(self) -> list[dict[str, Any]]:
        if not self.architecture:
            return []
        return list(self.architecture.get("roles", []))

    @property
    def role_map(self) -> dict[str, dict[str, Any]]:
        """Roles keyed by id for panels that join architecture and ablation data."""

        return {
            str(role.get("id")): role
            for role in self.roles
            if isinstance(role, dict) and role.get("id")
        }

    @property
    def n_roles(self) -> int:
        return len(self.roles)

    @property
    def full_execution(self) -> dict[str, Any] | None:
        return self.executions.get(FULL_VARIANT)

    @property
    def cases(self) -> list[dict[str, Any]]:
        """Per-case results of the ``full`` variant, in arrival order."""

        execution = self.full_execution
        return list(execution.get("cases", [])) if execution else []

    @property
    def n_cases(self) -> int:
        execution = self.full_execution
        return int(execution.get("n_cases", 0)) if execution else 0

    @property
    def cases_passed(self) -> int:
        return sum(1 for case in self.cases if case.get("passed"))

    @property
    def selected_case(self) -> dict[str, Any] | None:
        """The first failed case, or the first case when the generation is clean."""

        return next((case for case in self.cases if not case.get("passed")), None) or (
            self.cases[0] if self.cases else None
        )

    @property
    def ablation_progress(self) -> tuple[int, int]:
        """Rows observed and roles expected while the ablation table streams."""

        return len(self.ablation_rows), len(self.ablation_roles)

    @property
    def reliability_pass3(self) -> float | None:
        """The final-generation pass³ value, regardless of event arrival order."""

        if self.metrics is not None and self.metrics.get("reliability_pass3") is not None:
            return float(self.metrics["reliability_pass3"])
        if self.reliability is not None and self.reliability.get("reliability_pass3") is not None:
            return float(self.reliability["reliability_pass3"])
        return None

    @property
    def pass_rate(self) -> float | None:
        if self.metrics is not None and "pass_rate" in self.metrics:
            return float(self.metrics["pass_rate"])
        execution = self.full_execution
        if execution is not None and "pass_rate" in execution:
            return float(execution["pass_rate"])
        if self.cases:
            return self.cases_passed / len(self.cases)
        return None

    @property
    def cost_usd(self) -> float | None:
        """Cost of the full execution represented by this generation."""

        if self.metrics is not None and "cost_usd" in self.metrics:
            return float(self.metrics["cost_usd"])
        execution = self.full_execution
        if execution is not None and "cost_usd" in execution:
            return float(execution["cost_usd"])
        if execution is not None and execution.get("cases"):
            return sum(float(case.get("cost_usd", 0.0)) for case in execution["cases"])
        return None

    @property
    def total_spend_usd(self) -> float | None:
        """Observed spend for the generation, including repeat and baseline runs."""

        total = 0.0
        seen = False
        for execution in self.executions.values():
            if "cost_usd" in execution:
                total += float(execution["cost_usd"])
                seen = True
            elif execution.get("cases"):
                total += sum(float(case.get("cost_usd", 0.0)) for case in execution["cases"])
                seen = True
        if self.baseline is not None and "cost_usd" in self.baseline:
            total += float(self.baseline["cost_usd"])
            seen = True
        return total if seen else None

    @property
    def ablation_rows(self) -> list[dict[str, Any]]:
        if not self.ablation:
            return []
        return list(self.ablation.get("rows", []))

    @property
    def ablation_roles(self) -> list[str]:
        if not self.ablation:
            return []
        return list(self.ablation.get("roles", []))

    @property
    def witnesses(self) -> list[str]:
        if not self.ablation:
            return []
        return list(self.ablation.get("witnesses", []))

    @property
    def noise_rate(self) -> float | None:
        """Noise floor carried by ``ablation.started``, when the contract has it."""

        if not self.ablation or "noise_rate" not in self.ablation:
            return None
        try:
            return float(self.ablation["noise_rate"])
        except (TypeError, ValueError):
            return None

    @property
    def structural_fidelity(self) -> float | None:
        if self.ablation is not None and "structural_fidelity" in self.ablation:
            return float(self.ablation["structural_fidelity"])
        if self.metrics is not None and "structural_fidelity" in self.metrics:
            return float(self.metrics["structural_fidelity"])
        return None


class RunView:
    """One immutable snapshot of everything the screen needs to paint."""

    def __init__(
        self,
        state: State | None,
        *,
        selected_generation: int | None = None,
        selected_case_id: str | None = None,
        mode: str = "live",
        speed: float = 1.0,
        paused: bool = False,
        finished: bool = False,
        elapsed_s: float | None = None,
        status: str | None = None,
        compare: dict[str, Any] | None = None,
    ):
        self.state = state
        self.mode = mode
        self.speed = speed
        self.paused = paused
        self.finished = finished
        self.elapsed_s = elapsed_s
        self.status = status
        self.compare = compare
        self._generations = {
            snapshot.generation: GenerationView(
                generation=snapshot.generation,
                architecture=snapshot.architecture,
                executions=snapshot.executions,
                ablation=snapshot.ablation,
                baseline=snapshot.baseline,
                metrics=(snapshot.metrics.model_dump(mode="json") if snapshot.metrics else None),
                reliability=snapshot.reliability,
                diagnosis=snapshot.diagnosis,
                mutation=snapshot.mutation,
                reverted=snapshot.reverted,
            )
            for snapshot in (state.generations.values() if state else ())
        }
        self.selected_generation = self._resolve_selection(selected_generation)
        self.selected_case_id = self._resolve_case_selection(selected_case_id)

    def _resolve_selection(self, requested: int | None) -> int | None:
        if requested is not None and requested in self._generations:
            return requested
        if self.state is not None and self.state.current_generation in self._generations:
            return self.state.current_generation
        if self._generations:
            return max(self._generations)
        return None

    def _resolve_case_selection(self, requested: str | None) -> str | None:
        generation = self.selected
        if generation is None:
            return None
        if requested is not None and any(
            str(case.get("case_id")) == requested for case in generation.cases
        ):
            return requested
        selected = generation.selected_case
        return str(selected.get("case_id")) if selected else None

    # -- generations ---------------------------------------------------

    @property
    def generations(self) -> list[GenerationView]:
        return [self._generations[key] for key in sorted(self._generations)]

    @property
    def generation_numbers(self) -> list[int]:
        return sorted(self._generations)

    def generation(self, number: int | None) -> GenerationView | None:
        if number is None:
            return None
        return self._generations.get(number)

    @property
    def selected(self) -> GenerationView | None:
        return self.generation(self.selected_generation)

    @property
    def selected_case(self) -> dict[str, Any] | None:
        """The case highlighted by the shared case cursor."""

        if self.selected_case_id is not None:
            selected = self.case(self.selected_case_id)
            if selected is not None:
                return selected
        return self.selected.selected_case if self.selected is not None else None

    def case(self, case_id: str | None) -> dict[str, Any] | None:
        """Return a selected case without making panels know execution storage."""

        if case_id is None or self.selected is None:
            return None
        return next(
            (case for case in self.selected.cases if case.get("case_id") == case_id),
            None,
        )

    @property
    def max_generations(self) -> int | None:
        if self.state is None:
            return None
        configured = self.state.config.get("max_generations")
        if configured is None:
            return None
        return int(configured)

    @property
    def best_generation(self) -> int | None:
        return self.state.best_generation if self.state else None

    # -- run identity --------------------------------------------------

    @property
    def task_name(self) -> str:
        task = (self.state.task if self.state else None) or {}
        return str(task.get("name", ""))

    @property
    def task_label(self) -> str:
        """``fx_recon_a · FX revaluation`` — pack name plus a short human string."""

        task = (self.state.task if self.state else None) or {}
        name = str(task.get("name", ""))
        goal = _short_goal(str(task.get("goal", "")))
        if name and goal:
            return f"{name} · {goal}"
        return name or goal

    @property
    def run_index(self) -> int:
        """1 for a cold run, 2+ once earlier runs have left lessons behind."""

        run_name = self.state.run_name if self.state else ""
        match = _RUN_INDEX.search(run_name or "")
        if match:
            return max(1, int(match.group(1)))
        return 2 if self.lessons_loaded else 1

    @property
    def lessons(self) -> list[Lesson]:
        return list(self.state.lessons) if self.state else []

    @property
    def lessons_loaded(self) -> list[Lesson]:
        return list(self.state.lessons_loaded) if self.state else []

    @property
    def lessons_written(self) -> list[Lesson]:
        loaded = {lesson.id for lesson in self.lessons_loaded}
        return [lesson for lesson in self.lessons if lesson.id not in loaded]

    @property
    def spend_usd(self) -> float:
        return sum(generation.total_spend_usd or 0.0 for generation in self.generations)

    @property
    def completed(self) -> bool:
        return bool(self.state.completed) if self.state else False

    # -- header --------------------------------------------------------

    @property
    def run_label(self) -> str:
        """``run 1 · lessons 0`` or ``run 2/2 · lessons loaded 3`` (04 §1b)."""

        loaded = len(self.lessons_loaded)
        if loaded:
            return f"run {self.run_index}/{self.run_index} · lessons loaded {loaded}"
        return f"run {self.run_index} · lessons 0"

    @property
    def generation_label(self) -> str:
        selected = self.selected_generation
        if selected is None:
            return "g—"
        maximum = self.max_generations
        return f"g{selected}/g{maximum}" if maximum else f"g{selected}"

    @property
    def mode_badge(self) -> str:
        if self.completed or self.finished:
            return "■ DONE"
        if self.mode == "replay":
            if self.paused:
                return "⏸ REPLAY PAUSED"
            speed = f"{self.speed:g}"
            return f"⏵ REPLAY ×{speed}"
        return "● LIVE"


__all__ = ["FULL_VARIANT", "GenerationView", "RunView"]
