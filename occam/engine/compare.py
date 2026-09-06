"""Run-over-run comparison derived from immutable event logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from occam.store.reader import EventReader
from occam.store.reducer import reduce


def _generation(state: Any, number: int) -> Any:
    return state.generations.get(f"g{number:03d}")


def _summary(run_dir: str | Path) -> dict[str, Any]:
    events = EventReader(run_dir).read()
    state = reduce(events)
    generation_numbers = sorted(int(key[1:]) for key in state.generations)
    if not generation_numbers:
        raise ValueError(f"run has no generations: {run_dir}")
    first = _generation(state, generation_numbers[0])
    best_number = (
        state.best_generation if state.best_generation is not None else generation_numbers[-1]
    )
    final = _generation(state, best_number) or _generation(state, generation_numbers[-1])
    if first is None or final is None or first.metrics is None or final.metrics is None:
        raise ValueError(f"run has incomplete metrics: {run_dir}")
    n_cases = int((state.task or {}).get("n_cases", 0))
    first_architecture = first.architecture or {}
    final_architecture = final.architecture or {}
    reliability = final.metrics.reliability_pass3
    if reliability is None and final.reliability is not None:
        reliability = float(final.reliability.get("reliability_pass3", 0.0))
    return {
        "g0_pass_rate": first.metrics.pass_rate,
        "final_pass_rate": final.metrics.pass_rate,
        "g0_tool_calls_per_case": first.metrics.tool_calls_per_case,
        "g0_cost_per_case": first.metrics.cost_usd / n_cases if n_cases else 0.0,
        "generations_to_plateau": best_number,
        "roles_at_g0": len(first_architecture.get("roles", [])),
        "roles_final": len(final_architecture.get("roles", [])),
        "reliability_pass3": reliability,
        "lessons_loaded": len(state.lessons_loaded),
        "lessons_written": sum(1 for event in events if event.type == "lesson.written"),
    }


def compare_runs(
    run1: str | Path,
    run2: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Build the compare table and optionally persist it into run 2."""

    left = _summary(run1)
    right = _summary(run2)
    delta_keys = (
        "g0_pass_rate",
        "final_pass_rate",
        "g0_tool_calls_per_case",
        "g0_cost_per_case",
        "generations_to_plateau",
        "reliability_pass3",
    )
    delta: dict[str, Any] = {}
    for key in delta_keys:
        value = right.get(key)
        previous = left.get(key)
        if isinstance(value, (int, float)) and isinstance(previous, (int, float)):
            delta[key] = value - previous
        else:
            delta[key] = None
    payload = {"run1": left, "run2": right, "delta": delta}
    if write:
        destination = Path(run2) / "compare.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def format_compare(payload: dict[str, Any]) -> str:
    """Render a compact human table without adding a runtime dependency."""

    left, right, delta = payload["run1"], payload["run2"], payload["delta"]
    rows = [
        ("g0 pass rate", "g0_pass_rate"),
        ("final pass rate", "final_pass_rate"),
        ("g0 calls/case", "g0_tool_calls_per_case"),
        ("g0 cost/case", "g0_cost_per_case"),
        ("generations to plateau", "generations_to_plateau"),
        ("roles at g0 → final", "roles_at_g0"),
        ("pass³ (final)", "reliability_pass3"),
        ("lessons loaded", "lessons_loaded"),
        ("lessons written", "lessons_written"),
    ]
    lines = ["metric                    run1       run2          Δ"]
    for label, key in rows:
        if key == "roles_at_g0":
            left_value = f"{left[key]} → {left['roles_final']}"
            right_value = f"{right[key]} → {right['roles_final']}"
            delta_value = "—"
        else:
            left_value = _format_value(left.get(key))
            right_value = _format_value(right.get(key))
            delta_value = _format_value(delta.get(key))
        lines.append(f"{label:<26}{left_value:>10} {right_value:>10} {delta_value:>12}")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.5f}"
        return f"{value:.2f}"
    return str(value)


__all__ = ["compare_runs", "format_compare"]
