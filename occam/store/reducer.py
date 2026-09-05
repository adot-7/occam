"""Pure reduction of the append-only event stream into a run snapshot."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from occam.core.models import Event, GenerationState, State


def _generation_key(generation: int) -> str:
    return f"g{generation:03d}"


def _copy_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy event data while converting any nested Pydantic models to JSON values."""

    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, BaseModel):
            result[key] = value.model_dump(mode="json")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _generation(state: State, generation: int) -> GenerationState:
    key = _generation_key(generation)
    if key not in state.generations:
        state.generations[key] = GenerationState(generation=generation)
    return state.generations[key]


def reduce(events: Iterable[Event | Mapping[str, Any]]) -> State:
    """Apply events in order and return a deterministic state snapshot.

    The reducer has no clock, filesystem, or provider dependencies. It accepts
    either validated :class:`Event` objects or their serialized mappings so the
    engine and the TUI can share exactly one state transition function.
    """

    materialized = list(events)
    if not materialized:
        return State(run_id="")

    first = materialized[0]
    first_event = first if isinstance(first, Event) else Event.model_validate(first)
    state = State(run_id=first_event.run_id)
    expected_seq = 0

    for raw_event in materialized:
        event = raw_event if isinstance(raw_event, Event) else Event.model_validate(raw_event)
        if event.run_id != state.run_id:
            raise ValueError(
                f"event seq {event.seq} belongs to run {event.run_id!r}, expected {state.run_id!r}"
            )
        if event.seq != expected_seq:
            raise ValueError(f"expected event seq {expected_seq}, got {event.seq}")
        expected_seq += 1
        data = _copy_data(event.data)
        event_type = event.type

        if event_type == "run.started":
            state.task = data["task"]
            state.config = data["config"]
        elif event_type == "architecture.proposed":
            generation = int(data["generation"])
            current = _generation(state, generation)
            current.architecture = data["architecture"]
            state.current_generation = generation
        elif event_type == "execution.started":
            generation = int(data["generation"])
            variant = str(data["variant"])
            current = _generation(state, generation)
            current.executions[variant] = {
                "generation": generation,
                "variant": variant,
                "n_cases": int(data["n_cases"]),
                "cases": [],
            }
            state.current_generation = generation
        elif event_type == "execution.case":
            generation = int(data["generation"])
            variant = str(data["variant"])
            current = _generation(state, generation)
            execution = current.executions.setdefault(
                variant,
                {"generation": generation, "variant": variant, "n_cases": 0, "cases": []},
            )
            execution.setdefault("cases", []).append(data)
        elif event_type == "execution.completed":
            generation = int(data["generation"])
            variant = str(data["variant"])
            current = _generation(state, generation)
            execution = current.executions.setdefault(
                variant,
                {"generation": generation, "variant": variant, "n_cases": 0, "cases": []},
            )
            execution.update(data)
        elif event_type == "ablation.started":
            generation = int(data["generation"])
            current = _generation(state, generation)
            current.ablation = {**data, "rows": []}
            state.current_generation = generation
        elif event_type == "ablation.role":
            generation = int(data["generation"])
            current = _generation(state, generation)
            if current.ablation is None:
                current.ablation = {"generation": generation, "roles": [], "rows": []}
            current.ablation.setdefault("rows", []).append(data)
        elif event_type == "ablation.completed":
            generation = int(data["generation"])
            current = _generation(state, generation)
            if current.ablation is None:
                current.ablation = {"generation": generation, "rows": []}
            current.ablation.update(data)
        elif event_type == "baseline.completed":
            generation = int(data["generation"])
            current = _generation(state, generation)
            current.baseline = data
        elif event_type == "diagnosis.emitted":
            generation = int(data["generation"])
            current = _generation(state, generation)
            current.diagnosis = data
            state.diagnoses.append(data)
        elif event_type == "mutation.applied":
            generation = int(data["generation"])
            current = _generation(state, generation)
            current.mutation = data
            state.mutations.append(data)
            state.current_generation = generation
        elif event_type == "mutation.reverted":
            generation = int(data["generation"])
            current = _generation(state, generation)
            current.reverted = True
            current.mutation = data
            state.mutations.append(data)
        elif event_type == "metrics.snapshot":
            generation = int(data["generation"])
            current = _generation(state, generation)
            current.metrics = data
            state.current_generation = generation
        elif event_type == "run.completed":
            state.best_generation = int(data["best_generation"])
            state.summary = data["summary"]
            state.completed = True
        elif event_type == "log":
            state.logs.append(data)

        state.last_seq = event.seq

    if state.best_generation is None and state.generations:
        state.best_generation = max(
            generation.generation for generation in state.generations.values()
        )
    return State.model_validate(state.model_dump(mode="python"))


def state_json_bytes(state: State) -> bytes:
    """Serialize a state in one canonical form for byte-level comparisons."""

    payload = state.model_dump(mode="json", exclude_none=False)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return serialized.encode("utf-8")
