"""Pure reduction of the append-only event stream into a run snapshot."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from occam.core.models import Event, GenerationState, Lesson, MetricsSnapshot, State
from occam.store.schema import validate_event


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


def _materialize_event(raw_event: Event | Mapping[str, Any]) -> Event:
    """Validate both the typed envelope and its event-specific JSON contract."""

    event = raw_event if isinstance(raw_event, Event) else Event.model_validate(raw_event)
    validate_event(event.model_dump(mode="json", exclude_none=False))
    return event


class Reduction:
    """Incremental driver over the one state transition function.

    :func:`reduce` folds a whole stream in one call; a follower that receives
    events a few at a time (the TUI tailing ``events.jsonl``) pushes them
    through this instead of re-reducing everything it has seen.  Both paths run
    exactly the same transition, so their states cannot diverge.
    """

    def __init__(self) -> None:
        self._state: State | None = None
        self._expected_seq = 0

    def prime(self, state: State) -> State:
        """Continue reduction from an already validated state snapshot.

        Live followers load ``state.json`` for an instant first paint.  Priming
        the driver with that snapshot lets the next event be applied directly
        instead of replaying the entire log from sequence zero.
        """

        if not isinstance(state, State):
            raise TypeError("state must be a State")
        self._state = State.model_validate(state.model_dump(mode="python"))
        self._expected_seq = self._state.last_seq + 1
        return self.snapshot()

    @property
    def last_seq(self) -> int:
        """Sequence number of the last applied event, or ``-1``."""

        return self._expected_seq - 1

    def _push(self, raw_event: Event | Mapping[str, Any]) -> None:
        event = _materialize_event(raw_event)
        if self._state is None:
            self._state = State(run_id=event.run_id)
        _apply(self._state, event, self._expected_seq)
        self._expected_seq += 1

    def push(self, raw_event: Event | Mapping[str, Any]) -> State:
        """Apply one event and return the snapshot that follows it."""

        self._push(raw_event)
        return self.snapshot()

    def extend(self, events: Iterable[Event | Mapping[str, Any]]) -> State:
        """Apply a batch of events and return one snapshot for the batch."""

        for raw_event in events:
            self._push(raw_event)
        return self.snapshot()

    def snapshot(self) -> State:
        """Return an isolated, validated copy of the current state."""

        if self._state is None:
            raise ValueError("cannot reduce an empty event stream")
        state = State.model_validate(self._state.model_dump(mode="python"))
        if state.best_generation is None and state.generations:
            state.best_generation = max(
                generation.generation for generation in state.generations.values()
            )
        return state


def _apply(state: State, event: Event, expected_seq: int) -> None:
    """Fold one validated event into ``state`` in place."""

    if event.run_id != state.run_id:
        raise ValueError(
            f"event seq {event.seq} belongs to run {event.run_id!r}, expected {state.run_id!r}"
        )
    if event.seq != expected_seq:
        raise ValueError(f"expected event seq {expected_seq}, got {event.seq}")
    data = _copy_data(event.data)
    event_type = event.type

    if event_type == "run.started":
        state.task = data["task"]
        state.config = data["config"]
        state.run_name = str(data["run_name"])
        state.memory_ns = str(data["memory_ns"])
        state.lessons_loaded = [Lesson.model_validate(lesson) for lesson in data["lessons_loaded"]]
        state.lessons = copy.deepcopy(state.lessons_loaded)
    elif event_type == "lesson.written":
        lesson = Lesson.model_validate(data["lesson"])
        state.lessons.append(lesson)
    elif event_type == "reliability.completed":
        generation = int(data["generation"])
        current = _generation(state, generation)
        current.reliability = data
        state.current_generation = generation
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
        current.revert = data
        if current.mutation is not None:
            current.mutation = {
                **current.mutation,
                "reverted": True,
                "revert": data,
            }
        state.mutations.append(data)
    elif event_type == "metrics.snapshot":
        generation = int(data["generation"])
        current = _generation(state, generation)
        current.metrics = MetricsSnapshot.model_validate(data)
        state.current_generation = generation
    elif event_type == "run.completed":
        state.best_generation = int(data["best_generation"])
        state.summary = data["summary"]
        state.completed = True
    elif event_type == "log":
        state.logs.append(data)

    state.last_seq = event.seq


def reduce(events: Iterable[Event | Mapping[str, Any]]) -> State:
    """Apply events in order and return a deterministic state snapshot.

    The reducer has no clock, filesystem, or provider dependencies. It accepts
    either validated :class:`Event` objects or their serialized mappings so the
    engine and the TUI can share exactly one state transition function.
    """

    return Reduction().extend(events)


def state_json_bytes(state: State) -> bytes:
    """Serialize a state in one canonical form for byte-level comparisons."""

    payload = state.model_dump(mode="json", exclude_none=False)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return serialized.encode("utf-8")
