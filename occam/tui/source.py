"""Event sources that feed the TUI: recorded replay and live tail.

Both sources hand the app the same thing — batches of validated
:class:`~occam.core.models.Event` objects read from ``runs/<run_id>/events.jsonl``
— so the app cannot tell replay from live (`01 §1`).  Neither source ever writes
into the run directory.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from occam.core.models import Event, State
from occam.store.reader import EventReader

EventSink = Callable[[Sequence[Event]], None]

MIN_SPEED = 0.25
MAX_SPEED = 64.0
DEFAULT_MAX_GAP_S = 2.0


class ReplayTargetNotFound(ValueError):
    """Raised when ``--to-gen`` or ``--at`` matches no event in the log."""


def _generation_of(event: Event) -> int | None:
    value = event.data.get("generation")
    return int(value) if isinstance(value, int) else None


def generation_start_index(events: Sequence[Event], generation: int) -> int:
    """Index of the first event belonging to ``generation``."""

    for index, event in enumerate(events):
        if _generation_of(event) == generation:
            return index
    raise ReplayTargetNotFound(f"no events for generation {generation}")


def fast_forward_index(
    events: Sequence[Event],
    *,
    to_gen: int | None = None,
    at: str | None = None,
) -> int:
    """Index of the last event to emit instantly; ``-1`` plays from the start.

    ``--to-gen N`` alone stops just *before* generation ``N`` opens, so the
    generation then plays normally from its first event (`04 §4`).  ``--at
    <type>`` lands *on* the first event of that type — inside ``--to-gen`` when
    both are given — so the frame already shows that event's effect.
    """

    if at is None:
        if to_gen is None:
            return -1
        return generation_start_index(events, to_gen) - 1

    start = 0 if to_gen is None else generation_start_index(events, to_gen)
    for index in range(start, len(events)):
        event = events[index]
        if event.type != at:
            continue
        if to_gen is not None:
            generation = _generation_of(event)
            if generation is not None and generation != to_gen:
                continue
        return index
    scope = "" if to_gen is None else f" within generation {to_gen}"
    raise ReplayTargetNotFound(f"no {at!r} event{scope}")


@dataclass(frozen=True)
class ReplayOptions:
    """Command-line shape of ``occam replay``."""

    speed: float = 1.0
    to_gen: int | None = None
    at: str | None = None
    paused: bool = False
    max_gap_s: float = DEFAULT_MAX_GAP_S

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError("--speed must be greater than zero")
        if self.max_gap_s < 0:
            raise ValueError("max_gap_s must not be negative")
        if self.to_gen is not None and self.to_gen < 0:
            raise ValueError("--to-gen must not be negative")


class EventSource(ABC):
    """Something that pushes batches of events at the app."""

    mode: str = "live"

    def initial_state(self) -> State | None:
        """A snapshot for the first paint, when one is available and honest."""

        return None

    @abstractmethod
    async def run(self, sink: EventSink) -> None:
        """Push every event this source has, then return."""


class ReplaySource(EventSource):
    """Play a recorded ``events.jsonl`` back at a chosen speed."""

    mode = "replay"

    def __init__(
        self,
        run_dir: str | Path,
        options: ReplayOptions | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.run_dir = Path(run_dir)
        self.options = options or ReplayOptions()
        self._sleep = sleep
        self._events = EventReader(self.run_dir).read()
        self._speed = self.options.speed
        self._paused = self.options.paused
        self._steps = 0
        self._wake = asyncio.Event()
        self._finished = asyncio.Event()
        self.emitted = 0
        # Validate the landing point eagerly so the CLI can report a bad
        # --to-gen/--at before it clears the screen.
        self.fast_forward_to = fast_forward_index(
            self._events, to_gen=self.options.to_gen, at=self.options.at
        )

    # -- transport controls --------------------------------------------

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    @property
    def total_events(self) -> int:
        return len(self._events)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._steps = 0
        self._wake.set()

    def toggle_pause(self) -> bool:
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused

    def step(self) -> None:
        """Advance exactly one event, pausing first if playback is running."""

        self._paused = True
        self._steps += 1
        self._wake.set()

    def set_speed(self, speed: float) -> float:
        self._speed = min(MAX_SPEED, max(MIN_SPEED, speed))
        return self._speed

    def nudge_speed(self, factor: float) -> float:
        return self.set_speed(self._speed * factor)

    async def wait_finished(self) -> None:
        await self._finished.wait()

    # -- playback -------------------------------------------------------

    async def _gate(self) -> bool:
        """Block while paused; return True when released by a single step."""

        while self._paused and self._steps == 0:
            self._wake.clear()
            await self._wake.wait()
        if self._paused and self._steps > 0:
            self._steps -= 1
            return True
        return False

    def _delay(self, index: int) -> float:
        if index <= 0:
            return 0.0
        gap = (self._events[index].ts - self._events[index - 1].ts).total_seconds()
        gap = max(0.0, min(gap, self.options.max_gap_s))
        return gap / self._speed

    async def run(self, sink: EventSink) -> None:
        head = self.fast_forward_to + 1
        if head:
            # The fast-forward is silent: one batch, one repaint, no sleeps.
            sink(self._events[:head])
            self.emitted = head
        index = head
        while index < len(self._events):
            stepped = await self._gate()
            if not stepped:
                delay = self._delay(index)
                if delay > 0:
                    await self._sleep(delay)
            sink([self._events[index]])
            self.emitted = index + 1
            index += 1
        self._finished.set()


class LiveSource(EventSource):
    """Tail a run directory that the engine may still be writing to."""

    mode = "live"

    def __init__(
        self,
        run_dir: str | Path,
        *,
        poll_interval: float = 0.25,
        follow: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.poll_interval = poll_interval
        self.follow = follow
        self._reader = EventReader(self.run_dir)
        if not self._reader.events_path.exists() and not self._reader.state_path.exists():
            raise FileNotFoundError(f"event log not found: {self._reader.events_path}")
        self._finished = asyncio.Event()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def initial_state(self) -> State | None:
        """Load ``state.json`` for an instant first paint when it exists."""

        try:
            return self._reader.load_state()
        except (OSError, ValueError):
            return None

    async def wait_finished(self) -> None:
        await self._finished.wait()

    async def run(self, sink: EventSink) -> None:
        next_seq = 0
        while True:
            events = await asyncio.to_thread(self._reader.read, live=True)
            fresh = [event for event in events if event.seq >= next_seq]
            if fresh:
                next_seq = fresh[-1].seq + 1
                sink(fresh)
                if any(event.type == "run.completed" for event in fresh):
                    break
            if not self.follow:
                break
            await asyncio.sleep(self.poll_interval)
        self._finished.set()


__all__ = [
    "DEFAULT_MAX_GAP_S",
    "EventSink",
    "EventSource",
    "LiveSource",
    "MAX_SPEED",
    "MIN_SPEED",
    "ReplayOptions",
    "ReplaySource",
    "ReplayTargetNotFound",
    "fast_forward_index",
    "generation_start_index",
]
