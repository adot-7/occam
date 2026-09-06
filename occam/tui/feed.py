"""Fold observed events into view state with the shared pure reducer."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from occam.core.models import Event, State
from occam.store.reducer import Reduction


class StateFeed:
    """Rebuild run state from the events the TUI has observed.

    The TUI implements no state transitions of its own: it pushes events
    through :class:`occam.store.reducer.Reduction`, the same pure transition
    the engine uses to write ``state.json``.  Replay and live therefore cannot
    diverge from each other or from the engine.
    """

    def __init__(self) -> None:
        self._reduction = Reduction()
        self._first_ts: datetime | None = None
        self._last_ts: datetime | None = None
        self._count = 0
        self.state: State | None = None
        self.primed = False

    @property
    def count(self) -> int:
        """How many events have been folded in."""

        return self._count

    @property
    def last_seq(self) -> int:
        return self._reduction.last_seq

    def prime(self, state: State) -> State:
        """Paint a snapshot loaded from ``state.json`` before any event arrives.

        The snapshot is only a first frame; the next :meth:`apply` replaces it
        with state derived from the events themselves.
        """

        if self._count:
            return self.state  # type: ignore[return-value]
        self.state = state
        self.primed = True
        return state

    def apply(self, events: Sequence[Event]) -> State:
        """Fold a batch of events in and return the new state."""

        if not events:
            if self.state is None:
                raise ValueError("no events applied yet")
            return self.state
        self.state = self._reduction.extend(events)
        for event in events:
            if self._first_ts is None:
                self._first_ts = event.ts
            self._last_ts = event.ts
        self._count += len(events)
        self.primed = False
        return self.state

    def elapsed_s(self) -> float:
        """Run-clock seconds between the first and last observed event."""

        if self._first_ts is None or self._last_ts is None:
            return 0.0
        return (self._last_ts - self._first_ts).total_seconds()


__all__ = ["StateFeed"]
