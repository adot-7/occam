"""The Occam Textual app — a strictly read-only view of a run directory.

Engine and TUI share nothing but ``runs/<run_id>/`` (`01 §1`): this package
never imports :mod:`occam.engine`, never calls an LLM, and never writes into a
run directory.  It tails ``events.jsonl``, folds the events through the shared
pure reducer in :mod:`occam.store.reducer`, and paints the result.
"""

from occam.tui.app import InspectRequested, OccamApp, StateChanged
from occam.tui.feed import StateFeed
from occam.tui.source import (
    EventSource,
    LiveSource,
    ReplayOptions,
    ReplaySource,
    ReplayTargetNotFound,
)
from occam.tui.viewmodel import GenerationView, RunView

__all__ = [
    "EventSource",
    "GenerationView",
    "InspectRequested",
    "LiveSource",
    "OccamApp",
    "ReplayOptions",
    "ReplaySource",
    "ReplayTargetNotFound",
    "RunView",
    "StateChanged",
    "StateFeed",
]
