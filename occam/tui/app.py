"""The Occam Textual app: a read-only view of one run directory.

The app never imports the engine, never calls an LLM and never writes into the
run directory.  It receives batches of events from an
:class:`~occam.tui.source.EventSource`, folds them through the shared pure
reducer, and repaints panels from the resulting :class:`~occam.tui.viewmodel.RunView`.
Replay and live differ only in which source is attached (`01 §1`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

from occam.core.models import Event, State
from occam.tui.feed import StateFeed
from occam.tui.panels import (
    PANEL_TYPES,
    DiagnosisFeed,
    FooterBar,
    HeaderBar,
    LineagePanel,
    ViewPanel,
)
from occam.tui.source import EventSource, LiveSource, ReplayOptions, ReplaySource
from occam.tui.viewmodel import RunView

COMPARE_FILENAME = "compare.json"


class StateChanged(Message):
    """A new reduced state is available (04 §4)."""

    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state


class InspectRequested(Message):
    """WP-09 seam: ``i`` asks for the Case Inspector modal (04 §3.6)."""

    def __init__(self, case_id: str | None = None) -> None:
        super().__init__()
        self.case_id = case_id


class HelpScreen(ModalScreen[None]):
    """The ``?`` key list."""

    BINDINGS = [
        Binding("escape,question_mark,q", "dismiss", "close", show=False),
    ]

    HELP = """\
OCCAM — read-only run viewer

  ←/→      previous / next generation
  a c d l  focus ablation · cases · diagnosis · lessons
  b        toggle the cost-matched baseline panel
  i        inspect the selected case
  space    pause / resume replay
  .        step one event
  +/-      replay speed
  ?        this help
  q        quit
"""

    def compose(self) -> ComposeResult:
        yield Static(self.HELP, id="help-body")


class OccamApp(App[None]):
    """One screen over one run directory."""

    CSS_PATH = "occam.tcss"
    TITLE = "OCCAM"

    # The shell keymap (04 §5) owns these keys everywhere: panels such as the
    # lineage Tree and the diagnosis log bind arrows and space themselves, so
    # the app's bindings take priority over whatever currently has focus.
    BINDINGS = [
        Binding("q,ctrl+c", "quit", "quit"),
        Binding("left", "prev_generation", "prev gen", priority=True),
        Binding("right", "next_generation", "next gen", priority=True),
        Binding("a", "focus_panel('ablation')", "ablation", priority=True),
        Binding("c", "focus_panel('cases')", "cases", priority=True),
        Binding("d", "focus_panel('diagnosis')", "diagnosis", priority=True),
        Binding("l", "focus_panel('lessons')", "lessons", priority=True),
        Binding("b", "toggle_baseline", "baseline", priority=True),
        Binding("i", "inspect", "inspect", priority=True),
        Binding("space", "toggle_pause", "pause", priority=True),
        Binding("full_stop", "step", "step", priority=True),
        Binding("plus,equals_sign", "speed_up", "faster", priority=True),
        Binding("minus", "speed_down", "slower", priority=True),
        Binding("question_mark", "help", "help"),
    ]

    def __init__(
        self,
        run_dir: str | Path,
        *,
        source: EventSource | None = None,
        replay: ReplayOptions | None = None,
    ):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.source = source or (
            ReplaySource(self.run_dir, replay) if replay is not None else LiveSource(self.run_dir)
        )
        self.feed = StateFeed()
        self.compare = self._load_compare()
        self.selected_generation: int | None = None
        self.pinned_generation = False
        self.view = self._build_view()

    # -- composition ----------------------------------------------------

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Horizontal(id="body"):
            with Vertical(id="col-left"):
                yield LineagePanel()
                yield PANEL_TYPES["architecture"](id="architecture")
            with Vertical(id="col-centre"):
                yield PANEL_TYPES["ablation"](id="ablation")
                yield PANEL_TYPES["cases"](id="cases")
                yield DiagnosisFeed()
            with Vertical(id="col-right"):
                yield PANEL_TYPES["generation-metrics"](id="generation-metrics")
                yield PANEL_TYPES["lessons"](id="lessons")
                yield PANEL_TYPES["baseline"](id="baseline")
        yield PANEL_TYPES["compare"](id="compare")
        yield PANEL_TYPES["metrics"](id="metrics")
        yield FooterBar(id="footer")

    def on_mount(self) -> None:
        self.query_one("#baseline").display = False
        self.refresh_view()
        self.run_worker(self._drive(), name="event-source", exclusive=True)

    # -- run directory --------------------------------------------------

    def _load_compare(self) -> dict[str, Any] | None:
        """Read the optional run-over-run comparison written by ``occam compare``."""

        path = self.run_dir / COMPARE_FILENAME
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    # -- event plumbing -------------------------------------------------

    async def _drive(self) -> None:
        initial = self.source.initial_state()
        if initial is not None:
            history = tuple(
                event for event in self.source.initial_events() if event.seq <= initial.last_seq
            )
            self.feed.prime(initial, history=history)
            self.query_one(DiagnosisFeed).prime(history)
            self.post_message(StateChanged(initial))
        await self.source.run(self._on_events)
        if self.feed.state is not None:
            # Sources set ``finished`` immediately after their final sink call;
            # repaint once more so the mode badge reflects that transition even
            # for logs without a run.completed event.
            self.post_message(StateChanged(self.feed.state))
        else:
            # A source can fail before producing its first event; refresh the
            # header so a bounded-tail error is still visible.
            self.refresh_view()

    def _on_events(self, events: Sequence[Event]) -> None:
        """Sink handed to the source; runs on the app's event loop."""

        batch = list(events)
        state = self.feed.apply(batch)
        self.query_one(DiagnosisFeed).ingest(batch)
        self.post_message(StateChanged(state))

    def on_state_changed(self, message: StateChanged) -> None:
        del message  # the feed already holds the authoritative state
        self.refresh_view()

    def on_inspect_requested(self, message: InspectRequested) -> None:
        """Default handler; WP-09 mounts the Case Inspector modal instead."""

        del message
        self.notify("Case Inspector arrives with the cases grid.", timeout=2)

    # -- painting -------------------------------------------------------

    def _build_view(self) -> RunView:
        source = self.source
        return RunView(
            self.feed.state,
            selected_generation=self.selected_generation if self.pinned_generation else None,
            mode=source.mode,
            speed=getattr(source, "speed", 1.0),
            paused=getattr(source, "paused", False),
            finished=getattr(source, "finished", False),
            elapsed_s=self.feed.elapsed_s(),
            status=getattr(source, "status", None),
            compare=self.compare,
        )

    def refresh_view(self) -> None:
        self.view = self._build_view()
        self.selected_generation = self.view.selected_generation
        for panel in self.query(".view-panel"):
            if isinstance(panel, ViewPanel):
                panel.update_view(self.view)

    # -- generation navigation ------------------------------------------

    def _select(self, generation: int) -> None:
        self.pinned_generation = True
        self.selected_generation = generation
        self.refresh_view()

    def action_prev_generation(self) -> None:
        self._step_generation(-1)

    def action_next_generation(self) -> None:
        self._step_generation(1)

    def _step_generation(self, delta: int) -> None:
        numbers = self.view.generation_numbers
        if not numbers:
            return
        current = self.view.selected_generation
        index = numbers.index(current) if current in numbers else 0
        self._select(numbers[max(0, min(len(numbers) - 1, index + delta))])

    def on_tree_node_selected(self, event: Any) -> None:
        generation = getattr(event.node, "data", None)
        if isinstance(generation, int):
            self._select(generation)

    # -- transport ------------------------------------------------------

    def action_toggle_pause(self) -> None:
        source = self.source
        if isinstance(source, ReplaySource):
            source.toggle_pause()
            self.refresh_view()

    def action_step(self) -> None:
        source = self.source
        if isinstance(source, ReplaySource):
            source.step()
            self.refresh_view()

    def action_speed_up(self) -> None:
        self._nudge_speed(2.0)

    def action_speed_down(self) -> None:
        self._nudge_speed(0.5)

    def _nudge_speed(self, factor: float) -> None:
        source = self.source
        if isinstance(source, ReplaySource):
            source.nudge_speed(factor)
            self.refresh_view()

    # -- panels ---------------------------------------------------------

    def action_focus_panel(self, panel_id: str) -> None:
        try:
            panel = self.query_one(f"#{panel_id}")
        except NoMatches:  # a panel WP-09 has not mounted must never crash the shell
            return
        if panel.display:
            panel.focus()

    def action_toggle_baseline(self) -> None:
        panel = self.query_one("#baseline")
        panel.display = not panel.display

    def action_inspect(self) -> None:
        self.post_message(InspectRequested())

    def action_help(self) -> None:
        self.push_screen(HelpScreen())


__all__ = ["HelpScreen", "InspectRequested", "OccamApp", "StateChanged"]
