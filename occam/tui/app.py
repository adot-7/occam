"""The Occam Textual app: a read-only view of one run directory.

The app never imports the engine, never calls an LLM and never writes into the
run directory. It receives batches of events from an
:class:`~occam.tui.source.EventSource`, folds them through the shared pure
reducer, and repaints panels from the resulting :class:`~occam.tui.viewmodel.RunView`.
Replay and live differ only in which source is attached (01 §1).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from occam.core.models import Event, Lesson, State
from occam.tui.feed import StateFeed
from occam.tui.palette import AMBER, AMBER_HI, CYAN, DIM, FG, FG_BRIGHT, GREEN, RED
from occam.tui.panels import (
    PANEL_TYPES,
    CaseSelectionChanged,
    CasesPanel,
    DiagnosisFeed,
    FooterBar,
    HeaderBar,
    LessonEvidenceRequested,
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
    """Ask the app to open the read-only Case Inspector."""

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
  i/enter  inspect the selected case
  space    pause / resume replay
  .        step one event
  +/-      replay speed
  ?        this help
  q        quit
"""

    def compose(self) -> ComposeResult:
        yield Static(self.HELP, id="help-body")


class CaseInspector(ModalScreen[None]):
    """A compact, read-only case and tool-trace inspector (04 §3.6).

    The event schema intentionally keeps ``execution.case`` small. When a
    richer result record is present in that projection, this screen expands the
    per-role and raw tool fields. For summary-only fixture events it says so
    explicitly while retaining the highlighted requested-date → rate-date row;
    it never manufactures a response that was not recorded.
    """

    BINDINGS = [
        Binding("escape,q", "dismiss", "close", show=False),
    ]

    def __init__(
        self,
        view: RunView,
        case_id: str | None,
        *,
        lesson: Lesson | None = None,
        highlight: bool = False,
    ) -> None:
        super().__init__()
        self.view = view
        self.case_id = case_id
        self.lesson = lesson
        self.highlight = highlight

    def compose(self) -> ComposeResult:
        yield Static(id="inspector-body")

    def on_mount(self) -> None:
        self.query_one("#inspector-body", Static).update(self.render_inspector())

    def render_inspector(self) -> Text:
        case = self.view.case(self.case_id)
        selected = self.view.selected
        text = Text()
        text.append("CASE INSPECTOR", style=f"bold {CYAN}")
        if selected is not None:
            text.append(f"  {selected.label}", style=DIM)
        text.append("\n\n", style=DIM)
        if case is None:
            text.append("case trace unavailable  ", style=f"bold {AMBER}")
            text.append("(not present in this run's event projection)\n", style=DIM)
        else:
            text.append(f"{self.case_id}  ", style=f"bold {FG_BRIGHT}")
            passed = bool(case.get("passed"))
            text.append("PASS\n" if passed else "FAIL\n", style=GREEN if passed else RED)
        text.append(
            f"cost {_money(case.get('cost_usd') if case else None)}  ·  "
            f"latency {case.get('latency_s', '—') if case else '—'}s\n",
            style=FG,
        )
        if self.lesson is not None:
            self._append_lesson_evidence(text, self.lesson, case)
        if case is not None:
            self._append_sub_results(text, case)
            self._append_optional_payload(text, case)
        self._append_tool_trace(text, case)
        return text

    @staticmethod
    def _append_lesson_evidence(text: Text, lesson: Lesson, case: dict[str, Any] | None) -> None:
        text.append("\nLESSON EVIDENCE\n", style=f"bold {CYAN}")
        badge = "tool_note" if lesson.kind == "tool_note" else "domain_rule"
        text.append(f"{lesson.id}  [{badge}]", style=f"bold {AMBER}")
        if lesson.tool:
            text.append(f"  tool={lesson.tool}", style=CYAN)
        text.append("\n", style=FG)
        text.append(f"{lesson.text}\n", style=FG)
        evidence = lesson.evidence if isinstance(lesson.evidence, dict) else {}
        case_ids = evidence.get("case_ids") or []
        trace_refs = evidence.get("trace_refs") or []
        text.append(
            f"evidence cases  {', '.join(str(item) for item in case_ids) or 'unavailable'}\n",
            style=FG if case_ids else DIM,
        )
        text.append(
            f"trace refs  {', '.join(str(item) for item in trace_refs) or 'unavailable'}\n",
            style=FG if trace_refs else DIM,
        )
        if case is not None:
            text.append(f"linked case  {case.get('case_id', 'unavailable')}\n", style=GREEN)
        elif case_ids:
            text.append(
                "linked case trace  unavailable in this run's compact event projection\n",
                style=DIM,
            )
        else:
            text.append("linked case trace  unavailable (evidence.case_ids absent)\n", style=DIM)

    @staticmethod
    def _append_sub_results(text: Text, case: dict[str, Any]) -> None:
        sub_results = case.get("sub_results") or {}
        if not isinstance(sub_results, dict) or not sub_results:
            text.append("sub_results  —  not carried in this execution.case summary\n", style=DIM)
            return
        passed = sum(bool(value) for value in sub_results.values())
        text.append(f"sub_results  {passed}/{len(sub_results)} invoices pass\n", style=FG)
        for invoice, result in sub_results.items():
            text.append(f"  {'✓' if result else '✗'} {invoice}\n", style=GREEN if result else RED)

    @staticmethod
    def _append_optional_payload(text: Text, case: dict[str, Any]) -> None:
        for label in ("input", "expected", "answer"):
            value = case.get(label)
            if value is not None:
                text.append(f"{label}  {_clip(value, 120)}\n", style=FG)

    def _append_tool_trace(self, text: Text, case: dict[str, Any] | None) -> None:
        text.append("\nTOOL RESPONSE HIGHLIGHT\n", style=f"bold {AMBER}")
        if case is None:
            text.append(
                "unavailable — no linked case/tool trace is recorded in this run.\n",
                style=DIM,
            )
            return
        calls = _tool_calls(case)
        if not calls:
            text.append(
                "requested_date  →  rate_date\n",
                style=f"bold {AMBER_HI} on #231908",
            )
            text.append(
                "raw role/tool fields are not present in this fixture's compact event contract; "
                "the labels above are the trace fields the Inspector highlights when a full "
                "result is recorded.\n",
                style=DIM,
            )
            return
        for call in calls:
            tool = str(call.get("tool") or call.get("name") or "tool")
            text.append(f"{tool}\n", style=f"bold {CYAN}")
            requested = _first_value(call, "requested_date")
            response = call.get("response") or call.get("raw_response") or call
            rate_date = _first_value(response, "rate_date")
            if requested is not None or rate_date is not None:
                text.append("  requested_date ", style=FG)
                text.append(str(requested or "—"), style=f"bold {AMBER_HI} on #231908")
                text.append("  →  rate_date ", style=FG)
                text.append(str(rate_date or "—"), style=f"bold {AMBER_HI} on #231908")
                text.append("\n", style=FG)
            for field in ("rate", "status", "bytes", "cached", "error"):
                value = _first_value(response, field)
                if value is not None:
                    text.append(f"  {field}: {value}\n", style=DIM if field != "error" else RED)


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.4f}"
    except (TypeError, ValueError):
        return "—"


def _clip(value: Any, width: int) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _first_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _first_value(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_value(value, key)
            if found is not None:
                return found
    return None


def _tool_calls(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize optional per-role trace shapes without coupling to the engine."""

    calls: list[dict[str, Any]] = []
    direct = case.get("tool_calls") or case.get("raw_tool_responses")
    if isinstance(direct, list):
        calls.extend(item for item in direct if isinstance(item, dict))
    per_role = case.get("per_role")
    if isinstance(per_role, dict):
        for role_id, trace in per_role.items():
            if not isinstance(trace, dict):
                continue
            role_calls = trace.get("tool_calls") or trace.get("raw_tool_responses") or []
            if isinstance(role_calls, list):
                for call in role_calls:
                    if isinstance(call, dict):
                        calls.append({"role_id": role_id, **call})
    return calls


class OccamApp(App[None]):
    """One screen over one run directory."""

    CSS_PATH = "occam.tcss"
    TITLE = "OCCAM"

    BINDINGS = [
        Binding("q,ctrl+c", "quit", "quit"),
        Binding("left", "prev_generation", "prev gen", priority=True),
        Binding("right", "next_generation", "next gen", priority=True),
        Binding("a", "focus_panel('ablation')", "ablation", priority=True),
        Binding("c", "focus_panel('cases')", "cases", priority=True),
        Binding("d", "focus_panel('diagnosis')", "diagnosis", priority=True),
        Binding("l", "focus_panel('lessons')", "lessons", priority=True),
        Binding("b", "toggle_baseline", "baseline", priority=True),
        Binding("i,enter", "inspect", "inspect", priority=True, show=False),
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
        self.selected_case_id: str | None = None
        self.pinned_generation = False
        self.view = self._build_view()

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Horizontal(id="body"):
            with Vertical(id="col-left"):
                yield LineagePanel()
                yield PANEL_TYPES["structural"](id="structural")
            with Vertical(id="col-centre"):
                yield PANEL_TYPES["ablation"](id="ablation")
                yield PANEL_TYPES["cases"](id="cases")
                yield PANEL_TYPES["evidence"](id="evidence")
            with Vertical(id="col-right"):
                yield PANEL_TYPES["architecture"](id="architecture")
                yield PANEL_TYPES["generation-metrics"](id="generation-metrics")
                yield PANEL_TYPES["lessons"](id="lessons")
                yield PANEL_TYPES["baseline"](id="baseline")
        yield PANEL_TYPES["compare"](id="compare")
        yield DiagnosisFeed()
        yield PANEL_TYPES["metrics"](id="metrics")
        yield FooterBar(id="footer")

    def on_mount(self) -> None:
        self.query_one("#baseline").display = False
        self.refresh_view()
        self.run_worker(self._drive(), name="event-source", exclusive=True)

    def _load_compare(self) -> dict[str, Any] | None:
        """Read the optional run-over-run comparison written by ``occam compare``."""

        path = self.run_dir / COMPARE_FILENAME
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

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
            self.post_message(StateChanged(self.feed.state))
        else:
            self.refresh_view()

    def _on_events(self, events: Sequence[Event]) -> None:
        batch = list(events)
        state = self.feed.apply(batch)
        self.query_one(DiagnosisFeed).ingest(batch)
        self.post_message(StateChanged(state))

    def on_state_changed(self, message: StateChanged) -> None:
        del message
        self.refresh_view()

    def on_inspect_requested(self, message: InspectRequested) -> None:
        self.open_case_inspector(message.case_id)

    def on_case_selection_changed(self, message: CaseSelectionChanged) -> None:
        self.selected_case_id = message.case_id
        self.refresh_view()

    def on_lesson_evidence_requested(self, message: LessonEvidenceRequested) -> None:
        self.open_lesson_evidence(message.lesson_id)

    def on_button_pressed(self, message: Button.Pressed) -> None:
        button_id = message.button.id or ""
        if not button_id.startswith("lesson-evidence-"):
            return
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if 0 <= index < len(self.view.lessons):
            self.open_lesson_evidence(self.view.lessons[index].id)

    def _build_view(self) -> RunView:
        return RunView(
            self.feed.state,
            selected_generation=self.selected_generation if self.pinned_generation else None,
            selected_case_id=self.selected_case_id,
            mode=self.source.mode,
            speed=getattr(self.source, "speed", 1.0),
            paused=getattr(self.source, "paused", False),
            finished=getattr(self.source, "finished", False),
            elapsed_s=self.feed.elapsed_s(),
            status=getattr(self.source, "status", None),
            compare=self.compare,
        )

    def refresh_view(self) -> None:
        self.view = self._build_view()
        self.selected_generation = self.view.selected_generation
        self.selected_case_id = self.view.selected_case_id
        for panel in self.query(".view-panel"):
            if isinstance(panel, ViewPanel):
                panel.update_view(self.view)

    def _select(self, generation: int) -> None:
        self.pinned_generation = True
        self.selected_generation = generation
        self.refresh_view()

    def action_prev_generation(self) -> None:
        if self._move_focused_case(-1):
            return
        self._step_generation(-1)

    def action_next_generation(self) -> None:
        if self._move_focused_case(1):
            return
        self._step_generation(1)

    def _move_focused_case(self, delta: int) -> bool:
        focused = self.focused
        if not isinstance(focused, CasesPanel):
            return False
        focused.move_cursor(delta)
        return True

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

    def action_focus_panel(self, panel_id: str) -> None:
        try:
            panel = self.query_one(f"#{panel_id}")
        except NoMatches:
            return
        if panel.display:
            panel.focus()

    def action_toggle_baseline(self) -> None:
        try:
            panel = self.query_one("#baseline")
        except NoMatches:
            return
        panel.display = not panel.display

    def action_inspect(self) -> None:
        self.open_case_inspector()

    def open_case_inspector(self, case_id: str | None = None) -> None:
        if case_id is None:
            selected = self.view.selected_case
            case_id = str(selected.get("case_id")) if selected else None
        if case_id is None:
            self.notify("No case is available for inspection.", severity="warning", timeout=2)
            return
        self.push_screen(CaseInspector(self.view, case_id))

    def open_lesson_evidence(self, lesson_id: str) -> None:
        lesson = next((item for item in self.view.lessons if item.id == lesson_id), None)
        if lesson is None:
            self.notify("Lesson evidence is not available.", severity="warning", timeout=2)
            return
        evidence = lesson.evidence if isinstance(lesson.evidence, dict) else {}
        case_ids = evidence.get("case_ids") or []
        case_id = next(
            (
                str(candidate)
                for candidate in case_ids
                if self.view.case(str(candidate)) is not None
            ),
            None,
        )
        self.push_screen(CaseInspector(self.view, case_id, lesson=lesson, highlight=True))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())


__all__ = ["CaseInspector", "HelpScreen", "InspectRequested", "OccamApp", "StateChanged"]
