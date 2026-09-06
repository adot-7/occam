"""Dense, read-only WP-09 panels for the Occam run viewer.

The widgets in this module are deliberately projections of ``RunView``. They
do not load task packs, call tools, or reach into the engine. That keeps the
TUI and replay on the same event/state contract while allowing the screen to
be substantially richer than the WP-07 shell.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from rich.console import RenderableType
from rich.text import Text
from textual.events import Key
from textual.message import Message
from textual.widgets import Button, DataTable, ProgressBar, RichLog, Static, Tree
from textual.widgets.tree import TreeNode

from occam.core.models import Event
from occam.tui.palette import (
    ACCENT,
    AMBER,
    AMBER_BG,
    AMBER_HI,
    CYAN,
    DIM,
    FG,
    FG_BRIGHT,
    GREEN,
    GREEN_BG,
    MAGENTA,
    PURPLE,
    RED,
    RED_BG,
    RED_DIM,
    SELECTED,
    WIRE,
)
from occam.tui.viewmodel import GenerationView, RunView

GAP = "   "


@runtime_checkable
class ViewPanel(Protocol):
    """Anything the app repaints when the reduced state changes."""

    def update_view(self, view: RunView) -> None: ...


class CaseSelectionChanged(Message):
    """The case cursor moved and the app should rebuild its shared view."""

    def __init__(self, case_id: str | None) -> None:
        super().__init__()
        self.case_id = case_id


class LessonEvidenceRequested(Message):
    """Open the selected lesson's evidence in the read-only inspector."""

    def __init__(self, lesson_id: str) -> None:
        super().__init__()
        self.lesson_id = lesson_id


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.4f}"


def _ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _bar(value: float | None, width: int = 6, *, full: str = "▓", empty: str = "░") -> str:
    """Return a tiny token-sized bar for a normalized value."""

    if value is None:
        return empty * width
    filled = max(0, min(width, round(value * width)))
    return full * filled + empty * (width - filled)


def _cell(value: str, colour: str = FG, *, background: str | None = None) -> Text:
    style = colour
    if background:
        style = f"{style} on {background}"
    return Text(value, style=style)


def _justification_colour(justification: str) -> str:
    return {
        "parallel": CYAN,
        "context_isolation": PURPLE,
        "verification": AMBER,
        "control": GREEN,
        "ensemble": MAGENTA,
    }.get(justification, DIM)


def _verdict_label(verdict: str) -> str:
    return {
        "load_bearing": "● LOAD-BEARING",
        "witness": "✗ WITNESS",
        "harmful": "⚠ HARMFUL",
        "uncertain": "? UNCERTAIN",
    }.get(verdict, verdict.upper() or "PENDING")


def _verdict_colour(verdict: str) -> tuple[str, str | None]:
    return {
        "load_bearing": (GREEN, GREEN_BG),
        "witness": (RED, RED_BG),
        "harmful": (RED, "#1a0000"),
        "uncertain": (AMBER, AMBER_BG),
    }.get(verdict, (DIM, None))


def _clip(value: Any, width: int) -> str:
    text = str(value)
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


class Panel(Static):
    """A titled region whose only input is a :class:`RunView`."""

    DEFAULT_CLASSES = "view-panel"
    can_focus = True
    title_text = ""

    def on_mount(self) -> None:
        self.border_title = self.title_text

    def update_view(self, view: RunView) -> None:
        self.border_title = self.panel_title(view)
        self.update(self.render_view(view))

    def panel_title(self, view: RunView) -> str:
        del view
        return self.title_text

    def render_view(self, view: RunView) -> RenderableType:
        del view
        return ""


class HeaderBar(Static):
    """The compact product header from the Figma frame."""

    DEFAULT_CLASSES = "view-panel"

    def update_view(self, view: RunView) -> None:
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append("OCCAM", style=f"bold {ACCENT}")
        if view.task_label:
            text.append("  ")
            text.append(view.task_label, style=FG_BRIGHT)
        text.append("  │  ", style=DIM)
        text.append(view.run_label, style=GREEN if view.lessons_loaded else DIM)
        text.append("  │  ", style=DIM)
        text.append(view.generation_label, style=DIM)
        text.append("  │  ", style=DIM)
        badge = AMBER if view.mode == "replay" else GREEN
        text.append(view.mode_badge, style=f"bold {badge}")
        text.append("  ", style=DIM)
        elapsed = "elapsed —" if view.elapsed_s is None else f"elapsed {view.elapsed_s:,.0f}s"
        text.append(elapsed, style=DIM)
        text.append("  ", style=DIM)
        text.append(_money(view.spend_usd), style=FG)
        if view.status:
            text.append("  ", style=DIM)
            status_style = RED if "error" in view.status or "incomplete" in view.status else AMBER
            text.append(view.status, style=status_style)
        self.update(text)


class LineagePanel(Tree[int]):
    """Compact generation lineage with current and best markers."""

    DEFAULT_CLASSES = "view-panel"

    def __init__(self) -> None:
        super().__init__("lineage", id="lineage")
        self.show_root = False
        self.guide_depth = 2
        self.border_title = "LINEAGE"
        self._signature: tuple[Any, ...] | None = None

    def update_view(self, view: RunView) -> None:
        generations = view.generations
        self.border_title = f"LINEAGE · {len(generations)} GEN"
        signature = tuple(
            (
                generation.generation,
                generation.n_roles,
                generation.pass_rate,
                generation.cost_usd,
                generation.reverted,
                generation.generation == view.best_generation,
                generation.generation == view.selected_generation,
                generation.mutation.get("type") if generation.mutation else None,
            )
            for generation in generations
        )
        if signature == self._signature:
            return
        self._signature = signature
        self.clear()
        node: TreeNode[int] = self.root
        for generation in generations:
            node = node.add(self._label(generation, view), data=generation.generation, expand=True)
            if generation.generation == view.selected_generation:
                move_cursor = getattr(self, "move_cursor", None)
                if move_cursor is not None:
                    move_cursor(node)
                else:  # Textual 0.70 compatibility.
                    self.cursor_line = node.line

    @staticmethod
    def _label(generation: GenerationView, view: RunView) -> Text:
        if generation.reverted:
            marker, style = "○", DIM
        elif generation.generation == view.selected_generation:
            marker, style = "●", f"bold {CYAN}"
        else:
            marker, style = "·", FG
        label = Text(f"{marker} {generation.label}", style=style)
        if generation.generation == view.best_generation:
            label.append(" ★", style=AMBER)
        label.append(f"  {generation.n_roles} roles", style=DIM)
        label.append(f"  {_percent(generation.pass_rate)}", style=GREEN)
        label.append(f"  {_money(generation.cost_usd)}", style=DIM)
        if generation.reverted:
            label.append("  reverted", style=DIM)
        if generation.mutation and generation.mutation.get("type") == "prune":
            label.append(f"  ✗ {generation.mutation.get('target_role', '')}", style=RED)
        return label


class StructuralFidelityPanel(Panel):
    """Small left-column summary that keeps SF visible while the table is busy."""

    title_text = "STRUCT. FIDELITY"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None:
            return Text("SF  —", style=DIM)
        fidelity = generation.structural_fidelity
        text = Text("SF  ", style=DIM)
        text.append(_ratio(fidelity), style=f"bold {CYAN}")
        text.append(f"  {_bar(fidelity, 8)}\n", style=CYAN)
        witnesses = len(generation.witnesses)
        waste = 1.0 - (fidelity or 0.0)
        text.append(f"{witnesses} witness", style=RED if witnesses else GREEN)
        text.append(f"  ·  {waste:.0%} spend at risk", style=DIM)
        return text


class AblationPanel(Panel):
    """The DataTable hero: role influence, cost and verdict in one scan."""

    title_text = "ABLATION TABLE"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._table_signature: tuple[Any, ...] | None = None

    def compose(self):
        yield DataTable(
            show_row_labels=False,
            show_cursor=False,
            cursor_type="row",
            zebra_stripes=False,
            cell_padding=2,
            id="ablation-table",
        )
        yield Static(id="ablation-callout")
        yield Static(id="ablation-progress-label")
        yield ProgressBar(total=1, show_eta=False, show_percentage=False, id="ablation-progress")
        yield Static(id="ablation-footer")

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        if generation is None:
            return self.title_text
        return (
            f"{self.title_text}  ·  {generation.label}  ·  {generation.n_roles} ROLES"
            "  ·  LOO APPROX."
        )

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or generation.ablation is None:
            return Text("ablated —/— roles", style=DIM)
        rows = generation.ablation_rows
        roles = generation.ablation_roles or [row.get("role_id", "?") for row in rows]
        text = Text(f"ablated {len(rows)}/{len(roles)} roles", style=FG)
        if generation.noise_rate is not None:
            text.append("  ·  noise floor ", style=DIM)
            text.append(_ratio(generation.noise_rate), style=FG)
        if generation.structural_fidelity is not None:
            text.append("  ·  SF ", style=DIM)
            text.append(_ratio(generation.structural_fidelity), style=CYAN)
        witnesses = generation.witnesses
        if witnesses:
            text.append(f"  ·  {len(witnesses)} witness", style=RED)
        return text

    def update_view(self, view: RunView) -> None:
        self.border_title = self.panel_title(view)
        self.update(self.render_view(view))
        try:
            table = self.query_one("#ablation-table", DataTable)
            callout = self.query_one("#ablation-callout", Static)
            progress_label = self.query_one("#ablation-progress-label", Static)
            progress = self.query_one("#ablation-progress", ProgressBar)
            footer = self.query_one("#ablation-footer", Static)
        except Exception:  # pragma: no cover - only possible before child mount.
            return

        generation = view.selected
        rows = generation.ablation_rows if generation is not None else []
        roles = generation.role_map if generation is not None else {}
        signature = tuple(
            (
                row.get("role_id"),
                row.get("influence"),
                row.get("influence_ci", {}).get("lo"),
                row.get("influence_ci", {}).get("hi"),
                row.get("divergence"),
                row.get("cost_share"),
                row.get("verdict"),
            )
            for row in rows
        )
        if signature != self._table_signature:
            self._table_signature = signature
            table.clear(columns=True)
            table.add_column("ROLE", width=17, key="role")
            table.add_column("JUSTIFICATION", width=15, key="justification")
            table.add_column("INFLUENCE", width=15, key="influence")
            table.add_column("95% CI", width=17, key="ci")
            table.add_column("COST", width=11, key="cost")
            table.add_column("DIVERGENCE", width=12, key="divergence")
            table.add_column("VERDICT", width=17, key="verdict")
            for row in rows:
                role_id = str(row.get("role_id", "?"))
                role = roles.get(role_id, {})
                name = str(role.get("name", role_id))
                justification = str(role.get("justification", "unspecified"))
                verdict = str(row.get("verdict", "pending"))
                verdict_colour, verdict_bg = _verdict_colour(verdict)
                ci = row.get("influence_ci") or {}
                influence = _safe_float(row.get("influence"))
                divergence = _safe_float(row.get("divergence"))
                cost_share = _safe_float(row.get("cost_share"))
                table.add_row(
                    _cell(_clip(name, 17), verdict_colour, background=verdict_bg),
                    _cell(
                        _clip(justification, 14),
                        _justification_colour(justification),
                        background=verdict_bg,
                    ),
                    _cell(
                        f"{_bar(abs(influence), 5)} {influence:+.2f}",
                        GREEN if influence >= 0 else RED,
                        background=verdict_bg,
                    ),
                    _cell(
                        f"[{_safe_float(ci.get('lo')):+.2f},{_safe_float(ci.get('hi')):+.2f}]",
                        FG,
                        background=verdict_bg,
                    ),
                    _cell(
                        f"{_bar(cost_share, 4)} {_percent(cost_share)}",
                        FG_BRIGHT if cost_share >= 0.2 else FG,
                        background=verdict_bg,
                    ),
                    _cell(f"{_bar(divergence, 4)} {divergence:.2f}", FG, background=verdict_bg),
                    _cell(_verdict_label(verdict), verdict_colour, background=verdict_bg),
                    key=role_id,
                )

        expected = len(generation.ablation_roles) if generation is not None else 0
        observed = len(rows)
        progress.total = max(1, expected)
        progress.progress = min(expected, observed)
        progress_label.update(
            Text(
                "ablation complete"
                if expected and observed >= expected
                else f"ablating role {observed + 1}/{expected}",
                style=GREEN if expected and observed >= expected else AMBER,
            )
        )
        footer.update(self._footer(generation, observed, expected))
        callout_text = self._callout(generation)
        callout.update(callout_text)
        callout.display = bool(callout_text.plain.strip())

    @staticmethod
    def _footer(generation: GenerationView | None, observed: int, expected: int) -> Text:
        text = Text(f"ablated {observed}/{expected} roles", style=DIM)
        if generation is None:
            return text
        if generation.noise_rate is not None:
            text.append(f"   ·   noise {generation.noise_rate:.2f}", style=DIM)
        if generation.structural_fidelity is not None:
            text.append(f"   ·   SF {generation.structural_fidelity:.2f}", style=CYAN)
        if generation.witnesses:
            text.append(f"   ·   {len(generation.witnesses)} witness", style=RED)
        return text

    @staticmethod
    def _callout(generation: GenerationView | None) -> Text:
        if generation is None or not generation.ablation_rows:
            return Text("", style=DIM)
        witness = next(
            (row for row in generation.ablation_rows if row.get("verdict") == "witness"), None
        )
        if witness is None:
            return Text("", style=DIM)
        cases = int(
            round(_safe_float(generation.ablation.get("n_cases")) if generation.ablation else 0)
        )
        changed = int(round(_safe_float(witness.get("divergence")) * cases))
        role = witness.get("role_id", "role")
        return Text(
            f"✗ {role} changed {changed}/{cases} answers beyond noise floor — "
            "near-zero causal influence at full inference cost",
            style=f"{RED} on {RED_BG}",
        )


class CasesPanel(Panel):
    """The case row, with a keyboard-selectable cell and sub-result summary."""

    title_text = "EVAL CASES"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._case_ids: list[str] = []
        self._cursor = 0
        self._view: RunView | None = None

    def compose(self):
        yield Static(id="cases-summary")
        yield Static(id="cases-grid")
        yield Static(id="cases-detail")

    @property
    def selected_case_id(self) -> str | None:
        if not self._case_ids:
            return None
        return self._case_ids[max(0, min(self._cursor, len(self._case_ids) - 1))]

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        if generation is None or not generation.cases:
            return self.title_text
        return f"{self.title_text}  ·  {generation.cases_passed}/{len(generation.cases)} PASS"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or not generation.cases:
            return Text("no cases executed yet", style=DIM)
        selected = self.selected_case_id
        text = Text()
        for index, case in enumerate(generation.cases):
            case_id = str(case.get("case_id", "?"))
            passed = bool(case.get("passed"))
            marker = "✓" if passed else "✗"
            colour = GREEN if passed else RED
            if case_id == selected:
                text.append(f"[{marker}]", style=f"bold {colour} on {SELECTED}")
            else:
                text.append(f" {marker} ", style=colour)
            if index != len(generation.cases) - 1:
                text.append(" ", style=DIM)
        case = next((item for item in generation.cases if item.get("case_id") == selected), None)
        if case is not None:
            text.append("\n")
            text.append(self._case_detail(case))
        return text

    def update_view(self, view: RunView) -> None:
        self._view = view
        self.border_title = self.panel_title(view)
        generation = view.selected
        cases = generation.cases if generation is not None else []
        old_id = self.selected_case_id
        self._case_ids = [str(case.get("case_id", "?")) for case in cases]
        shared_id = view.selected_case_id
        if shared_id in self._case_ids:
            self._cursor = self._case_ids.index(shared_id)
        elif old_id in self._case_ids:
            self._cursor = self._case_ids.index(old_id)
        else:
            failed = next((index for index, case in enumerate(cases) if not case.get("passed")), 0)
            self._cursor = failed
        self.update(self.render_view(view))
        try:
            self.query_one("#cases-summary", Static).update(self._summary(view))
            self.query_one("#cases-grid", Static).update(self._grid(view))
            selected = next(
                (case for case in cases if case.get("case_id") == self.selected_case_id), None
            )
            self.query_one("#cases-detail", Static).update(
                self._case_detail(selected) if selected else ""
            )
        except Exception:  # pragma: no cover - only possible before child mount.
            pass

    def _grid(self, view: RunView) -> Text:
        generation = view.selected
        if generation is None:
            return Text("", style=DIM)
        selected = self.selected_case_id
        text = Text()
        for index, case in enumerate(generation.cases):
            case_id = str(case.get("case_id", "?"))
            passed = bool(case.get("passed"))
            colour = GREEN if passed else RED
            marker = "✓" if passed else "✗"
            text.append(f"{index + 1:02d}", style=DIM)
            text.append(
                marker,
                style=(f"bold {colour} on {SELECTED}" if case_id == selected else colour),
            )
            if index != len(generation.cases) - 1:
                text.append(" ", style=DIM)
        return text

    @staticmethod
    def _summary(view: RunView) -> Text:
        generation = view.selected
        if generation is None or not generation.cases:
            return Text("EVAL CASES  —", style=DIM)
        total = len(generation.cases)
        passed = generation.cases_passed
        failed = total - passed
        text = Text("EVAL CASES  ", style=f"bold {FG_BRIGHT}")
        text.append(f"{passed}/{total} PASS", style=GREEN)
        if failed:
            text.append(f"  ·  {failed} FAIL", style=RED)
        text.append(f"  ·  selected {view.selected_case_id or '—'}", style=DIM)
        return text

    @staticmethod
    def _case_detail(case: dict[str, Any] | None) -> Text:
        if not case:
            return Text("no case selected", style=DIM)
        case_id = str(case.get("case_id", "?"))
        passed = bool(case.get("passed"))
        state = "PASS" if passed else "FAIL"
        text = Text(f"{case_id} · {state}", style=GREEN if passed else RED)
        sub_results = case.get("sub_results") or {}
        if isinstance(sub_results, dict) and sub_results:
            sub_passed = sum(bool(value) for value in sub_results.values())
            failed = [str(key) for key, value in sub_results.items() if not value]
            text.append(f" · {sub_passed}/{len(sub_results)} invoices ✓", style=FG)
            if failed:
                text.append(f" · {', '.join(failed[:2])} ✗", style=RED)
        else:
            text.append(" · per-invoice results in case trace", style=DIM)
        text.append(" · inspect ↗", style=f"underline {AMBER_HI}")
        return text

    def on_key(self, event: Key) -> None:
        if not self._case_ids:
            return
        if event.key in {"up", "left"}:
            self._cursor = max(0, self._cursor - 1)
            event.stop()
            self._selection_changed()
        elif event.key in {"down", "right"}:
            self._cursor = min(len(self._case_ids) - 1, self._cursor + 1)
            event.stop()
            self._selection_changed()
        elif event.key == "enter":
            event.stop()
            self._selection_changed()
            action = getattr(self.app, "open_case_inspector", None)
            if action is not None:
                action(self.selected_case_id)

    def move_cursor(self, delta: int) -> None:
        """Move the case cursor for app-level arrow bindings."""

        if not self._case_ids:
            return
        self._cursor = max(0, min(len(self._case_ids) - 1, self._cursor + delta))
        self._selection_changed()

    def _selection_changed(self) -> None:
        if self._view is None:
            return
        self.update(self.render_view(self._view))
        try:
            cases = self._view.selected.cases if self._view.selected is not None else []
            selected = next(
                (case for case in cases if case.get("case_id") == self.selected_case_id), None
            )
            self.query_one("#cases-summary", Static).update(self._summary(self._view))
            self.query_one("#cases-grid", Static).update(self._grid(self._view))
            self.query_one("#cases-detail", Static).update(
                self._case_detail(selected) if selected else ""
            )
        except Exception:  # pragma: no cover - only possible before child mount.
            pass
        self.post_message(CaseSelectionChanged(self.selected_case_id))


class CaseEvidencePanel(Panel):
    """Selected-case trace summary below the case row.

    The compact ``execution.case`` event deliberately carries verdict, cost,
    and latency only.  This panel keeps the trace affordance visible without
    pretending those omitted fields are available in a fixture; richer
    projections are rendered when a producer supplies them.
    """

    title_text = "CASE TRACE"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        case = view.selected_case
        if generation is None or case is None:
            return Text("select a case to inspect", style=DIM)

        case_id = str(case.get("case_id", "?"))
        passed = bool(case.get("passed"))
        text = Text(f"{case_id}  ·  {'PASS' if passed else 'FAIL'}", style=GREEN if passed else RED)
        text.append(
            f"   cost {_money(case.get('cost_usd'))}   latency {case.get('latency_s', '—')}s\n",
            style=FG,
        )
        self._append_sub_results(text, case)
        text.append("TOOL RESPONSE HIGHLIGHT\n", style=f"bold {AMBER}")
        calls = _trace_calls(case)
        if not calls:
            text.append(
                "requested_date  →  rate_date   ",
                style=f"bold {AMBER_HI} on {AMBER_BG}",
            )
            text.append(
                "raw response fields are not present in this compact fixture; "
                "press i for the read-only inspector",
                style=DIM,
            )
            return text
        for call in calls[:2]:
            tool = str(call.get("tool") or call.get("name") or "tool")
            requested = _first_nested(call, "requested_date")
            response = call.get("response") or call.get("raw_response") or call
            rate_date = _first_nested(response, "rate_date")
            text.append(f"{tool}  ", style=f"bold {CYAN}")
            text.append(
                f"{requested or '—'} → {rate_date or '—'}",
                style=f"bold {AMBER_HI} on {AMBER_BG}",
            )
            text.append("\n", style=FG)
        return text

    @staticmethod
    def _append_sub_results(text: Text, case: dict[str, Any]) -> None:
        sub_results = case.get("sub_results") or {}
        if not isinstance(sub_results, dict) or not sub_results:
            text.append("per-invoice breakdown is available in the full case trace\n", style=DIM)
            return
        passed = sum(bool(value) for value in sub_results.values())
        text.append(f"invoices  {passed}/{len(sub_results)} pass", style=FG)
        failed = [str(key) for key, value in sub_results.items() if not value]
        if failed:
            text.append(f"  ·  failed {', '.join(failed[:3])}", style=RED)
        text.append("\n", style=FG)


class ArchitecturePanel(Panel):
    """A compact box-drawn DAG with the pruned role called out."""

    title_text = "ARCHITECTURE DAG"

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        return (
            self.title_text if generation is None else f"{self.title_text}  ·  {generation.label}"
        )

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or not generation.roles:
            return Text("no architecture proposed yet", style=DIM)
        text = Text()
        task_name = view.task_name or "task"
        text.append(f"{task_name}\n", style=f"bold {ACCENT}")
        text.append("│\n", style=WIRE)
        role_ids = {str(role.get("id")) for role in generation.roles}
        for index, role in enumerate(generation.roles):
            role_id = str(role.get("id", "?"))
            name = str(role.get("name", role_id))
            justification = str(role.get("justification", "unspecified"))
            prefix = "├─" if index < len(generation.roles) - 1 else "└─"
            text.append(f"{prefix} ", style=WIRE)
            text.append(_clip(name, 18), style=FG_BRIGHT)
            text.append(
                f"  [{_clip(justification, 11)}]\n",
                style=_justification_colour(justification),
            )
            inputs = [str(value) for value in role.get("inputs", []) if value != "task"]
            if inputs:
                text.append(f"│  ← {', '.join(_clip(value, 10) for value in inputs)}\n", style=DIM)
        mutation = generation.mutation or {}
        if mutation.get("type") == "prune":
            target = str(mutation.get("target_role", "role"))
            text.append(f"✗ {_clip(target, 18)}  (pruned)\n", style=f"strike {RED_DIM}")
        elif generation.generation > 0:
            previous = view.generation(generation.generation - 1)
            if previous is not None:
                previous_ids = {str(role.get("id")) for role in previous.roles}
                for removed in sorted(previous_ids - role_ids):
                    text.append(f"✗ {_clip(removed, 18)}  (pruned)\n", style=f"strike {RED_DIM}")
        return text


class GenerationMetricsPanel(Panel):
    """Selected-generation metrics in the right rail."""

    title_text = "GENERATION METRICS"

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        return self.title_text if generation is None else f"{generation.label.upper()}  METRICS"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or generation.metrics is None:
            return Text("no metrics yet", style=DIM)
        metrics = generation.metrics
        text = Text()
        text.append("ACCURACY  ", style=DIM)
        text.append(f"{metrics['pass_rate']:.1%}\n", style=f"bold {GREEN}")
        text.append("COST (L-R EQ.)  ", style=DIM)
        text.append(f"{_money(metrics['cost_usd'])}\n", style=AMBER)
        text.append("LATENCY  ", style=DIM)
        text.append(f"{metrics['latency_s_mean']:.1f}s\n", style=FG)
        text.append("CALLS/CASE  ", style=DIM)
        text.append(f"{metrics['tool_calls_per_case']:.1f}\n", style=CYAN)
        text.append("REL³  ", style=DIM)
        reliability = generation.reliability_pass3
        text.append("—\n" if reliability is None else f"{reliability:.2f}\n", style=PURPLE)
        text.append("SF  ", style=DIM)
        text.append(
            f"{_ratio(generation.structural_fidelity)} {_bar(generation.structural_fidelity, 5)}\n",
            style=CYAN,
        )
        baseline = generation.baseline
        if baseline is not None:
            comparison = metrics.get("vs_baseline", {})
            text.append("VS COT-SC  ", style=MAGENTA)
            text.append(
                f"k={baseline.get('k', '?')}  Δpass {comparison.get('pass_delta', 0):+.2f}\n",
                style=MAGENTA,
            )
        return text


class LessonsPanel(Panel):
    """Loaded/written lessons with their evidence pointers visible."""

    title_text = "LESSONS"
    MAX_LESSON_ROWS = 8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._view: RunView | None = None
        self._cursor = 0

    def compose(self):
        for index in range(self.MAX_LESSON_ROWS):
            yield Button(
                "",
                id=f"lesson-evidence-{index}",
                classes="lesson-evidence",
                variant="default",
            )

    def panel_title(self, view: RunView) -> str:
        loaded = len(view.lessons_loaded)
        written = len(view.lessons_written)
        return f"{self.title_text}  ·  {loaded} loaded  ·  {written} written"

    def render_view(self, view: RunView) -> RenderableType:
        if not view.lessons:
            return Text("no lessons yet", style=DIM)
        loaded_ids = {lesson.id for lesson in view.lessons_loaded}
        text = Text(
            f"LESSONS  {len(view.lessons_loaded)} loaded  ·  {len(view.lessons_written)} written\n",
            style=f"bold {FG_BRIGHT}",
        )
        for index, lesson in enumerate(view.lessons):
            text.append(self._lesson_row(lesson, lesson.id in loaded_ids))
            if index != len(view.lessons) - 1:
                text.append("\n", style=WIRE)
        return text

    def update_view(self, view: RunView) -> None:
        self._view = view
        self._cursor = min(self._cursor, max(0, len(view.lessons) - 1))
        self.border_title = self.panel_title(view)
        self.update(self.render_view(view))
        for index in range(self.MAX_LESSON_ROWS):
            try:
                button = self.query_one(f"#lesson-evidence-{index}", Button)
            except Exception:  # pragma: no cover - only possible before child mount.
                return
            if index >= len(view.lessons):
                button.display = False
                continue
            button.display = True
            lesson = view.lessons[index]
            button.label = self._evidence_row(lesson)

    @staticmethod
    def _lesson_row(lesson: Any, loaded: bool) -> Text:
        badge = "tool" if lesson.kind == "tool_note" else "rule"
        badge_colour = CYAN if badge == "tool" else AMBER
        text = Text()
        text.append(f"[{badge}] ", style=badge_colour)
        if lesson.tool:
            text.append(f"{lesson.tool}  ", style=f"bold {badge_colour}")
        text.append("● loaded" if loaded else "+ written", style=GREEN if loaded else DIM)
        text.append("\n", style=DIM)
        text.append(f"  {_clip(lesson.text, 62)}", style=FG)
        return text

    @staticmethod
    def _evidence_row(lesson: Any) -> Text:
        evidence = lesson.evidence.get("case_ids", [])
        available = bool(evidence)
        label = ", ".join(str(item) for item in evidence[:3]) if available else "unavailable"
        text = Text("↗ evidence ▸ ", style=f"bold {CYAN}" if available else DIM)
        text.append(label, style=FG if available else RED)
        return text

    def on_key(self, event: Key) -> None:
        if self._view is None or not self._view.lessons:
            return
        if event.key == "up":
            self._cursor = max(0, self._cursor - 1)
            event.stop()
        elif event.key == "down":
            self._cursor = min(len(self._view.lessons) - 1, self._cursor + 1)
            event.stop()
        elif event.key == "enter":
            event.stop()
            self._emit_lesson()

    def _emit_lesson(self) -> None:
        if self._view is None or not self._view.lessons:
            return
        self.post_message(LessonEvidenceRequested(self._view.lessons[self._cursor].id))


class ComparePanel(Panel):
    """Run-over-run strip populated from the committed ``compare.json``."""

    title_text = "COMPARE"

    def render_view(self, view: RunView) -> RenderableType:
        if not view.compare:
            return Text("single run  ·  no comparison recorded", style=DIM)
        run1 = view.compare.get("run1", {})
        run2 = view.compare.get("run2", {})
        text = Text()
        text.append(f"run1 g0 {float(run1.get('g0_pass_rate', 0)):.2f}", style=RED)
        text.append("  →  ", style=DIM)
        text.append(f"run2 g0 {float(run2.get('g0_pass_rate', 0)):.2f}", style=GREEN)
        text.append("   │   calls/case ", style=DIM)
        text.append(
            f"{run1.get('g0_tool_calls_per_case', '—')} → "
            f"{run2.get('g0_tool_calls_per_case', '—')}",
            style=CYAN,
        )
        text.append("   │   gens to plateau ", style=DIM)
        text.append(
            f"{run1.get('generations_to_plateau', '—')} → "
            f"{run2.get('generations_to_plateau', '—')}",
            style=AMBER,
        )
        text.append("   │   ↑ lessons loaded from run 1", style=GREEN)
        return text


class MetricsStrip(Panel):
    """Bottom evidence strip: pass, cost, latency, calls/case and rel³."""

    title_text = "METRICS"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or generation.metrics is None:
            return Text("no metrics yet", style=DIM)
        metrics = generation.metrics
        reliability = generation.reliability_pass3
        text = Text()
        text.append("ACCURACY  ", style=DIM)
        text.append(f"{metrics['pass_rate']:.1%}  {_bar(metrics['pass_rate'])}", style=GREEN)
        text.append("    COST (L-R EQ.)  ", style=DIM)
        text.append(f"{_money(metrics['cost_usd'])}", style=AMBER)
        text.append("    LATENCY  ", style=DIM)
        text.append(f"{metrics['latency_s_mean']:.1f}s", style=FG)
        text.append("    CALLS/CASE  ", style=DIM)
        text.append(f"{metrics['tool_calls_per_case']:.1f}", style=CYAN)
        text.append("    REL³  ", style=DIM)
        text.append("—" if reliability is None else f"{reliability:.2f}", style=PURPLE)
        text.append("\nVS COT-SC  ", style=MAGENTA)
        baseline = generation.baseline
        if baseline is None:
            text.append("not recorded", style=DIM)
        else:
            comparison = metrics.get("vs_baseline", {})
            text.append(
                f"k={baseline.get('k', '?')}  pass {baseline.get('pass_rate', 0):.2f}"
                f"  ·  Δpass {comparison.get('pass_delta', 0):+.2f}"
                f"  ·  cost ×{comparison.get('cost_ratio', 0):.2f}",
                style=MAGENTA,
            )
        return text


class BaselinePanel(Panel):
    """Cost-matched CoT-SC runs, toggled with ``b``."""

    title_text = "BASELINE"

    def render_view(self, view: RunView) -> RenderableType:
        rows = [
            (generation, generation.baseline)
            for generation in view.generations
            if generation.baseline is not None
        ]
        if not rows:
            return Text("no baseline recorded", style=DIM)
        text = Text()
        for generation, baseline in rows:
            text.append(
                f"{generation.label}  k={baseline.get('k', '?')}"
                f"  pass {baseline.get('pass_rate', 0):.2f}"
                f"  cost {_money(baseline.get('cost_usd'))}"
                f"  matched {_money(baseline.get('matched_to_cost_usd'))}\n",
                style=MAGENTA,
            )
        return text


class DiagnosisFeed(RichLog):
    """Streaming diagnosis, mutation and log feed."""

    DEFAULT_CLASSES = "view-panel"
    FEED_TYPES = frozenset({"diagnosis.emitted", "mutation.applied", "mutation.reverted", "log"})

    def __init__(self) -> None:
        super().__init__(id="diagnosis", wrap=True, markup=False, auto_scroll=True)
        self.border_title = "DIAGNOSIS"

    def update_view(self, view: RunView) -> None:
        generation = view.selected
        self.border_title = (
            "DIAGNOSIS" if generation is None else f"DIAGNOSIS  ·  {generation.label}"
        )

    def prime(self, events: Sequence[Event]) -> None:
        self.ingest(events)

    def ingest(self, events: Sequence[Event]) -> None:
        for event in events:
            if event.type in self.FEED_TYPES:
                self.write(self._line(event))

    @staticmethod
    def _line(event: Event) -> Text:
        data = event.data
        generation = data.get("generation")
        text = Text("▸ ", style=ACCENT)
        text.append(f"g{generation} " if generation is not None else "run ", style=DIM)
        if event.type == "diagnosis.emitted":
            text.append(str(data.get("text", "")), style=FG)
            mutation = data.get("chosen_mutation") or {}
            if mutation:
                target = f"{mutation.get('type')} {mutation.get('target_role', '')}".rstrip()
                text.append(f"  → {target}", style=AMBER)
        elif event.type == "mutation.applied":
            text.append(
                f"{data.get('type')} {data.get('target_role', '')} · {data.get('diff', '')}",
                style=CYAN,
            )
        elif event.type == "mutation.reverted":
            text.append(
                f"reverted → g{data.get('restored_to')} · {data.get('reason', '')}",
                style=RED,
            )
        else:
            text.append(str(data.get("message", "")), style=DIM)
        return text


class FooterBar(Static):
    """Keyboard map and the read-only boundary."""

    KEYS = (
        "←/→ gen   a ablation   c cases   d diagnosis   l lessons   b baseline"
        "   i inspect   space pause   . step   +/- speed   ? help   q quit"
    )

    def on_mount(self) -> None:
        self.update(Text(self.KEYS, style=DIM, no_wrap=True, overflow="ellipsis"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


PANEL_TYPES: dict[str, type[Panel]] = {
    "ablation": AblationPanel,
    "architecture": ArchitecturePanel,
    "baseline": BaselinePanel,
    "cases": CasesPanel,
    "evidence": CaseEvidencePanel,
    "compare": ComparePanel,
    "generation-metrics": GenerationMetricsPanel,
    "lessons": LessonsPanel,
    "metrics": MetricsStrip,
    "structural": StructuralFidelityPanel,
}


__all__ = [
    "PANEL_TYPES",
    "AblationPanel",
    "ArchitecturePanel",
    "BaselinePanel",
    "CaseSelectionChanged",
    "CasesPanel",
    "CaseEvidencePanel",
    "ComparePanel",
    "DiagnosisFeed",
    "FooterBar",
    "GenerationMetricsPanel",
    "HeaderBar",
    "LessonsPanel",
    "LineagePanel",
    "LessonEvidenceRequested",
    "MetricsStrip",
    "Panel",
    "StructuralFidelityPanel",
    "ViewPanel",
]


def _first_nested(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _first_nested(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_nested(value, key)
            if found is not None:
                return found
    return None


def _trace_calls(case: dict[str, Any]) -> list[dict[str, Any]]:
    direct = case.get("tool_calls") or case.get("raw_tool_responses")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    return []
