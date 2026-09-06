"""Shell panels and the seam WP-09's detailed panels plug into.

Every region of the screen is a widget that implements ``update_view(view)``.
The app calls it on each state change and never touches panel internals, so
WP-09 can replace a placeholder body — or a whole widget class, via
:data:`PANEL_TYPES` — without touching the app, the reducer or the replay driver.

WP-07 owns the header, the lineage tree, the diagnosis feed and the footer in
full.  The remaining panels render a summary derived from the same
:class:`~occam.tui.viewmodel.RunView` and are the WP-09 seams.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rich.console import RenderableType
from rich.text import Text
from textual.widgets import RichLog, Static, Tree
from textual.widgets.tree import TreeNode

from occam.core.models import Event
from occam.tui.palette import ACCENT, AMBER, CYAN, DIM, FG, GREEN, MAGENTA, RED
from occam.tui.viewmodel import GenerationView, RunView

GAP = "   "


@runtime_checkable
class ViewPanel(Protocol):
    """Anything the app repaints when the reduced state changes."""

    def update_view(self, view: RunView) -> None: ...


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.4f}"


def _ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


class Panel(Static):
    """A titled region of the shell.

    Subclasses override :meth:`render_view`, and may override
    :meth:`panel_title`.  ``update_view`` is deliberately the only entry point
    the app uses.
    """

    DEFAULT_CLASSES = "view-panel"
    can_focus = True
    title_text = ""

    def on_mount(self) -> None:
        self.border_title = self.title_text

    def update_view(self, view: RunView) -> None:
        self.border_title = self.panel_title(view)
        self.update(self.render_view(view))

    def panel_title(self, view: RunView) -> str:
        del view  # stable seam: WP-09 titles may depend on the view
        return self.title_text

    def render_view(self, view: RunView) -> RenderableType:
        del view
        return ""


class HeaderBar(Static):
    """``OCCAM  fx_recon_b · FX revaluation  run 2/2  lessons loaded 3``."""

    DEFAULT_CLASSES = "view-panel"

    def update_view(self, view: RunView) -> None:
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append("OCCAM", style=f"bold {ACCENT}")
        if view.task_label:
            text.append("  ")
            text.append(view.task_label, style=FG)
        text.append(GAP)
        loaded = len(view.lessons_loaded)
        if loaded:
            text.append(f"run {view.run_index}/{view.run_index}", style=DIM)
            text.append(GAP)
            text.append(f"lessons loaded {loaded}", style=GREEN)
        else:
            text.append(f"run {view.run_index}", style=DIM)
            text.append(GAP)
            text.append("lessons 0", style=DIM)
        text.append(GAP)
        text.append(view.generation_label, style=DIM)
        text.append(GAP)
        badge = AMBER if view.mode == "replay" else GREEN
        text.append(view.mode_badge, style=f"bold {badge}")
        text.append(GAP)
        text.append(f"{view.elapsed_s:,.0f}s", style=DIM)
        text.append(GAP)
        text.append(_money(view.spend_usd), style=DIM)
        self.update(text)


class LineagePanel(Tree[int]):
    """One node per generation (04 §3.2). Selecting a node moves the screen."""

    DEFAULT_CLASSES = "view-panel"

    def __init__(self) -> None:
        super().__init__("lineage", id="lineage")
        self.show_root = False
        self.guide_depth = 2
        self.border_title = "LINEAGE"
        self._signature: tuple[Any, ...] | None = None

    def update_view(self, view: RunView) -> None:
        generations = view.generations
        self.border_title = f"LINEAGE · {len(generations)} gen"
        signature = tuple(
            (
                generation.generation,
                generation.n_roles,
                generation.pass_rate,
                generation.cost_usd,
                generation.reverted,
                generation.generation == view.best_generation,
                generation.generation == view.selected_generation,
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
                # Move the highlight without selecting: a repaint must not
                # look like a user selection, which would pin the screen to
                # this generation.  ``move_cursor`` was added after Textual
                # 0.70; setting the reactive cursor line is its equivalent.
                move_cursor = getattr(self, "move_cursor", None)
                if move_cursor is not None:
                    move_cursor(node)
                else:  # Textual 0.70 compatibility
                    self.cursor_line = node.line

    @staticmethod
    def _label(generation: GenerationView, view: RunView) -> Text:
        if generation.reverted:
            marker, style = "○", DIM
        elif generation.generation == view.selected_generation:
            marker, style = "◉", f"bold {CYAN}"
        else:
            marker, style = "●", FG
        label = Text(f"{marker} {generation.label}", style=style)
        if generation.generation == view.best_generation:
            label.append(" ★", style=AMBER)
        label.append(f"  {generation.n_roles} roles", style=DIM)
        label.append(f"  {_ratio(generation.pass_rate)}", style=DIM)
        label.append(f"  {_money(generation.cost_usd)}", style=DIM)
        if generation.reverted:
            label.append("  reverted", style=DIM)
        return label


class AblationPanel(Panel):
    """WP-09 seam: replace the body with the hero ``DataTable`` (04 §3.4)."""

    title_text = "ABLATION TABLE"

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        if generation is None:
            return self.title_text
        return f"{self.title_text} · {generation.label} · {generation.n_roles} roles · LOO approx."

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or generation.ablation is None:
            return Text("no ablation for this generation yet", style=DIM)
        rows = generation.ablation_rows
        roles = generation.ablation_roles or [row["role_id"] for row in rows]
        text = Text()
        text.append(f"ablated {len(rows)}/{len(roles)} roles", style=FG)
        fidelity = generation.structural_fidelity
        if fidelity is not None:
            text.append("  ·  SF ", style=DIM)
            text.append(_ratio(fidelity), style=FG)
        witnesses = generation.witnesses
        if witnesses:
            text.append(f"  ·  {len(witnesses)} witness", style=RED)
        return text


class CasesPanel(Panel):
    """WP-09 seam: replace with the ✓/✗ grid and per-invoice sub-results."""

    title_text = "EVAL CASES"

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        if generation is None or not generation.cases:
            return self.title_text
        return f"{self.title_text} · {generation.cases_passed}/{len(generation.cases)} pass"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or not generation.cases:
            return Text("no cases executed yet", style=DIM)
        text = Text()
        for case in generation.cases:
            passed = bool(case.get("passed"))
            text.append("✓ " if passed else "✗ ", style=GREEN if passed else RED)
        return text


class ArchitecturePanel(Panel):
    """WP-09 seam: replace with the box-drawn DAG and prune strikethrough."""

    title_text = "ARCHITECTURE DAG"

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        return self.title_text if generation is None else f"{self.title_text} · {generation.label}"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or not generation.roles:
            return Text("no architecture proposed yet", style=DIM)
        text = Text()
        for role in generation.roles:
            text.append(f"{role.get('name', role.get('id', '?'))}\n", style=FG)
            text.append(f"  {role.get('justification', 'unspecified')}\n", style=DIM)
        return text


class GenerationMetricsPanel(Panel):
    """WP-09 seam: adds ``calls/case`` and ``rel³`` per 04 §1b."""

    title_text = "GENERATION"

    def panel_title(self, view: RunView) -> str:
        generation = view.selected
        return self.title_text if generation is None else generation.label.upper()

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or generation.metrics is None:
            return Text("no metrics yet", style=DIM)
        metrics = generation.metrics
        text = Text()
        text.append(f"accuracy   {metrics['pass_rate'] * 100:,.1f}%\n", style=GREEN)
        text.append(f"cost       {_money(metrics['cost_usd'])}\n", style=FG)
        text.append(f"latency    {metrics['latency_s_mean']:,.1f}s\n", style=FG)
        return text


class LessonsPanel(Panel):
    """WP-09 seam: replace with lesson rows, ``● loaded`` badges and evidence."""

    title_text = "LESSONS"

    def panel_title(self, view: RunView) -> str:
        loaded = len(view.lessons_loaded)
        written = len(view.lessons_written)
        return f"{self.title_text} · {loaded} loaded · {written} written"

    def render_view(self, view: RunView) -> RenderableType:
        if not view.lessons:
            return Text("no lessons yet", style=DIM)
        loaded_ids = {lesson.id for lesson in view.lessons_loaded}
        text = Text()
        for lesson in view.lessons:
            kind = "tool" if lesson.kind == "tool_note" else "rule"
            text.append(f"[{kind}] ", style=CYAN)
            text.append(
                "● loaded\n" if lesson.id in loaded_ids else "＋ written\n",
                style=DIM if lesson.id in loaded_ids else GREEN,
            )
        return text


class ComparePanel(Panel):
    """WP-09 seam: run-over-run strip, fed from ``compare.json`` (run 2 only)."""

    title_text = "COMPARE"

    def render_view(self, view: RunView) -> RenderableType:
        if not view.compare:
            return Text("single run — no comparison recorded", style=DIM)
        run1 = view.compare.get("run1", {})
        run2 = view.compare.get("run2", {})
        text = Text()
        text.append(
            f"run1 g0 {run1.get('g0_pass_rate', '—')} · run2 g0 {run2.get('g0_pass_rate', '—')}",
            style=FG,
        )
        text.append(
            f"   │   calls/case {run1.get('g0_tool_calls_per_case', '—')}"
            f" → {run2.get('g0_tool_calls_per_case', '—')}",
            style=DIM,
        )
        text.append(
            f"   │   gens to plateau {run1.get('generations_to_plateau', '—')}"
            f" → {run2.get('generations_to_plateau', '—')}",
            style=DIM,
        )
        return text


class MetricsStrip(Panel):
    """WP-09 seam: sparklines per 04 §3.7 plus the magenta baseline line."""

    title_text = "METRICS"

    def render_view(self, view: RunView) -> RenderableType:
        generation = view.selected
        if generation is None or generation.metrics is None:
            return Text("no metrics yet", style=DIM)
        metrics = generation.metrics
        text = Text()
        text.append(f"pass {metrics['pass_rate']:.2f}", style=GREEN)
        text.append(f"   cost {_money(metrics['cost_usd'])}", style=FG)
        text.append(f"   lat {metrics['latency_s_mean']:,.1f}s", style=FG)
        text.append(f"   calls/case {metrics['tool_calls_per_case']:,.1f}", style=FG)
        text.append(f"   SF {_ratio(metrics['structural_fidelity'])}", style=FG)
        reliability = metrics.get("reliability_pass3")
        if reliability is not None:
            text.append(f"   rel³ {reliability:.2f}", style=CYAN)
        baseline = generation.baseline
        if baseline is not None:
            comparison = metrics["vs_baseline"]
            text.append(
                f"\nvs CoT-SC(k={baseline.get('k', '?')}) pass {baseline.get('pass_rate', 0):.2f}"
                f" · Δpass {comparison['pass_delta']:+.2f}"
                f" · cost ×{comparison['cost_ratio']:.2f}",
                style=MAGENTA,
            )
        return text


class BaselinePanel(Panel):
    """Cost-matched CoT-SC runs per generation (04 §3.9); toggled with ``b``."""

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
    """Streams diagnosis, mutation and log events with a ``▸`` prefix (04 §3.8)."""

    DEFAULT_CLASSES = "view-panel"
    FEED_TYPES = frozenset({"diagnosis.emitted", "mutation.applied", "mutation.reverted", "log"})

    def __init__(self) -> None:
        super().__init__(id="diagnosis", wrap=True, markup=False, auto_scroll=True)
        self.border_title = "DIAGNOSIS"

    def update_view(self, view: RunView) -> None:
        generation = view.selected
        self.border_title = "DIAGNOSIS" if generation is None else f"DIAGNOSIS · {generation.label}"

    def ingest(self, events: list[Event]) -> None:
        """Append feed lines for the events that just arrived."""

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
    """The keymap line, and the standing reminder that this view is read-only."""

    KEYS = (
        "←/→ gen · a ablation · c cases · d diagnosis · l lessons · b baseline"
        " · i inspect · space pause · . step · +/- speed · ? help · q quit"
    )

    def on_mount(self) -> None:
        self.update(Text(self.KEYS, style=DIM, no_wrap=True, overflow="ellipsis"))


#: WP-09 swaps concrete classes in here; the app only knows the widget ids.
PANEL_TYPES: dict[str, type[Panel]] = {
    "ablation": AblationPanel,
    "architecture": ArchitecturePanel,
    "baseline": BaselinePanel,
    "cases": CasesPanel,
    "compare": ComparePanel,
    "generation-metrics": GenerationMetricsPanel,
    "lessons": LessonsPanel,
    "metrics": MetricsStrip,
}


__all__ = [
    "PANEL_TYPES",
    "AblationPanel",
    "ArchitecturePanel",
    "BaselinePanel",
    "CasesPanel",
    "ComparePanel",
    "DiagnosisFeed",
    "FooterBar",
    "GenerationMetricsPanel",
    "HeaderBar",
    "LessonsPanel",
    "LineagePanel",
    "MetricsStrip",
    "Panel",
    "ViewPanel",
]
