"""Shell, replay-driver and read-only-contract tests for the Occam TUI."""

from __future__ import annotations

import ast
import asyncio
import socket
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from occam.cli import app as cli_app
from occam.core.models import Event, GenerationState, State
from occam.store.reader import EventReader
from occam.store.reducer import Reduction, reduce, state_json_bytes
from occam.tui.app import OccamApp
from occam.tui.panels import AblationPanel, DiagnosisFeed, HeaderBar
from occam.tui.source import (
    LiveSource,
    ReplayOptions,
    ReplaySource,
    ReplayTargetNotFound,
    fast_forward_index,
)
from occam.tui.viewmodel import RunView

ROOT = Path(__file__).resolve().parents[1]
RUN1 = ROOT / "fixtures" / "demo_run1"
RUN2 = ROOT / "fixtures" / "demo_run2"
FIXTURES = (RUN1, RUN2)
RUNNER = CliRunner()

#: A first paint must land well inside the demo's patience budget.
FIRST_PAINT_BUDGET_S = 2.0


async def _instant(_delay: float) -> None:
    """Replace the replay clock so tests are not bound to wall time."""

    await asyncio.sleep(0)


def _events(run_dir: Path) -> list[Event]:
    return EventReader(run_dir).read()


def _replay(run_dir: Path, **options: Any) -> ReplaySource:
    return ReplaySource(run_dir, ReplayOptions(**options), sleep=_instant)


def _header(app: OccamApp) -> str:
    return app.query_one(HeaderBar).renderable.plain


def _panel_text(app: OccamApp, panel_id: str) -> str:
    return app.query_one(f"#{panel_id}").renderable.plain


def _drive(scenario: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
    return asyncio.run(scenario(*args, **kwargs))


async def _run_app(app: OccamApp, body: Callable[[Any], Awaitable[Any]]) -> Any:
    async with app.run_test() as pilot:
        return await body(pilot)


# -- replay driver ------------------------------------------------------


@pytest.mark.parametrize("run_dir", FIXTURES, ids=lambda path: path.name)
def test_replay_reaches_the_end_of_both_fixtures(run_dir: Path) -> None:
    expected = _events(run_dir)

    async def scenario() -> tuple[int, bool, bytes]:
        source = _replay(run_dir, speed=64.0)
        app = OccamApp(run_dir, source=source)

        async def body(pilot: Any) -> tuple[int, bool, bytes]:
            await asyncio.wait_for(source.wait_finished(), timeout=60)
            await pilot.pause()
            return (
                app.feed.count,
                app.feed.state.completed,
                state_json_bytes(app.feed.state),
            )

        return await _run_app(app, body)

    count, completed, replayed_state = _drive(scenario)

    assert count == len(expected)
    assert completed
    # Replay is indistinguishable from live: the same events, the same reducer,
    # therefore byte-identical state.
    assert replayed_state == state_json_bytes(reduce(expected))


@pytest.mark.parametrize("run_dir", FIXTURES, ids=lambda path: path.name)
def test_live_attach_reproduces_the_same_state_as_replay(run_dir: Path) -> None:
    async def scenario() -> bytes:
        source = LiveSource(run_dir, follow=False, poll_interval=0)
        app = OccamApp(run_dir, source=source)

        async def body(pilot: Any) -> bytes:
            await asyncio.wait_for(source.wait_finished(), timeout=60)
            await pilot.pause()
            return state_json_bytes(app.feed.state)

        return await _run_app(app, body)

    assert _drive(scenario) == state_json_bytes(reduce(_events(run_dir)))


def test_incremental_reducer_continues_from_a_snapshot() -> None:
    events = _events(RUN1)
    checkpoint = 15
    reduction = Reduction()

    assert reduction.prime(reduce(events[:checkpoint])).last_seq == checkpoint - 1
    actual = reduction.extend(events[checkpoint:])

    assert reduction.last_seq == len(events) - 1
    assert state_json_bytes(actual) == state_json_bytes(reduce(events))


def test_live_tail_starts_after_the_loaded_snapshot(tmp_path: Path) -> None:
    events = _events(RUN1)
    checkpoint = 15
    run_dir = tmp_path / "stale-snapshot"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_bytes((RUN1 / "events.jsonl").read_bytes())
    (run_dir / "state.json").write_bytes(state_json_bytes(reduce(events[:checkpoint])))

    async def scenario() -> tuple[list[int], bool]:
        source = LiveSource(run_dir, follow=False, poll_interval=0)
        loaded = source.initial_state()
        received: list[int] = []

        await source.run(lambda batch: received.extend(event.seq for event in batch))
        return received, loaded is not None

    received, loaded = _drive(scenario)
    assert loaded
    assert received == list(range(checkpoint, len(events)))


def test_live_attach_to_a_completed_snapshot_does_not_poll_forever(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed"
    run_dir.mkdir()
    events = _events(RUN1)
    (run_dir / "events.jsonl").write_bytes((RUN1 / "events.jsonl").read_bytes())
    (run_dir / "state.json").write_bytes(state_json_bytes(reduce(events)))

    async def scenario() -> tuple[bool, list[int]]:
        source = LiveSource(run_dir)
        assert source.initial_state() is not None
        received: list[int] = []
        await asyncio.wait_for(source.run(lambda batch: received.extend(e.seq for e in batch)), 1)
        return source.finished, received

    finished, received = _drive(scenario)
    assert finished
    assert received == []


def test_to_gen_fast_forwards_to_the_first_event_of_that_generation() -> None:
    events = _events(RUN1)
    index = fast_forward_index(events, to_gen=3)

    assert events[index].data.get("generation") != 3
    assert events[index + 1].data.get("generation") == 3
    assert all(event.data.get("generation") != 3 for event in events[: index + 1])

    async def scenario() -> tuple[int, int | None]:
        source = _replay(RUN1, to_gen=3, paused=True)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> tuple[int, int | None]:
            await pilot.pause()
            return app.feed.count, app.view.selected_generation

        return await _run_app(app, body)

    count, selected = _drive(scenario)
    assert count == index + 1
    assert selected == 2


def test_at_lands_on_the_first_event_of_that_type() -> None:
    events = _events(RUN1)
    index = fast_forward_index(events, at="ablation.completed")

    assert events[index].type == "ablation.completed"
    assert events[index].data["generation"] == 0

    async def scenario() -> tuple[int, str, str]:
        source = _replay(RUN1, at="ablation.completed", paused=True)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> tuple[int, str, str]:
            await pilot.pause()
            return (
                app.feed.count,
                _panel_text(app, "ablation"),
                app.query_one("#ablation").border_title,
            )

        return await _run_app(app, body)

    count, ablation, title = _drive(scenario)
    assert count == index + 1
    # The cold open lands on a fully populated ablation table.
    assert "ablated 5/5 roles" in ablation
    assert "1 witness" in ablation
    assert "g0" in title


def test_at_is_scoped_to_to_gen() -> None:
    events = _events(RUN1)

    index = fast_forward_index(events, to_gen=2, at="ablation.completed")

    assert events[index].type == "ablation.completed"
    assert events[index].data["generation"] == 2
    assert index > fast_forward_index(events, at="ablation.completed")


def test_ablation_noise_floor_renders_before_rows_arrive() -> None:
    state = State(
        run_id="run",
        current_generation=0,
        generations={
            "g000": GenerationState(
                generation=0,
                ablation={
                    "generation": 0,
                    "roles": ["r_rates"],
                    "rows": [],
                    "noise_rate": 0.1,
                },
            )
        },
    )
    view = RunView(state)

    assert view.selected is not None
    assert view.selected.noise_rate == pytest.approx(0.1)
    assert "ablated 0/1 roles" in AblationPanel().render_view(view).plain
    assert "noise floor 0.10" in AblationPanel().render_view(view).plain


def test_at_and_to_gen_reject_targets_that_are_not_in_the_log() -> None:
    events = _events(RUN2)

    with pytest.raises(ReplayTargetNotFound):
        fast_forward_index(events, to_gen=7)
    with pytest.raises(ReplayTargetNotFound):
        fast_forward_index(events, at="mutation.reverted")
    with pytest.raises(ReplayTargetNotFound):
        # demo_run2 has no lesson.written at all, let alone inside generation 1.
        fast_forward_index(events, to_gen=1, at="lesson.written")
    with pytest.raises(ReplayTargetNotFound):
        # Run completion is outside any generation and must not escape --to-gen.
        fast_forward_index(events, to_gen=1, at="run.completed")


def test_pause_starts_paused_and_step_advances_exactly_one_event() -> None:
    events = _events(RUN1)
    index = fast_forward_index(events, at="architecture.proposed")

    async def scenario() -> list[int]:
        source = _replay(RUN1, at="architecture.proposed", paused=True)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> list[int]:
            await pilot.pause()
            counts = [app.feed.count]
            # A paused replay stays on its frame however long we wait.
            await asyncio.sleep(0.05)
            await pilot.pause()
            counts.append(app.feed.count)
            await pilot.press("full_stop")
            await asyncio.sleep(0.05)
            await pilot.pause()
            counts.append(app.feed.count)
            await pilot.press("space")
            await asyncio.wait_for(source.wait_finished(), timeout=60)
            await pilot.pause()
            counts.append(app.feed.count)
            return counts

        return await _run_app(app, body)

    paused, still_paused, stepped, resumed = _drive(scenario)
    assert paused == still_paused == index + 1
    assert stepped == index + 2
    assert resumed == len(events)


def test_pause_interrupts_an_in_flight_replay_delay() -> None:
    async def scenario() -> list[int]:
        delay_started = asyncio.Event()
        calls = 0

        async def blocking_sleep(_delay: float) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                delay_started.set()
                await asyncio.Event().wait()

        source = ReplaySource(RUN1, ReplayOptions(), sleep=blocking_sleep)
        received: list[int] = []
        task = asyncio.create_task(source.run(lambda batch: received.extend(e.seq for e in batch)))
        await asyncio.wait_for(delay_started.wait(), 1)
        source.pause()
        await asyncio.sleep(0)
        assert received == [0]

        source.step()
        for _ in range(20):
            await asyncio.sleep(0)
            if len(received) == 2:
                break
        assert received == [0, 1]
        source.resume()
        await asyncio.wait_for(task, 2)
        return received

    assert _drive(scenario) == list(range(len(_events(RUN1))))


def test_space_pauses_a_running_replay() -> None:
    async def scenario() -> tuple[bool, bool]:
        source = _replay(RUN1, speed=1.0)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> tuple[bool, bool]:
            await pilot.pause()
            running = source.paused
            await pilot.press("space")
            await pilot.pause()
            return running, source.paused

        return await _run_app(app, body)

    running, paused = _drive(scenario)
    assert not running
    assert paused


def test_speed_keys_bound_the_replay_rate() -> None:
    source = _replay(RUN1, speed=4.0)

    assert source.nudge_speed(2.0) == 8.0
    assert source.nudge_speed(0.5) == 4.0
    for _ in range(12):
        source.nudge_speed(2.0)
    assert source.speed == 64.0
    for _ in range(12):
        source.nudge_speed(0.5)
    assert source.speed == 0.25


def test_replay_rejects_a_non_positive_speed() -> None:
    with pytest.raises(ValueError, match="--speed"):
        ReplayOptions(speed=0.0)


def test_long_gaps_are_capped_so_the_demo_never_stalls() -> None:
    source = _replay(RUN1, speed=1.0)
    delays = [source._delay(index) for index in range(1, source.total_events)]

    assert max(delays) <= ReplayOptions().max_gap_s
    assert min(delays) >= 0.0


# -- header, panels and first paint -------------------------------------


def test_demo_run2_header_reports_the_lessons_it_started_with() -> None:
    async def scenario() -> tuple[str, str]:
        source = _replay(RUN2, at="run.started", paused=True)
        app = OccamApp(RUN2, source=source)

        async def body(pilot: Any) -> tuple[str, str]:
            await pilot.pause()
            return _header(app), app.query_one("#lessons").border_title

        return await _run_app(app, body)

    header, lessons_title = _drive(scenario)
    assert "lessons loaded 3" in header
    assert "run 2/2" in header
    assert "fx_recon_b" in header
    assert "3 loaded" in lessons_title


def test_demo_run1_header_reports_a_cold_start() -> None:
    async def scenario() -> str:
        source = _replay(RUN1, at="run.started", paused=True)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> str:
            await pilot.pause()
            return _header(app)

        return await _run_app(app, body)

    header = _drive(scenario)
    assert "run 1" in header
    assert "lessons 0" in header
    assert "REPLAY" in header


def test_replay_finished_state_uses_the_done_badge_and_full_generation_cost() -> None:
    view = RunView(reduce(_events(RUN1)), mode="replay")
    generation = view.generation(0)

    assert view.mode_badge == "■ DONE"
    assert generation is not None
    assert generation.cost_usd == pytest.approx(0.0633)
    assert generation.total_spend_usd is not None
    assert generation.total_spend_usd > generation.cost_usd


@pytest.mark.parametrize("run_dir", FIXTURES, ids=lambda path: path.name)
def test_first_paint_is_offline_and_under_two_seconds(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any, *args: Any) -> Any:
        # Loopback is asyncio's own wakeup pipe on Windows; anything else would
        # be the TUI reaching for the network, which it must never do.
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError(f"the TUI must never open a network socket (to {host!r})")
        return real_connect(self, address, *args)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    async def scenario() -> tuple[float, str]:
        started = time.perf_counter()
        app = OccamApp(run_dir, replay=ReplayOptions(speed=1.0))

        async def body(pilot: Any) -> tuple[float, str]:
            await pilot.pause()
            return time.perf_counter() - started, _header(app)

        return await _run_app(app, body)

    elapsed, header = _drive(scenario)
    assert elapsed < FIRST_PAINT_BUDGET_S, f"first paint took {elapsed:.2f}s"
    # A frame with real content, not an empty shell.
    assert "OCCAM" in header
    assert "fx_recon" in header


def test_generation_keys_walk_the_lineage() -> None:
    async def scenario() -> list[int | None]:
        source = _replay(RUN1, speed=64.0)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> list[int | None]:
            await asyncio.wait_for(source.wait_finished(), timeout=60)
            await pilot.pause()
            seen = [app.view.selected_generation]
            for _ in range(3):
                await pilot.press("left")
            await pilot.pause()
            seen.append(app.view.selected_generation)
            await pilot.press("right")
            await pilot.pause()
            seen.append(app.view.selected_generation)
            return seen

        return await _run_app(app, body)

    final, walked_back, forward = _drive(scenario)
    assert final == 4
    assert walked_back == 1
    assert forward == 2


def test_diagnosis_feed_streams_the_narration_events() -> None:
    async def scenario() -> int:
        source = _replay(RUN1, speed=64.0)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> int:
            await asyncio.wait_for(source.wait_finished(), timeout=60)
            await pilot.pause()
            return len(app.query_one(DiagnosisFeed).lines)

        return await _run_app(app, body)

    expected = sum(1 for event in _events(RUN1) if event.type in DiagnosisFeed.FEED_TYPES)
    assert _drive(scenario) >= expected


def test_baseline_panel_toggles_with_b() -> None:
    async def scenario() -> list[bool]:
        source = _replay(RUN1, at="baseline.completed", paused=True)
        app = OccamApp(RUN1, source=source)

        async def body(pilot: Any) -> list[bool]:
            await pilot.pause()
            states = [app.query_one("#baseline").display]
            await pilot.press("b")
            await pilot.pause()
            states.append(app.query_one("#baseline").display)
            return states

        return await _run_app(app, body)

    hidden, shown = _drive(scenario)
    assert not hidden
    assert shown


# -- the read-only contract ---------------------------------------------


@pytest.mark.parametrize("run_dir", FIXTURES, ids=lambda path: path.name)
def test_replay_never_writes_into_the_run_directory(run_dir: Path) -> None:
    def fingerprint() -> set[tuple[str, int, int]]:
        return {
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(run_dir.iterdir())
        }

    before = fingerprint()

    async def scenario() -> None:
        source = _replay(run_dir, speed=64.0)
        app = OccamApp(run_dir, source=source)

        async def body(pilot: Any) -> None:
            await asyncio.wait_for(source.wait_finished(), timeout=60)
            await pilot.pause()

        await _run_app(app, body)

    _drive(scenario)
    assert fingerprint() == before


def test_the_tui_package_never_imports_the_engine() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "occam" / "tui").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders.extend(
                f"{path.name}: {name}"
                for name in names
                if name.startswith(("occam.engine", "occam.llm", "occam.tools"))
            )
    assert offenders == []


def test_the_tui_package_opens_no_file_for_writing() -> None:
    write_modes = {"w", "a", "x", "wb", "ab", "xb", "w+", "a+", "r+"}
    offenders: list[str] = []
    for path in sorted((ROOT / "occam" / "tui").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name not in {"open", "write_text", "write_bytes", "mkdir", "touch", "unlink"}:
                continue
            if name != "open":
                offenders.append(f"{path.name}: {name}()")
                continue
            modes = [
                argument.value
                for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
                if isinstance(argument, ast.Constant) and argument.value in write_modes
            ]
            offenders.extend(f"{path.name}: open(mode={mode!r})" for mode in modes)
    assert offenders == []


# -- CLI ----------------------------------------------------------------


def test_replay_command_reports_a_missing_generation() -> None:
    result = RUNNER.invoke(cli_app, ["replay", str(RUN2), "--to-gen", "9"])

    assert result.exit_code != 0
    assert "no events for generation 9" in result.output


def test_replay_command_reports_a_missing_event_type() -> None:
    result = RUNNER.invoke(cli_app, ["replay", str(RUN2), "--at", "lesson.written"])

    assert result.exit_code != 0
    assert "lesson.written" in result.output


def test_replay_command_reports_a_missing_run_directory(tmp_path: Path) -> None:
    result = RUNNER.invoke(cli_app, ["replay", str(tmp_path / "nowhere")])

    assert result.exit_code != 0
    assert "event log not found" in result.output


def test_tui_command_reports_a_missing_run_directory(tmp_path: Path) -> None:
    result = RUNNER.invoke(cli_app, ["tui", "--run", str(tmp_path / "nowhere")])

    assert result.exit_code != 0
    assert "event log not found" in result.output
