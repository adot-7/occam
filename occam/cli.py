"""Command-line entry points for the repository skeleton."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

# Initialize before importing any command implementation that could construct
# an OpenAI provider.  The provider itself remains lazy for library users.
from occam.llm.tracing import initialize as initialize_tracing
from occam.llm.tracing import shutdown as shutdown_tracing
from occam.memory.lessons import LessonStore, LessonStoreError
from occam.store.reader import EventReader
from occam.store.reducer import reduce, state_json_bytes
from occam.store.schema import validate_state

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type checkers
    from occam.tui.app import OccamApp

#: Test/CI affordances.  The documented interface is the flags below; these let
#: an automated check drive the real command without a terminal.
HEADLESS_ENV = "OCCAM_TUI_HEADLESS"
EXIT_AFTER_ENV = "OCCAM_TUI_EXIT_AFTER"

initialize_tracing()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Occam architecture engineering tools.",
)
llm_app = typer.Typer(add_completion=False, help="LLM provider utilities.")
lessons_app = typer.Typer(add_completion=False, help="Read and reset learned lessons.")
app.add_typer(llm_app, name="llm")
app.add_typer(lessons_app, name="lessons")


@app.callback()
def _root() -> None:
    """Group Occam's subcommands."""


@lessons_app.command("show")
def lessons_show(
    memory: Annotated[
        Path,
        typer.Option("--memory", help="Memory namespace containing lessons.jsonl."),
    ],
) -> None:
    """Print the deterministic Markdown view of a lesson namespace."""

    try:
        markdown = LessonStore(memory).show()
    except LessonStoreError as exc:
        raise typer.BadParameter(str(exc), param_hint="--memory") from exc
    typer.echo(markdown, nl=False)


@lessons_app.command("reset")
def lessons_reset(
    memory: Annotated[
        Path,
        typer.Option("--memory", help="Memory namespace containing lessons.jsonl."),
    ],
) -> None:
    """Clear exactly one lesson namespace for a clean run."""

    try:
        removed = LessonStore(memory).reset()
    except LessonStoreError as exc:
        raise typer.BadParameter(str(exc), param_hint="--memory") from exc
    typer.echo(f"reset {memory}: removed {removed} lesson(s)")


@app.command("run")
def run_command(
    task: Annotated[str, typer.Option("--task", help="Task-pack directory or committed pack name")],
    run_name: Annotated[str, typer.Option("--run-name", help="Stable run id and artifact name")],
    memory: Annotated[
        Path | None,
        typer.Option("--memory", help="Lesson namespace; defaults to the task manifest value"),
    ] = None,
    max_gens: Annotated[
        int, typer.Option("--max-gens", min=1, help="Maximum architecture generations")
    ] = 6,
    cases: Annotated[
        int | None, typer.Option("--cases", min=1, help="Number of cases from the pack")
    ] = None,
    ablate_cases: Annotated[
        int, typer.Option("--ablate-cases", min=1, help="Paired cases per role knockout")
    ] = 10,
    pass3: Annotated[
        bool, typer.Option("--pass3", help="Run three independent final-generation passes")
    ] = False,
    out: Annotated[Path, typer.Option("--out", help="Parent directory for run artifacts")] = Path(
        "runs"
    ),
    launch_tui: Annotated[
        bool, typer.Option("--tui", help="Attach the read-only TUI after the run")
    ] = False,
) -> None:
    """Architect, execute, ablate, diagnose, mutate, and record one run."""

    from occam.engine.loop import RunConfig, run_task

    try:
        outcome = run_task(
            task,
            RunConfig(
                run_name=run_name,
                out=out,
                memory=memory,
                max_generations=max_gens,
                n_cases=cases,
                ablate_cases=ablate_cases,
                pass3=pass3,
            ),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = outcome.summary
    typer.echo(f"run_id: {outcome.run_id}")
    typer.echo(f"run_dir: {outcome.run_dir}")
    typer.echo(
        f"completed: best=g{outcome.best_generation} generations={summary['generations']} "
        f"pass={summary['final_pass_rate']:.3f} cost=${summary['final_cost_usd']:.6f} "
        f"calls/case={summary['calls_per_case_final']:.2f}"
    )
    provider_calls = summary.get("provider_calls")
    typer.echo(
        f"provider_calls: {provider_calls if provider_calls is not None else 'unavailable'}; "
        f"displayed_cost_usd=${summary['displayed_cost_usd']:.6f}; "
        f"billed_cost_usd=${summary['billed_cost_usd']:.6f}"
    )
    if launch_tui:
        from occam.tui.app import OccamApp

        _launch(OccamApp(outcome.run_dir))


def _resolve_run_argument(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate
    named = Path("runs") / value
    if named.is_dir():
        return named
    raise ValueError(f"run directory not found: {value}")


@app.command("compare")
def compare_command(
    run_dir_1: Annotated[str, typer.Argument(help="First run directory or run name")],
    run_dir_2: Annotated[str, typer.Argument(help="Second run directory or run name")],
) -> None:
    """Compare two completed runs and write compare.json into the second."""

    from occam.engine.compare import compare_runs, format_compare

    try:
        first = _resolve_run_argument(run_dir_1)
        second = _resolve_run_argument(run_dir_2)
        payload = compare_runs(first, second)
    except (OSError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_compare(payload))
    typer.echo(f"compare_json: {second / 'compare.json'}")


@app.command("validate")
def validate_run(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing events.jsonl")],
) -> None:
    """Validate a run's events and prove that reduction is byte-identical."""

    try:
        reader = EventReader(run_dir)
        events = reader.read()
        first_state = reduce(events)
        second_state = reduce(events)
        first_bytes = state_json_bytes(first_state)
        second_bytes = state_json_bytes(second_state)
        if first_bytes != second_bytes:
            raise ValueError("reducer is not deterministic: state bytes differ")

        validate_state(json.loads(first_bytes))

        if reader.state_path.exists():
            actual = reader.state_path.read_bytes()
            payload = json.loads(actual)
            validate_state(payload)
            if actual != first_bytes:
                raise ValueError("state.json does not match the canonical reduced state")
            snapshot_message = "state.json matches"
        else:
            snapshot_message = "state.json absent"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run_dir") from exc

    typer.echo(f"valid: {len(events)} events in {run_dir}")
    typer.echo(
        f"deterministic: yes ({len(first_bytes)} state bytes; "
        f"derived state schema validated; {snapshot_message})"
    )


def _launch(app_instance: OccamApp) -> None:
    """Run a Textual app, honouring the headless/auto-exit test affordances."""

    import asyncio

    headless = os.environ.get(HEADLESS_ENV) == "1"
    exit_after = os.environ.get(EXIT_AFTER_ENV)
    auto_pilot = None
    if exit_after:
        seconds = float(exit_after)

        async def auto_pilot(pilot: Any) -> None:  # noqa: RUF029 - Textual API shape
            await asyncio.sleep(seconds)
            pilot.app.exit()

    app_instance.run(headless=headless, auto_pilot=auto_pilot)


@app.command("tui")
def tui(
    run: Annotated[Path, typer.Option("--run", help="Run directory to attach to")],
) -> None:
    """Attach the read-only TUI to a live or finished run."""

    from occam.tui.app import OccamApp

    try:
        _launch(OccamApp(run))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--run") from exc


@app.command("replay")
def replay(
    run_dir: Annotated[Path, typer.Argument(help="Recorded run directory to replay")],
    speed: Annotated[
        float, typer.Option("--speed", help="Playback speed multiplier", min=0.0)
    ] = 1.0,
    to_gen: Annotated[
        int | None,
        typer.Option("--to-gen", help="Fast-forward silently to this generation"),
    ] = None,
    at: Annotated[
        str | None,
        typer.Option("--at", help="Fast-forward to the first event of this type"),
    ] = None,
    pause: Annotated[
        bool, typer.Option("--pause", help="Start paused, for recording a static frame")
    ] = False,
) -> None:
    """Replay a recorded run into the TUI, indistinguishably from live."""

    from occam.tui.app import OccamApp
    from occam.tui.source import ReplayOptions

    try:
        options = ReplayOptions(speed=speed, to_gen=to_gen, at=at, paused=pause)
        instance = OccamApp(run_dir, replay=options)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run_dir") from exc
    _launch(instance)


@llm_app.command("ping")
def llm_ping(
    model_key: Annotated[str, typer.Argument(help="Configured model key to probe")],
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass the shared completion cache."),
    ] = False,
) -> None:
    """Make a completion and verify native function calling for one model."""

    # Keep this import inside the command.  Provider modules import their SDKs
    # lazily so WP-15 can initialise Neatlogs before openai is imported.
    from occam.llm import (
        LLMClient,
        LLMError,
        MissingCredentialsError,
        load_model_configs,
        safe_provider_error_metadata,
    )

    try:
        configs = load_model_configs()
        if model_key not in configs:
            available = ", ".join(sorted(configs))
            raise typer.BadParameter(f"unknown model key {model_key!r}; choose from {available}")
        config = configs[model_key]
        client = LLMClient(configs)
        completion = client.complete(
            model_key,
            [{"role": "user", "content": "Call ping_tool with value 'ok'."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "ping_tool",
                        "description": "A no-op health check function.",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            use_cache=not no_cache,
        )
    except MissingCredentialsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except LLMError as exc:
        metadata = safe_provider_error_metadata(exc)
        suffix = f"; {metadata}" if metadata is not None else ""
        typer.echo(f"LLM ping failed: {type(exc).__name__}{suffix}", err=True)
        raise typer.Exit(code=2) from exc
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"model: {model_key} ({config.model})")
    typer.echo(
        f"completion: ok; tokens_in={completion.tokens_in}; "
        f"tokens_out={completion.tokens_out}; cost_usd={completion.cost_usd:.8f}; "
        f"cost_basis={completion.cost_label}; cached={completion.cached}"
    )
    if not completion.tool_calls:
        typer.echo("native_tool_calls: no", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"native_tool_calls: yes ({len(completion.tool_calls)})")


def main() -> None:
    """Invoke Typer's application."""

    try:
        app()
    finally:
        # ``occam`` commands are short-lived processes; make the final case's
        # spans visible before the interpreter exits.
        shutdown_tracing()


if __name__ == "__main__":
    main()
