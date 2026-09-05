"""Command-line entry points for the repository skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from occam.store.reader import EventReader
from occam.store.reducer import reduce, state_json_bytes
from occam.store.schema import validate_state

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Occam architecture engineering tools.",
)


@app.callback()
def _root() -> None:
    """Group Occam's subcommands."""


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

        if reader.state_path.exists():
            actual = reader.state_path.read_bytes()
            payload = json.loads(actual)
            validate_state(payload)
            if actual != first_bytes:
                raise ValueError("state.json does not match the canonical reduced state")
            snapshot_message = "state.json matches"
        else:
            snapshot_message = "state.json absent (reduced state validated in memory)"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run_dir") from exc

    typer.echo(f"valid: {len(events)} events in {run_dir}")
    typer.echo(f"deterministic: yes ({len(first_bytes)} state bytes; {snapshot_message})")


def main() -> None:
    """Invoke Typer's application."""

    app()


if __name__ == "__main__":
    main()
