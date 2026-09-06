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
llm_app = typer.Typer(add_completion=False, help="LLM provider utilities.")
app.add_typer(llm_app, name="llm")


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


@llm_app.command("ping")
def llm_ping(
    model_key: Annotated[str, typer.Argument(help="Configured model key to probe")],
) -> None:
    """Make a completion and verify native function calling for one model."""

    # Keep this import inside the command.  Provider modules import their SDKs
    # lazily so WP-15 can initialise Neatlogs before openai is imported.
    from occam.llm import LLMClient, LLMError, MissingCredentialsError, load_model_configs

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
        )
    except MissingCredentialsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except LLMError as exc:
        typer.echo(f"LLM ping failed: {type(exc).__name__}", err=True)
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

    app()


if __name__ == "__main__":
    main()
