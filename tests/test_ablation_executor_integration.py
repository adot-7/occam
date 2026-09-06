"""Integration coverage for WP-08 through the current WP-05 executor seam."""

from __future__ import annotations

import json
from pathlib import Path

from occam.core.models import Architecture, Case, Role
from occam.engine.ablation import ablate, select_ablation_subset
from occam.engine.emit import writer_sink
from occam.engine.executor import Executor
from occam.store.reader import EventReader
from occam.store.writer import EventWriter
from occam.tools.fx import FXClient
from occam.tools.registry import ToolRegistry
from tests.executor_doubles import ScriptedProvider, build_client, text_response, tool_call_response


def _architecture() -> Architecture:
    source = Role(
        id="r_source",
        name="Source",
        justification="context_isolation",
        model="worker_fast",
        system_prompt="ROLE: source",
        tools=[],
        inputs=["task"],
        output_key="source_output",
    )
    final = Role(
        id="r_final",
        name="Final",
        justification="control",
        model="worker_fast",
        system_prompt="ROLE: final",
        tools=[],
        inputs=["r_source"],
        output_key="final_output",
    )
    return Architecture(
        id="g002",
        parent_id=None,
        roles=[source, final],
        final_role="r_final",
    )


def test_ablation_uses_fresh_stratified_repeat_and_current_executor_events(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        lambda call: text_response("changed" if "[no input from Source]" in call.user else "answer")
    )
    run_dir = tmp_path / "run"
    writer = EventWriter(run_dir, run_id="run_test")
    executor = Executor(
        llm=build_client(provider, tmp_path / "cache"),
        grader=lambda answer, _expected: {"passed": answer == "answer", "sub_results": {}},
        writer=writer,
        run_dir=run_dir,
        case_concurrency=1,
        model_concurrency=1,
    )
    cases = [
        Case(id="c1", input="first", expected={}, meta={"has_weekend_or_holiday": True}),
        Case(id="c2", input="second", expected={}, meta={}),
        Case(id="c3", input="third", expected={}, meta={"has_bank_fee": True}),
    ]
    architecture = _architecture()
    sink = writer_sink(writer)

    full = executor.run_variant(
        architecture, cases, variant="full", generation=2, grader=executor.grader
    )
    assert full.pass_rate == 1.0
    provider.reset()

    subset = select_ablation_subset(cases, 2)
    assert [case.id for case in subset] == ["c1", "c2"]
    table = ablate(
        architecture,
        subset,
        runner=executor,
        generation=2,
        full=full,
        sink=sink,
    )

    assert table.noise_rate == 0.0
    assert [row.role_id for row in table.rows] == ["r_source", "r_final"]
    assert table.structural_fidelity == 1.0
    assert all(row.verdict == "load_bearing" for row in table.rows)

    repeat_path = run_dir / "generations" / "g002" / "results.full_repeat.jsonl"
    repeat_results = [json.loads(line) for line in repeat_path.read_text().splitlines()]
    assert [result["case_id"] for result in repeat_results] == ["c1", "c2"]
    assert all(
        trace["cached"] is False
        for result in repeat_results
        for trace in result["per_role"].values()
    )
    assert all(
        trace["cost_usd"] > 0 for result in repeat_results for trace in result["per_role"].values()
    )
    assert all(
        trace["billed_cost_usd"] == 0.0 and trace["cost_label"] == "list-rate-equivalent"
        for result in repeat_results
        for trace in result["per_role"].values()
    )
    for role_id in ("r_source", "r_final"):
        knockout_path = run_dir / "generations" / "g002" / f"results.ablate_{role_id}.jsonl"
        knockout_results = [json.loads(line) for line in knockout_path.read_text().splitlines()]
        assert [result["case_id"] for result in knockout_results] == ["c1", "c2"]
        assert all(role_id not in result["per_role"] for result in knockout_results)
    # The repeat bypassed the LLM cache and did not overwrite the canonical
    # full-run entries: four repeat calls plus one unique sentinel prompt.  The
    # second identical sentinel prompt is itself a valid content-cache hit.
    assert provider.count == 5

    events = EventReader(run_dir).read()
    repeat_completed = next(
        index
        for index, event in enumerate(events)
        if event.type == "execution.completed"
        and event.data["generation"] == 2
        and event.data["variant"] == "full_repeat"
    )
    started = next(index for index, event in enumerate(events) if event.type == "ablation.started")
    first_row = next(index for index, event in enumerate(events) if event.type == "ablation.role")
    completed = next(
        index for index, event in enumerate(events) if event.type == "ablation.completed"
    )
    assert repeat_completed < started < first_row < completed
    started_event = events[started]
    assert started_event.data["case_ids"] == ["c1", "c2"]
    assert started_event.data["n_cases"] == 2
    assert started_event.data["noise_rate"] == 0.0
    assert [event.data["role_id"] for event in events if event.type == "ablation.role"] == [
        "r_source",
        "r_final",
    ]

    # The fresh repeat was deliberately not written into the regular cache.
    provider.reset()
    executor.run_variant(architecture, cases, variant="full", generation=3, grader=executor.grader)
    assert provider.count == 0
    writer.close()


def test_cache_bypass_reaches_fan_out_role_completions(tmp_path: Path) -> None:
    registry = ToolRegistry(
        fx_client=FXClient(cache_dir=tmp_path / "fx_cache"),
        fan_out_max_workers=2,
    )
    provider = ScriptedProvider(
        lambda call: (
            tool_call_response([("fan_out", {"subtasks": ["one", "two"]})])
            if call.user == "### task\nparent"
            and not any(message["role"] == "tool" for message in call.messages)
            else text_response(call.user.splitlines()[-1])
        )
    )
    architecture = Architecture(
        id="fanout",
        parent_id=None,
        roles=[
            Role(
                id="parent",
                name="Parent",
                justification="parallel",
                model="worker_fast",
                system_prompt="ROLE: parent",
                tools=["fan_out"],
                inputs=["task"],
                output_key="answer",
            )
        ],
        final_role="parent",
    )
    executor = Executor(
        llm=build_client(provider, tmp_path / "llm_cache"),
        tools=registry.bindings(),
        case_concurrency=1,
        model_concurrency=4,
    )
    cases = [Case(id="c1", input="parent", expected={})]

    full = executor.run_variant(architecture, cases, variant="full")
    assert full.results[0].answer == "parent"
    assert provider.count == 4  # parent, two branches, parent follow-up

    provider.reset()
    repeat = executor.run_variant(architecture, cases, variant="full_repeat", use_cache=False)
    assert repeat.results[0].answer == "parent"
    assert repeat.results[0].per_role["parent"].cached is False
    assert provider.count == 4  # bypass includes both nested branch completions

    provider.reset()
    cached = executor.run_variant(architecture, cases, variant="full")
    assert cached.results[0].per_role["parent"].cached is True
    assert provider.count == 0  # the bypass did not replace canonical entries
