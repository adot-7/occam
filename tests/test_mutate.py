"""Every v3 mutation type gets a focused immutable-lineage test."""

from __future__ import annotations

from typing import Any

import pytest

from occam.core import Architecture, Role
from occam.engine.mutate import (
    Mutation,
    MutationError,
    apply_mutation,
    revert_mutation,
)

AVAILABLE_TOOLS = ("fx_rate", "fx_series", "python_exec")


def role(role_id: str, **overrides: Any) -> Role:
    payload = {
        "id": role_id,
        "name": role_id.replace("_", " ").title(),
        "justification": "control",
        "model": "worker_fast",
        "system_prompt": f"Prompt for {role_id}.",
        "tools": [],
        "inputs": ["task"],
        "output_key": f"{role_id}_out",
    }
    payload.update(overrides)
    return Role.model_validate(payload)


def architecture() -> Architecture:
    return Architecture(
        id="g000",
        parent_id=None,
        roles=[
            role("r_parse", justification="context_isolation"),
            role("r_rates", justification="parallel", inputs=["r_parse"], tools=["fx_rate"]),
            role(
                "r_calc",
                inputs=["r_parse", "r_rates"],
                tools=["python_exec"],
            ),
            role("r_report", inputs=["r_calc"], justification="control"),
        ],
        final_role="r_report",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        Mutation(type="prune", target_role="r_rates"),
        Mutation(type="rewrite_prompt", target_role="r_calc", prompt="New calculator prompt."),
        Mutation(type="rebind_tools", target_role="r_rates", tools=["fx_series"]),
        Mutation(type="split", target_role="r_rates"),
        Mutation(type="merge", target_role="r_parse", merge_with="r_rates"),
        Mutation(type="set_memory", target_role="r_calc", memory="summary"),
        Mutation(type="set_retry", target_role="r_calc", max_turns=9),
        Mutation(type="collapse", target_role="r_report"),
    ],
)
def test_every_mutation_type_creates_a_parent_linked_child(mutation: Mutation) -> None:
    parent = architecture()
    result = apply_mutation(
        parent,
        mutation,
        role_verdicts={"r_rates": "witness"},
        available_tools=AVAILABLE_TOOLS,
    )

    assert result.architecture.id == "g001"
    assert result.architecture.parent_id == parent.id
    assert result.parent == parent
    assert result.diff
    assert result.event_data()["generation"] == 1
    assert result.event_data()["parent"] == 0
    assert result.revert() == parent


def test_prune_rewires_consumers_and_never_mutates_parent() -> None:
    parent = architecture()
    result = apply_mutation(
        parent,
        Mutation(type="prune", target_role="r_rates"),
        role_verdicts={"r_rates": "witness"},
    )

    assert [role.id for role in result.architecture.roles] == ["r_parse", "r_calc", "r_report"]
    calc = next(role for role in result.architecture.roles if role.id == "r_calc")
    assert calc.inputs == ["r_parse"]
    assert any(role.id == "r_rates" for role in parent.roles)


def test_prune_rejects_non_prunable_verdicts() -> None:
    with pytest.raises(MutationError, match="non-prunable"):
        apply_mutation(
            architecture(),
            Mutation(type="prune", target_role="r_rates"),
            role_verdicts={"r_rates": "load_bearing"},
        )


def test_prune_requires_explicit_verdict_evidence() -> None:
    with pytest.raises(MutationError, match="explicit role verdict evidence"):
        apply_mutation(
            architecture(),
            Mutation(type="prune", target_role="r_rates"),
        )


def test_split_replaces_target_in_downstream_inputs() -> None:
    result = apply_mutation(architecture(), Mutation(type="split", target_role="r_rates"))
    ids = [role.id for role in result.architecture.roles]
    calc = next(role for role in result.architecture.roles if role.id == "r_calc")

    assert ids == ["r_parse", "r_rates_a", "r_rates_b", "r_calc", "r_report"]
    assert calc.inputs == ["r_parse", "r_rates_a", "r_rates_b"]


def test_merge_rewires_downstream_and_collapses_tool_bindings() -> None:
    result = apply_mutation(
        architecture(),
        Mutation(type="merge", target_role="r_parse", merge_with="r_rates"),
        available_tools=AVAILABLE_TOOLS,
    )
    ids = [role.id for role in result.architecture.roles]
    merged = next(role for role in result.architecture.roles if role.id == "r_parse")
    calc = next(role for role in result.architecture.roles if role.id == "r_calc")

    assert ids == ["r_parse", "r_calc", "r_report"]
    assert merged.inputs == ["task"]
    assert merged.tools == ["fx_rate"]
    assert calc.inputs == ["r_parse"]


def test_rebind_is_manifest_limited_and_reversion_emits_event() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    result = apply_mutation(
        architecture(),
        Mutation(type="rebind_tools", target_role="r_rates", tools=["fx_series"]),
        available_tools=AVAILABLE_TOOLS,
        event_sink=lambda event_type, data: events.append((event_type, dict(data))),
    )
    assert next(role for role in result.architecture.roles if role.id == "r_rates").tools == [
        "fx_series"
    ]
    revert_mutation(
        result,
        reason="new roles failed ablation",
        event_sink=lambda t, d: events.append((t, dict(d))),
    )
    assert [event[0] for event in events] == ["mutation.applied", "mutation.reverted"]
    assert events[1][1] == {
        "generation": 1,
        "restored_to": 0,
        "reason": "new roles failed ablation",
    }


def test_set_memory_and_retry_accept_mapping_aliases() -> None:
    memory_result = apply_mutation(
        architecture(),
        {"type": "set_memory", "role_id": "r_calc", "value": "scratchpad"},
    )
    retry_result = apply_mutation(
        architecture(),
        {"type": "set_retry", "target_role": "r_calc", "value": 4},
    )
    assert next(role for role in memory_result.roles if role.id == "r_calc").memory == "scratchpad"
    assert next(role for role in retry_result.roles if role.id == "r_calc").max_turns == 4


def test_collapse_leaves_one_direct_solver() -> None:
    result = apply_mutation(architecture(), Mutation(type="collapse", target_role="r_report"))
    assert len(result.roles) == 1
    assert result.final_role == "r_report"
    assert result.roles[0].inputs == ["task"]
