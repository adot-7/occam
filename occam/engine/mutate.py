"""Closed, immutable architecture mutations and their lineage events.

Diagnosis chooses one menu item; this module applies only that item.  Every
operation creates a fresh architecture whose ``parent_id`` is the old
architecture id.  The parent is retained in :class:`MutationApplication`, so
the loop can revert a trial branch without reconstructing it from a diff.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from occam.core.models import Architecture, MemoryPolicy, Role
from occam.engine.executor import ArchitectureError, validate_architecture

MutationType = Literal[
    "prune",
    "rewrite_prompt",
    "rebind_tools",
    "split",
    "merge",
    "set_memory",
    "set_retry",
    "collapse",
]
PRUNABLE_VERDICTS = frozenset({"witness", "harmful"})
_GENERATION_ID = re.compile(r"^g(\d+)$")


class MutationError(ValueError):
    """A requested mutation is not legal for the supplied architecture."""


class Mutation(BaseModel):
    """Strict representation of one entry in the v3 mutation menu.

    The event contract stores only ``type``, ``target_role`` and ``rationale``;
    the optional fields carry the concrete edit between diagnosis and apply.
    ``from_mapping`` also accepts the common names used by structured model
    responses (``new_prompt``, ``other_role``, and ``new_roles``).
    """

    model_config = ConfigDict(extra="forbid")

    type: MutationType
    target_role: str = Field(min_length=1)
    rationale: str = ""
    prompt: str | None = None
    new_prompt: str | None = None
    tools: list[str] | None = None
    add_tools: list[str] = Field(default_factory=list)
    remove_tools: list[str] = Field(default_factory=list)
    split_roles: list[dict[str, Any]] | None = None
    new_roles: list[dict[str, Any]] | None = None
    merge_with: str | None = None
    other_role: str | None = None
    memory: MemoryPolicy | None = None
    max_turns: int | None = Field(default=None, ge=1)
    retry_policy: dict[str, Any] | None = None
    value: Any = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Mutation:
        """Normalize diagnosis-friendly aliases before strict validation."""

        payload = dict(raw)
        if "target_role" not in payload and "role_id" in payload:
            payload["target_role"] = payload.pop("role_id")
        if "prompt" not in payload:
            for key in ("new_prompt", "replacement_prompt", "system_prompt"):
                if key in payload:
                    payload["prompt"] = payload[key]
                    break
        if "merge_with" not in payload:
            for key in ("other_role", "merge_role", "source_role"):
                if key in payload:
                    payload["merge_with"] = payload[key]
                    break
        if "split_roles" not in payload and "new_roles" in payload:
            payload["split_roles"] = payload["new_roles"]
        if "max_turns" not in payload:
            retry = payload.get("retry_policy")
            if isinstance(retry, Mapping) and "max_turns" in retry:
                payload["max_turns"] = retry["max_turns"]
            elif payload.get("type") == "set_retry" and isinstance(payload.get("value"), int):
                payload["max_turns"] = payload["value"]
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise MutationError(f"invalid mutation: {exc}") from exc


@dataclass(frozen=True)
class MutationApplication:
    """A new architecture plus the exact parent needed for reversal."""

    parent: Architecture
    architecture: Architecture
    mutation: Mutation
    diff: str
    generation: int

    @property
    def new_architecture(self) -> Architecture:
        """Readable alias used by loop callers."""

        return self.architecture

    def event_data(self) -> dict[str, Any]:
        """Return the schema-compatible ``mutation.applied`` payload."""

        return {
            "generation": self.generation,
            "parent": _generation_number(self.parent.id),
            "type": self.mutation.type,
            "target_role": self.mutation.target_role,
            "diff": self.diff,
        }

    def reverted_event_data(self, reason: str = "trial mutation reverted") -> dict[str, Any]:
        """Return the schema-compatible reversal payload."""

        reason = reason.strip()
        if not reason:
            raise MutationError("reversion reason must not be blank")
        return {
            "generation": self.generation,
            "restored_to": _generation_number(self.parent.id),
            "reason": reason,
        }

    def revert(self) -> Architecture:
        """Restore the immutable parent architecture."""

        return self.parent

    def __getattr__(self, name: str) -> Any:
        """Make result use ergonomic for callers expecting architecture fields."""

        return getattr(self.architecture, name)


EventSink = Callable[[str, Mapping[str, Any]], None]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _generation_number(architecture_id: str) -> int:
    match = _GENERATION_ID.fullmatch(architecture_id)
    if match is None:
        raise MutationError(f"architecture id must look like g000, got {architecture_id!r}")
    return int(match.group(1))


def _next_generation(parent: Architecture, requested: int | None) -> int:
    parent_generation = _generation_number(parent.id)
    generation = parent_generation + 1 if requested is None else requested
    if generation <= parent_generation:
        raise MutationError(
            f"new generation {generation} must be greater than parent {parent_generation}"
        )
    return generation


def _role_index(architecture: Architecture) -> dict[str, Role]:
    index = {role.id: role for role in architecture.roles}
    if len(index) != len(architecture.roles):
        raise MutationError("architecture contains duplicate role ids")
    return index


def _target_role(architecture: Architecture, role_id: str) -> Role:
    try:
        return _role_index(architecture)[role_id]
    except KeyError as exc:
        raise MutationError(f"unknown target role {role_id!r}") from exc


def _replace_inputs(inputs: Sequence[str], old: str, replacements: Sequence[str]) -> list[str]:
    result: list[str] = []
    for source in inputs:
        candidates = replacements if source == old else [source]
        for candidate in candidates:
            if candidate not in result:
                result.append(candidate)
    return result


def _role_update(role: Role, **updates: Any) -> Role:
    payload = role.model_dump(mode="python")
    payload.update(updates)
    try:
        return Role.model_validate(payload)
    except ValidationError as exc:
        raise MutationError(f"invalid role mutation for {role.id!r}: {exc}") from exc


def _new_architecture(
    parent: Architecture,
    *,
    generation: int,
    roles: Sequence[Role],
    final_role: str,
    diff: str,
) -> Architecture:
    payload = parent.model_dump(mode="python")
    payload.update(
        {
            "id": f"g{generation:03d}",
            "parent_id": parent.id,
            "roles": [role.model_dump(mode="python") for role in roles],
            "final_role": final_role,
            "notes": "\n".join(part for part in (parent.notes, diff) if part),
        }
    )
    try:
        architecture = Architecture.model_validate(payload)
        validate_architecture(architecture)
    except (ValidationError, ArchitectureError) as exc:
        raise MutationError(f"mutation produced an invalid architecture: {exc}") from exc
    return architecture


def _ensure_tools(
    tools: Sequence[str], *, available_tools: Sequence[str] | None, role_id: str
) -> list[str]:
    result: list[str] = []
    for tool in tools:
        if not isinstance(tool, str) or not tool.strip():
            raise MutationError(f"role {role_id!r} has an empty tool binding")
        if tool not in result:
            result.append(tool)
    if available_tools is not None:
        allowed = set(available_tools)
        unknown = sorted(set(result) - allowed)
        if unknown:
            raise MutationError(
                f"role {role_id!r} binds tools outside the task manifest: {', '.join(unknown)}"
            )
    return result


def _split_roles(
    target: Role,
    mutation: Mutation,
    *,
    available_tools: Sequence[str] | None,
) -> list[Role]:
    raw = mutation.split_roles if mutation.split_roles is not None else mutation.new_roles
    if raw is not None:
        if len(raw) != 2:
            raise MutationError("split requires exactly two replacement roles")
        try:
            replacements = [Role.model_validate(item) for item in raw]
        except ValidationError as exc:
            raise MutationError(f"split replacement is not a valid role: {exc}") from exc
        if any(role.id == target.id for role in replacements):
            raise MutationError("split replacement roles must have new ids")
        for role in replacements:
            _ensure_tools(role.tools, available_tools=available_tools, role_id=role.id)
        return replacements
    first = _role_update(
        target,
        id=f"{target.id}_a",
        name=f"{target.name} A",
        justification="parallel",
        output_key=f"{target.output_key}_a",
    )
    second = _role_update(
        target,
        id=f"{target.id}_b",
        name=f"{target.name} B",
        justification="context_isolation",
        output_key=f"{target.output_key}_b",
    )
    return [first, second]


def _apply_operation(
    parent: Architecture,
    mutation: Mutation,
    *,
    available_tools: Sequence[str] | None,
) -> tuple[list[Role], str, str]:
    """Return ``(roles, final_role, human_diff)`` for one menu operation."""

    index = _role_index(parent)
    operation = mutation.type
    if operation == "collapse":
        chosen = index.get(mutation.target_role)
        if chosen is None and mutation.target_role in {"*", "all"}:
            chosen = index[parent.final_role]
        if chosen is None:
            raise MutationError(f"unknown collapse target role {mutation.target_role!r}")
        all_tools = _ensure_tools(
            [tool for role in parent.roles for tool in role.tools],
            available_tools=available_tools,
            role_id=chosen.id,
        )
        prompts = "\n\n".join(
            f"Former role {role.name}:\n{role.system_prompt}" for role in parent.roles
        )
        collapsed = _role_update(
            chosen,
            name="Collapsed solver",
            justification="control",
            system_prompt=prompts,
            tools=all_tools,
            inputs=["task"],
        )
        diff = f"collapse {len(parent.roles)} roles into {collapsed.name} ({collapsed.id})"
        return [collapsed], collapsed.id, diff

    target = _target_role(parent, mutation.target_role)
    roles = list(parent.roles)
    target_position = next(index for index, role in enumerate(roles) if role.id == target.id)

    if operation == "prune":
        replacements = list(target.inputs)
        if parent.final_role == target.id:
            candidate_final = next(
                (source for source in replacements if source != "task" and source in index),
                None,
            )
            if candidate_final is None:
                raise MutationError("cannot prune the final role without an upstream final role")
        else:
            candidate_final = parent.final_role
        rewritten = [
            _role_update(role, inputs=_replace_inputs(role.inputs, target.id, replacements))
            if role.id != target.id
            else None
            for role in roles
        ]
        result_roles = [role for role in rewritten if role is not None]
        diff = f"prune {target.id} ({target.name}); rewire consumers to {replacements or ['task']}"
        return result_roles, candidate_final, diff

    if operation == "rewrite_prompt":
        prompt = mutation.prompt or mutation.new_prompt
        if prompt is None and isinstance(mutation.value, str):
            prompt = mutation.value
        if prompt is None:
            rationale = mutation.rationale.strip()
            if not rationale:
                raise MutationError("rewrite_prompt requires prompt or rationale")
            prompt = f"{target.system_prompt.rstrip()}\n\nAdditional instruction:\n{rationale}"
        if not prompt.strip():
            raise MutationError("rewrite_prompt cannot produce a blank prompt")
        roles[target_position] = _role_update(target, system_prompt=prompt)
        diff = f"rewrite prompt for {target.id} ({target.name})"
        return roles, parent.final_role, diff

    if operation == "rebind_tools":
        selected = mutation.tools
        if selected is None:
            selected = [tool for tool in target.tools if tool not in mutation.remove_tools]
            for tool in mutation.add_tools:
                if tool not in selected:
                    selected.append(tool)
        selected = _ensure_tools(selected, available_tools=available_tools, role_id=target.id)
        roles[target_position] = _role_update(target, tools=selected)
        diff = f"rebind tools on {target.id}: {', '.join(selected) or 'none'}"
        return roles, parent.final_role, diff

    if operation == "split":
        replacements = _split_roles(
            target,
            mutation,
            available_tools=available_tools,
        )
        replacement_ids = [role.id for role in replacements]
        existing_ids = set(index)
        if existing_ids.intersection(replacement_ids):
            raise MutationError("split replacement role id already exists")
        result_roles = []
        for role in roles:
            if role.id == target.id:
                result_roles.extend(replacements)
            else:
                result_roles.append(
                    _role_update(
                        role,
                        inputs=_replace_inputs(role.inputs, target.id, replacement_ids),
                    )
                )
        final_role = replacement_ids[-1] if parent.final_role == target.id else parent.final_role
        diff = f"split {target.id} ({target.name}) into {', '.join(replacement_ids)}"
        return result_roles, final_role, diff

    if operation == "merge":
        other_id = mutation.merge_with or mutation.other_role
        if not other_id:
            raise MutationError("merge requires merge_with/other_role")
        if other_id == target.id:
            raise MutationError("merge requires two different roles")
        other = _target_role(parent, other_id)
        merged = _role_update(
            target,
            name=f"{target.name} + {other.name}",
            system_prompt=(
                f"{target.system_prompt.rstrip()}\n\n{other.system_prompt.rstrip()}"
            ).strip(),
            tools=_ensure_tools(
                [*target.tools, *other.tools],
                available_tools=available_tools,
                role_id=target.id,
            ),
            inputs=_replace_inputs(
                _replace_inputs([*target.inputs, *other.inputs], target.id, []), other.id, []
            ),
            max_turns=max(target.max_turns, other.max_turns),
        )
        result_roles = []
        for role in roles:
            if role.id == target.id:
                result_roles.append(merged)
            elif role.id == other.id:
                continue
            else:
                result_roles.append(
                    _role_update(role, inputs=_replace_inputs(role.inputs, other.id, [target.id]))
                )
        final_role = target.id if parent.final_role == other.id else parent.final_role
        diff = f"merge {target.id} ({target.name}) with {other.id} ({other.name})"
        return result_roles, final_role, diff

    if operation == "set_memory":
        policy = mutation.memory
        if policy is None and isinstance(mutation.value, str):
            policy = mutation.value  # type: ignore[assignment]
        if policy not in {"none", "scratchpad", "summary"}:
            raise MutationError("set_memory requires one of none, scratchpad, summary")
        roles[target_position] = _role_update(target, memory=policy)
        diff = f"set memory for {target.id} to {policy}"
        return roles, parent.final_role, diff

    if operation == "set_retry":
        max_turns = mutation.max_turns
        if (
            max_turns is None
            and isinstance(mutation.value, int)
            and not isinstance(mutation.value, bool)
        ):
            max_turns = mutation.value
        if max_turns is None:
            raise MutationError("set_retry requires max_turns")
        roles[target_position] = _role_update(target, max_turns=max_turns)
        diff = f"set max_turns for {target.id} to {max_turns}"
        return roles, parent.final_role, diff

    raise MutationError(f"unsupported mutation type {operation!r}")


def apply_mutation(
    architecture: Architecture,
    mutation: Mutation | Mapping[str, Any],
    *,
    role_verdicts: Mapping[str, str | Mapping[str, Any] | Any] | None = None,
    available_tools: Sequence[str] | None = None,
    generation: int | None = None,
    event_sink: EventSink | None = None,
) -> MutationApplication:
    """Apply one legal menu operation without mutating its parent.

    ``role_verdicts`` is supplied by ablation when pruning.  A prune requires
    explicit verdict evidence and is allowed only for ``witness`` or
    ``harmful``; uncertain and load-bearing roles can never be pruned by this
    boundary.
    """

    if not isinstance(mutation, Mutation):
        if not isinstance(mutation, Mapping):
            raise MutationError("mutation must be a Mutation or mapping")
        mutation = Mutation.from_mapping(mutation)
    if mutation.type == "prune":
        if role_verdicts is None:
            raise MutationError("prune requires explicit role verdict evidence")
        raw_verdict = role_verdicts.get(mutation.target_role)
        verdict = raw_verdict
        if isinstance(raw_verdict, Mapping):
            verdict = raw_verdict.get("verdict")
        else:
            verdict = getattr(raw_verdict, "verdict", raw_verdict)
        if verdict not in PRUNABLE_VERDICTS:
            raise MutationError(
                f"prune target {mutation.target_role!r} has non-prunable verdict {verdict!r}"
            )
    next_generation = _next_generation(architecture, generation)
    roles, final_role, diff = _apply_operation(
        architecture,
        mutation,
        available_tools=available_tools,
    )
    result = MutationApplication(
        parent=architecture,
        architecture=_new_architecture(
            architecture,
            generation=next_generation,
            roles=roles,
            final_role=final_role,
            diff=diff,
        ),
        mutation=mutation,
        diff=diff,
        generation=next_generation,
    )
    if event_sink is not None:
        event_sink("mutation.applied", result.event_data())
    return result


def revert_mutation(
    application: MutationApplication,
    *,
    reason: str = "trial mutation reverted",
    event_sink: EventSink | None = None,
) -> Architecture:
    """Restore a mutation's exact parent and optionally emit its reversal."""

    if event_sink is not None:
        event_sink("mutation.reverted", application.reverted_event_data(reason))
    return application.revert()


def mutate(
    architecture: Architecture,
    mutation: Mutation | Mapping[str, Any],
    **kwargs: Any,
) -> MutationApplication:
    """Alias matching the engine vocabulary used by the PRD."""

    return apply_mutation(architecture, mutation, **kwargs)


def apply_architecture_mutation(
    architecture: Architecture,
    mutation: Mutation | Mapping[str, Any],
    **kwargs: Any,
) -> Architecture:
    """Return only the child architecture for simple loop callers."""

    return apply_mutation(architecture, mutation, **kwargs).architecture


__all__ = [
    "Mutation",
    "MutationApplication",
    "MutationError",
    "MutationType",
    "PRUNABLE_VERDICTS",
    "apply_architecture_mutation",
    "apply_mutation",
    "mutate",
    "revert_mutation",
]
