"""The strict first-generation architecture proposal.

The architect is intentionally a narrow boundary: it reads the task manifest
and active lessons, asks the configured ``architect`` model for one JSON
architecture, validates that response before it can reach the executor, and
optionally emits the schema-compatible ``architecture.proposed`` event.  It
does not diagnose failures or write lessons; those responsibilities belong to
WP-10.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from occam.core.models import Architecture, Lesson, Task, ToolSpec
from occam.engine.executor import ArchitectureError, validate_architecture
from occam.llm.client import LLMClient
from occam.memory.lessons import LessonStore, LessonStoreError, validate_lesson
from occam.store.schema import validate_event
from occam.tools.registry import append_tool_note, build_spec

EventSink = Callable[[str, Mapping[str, Any]], None]

DOMAIN_RULE_HEADING = "Known conventions and facts about this task (learned in earlier runs)"
ARCHITECT_MODEL_KEY = "architect"
MIN_ROLES = 3
MAX_ROLES = 6


class ArchitectError(RuntimeError):
    """The architect could not produce a safe, executable architecture."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _value(task: Task | Mapping[str, Any], name: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _tool_names(task: Task | Mapping[str, Any]) -> list[str]:
    raw = _value(task, "tools", ())
    if isinstance(raw, Mapping):
        raw = raw.keys()
    names: list[str] = []
    for item in raw or ():
        if isinstance(item, str):
            name = item
        elif isinstance(item, Mapping):
            name = item.get("name")
        else:
            name = getattr(item, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ArchitectError("task manifest contains a tool without a name")
        if name not in names:
            names.append(name)
    return names


def _tool_from_entry(name: str, entry: Any) -> ToolSpec:
    if isinstance(entry, ToolSpec):
        return entry
    candidate = getattr(entry, "spec", None)
    if candidate is not None:
        entry = candidate
    if isinstance(entry, ToolSpec):
        return entry
    if isinstance(entry, Mapping):
        payload = dict(entry)
        payload.setdefault("name", name)
        return ToolSpec.model_validate(payload)
    raise ArchitectError(f"tool {name!r} has no ToolSpec")


def _manifest_entries(task: Task | Mapping[str, Any]) -> dict[str, Any]:
    for key in ("tool_specs", "tool_manifest", "available_tools"):
        raw = _value(task, key)
        if isinstance(raw, Mapping):
            return {str(name): entry for name, entry in raw.items()}
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            entries: dict[str, Any] = {}
            for entry in raw:
                name = (
                    entry.get("name")
                    if isinstance(entry, Mapping)
                    else getattr(entry, "name", None)
                )
                if isinstance(name, str):
                    entries[name] = entry
            if entries:
                return entries
    raw_tools = _value(task, "tools", ())
    if isinstance(raw_tools, Sequence) and not isinstance(raw_tools, (str, bytes, bytearray)):
        entries = {}
        for entry in raw_tools:
            if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                entries[entry["name"]] = entry
        if entries:
            return entries
    return {}


def _registry_spec(registry: Any, name: str) -> ToolSpec | None:
    if registry is None:
        return None
    try:
        spec_method = getattr(registry, "spec", None)
        if callable(spec_method):
            return _tool_from_entry(name, spec_method(name))
        entry = registry[name]
    except (KeyError, TypeError, AttributeError):
        return None
    return _tool_from_entry(name, entry)


def _lesson_list(raw: Iterable[Lesson | Mapping[str, Any]]) -> list[Lesson]:
    lessons: list[Lesson] = []
    seen: set[str] = set()
    for item in raw:
        lesson = validate_lesson(item)
        if lesson.status != "active":
            continue
        if lesson.id in seen:
            raise ArchitectError(f"duplicate active lesson id {lesson.id!r}")
        seen.add(lesson.id)
        lessons.append(lesson)
    return lessons


def _architect_response_schema() -> dict[str, Any]:
    """The ``Architecture`` schema handed to the model, minus engine-owned ids.

    ``id`` and ``parent_id`` are the engine's own generation bookkeeping
    (``01`` section 2: ``g000``, ``g001``, ...), assigned by :meth:`Architect
    ._parse_architecture` after the model responds. Excluding them here means
    the model is never asked to invent a value it cannot get right and that
    would otherwise abort the run if it guessed wrong.
    """

    schema = Architecture.model_json_schema()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("id", None)
        properties.pop("parent_id", None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [name for name in required if name not in ("id", "parent_id")]
    return schema


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return value


class ArchitectContext:
    """The exact manifest/context sent to the architect model."""

    def __init__(
        self,
        *,
        task: Mapping[str, Any],
        tools: Sequence[ToolSpec],
        lessons: Sequence[Lesson],
        messages: Sequence[Mapping[str, Any]],
        response_schema: Mapping[str, Any],
    ) -> None:
        self.task = dict(task)
        self.tools = list(tools)
        self.lessons = list(lessons)
        self.messages = [dict(message) for message in messages]
        self.response_schema = dict(response_schema)

    @property
    def tool_notes(self) -> list[Lesson]:
        """Tool lessons that were appended to the manifest descriptions."""

        return [lesson for lesson in self.lessons if lesson.kind == "tool_note"]

    @property
    def domain_rules(self) -> list[Lesson]:
        """Domain lessons passed under the explicit architect heading."""

        return [lesson for lesson in self.lessons if lesson.kind == "domain_rule"]


class Architect:
    """Build and validate generation-zero architectures."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        registry: Any = None,
        memory: str | Path | None = None,
        lessons_store: LessonStore | None = None,
        event_sink: EventSink | None = None,
        writer: Any = None,
        model_key: str = ARCHITECT_MODEL_KEY,
        allow_llm_control: bool = False,
    ) -> None:
        self.llm = llm or LLMClient()
        self.registry = registry
        self.memory = memory
        self.lessons_store = lessons_store
        self.event_sink = event_sink
        self.writer = writer
        self.model_key = model_key
        # The router convention is explicitly unresolved in OPEN-QUESTIONS.
        # Keep it opt-in so a malformed/ambiguous proposal cannot silently
        # change execution semantics.
        self.allow_llm_control = allow_llm_control
        self.last_context: ArchitectContext | None = None
        self.last_lessons: list[Lesson] = []

    def load_lessons(
        self,
        task: Task | Mapping[str, Any],
        *,
        memory: str | Path | None = None,
    ) -> list[Lesson]:
        """Load active lessons immediately before a proposal is built."""

        namespace = memory if memory is not None else self.memory
        if namespace is None:
            namespace = _value(task, "memory", "") or None
        if namespace is None or not str(namespace):
            return []
        store = self.lessons_store or LessonStore(namespace)
        try:
            return _lesson_list(store.load(active_only=True))
        except LessonStoreError as exc:
            raise ArchitectError(str(exc)) from exc

    def propose(
        self,
        task: Task | Mapping[str, Any],
        *,
        generation: int = 0,
        lessons: Iterable[Lesson | Mapping[str, Any]] | None = None,
        memory: str | Path | None = None,
        memory_ns: str | Path | None = None,
        parent_id: str | None = None,
    ) -> Architecture:
        """Ask for one strict architecture and emit ``architecture.proposed``.

        At generation zero, omitting ``lessons`` always reads the namespace's
        JSONL file.  Passing an explicit list is useful for deterministic unit
        tests and for a future loop that has already loaded ``run.started``'s
        lesson snapshot.
        """

        if generation < 0:
            raise ArchitectError("generation must be non-negative")
        if memory is not None and memory_ns is not None:
            raise ArchitectError("pass only one of memory and memory_ns")
        resolved_memory = memory if memory is not None else memory_ns
        loaded = (
            self.load_lessons(task, memory=resolved_memory)
            if lessons is None
            else _lesson_list(lessons)
        )
        if generation == 0 and parent_id is not None:
            raise ArchitectError("generation zero cannot have a parent architecture")
        if generation > 0 and parent_id is None:
            raise ArchitectError("later generations require a parent architecture id")
        self.last_lessons = loaded

        tool_names = _tool_names(task)
        specs = self._build_tool_specs(task, tool_names, loaded)
        manifest = self._build_manifest(task, specs)
        messages = self._build_messages(manifest, loaded)
        response_schema = _architect_response_schema()
        self.last_context = ArchitectContext(
            task=manifest,
            tools=specs,
            lessons=loaded,
            messages=messages,
            response_schema=response_schema,
        )
        completion = self._complete(messages, response_schema)
        architecture = self._parse_architecture(
            completion, generation=generation, parent_id=parent_id
        )
        self._validate_proposal(architecture, tool_names, task)
        event_data = {
            "architecture": architecture.model_dump(mode="json", exclude_none=False),
            "generation": generation,
        }
        self._validate_event_compatibility(event_data)
        self._emit("architecture.proposed", event_data)
        return architecture

    def _build_tool_specs(
        self,
        task: Task | Mapping[str, Any],
        names: Sequence[str],
        lessons: Sequence[Lesson],
    ) -> list[ToolSpec]:
        entries = _manifest_entries(task)
        notes_by_tool: dict[str, list[str]] = {}
        for lesson in lessons:
            if lesson.kind == "tool_note" and lesson.tool in names:
                notes_by_tool.setdefault(lesson.tool, []).append(lesson.text)

        specs: list[ToolSpec] = []
        for name in names:
            base = entries.get(name)
            if base is None:
                base = _registry_spec(self.registry, name)
            if base is None:
                try:
                    base = build_spec(name)
                except (KeyError, ValueError) as exc:
                    raise ArchitectError(
                        f"no ToolSpec is available for task tool {name!r}"
                    ) from exc
            spec = _tool_from_entry(name, base)
            description = spec.description
            for note in notes_by_tool.get(name, ()):
                description = append_tool_note(description, note)
            specs.append(
                ToolSpec(
                    name=spec.name,
                    description=description,
                    parameters=json.loads(json.dumps(spec.parameters)),
                )
            )
        return specs

    @staticmethod
    def _build_manifest(
        task: Task | Mapping[str, Any], specs: Sequence[ToolSpec]
    ) -> dict[str, Any]:
        if isinstance(task, Mapping):
            manifest = {str(key): _jsonable(value) for key, value in task.items()}
        else:
            manifest = _jsonable(task.model_dump(mode="json"))
        manifest["tools"] = [spec.model_dump(mode="json") for spec in specs]
        return manifest

    @staticmethod
    def _build_messages(
        manifest: Mapping[str, Any], lessons: Sequence[Lesson]
    ) -> list[dict[str, str]]:
        system = (
            "You are Occam's architecture architect. Design a compact multi-agent DAG for the "
            "task below. Return exactly one JSON object matching the supplied Architecture "
            "schema and no prose. Use 3 to 6 roles, tag every role with a justification, "
            "bind only manifest tools, and prefer deterministic control. Put reusable task "
            "instructions in the system_prompt of the roles that need them."
        )
        sections = [
            "TASK AND TOOL MANIFEST",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            "ARCHITECTURE OUTPUT SCHEMA",
            json.dumps(_architect_response_schema(), ensure_ascii=False, sort_keys=True),
        ]
        domain_rules = [lesson for lesson in lessons if lesson.kind == "domain_rule"]
        if domain_rules:
            sections.extend(
                [
                    DOMAIN_RULE_HEADING,
                    *[f"- {lesson.text}" for lesson in domain_rules],
                    "Place each applicable rule in the prompts of the roles that need it; do "
                    "not invent additional facts.",
                ]
            )
        sections.append(
            "The manifest tool descriptions already include any learned tool notes. Preserve "
            "their meaning and do not hardcode facts that are absent from the manifest."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def _complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        response_schema: Mapping[str, Any],
    ) -> Any:
        try:
            return self.llm.complete(
                self.model_key,
                messages,
                response_schema=response_schema,
                temperature=0.0,
            )
        except TypeError as exc:
            # Keep the public boundary friendly to tiny deterministic test
            # doubles while the production LLMClient always takes the strict
            # response_schema keyword.
            try:
                return self.llm.complete(self.model_key, messages, None, response_schema)
            except TypeError:
                raise ArchitectError(f"architect completion failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - normalize provider failures at this boundary
            raise ArchitectError(
                f"architect completion failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _completion_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, Mapping):
            if completion.get("tool_calls"):
                raise ArchitectError("architect output must be JSON text, not native tool calls")
            return str(completion.get("text") or completion.get("content") or "")
        if getattr(completion, "tool_calls", None):
            raise ArchitectError("architect output must be JSON text, not native tool calls")
        return str(getattr(completion, "text", "") or "")

    def _parse_architecture(
        self,
        completion: Any,
        *,
        generation: int,
        parent_id: str | None,
    ) -> Architecture:
        text = self._completion_text(completion).strip()
        if not text:
            raise ArchitectError("architect returned empty output")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArchitectError("architect output must be one JSON object with no prose") from exc
        if not isinstance(payload, Mapping):
            raise ArchitectError("architect output must be a JSON object")
        # `id`/`parent_id` are engine-assigned generation bookkeeping, not
        # something the model can be expected to guess correctly - overwrite
        # whatever (if anything) it returned rather than aborting the run
        # over a plausible-looking value like "fx_recon_a_v1".
        payload = {**payload, "id": f"g{generation:03d}", "parent_id": parent_id}
        try:
            architecture = Architecture.model_validate(payload)
        except ValidationError as exc:
            raise ArchitectError(f"architect output failed strict validation: {exc}") from exc
        return architecture

    def _validate_proposal(
        self,
        architecture: Architecture,
        allowed_tools: Sequence[str],
        task: Task | Mapping[str, Any],
    ) -> None:
        if not MIN_ROLES <= len(architecture.roles) <= MAX_ROLES:
            raise ArchitectError(f"architecture must contain {MIN_ROLES} to {MAX_ROLES} roles")
        if architecture.control == "llm" and not self.allow_llm_control:
            raise ArchitectError(
                "control='llm' is disabled: the router convention is unresolved in OPEN-QUESTIONS"
            )
        allowed = set(allowed_tools)
        output_keys: set[str] = set()
        role_ids: set[str] = set()
        for role in architecture.roles:
            if role.id in role_ids:
                raise ArchitectError(f"duplicate role id {role.id!r}")
            role_ids.add(role.id)
            if role.output_key in output_keys:
                raise ArchitectError(f"duplicate role output_key {role.output_key!r}")
            output_keys.add(role.output_key)
            unknown = sorted(set(role.tools) - allowed)
            if unknown:
                raise ArchitectError(
                    f"role {role.id!r} binds tools outside the task manifest: {', '.join(unknown)}"
                )
        self._validate_model_keys(architecture)
        try:
            validate_architecture(architecture)
        except ArchitectureError as exc:
            raise ArchitectError(f"architect proposed an invalid DAG: {exc}") from exc

        # Keep this explicit even though the helper above validates the graph;
        # it documents that a task's tool names, rather than the global registry,
        # are the only legal bindings.
        task_tools = set(_tool_names(task))
        if allowed != task_tools:
            raise ArchitectError("task tool manifest changed while validating the proposal")

    @staticmethod
    def _validate_event_compatibility(data: Mapping[str, Any]) -> None:
        """Reject proposals the event schema cannot persist before returning them."""

        envelope = {
            "ts": _now_iso(),
            "run_id": "architect-validation",
            "seq": 0,
            "type": "architecture.proposed",
            "data": dict(data),
        }
        try:
            validate_event(envelope)
        except ValueError as exc:
            raise ArchitectError(
                f"architect proposal is not compatible with the event schema: {exc}"
            ) from exc

    def _validate_model_keys(self, architecture: Architecture) -> None:
        configs = getattr(self.llm, "configs", None)
        if not isinstance(configs, Mapping) or not configs:
            return
        missing = sorted({role.model for role in architecture.roles if role.model not in configs})
        if missing:
            raise ArchitectError(
                f"architecture uses unconfigured model key(s): {', '.join(missing)}"
            )

    def _emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, dict(data))
            return
        if self.writer is None:
            return
        event = {"ts": _now_iso(), "type": event_type, "data": dict(data)}
        try:
            self.writer.append(event)
        except AttributeError:
            self.writer.write(event)


def propose_architecture(
    task: Task | Mapping[str, Any],
    *,
    architect: Architect | None = None,
    **kwargs: Any,
) -> Architecture:
    """Functional convenience wrapper around :class:`Architect`."""

    return (architect or Architect()).propose(task, **kwargs)


__all__ = [
    "ARCHITECT_MODEL_KEY",
    "Architect",
    "ArchitectContext",
    "ArchitectError",
    "DOMAIN_RULE_HEADING",
    "MAX_ROLES",
    "MIN_ROLES",
    "propose_architecture",
]
