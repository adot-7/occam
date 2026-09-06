"""The strict first-generation architecture proposal.

The architect is intentionally a narrow boundary: it reads the task manifest
and active lessons, asks the configured ``architect`` model for one JSON
architecture, validates that response before it can reach the executor, and
allows one bounded correction request when proposal validation fails. It
optionally emits the schema-compatible ``architecture.proposed`` event. It
does not diagnose failures or write lessons; those responsibilities belong to
WP-10.
"""

from __future__ import annotations

import json
import re
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
MAX_ARCHITECT_REPAIR_ATTEMPTS = 1

_JSON_FENCE = re.compile(r"```(?P<language>[^\r\n`]*)\r?\n(?P<body>.*?)```", re.DOTALL)
_WRAPPER_MARKERS = frozenset("{}[]`")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,79}$")
_SAFE_PROVIDER_FIELD = re.compile(r"^(status|type|code|parameter)=([A-Za-z0-9_.:-]{1,80})$")
_PROVIDER_FIELDS = ("status", "type", "code", "parameter")


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


def _architect_response_schema(
    role_model_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
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
    if role_model_keys is not None:
        definitions = schema.get("$defs")
        role_schema = definitions.get("Role") if isinstance(definitions, Mapping) else None
        role_properties = (
            role_schema.get("properties") if isinstance(role_schema, Mapping) else None
        )
        model_schema = (
            role_properties.get("model") if isinstance(role_properties, Mapping) else None
        )
        if not isinstance(model_schema, dict):
            raise ArchitectError("architect schema is missing the Role.model field")
        model_schema["enum"] = list(role_model_keys)
    return schema


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return value


def _output_shape_error(message: str) -> ArchitectError:
    """Return a safe, consistently classified model-output error."""

    return ArchitectError(f"architect output-shape failure: {message}")


def _reject_json_constant(value: str) -> Any:
    """Reject Python's non-standard NaN/Infinity JSON extensions."""

    raise ValueError(f"non-standard JSON constant {value}")


def _strict_json_object(text: str) -> dict[str, Any]:
    """Extract exactly one JSON object from a tightly wrapped completion.

    The model is allowed a Markdown JSON fence or short prose before/after one
    object because some providers add a human-readable preamble despite the
    response schema.  The wrapper itself may not contain JSON delimiters,
    fences, or another JSON value; this keeps extraction deterministic and
    prevents a valid nested/second object from being silently selected.
    """

    stripped = text.strip()
    if not stripped:
        raise _output_shape_error("expected exactly one JSON object; output was empty")

    fences = list(_JSON_FENCE.finditer(stripped))
    if "```" in stripped:
        if len(fences) != 1:
            raise _output_shape_error("expected exactly one complete JSON fence")
        fence = fences[0]
        language = fence.group("language").strip().lower()
        if language not in {"", "json"}:
            raise _output_shape_error("Markdown fence must contain JSON")
        outside = stripped[: fence.start()] + stripped[fence.end() :]
        if any(marker in outside for marker in _WRAPPER_MARKERS):
            raise _output_shape_error("found multiple or ambiguous JSON objects")
        candidate = fence.group("body").strip()
        try:
            payload = json.loads(candidate, parse_constant=_reject_json_constant)
        except (TypeError, ValueError) as exc:
            raise _output_shape_error("fenced content was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise _output_shape_error("expected one JSON object, not another JSON value")
        return payload

    start = stripped.find("{")
    if start < 0:
        raise _output_shape_error("expected exactly one JSON object; none was found")
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    try:
        payload, end = decoder.raw_decode(stripped, start)
    except json.JSONDecodeError as exc:
        raise _output_shape_error("the JSON object was malformed") from exc
    if not isinstance(payload, dict):
        raise _output_shape_error("expected one JSON object")

    outside = stripped[:start] + stripped[end:]
    if any(marker in outside for marker in _WRAPPER_MARKERS):
        raise _output_shape_error("found multiple or ambiguous JSON objects")
    # A second scalar JSON value is not prose around the object.  Prose such
    # as "Here is the architecture" is intentionally not valid JSON and is
    # therefore unaffected by this check.
    for wrapper in (stripped[:start].strip(), stripped[end:].strip()):
        if not wrapper:
            continue
        try:
            json.loads(wrapper, parse_constant=_reject_json_constant)
        except (TypeError, ValueError):
            continue
        raise _output_shape_error("found more than one JSON value")
    return payload


def _validation_category(error: ArchitectError) -> str:
    """Map an internal validation error to a concise repair-prompt category."""

    message = str(error)
    if "output-shape failure" in message:
        return "strict JSON output shape"
    if "strict validation" in message:
        return "Pydantic Architecture schema"
    if "event schema" in message:
        return "event schema compatibility"
    if "invalid DAG" in message or "unknown input" in message or "cycle" in message:
        return "DAG input contract"
    if "model key" in message or "configured role model" in message:
        return "configured role model keys"
    if "tool" in message.lower() or "ToolSpec" in message:
        return "task tool bindings"
    return "Architecture proposal contract"


def _safe_dependency_pairs(architecture: Architecture | None) -> tuple[tuple[str, str], ...]:
    """Return safe role/output-key pairs in the model-provided order."""

    if architecture is None:
        return ()
    return tuple(
        (role.id, role.output_key)
        for role in architecture.roles
        if _SAFE_IDENTIFIER.fullmatch(role.id) and _SAFE_IDENTIFIER.fullmatch(role.output_key)
    )


def _safe_validation_detail(error: ArchitectError, category: str) -> str:
    """Expose only safe identifiers that help the one repair attempt."""

    if category != "DAG input contract":
        return ""
    match = re.search(
        r"role ['\"]([^'\"]+)['\"] reads unknown input ['\"]([^'\"]+)['\"]",
        str(error),
    )
    if match is None:
        return ""
    role_id, input_name = match.groups()
    if not (_SAFE_IDENTIFIER.fullmatch(role_id) and _SAFE_IDENTIFIER.fullmatch(input_name)):
        return ""
    return f"role {role_id} referenced unknown input {input_name}"


def _safe_provider_metadata(error: BaseException) -> str | None:
    """Keep only the client's already-sanitized provider metadata fields."""

    match = re.search(r"provider_error\[([^\]\r\n]*)\]", str(error))
    if match is None:
        return None
    fields: dict[str, str] = {}
    for raw_field in match.group(1).split(";"):
        field = raw_field.strip()
        field_match = _SAFE_PROVIDER_FIELD.fullmatch(field)
        if field_match is None:
            return None
        name, value = field_match.groups()
        if name in fields:
            return None
        fields[name] = value
    if set(fields) != set(_PROVIDER_FIELDS):
        return None
    return (
        "provider_error[" + "; ".join(f"{name}={fields[name]}" for name in _PROVIDER_FIELDS) + "]"
    )


def _architect_provider_error(error: BaseException) -> ArchitectError:
    """Normalize provider failures without copying their message or payload."""

    metadata = _safe_provider_metadata(error)
    suffix = f"; {metadata}" if metadata is not None else ""
    return ArchitectError(f"architect provider failure: {type(error).__name__}{suffix}")


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
        role_model_keys = self._configured_role_model_keys()
        specs = self._build_tool_specs(task, tool_names, loaded)
        manifest = self._build_manifest(task, specs)
        messages = self._build_messages(manifest, loaded, role_model_keys)
        response_schema = _architect_response_schema(role_model_keys)
        self.last_context = ArchitectContext(
            task=manifest,
            tools=specs,
            lessons=loaded,
            messages=messages,
            response_schema=response_schema,
        )
        completion = self._complete(messages, response_schema)
        architecture: Architecture | None = None
        for repair_attempt in range(MAX_ARCHITECT_REPAIR_ATTEMPTS + 1):
            try:
                architecture = self._parse_architecture(
                    completion, generation=generation, parent_id=parent_id
                )
                self._validate_proposal(architecture, tool_names, task)
                event_data = {
                    "architecture": architecture.model_dump(mode="json", exclude_none=False),
                    "generation": generation,
                }
                self._validate_event_compatibility(event_data)
                break
            except ArchitectError as exc:
                if repair_attempt >= MAX_ARCHITECT_REPAIR_ATTEMPTS:
                    raise
                category = _validation_category(exc)
                messages = self._build_repair_messages(
                    messages,
                    category=category,
                    role_model_keys=role_model_keys,
                    dependency_pairs=_safe_dependency_pairs(architecture),
                    validation_detail=_safe_validation_detail(exc, category),
                )
                if self.last_context is not None:
                    self.last_context.messages = [dict(message) for message in messages]
                completion = self._complete(messages, response_schema)
        assert architecture is not None
        event_data = {
            "architecture": architecture.model_dump(mode="json", exclude_none=False),
            "generation": generation,
        }
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
        manifest: Mapping[str, Any],
        lessons: Sequence[Lesson],
        role_model_keys: Sequence[str],
    ) -> list[dict[str, str]]:
        system = (
            "You are Occam's architecture architect. Design a compact multi-agent DAG for the "
            "task below. Return exactly one JSON object matching the supplied Architecture "
            "schema and no prose. Use 3 to 6 roles, tag every role with a justification, "
            "bind only manifest tools, and prefer deterministic control. Every role's model "
            "must be one configured role model key from the allowlist below, never a provider "
            "model ID. Put reusable task instructions in the system_prompt of the roles that "
            "need them. For roles where tool calls are expected, set max_turns >= 16 so the "
            "role has room to finish its tool work; keep no-tool roles appropriately bounded. "
            "Declare simple role ids in topological order. Each later role's inputs must "
            'exactly equal an earlier declared role.id or the literal "task"; never use an '
            "output_key, role name, or invented descriptive alias such as parsed_case."
        )
        sections = [
            "TASK AND TOOL MANIFEST",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            "CONFIGURED ROLE MODEL KEYS (use these keys exactly; do not use provider model IDs)",
            json.dumps(list(role_model_keys), ensure_ascii=False),
            "DAG INPUT CONTRACT (role ids, not output keys)",
            'Declare roles in topological order. Each inputs item must be exactly "task" or '
            "the id of an earlier declared role. An output_key names stored context only and "
            "must never appear as an inputs token.",
            "ARCHITECTURE OUTPUT SCHEMA",
            json.dumps(
                _architect_response_schema(role_model_keys),
                ensure_ascii=False,
                sort_keys=True,
            ),
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

    @staticmethod
    def _build_repair_messages(
        messages: Sequence[Mapping[str, str]],
        *,
        category: str,
        role_model_keys: Sequence[str],
        dependency_pairs: Sequence[tuple[str, str]],
        validation_detail: str,
    ) -> list[dict[str, str]]:
        """Append one concise, sanitized correction request to the context."""

        known_role_ids = json.dumps(
            [role_id for role_id, _output_key in dependency_pairs], ensure_ascii=False
        )
        role_output_pairs = json.dumps(list(dependency_pairs), ensure_ascii=False)
        sections = [
            "ARCHITECTURE REPAIR: the previous proposal failed validation.",
            f"Validation category: {category}.",
        ]
        if validation_detail:
            sections.append(f"Sanitized validation detail: {validation_detail}.")
        sections.extend(
            [
                "Return a complete replacement as exactly one JSON object and no prose.",
                "Use only these configured role model keys:",
                json.dumps(list(role_model_keys), ensure_ascii=False) + ".",
                'Declare roles in topological order. Each role input must be exactly "task" or '
                "the id of an earlier declared role; never use an output_key, role name, or "
                "invented descriptive alias.",
                "Safe earlier role.id dependency tokens in proposal order:",
                known_role_ids + ".",
                "For reference only, role.id to output_key mapping (use only the first value "
                "in inputs; output_key is a context label, never an input token):",
                role_output_pairs + ".",
                "Bind tools only from the task manifest and satisfy the supplied schema.",
            ]
        )
        correction = " ".join(sections)
        return [
            *[dict(message) for message in messages],
            {"role": "user", "content": correction},
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
        except TypeError:
            # Keep the public boundary friendly to tiny deterministic test
            # doubles while the production LLMClient always takes the strict
            # response_schema keyword.
            try:
                return self.llm.complete(self.model_key, messages, None, response_schema)
            except Exception as fallback_exc:  # noqa: BLE001 - normalize provider failures
                raise _architect_provider_error(fallback_exc) from fallback_exc
        except Exception as exc:  # noqa: BLE001 - normalize provider failures at this boundary
            raise _architect_provider_error(exc) from exc

    @staticmethod
    def _completion_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, Mapping):
            if completion.get("tool_calls"):
                raise _output_shape_error("native tool calls are not JSON text")
            return str(completion.get("text") or completion.get("content") or "")
        if getattr(completion, "tool_calls", None):
            raise _output_shape_error("native tool calls are not JSON text")
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
            raise _output_shape_error("expected exactly one JSON object; output was empty")
        payload = _strict_json_object(text)
        # `id`/`parent_id` are engine-assigned generation bookkeeping, not
        # something the model can be expected to guess correctly - overwrite
        # whatever (if anything) it returned rather than aborting the run
        # over a plausible-looking value like "fx_recon_a_v1".
        payload = {**payload, "id": f"g{generation:03d}", "parent_id": parent_id}
        try:
            architecture = Architecture.model_validate(payload)
        except ValidationError as exc:
            locations = []
            for error in exc.errors(include_url=False):
                location = ".".join(str(part) for part in error.get("loc", ())) or "root"
                locations.append(f"{location}:{error.get('type', 'validation_error')}")
            summary = ", ".join(locations[:5])
            if len(locations) > 5:
                summary += f", +{len(locations) - 5} more"
            raise ArchitectError(
                f"architect proposal validation failure: strict validation ({summary})"
            ) from exc
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

    def _configured_role_model_keys(self) -> tuple[str, ...]:
        configs = getattr(self.llm, "configs", None)
        if not isinstance(configs, Mapping) or not configs:
            raise ArchitectError(
                "architect proposal validation failure: configured role model keys are unavailable"
            )
        excluded = {ARCHITECT_MODEL_KEY, self.model_key}
        keys = tuple(sorted(str(key) for key in configs if str(key) not in excluded))
        if not keys:
            raise ArchitectError(
                "architect proposal validation failure: no configured role model keys are available"
            )
        return keys

    def _validate_model_keys(self, architecture: Architecture) -> None:
        allowed = set(self._configured_role_model_keys())
        missing = sorted({role.model for role in architecture.roles if role.model not in allowed})
        if missing:
            available = ", ".join(sorted(allowed))
            raise ArchitectError(
                "architect proposal validation failure: role model key(s) are not configured: "
                f"{', '.join(missing)}; allowed role model keys: {available}"
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
