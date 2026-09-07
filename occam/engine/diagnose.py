"""Failure diagnosis, typed lesson writing, and the hard safety rules."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from occam.core.models import Architecture, Case, Lesson, RunResult
from occam.engine.ablation import AblationTable
from occam.engine.mutate import Mutation, MutationError
from occam.llm.client import LLMClient
from occam.memory.lessons import LessonStore, LessonStoreError, validate_lesson_text

DIAGNOSE_MODEL_KEY = "architect"
MAX_LESSONS_PER_RUN = 5
_WORD = re.compile(r"[a-z0-9]+")


class LessonProposal(BaseModel):
    """The small lesson shape accepted from the diagnosis model."""

    model_config = ConfigDict(extra="ignore")

    kind: str
    text: str
    tool: str | None = None
    case_ids: list[str] = Field(default_factory=list)


class DiagnosisPayload(BaseModel):
    """Structured diagnosis response; the mutation is validated again in code."""

    model_config = ConfigDict(extra="ignore")

    text: str = ""
    failure_summary: str = ""
    chosen_mutation: dict[str, Any] | None = None
    mutation: dict[str, Any] | None = None
    lessons: list[LessonProposal] = Field(default_factory=list)


@dataclass(frozen=True)
class DiagnosisResult:
    """A safe mutation plus the lessons that survived the write guard."""

    text: str
    failure_summary: str
    mutation: Mutation
    lessons: tuple[Lesson, ...] = ()
    rejected_lessons: tuple[str, ...] = ()

    def event_data(self, generation: int) -> dict[str, Any]:
        return {
            "generation": generation,
            "text": self.text[:600],
            "failure_summary": self.failure_summary,
            "chosen_mutation": {
                "type": self.mutation.type,
                "target_role": self.mutation.target_role,
                "rationale": self.mutation.rationale,
            },
        }


EventSink = Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _expected_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _expected_numbers(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _expected_numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            yield number


def case_expected_values(cases: Sequence[Case]) -> list[float]:
    """Collect expected totals and sub-results used by the near-value guard."""

    return [number for case in cases for number in _expected_numbers(case.expected)]


def lesson_leak_reasons(text: str, *, case_values: Iterable[float | int | str] = ()) -> list[str]:
    """Return guard failures instead of allowing a proposed lesson to escape."""

    try:
        validate_lesson_text(text, case_values=case_values)
    except LessonStoreError as exc:
        return [str(exc)]
    return []


def lesson_passes_leak_guard(
    text: str,
    *,
    case_values: Iterable[float | int | str] = (),
) -> bool:
    """Boolean convenience API used by unit tests and diagnosis callers."""

    return not lesson_leak_reasons(text, case_values=case_values)


def _tokens(text: str) -> Counter[str]:
    return Counter(_WORD.findall(text.lower()))


def cosine_similarity(left: str, right: str) -> float:
    """A dependency-free cosine similarity for concise lesson deduplication."""

    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 1.0 if left.strip().lower() == right.strip().lower() else 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class LessonWriter:
    """Append only general, evidence-backed lessons that pass the leak guard."""

    def __init__(
        self,
        store: LessonStore,
        *,
        run_id: str,
        active_lessons: Sequence[Lesson] = (),
        event_sink: EventSink | None = None,
        max_lessons: int = MAX_LESSONS_PER_RUN,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self._lessons = list(active_lessons)
        self.event_sink = event_sink
        self.max_lessons = max_lessons
        self.written: list[Lesson] = []
        self.rejected: list[str] = []

    def write(
        self,
        proposal: LessonProposal | Mapping[str, Any],
        *,
        generation: int,
        cases: Sequence[Case],
        default_case_ids: Sequence[str],
    ) -> Lesson | None:
        """Validate, deduplicate, append, and emit one lesson."""

        if len(self.written) >= self.max_lessons:
            self.rejected.append("per-run lesson limit reached")
            return None
        try:
            parsed = (
                proposal
                if isinstance(proposal, LessonProposal)
                else LessonProposal.model_validate(proposal)
            )
        except ValidationError as exc:
            self.rejected.append(f"invalid lesson proposal: {exc}")
            return None
        text = parsed.text.strip()
        values = case_expected_values(cases)
        reasons = lesson_leak_reasons(text, case_values=values)
        if reasons:
            self.rejected.extend(reasons)
            return None
        if parsed.kind not in {"tool_note", "domain_rule"}:
            self.rejected.append(f"unknown lesson kind: {parsed.kind}")
            return None
        tool = parsed.tool if parsed.kind == "tool_note" else None
        if parsed.kind == "tool_note" and not tool:
            self.rejected.append("tool_note lesson is missing a tool")
            return None
        if any(cosine_similarity(text, previous.text) > 0.9 for previous in self._lessons):
            self.rejected.append("duplicate lesson")
            return None
        case_ids = [case_id for case_id in parsed.case_ids if case_id in {c.id for c in cases}]
        if not case_ids:
            case_ids = list(default_case_ids)
        if not case_ids:
            self.rejected.append("lesson has no evidence case")
            return None
        lesson_id = self._lesson_id(parsed.kind, text)
        evidence = {
            "run_id": self.run_id,
            "generation": generation,
            "case_ids": case_ids,
            "trace_refs": [f"g{generation:03d}/{case_id}" for case_id in case_ids],
        }
        lesson = Lesson(
            id=lesson_id,
            kind=parsed.kind,  # type: ignore[arg-type]
            text=text,
            tool=tool,
            evidence=evidence,
            born={"run_id": self.run_id, "generation": generation},
        )
        try:
            stored = self.store.append(lesson)
        except LessonStoreError as exc:
            self.rejected.append(str(exc))
            return None
        self._lessons.append(stored)
        self.written.append(stored)
        if self.event_sink is not None:
            self.event_sink(
                "lesson.written",
                {"generation": generation, "lesson": stored.model_dump(mode="json")},
            )
        return stored

    def _lesson_id(self, kind: str, text: str) -> str:
        prefix = "tool" if kind == "tool_note" else "rule"
        slug = "_".join(_WORD.findall(text.lower()))[:45].strip("_") or "lesson"
        base = f"lesson_{prefix}_{slug}"
        existing = {lesson.id for lesson in self._lessons}
        candidate = base
        suffix = 2
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("text") or completion.get("content") or "")
    return str(getattr(completion, "text", "") or "")


def _parse_payload(completion: Any) -> DiagnosisPayload:
    text = _completion_text(completion).strip()
    if not text:
        raise ValueError("diagnosis returned empty output")
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    encoded = blocks[-1] if blocks else text
    try:
        payload = json.loads(encoded)
        return DiagnosisPayload.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError("diagnosis output must be one JSON object") from exc


def _role_rows(table: AblationTable | Sequence[Any]) -> list[Any]:
    return list(table.rows) if isinstance(table, AblationTable) else list(table)


def _safe_mutation(
    architecture: Architecture,
    requested: Mapping[str, Any] | None,
    rows: Sequence[Any],
    *,
    pass_rate: float | None = None,
) -> Mutation:
    """Apply code-level mutation safety before a choice reaches ``mutate.py``.

    A witness from an all-failed run is not actionable evidence, and removing a
    role from a two-role architecture would leave a one-role team.  In either
    case, keep the architecture intact and ask for a prompt rewrite instead.
    """

    witnesses = [row for row in rows if _value(row, "verdict") == "witness"]
    if witnesses and pass_rate is not None and pass_rate > 0.0 and len(architecture.roles) > 2:
        target = max(
            witnesses,
            key=lambda row: (float(_value(row, "cost_share", 0.0)), str(_value(row, "role_id"))),
        )
        rationale = str(_value(requested, "rationale", "") or "").strip()
        if not rationale:
            rationale = (
                "The highest-cost witness did not change answers beyond the measured noise floor."
            )
        return Mutation(
            type="prune",
            target_role=str(_value(target, "role_id")),
            rationale=rationale,
        )

    payload = dict(requested or {})
    if not payload:
        payload = {
            "type": "rewrite_prompt",
            "target_role": architecture.final_role,
            "rationale": "Review the failed cases and preserve the required answer format.",
        }
    if payload.get("type") == "prune":
        payload["type"] = "rewrite_prompt"
        payload.setdefault("target_role", architecture.final_role)
        payload.setdefault(
            "rationale", "Review the failed cases and preserve the required answer format."
        )
    if not payload.get("target_role"):
        payload["target_role"] = architecture.final_role
    try:
        return Mutation.from_mapping(payload)
    except MutationError:
        return Mutation(
            type="rewrite_prompt",
            target_role=architecture.final_role,
            rationale="Review the failed cases and preserve the required answer format.",
        )


def _diagnosis_messages(
    *,
    task: Any,
    generation: int,
    full: RunResult,
    table: AblationTable,
    cases: Sequence[Case],
    history: Sequence[Mapping[str, Any]],
    lessons: Sequence[Lesson],
) -> list[dict[str, str]]:
    cases_by_id = {case.id: case for case in cases}
    failed = [
        {
            "case_id": result.case_id,
            "input": cases_by_id[result.case_id].input if result.case_id in cases_by_id else None,
            "expected": cases_by_id[result.case_id].expected
            if result.case_id in cases_by_id
            else None,
            "answer": result.answer,
            "passed": result.passed,
            "sub_results": result.sub_results,
            "per_role": result.per_role,
        }
        for result in full.results
        if not result.passed
    ]
    system = (
        "You are Occam's failure diagnostician. Return one JSON object matching the supplied "
        "schema and no prose. Explain the observed failures, choose exactly one closed-menu "
        "mutation, and propose zero to three reusable typed lessons. Lessons must be general "
        "rules, never answers, dates, invoice ids, or case-specific numbers. Cite only the "
        "provided case ids in your evidence. If a witness appears, the runtime will enforce "
        "pruning the highest-cost witness."
    )
    context = {
        "task": _jsonable(task),
        "generation": generation,
        "failed_cases": _jsonable(failed),
        "ablation": _jsonable(table.model_dump(mode="json")),
        "history": _jsonable(history),
        "active_lessons": _jsonable([lesson.model_dump(mode="json") for lesson in lessons]),
        "lesson_schema": LessonProposal.model_json_schema(),
        "mutation_schema": Mutation.model_json_schema(),
        "response_schema": DiagnosisPayload.model_json_schema(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, sort_keys=True)},
    ]


def diagnose(
    task: Any,
    *,
    generation: int,
    full: RunResult,
    table: AblationTable,
    architecture: Architecture,
    cases: Sequence[Case],
    history: Sequence[Mapping[str, Any]] = (),
    lessons: Sequence[Lesson] = (),
    llm: Any | None = None,
    lesson_store: LessonStore | None = None,
    run_id: str = "run",
    event_sink: EventSink | None = None,
    model_key: str = DIAGNOSE_MODEL_KEY,
) -> DiagnosisResult:
    """Diagnose one generation and append only guard-approved lessons."""

    client = llm or LLMClient()
    messages = _diagnosis_messages(
        task=task.model_dump(mode="json") if hasattr(task, "model_dump") else task,
        generation=generation,
        full=full,
        table=table,
        cases=cases,
        history=history,
        lessons=lessons,
    )
    try:
        completion = client.complete(
            model_key,
            messages,
            response_schema=DiagnosisPayload.model_json_schema(),
            temperature=0.0,
        )
        payload = _parse_payload(completion)
    except Exception as exc:  # noqa: BLE001 - a diagnosis cannot lose the run
        payload = DiagnosisPayload(
            text=f"Diagnosis unavailable ({type(exc).__name__}); applying a safe prompt review.",
            failure_summary="The diagnosis provider did not return a usable structured response.",
            chosen_mutation={
                "type": "rewrite_prompt",
                "target_role": architecture.final_role,
                "rationale": "Review the failed cases and preserve the required answer format.",
            },
            lessons=[],
        )
    requested = payload.chosen_mutation or payload.mutation
    mutation = _safe_mutation(
        architecture,
        requested,
        _role_rows(table),
        pass_rate=full.pass_rate,
    )
    default_case_ids = [result.case_id for result in full.results if not result.passed]
    store = lesson_store or LessonStore(getattr(task, "memory", "memory"))
    pending_lesson_events: list[tuple[str, Mapping[str, Any]]] = []

    def collect_lesson_event(event_type: str, data: Mapping[str, Any]) -> None:
        pending_lesson_events.append((event_type, dict(data)))

    writer = LessonWriter(
        store,
        run_id=run_id,
        active_lessons=lessons,
        event_sink=collect_lesson_event,
    )
    for proposal in payload.lessons:
        writer.write(
            proposal,
            generation=generation,
            cases=cases,
            default_case_ids=default_case_ids,
        )
    result = DiagnosisResult(
        text=(payload.text or payload.failure_summary or "Diagnosis complete").strip()[:600],
        failure_summary=payload.failure_summary.strip(),
        mutation=mutation,
        lessons=tuple(writer.written),
        rejected_lessons=tuple(writer.rejected),
    )
    if event_sink is not None:
        event_sink("diagnosis.emitted", result.event_data(generation))
        for event_type, data in pending_lesson_events:
            event_sink(event_type, data)
    return result


__all__ = [
    "DIAGNOSE_MODEL_KEY",
    "DiagnosisPayload",
    "DiagnosisResult",
    "LessonProposal",
    "LessonWriter",
    "MAX_LESSONS_PER_RUN",
    "case_expected_values",
    "cosine_similarity",
    "diagnose",
    "lesson_leak_reasons",
    "lesson_passes_leak_guard",
]
