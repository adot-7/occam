"""Pydantic models for Occam's architecture, event, and task contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Justification = Literal[
    "parallel",
    "context_isolation",
    "verification",
    "ensemble",
    "control",
    "unspecified",
]
MemoryPolicy = Literal["none", "scratchpad", "summary"]
ControlMode = Literal["llm", "deterministic"]
Verdict = Literal["load_bearing", "witness", "harmful", "uncertain"]
CheckerName = Literal[
    "json_set_equal",
    "numeric_exact",
    "bfcl_ast",
    "exact",
    "llm_judge",
    "fx_total",
]
EventType = Literal[
    "run.started",
    "lesson.written",
    "reliability.completed",
    "architecture.proposed",
    "execution.started",
    "execution.case",
    "execution.completed",
    "ablation.started",
    "ablation.role",
    "ablation.completed",
    "baseline.completed",
    "diagnosis.emitted",
    "mutation.applied",
    "mutation.reverted",
    "metrics.snapshot",
    "run.completed",
    "log",
]


class ContractModel(BaseModel):
    """Base model with stable, explicit handling of unknown contract fields."""

    model_config = ConfigDict(extra="forbid")


class ToolSpec(ContractModel):
    """A tool binding available to a candidate role."""

    name: str
    description: str
    parameters: dict[str, Any]


class Role(ContractModel):
    """One node in an architecture DAG."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    justification: Justification
    model: str = Field(min_length=1)
    system_prompt: str
    tools: list[str]
    inputs: list[str]
    output_key: str = Field(min_length=1)
    memory: MemoryPolicy = "none"
    max_turns: int = Field(default=6, ge=1)


class Architecture(ContractModel):
    """A versioned, topologically sortable set of roles."""

    id: str = Field(min_length=1)
    parent_id: str | None
    roles: list[Role]
    final_role: str = Field(min_length=1)
    control: ControlMode = "deterministic"
    notes: str = ""


class Case(ContractModel):
    """An evaluation case from a task pack."""

    id: str
    input: str
    expected: Any
    meta: dict[str, Any] = Field(default_factory=dict)


def _validate_lesson_provenance(
    label: str,
    value: dict[str, Any],
    *,
    require_case_ids: bool = True,
) -> None:
    """Validate the stable provenance fields shared by lesson evidence."""

    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"lesson {label}.run_id must be a non-empty string")
    generation = value.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError(f"lesson {label}.generation must be a non-negative integer")
    if require_case_ids and "case_ids" not in value:
        raise ValueError("lesson evidence must include case_ids")
    case_ids = value.get("case_ids")
    if case_ids is not None:
        if not isinstance(case_ids, list) or any(
            not isinstance(case_id, str) or not case_id.strip() for case_id in case_ids
        ):
            raise ValueError(f"lesson {label}.case_ids must be a list of non-empty strings")
    trace_refs = value.get("trace_refs")
    if trace_refs is not None:
        if not isinstance(trace_refs, list) or any(
            not isinstance(trace_ref, str) or not trace_ref.strip() for trace_ref in trace_refs
        ):
            raise ValueError(f"lesson {label}.trace_refs must be a list of non-empty strings")


class Lesson(ContractModel):
    """A reusable, evidence-backed rule learned during a run."""

    id: str = Field(min_length=1)
    kind: Literal["tool_note", "domain_rule"]
    text: str = Field(min_length=1)
    tool: str | None = None
    evidence: dict[str, Any]
    born: dict[str, Any]
    status: Literal["active", "retired"] = "active"

    @field_validator("text")
    @classmethod
    def text_is_short_and_nonblank(cls, value: str) -> str:
        """Keep lessons at the concise, reusable-rule boundary from the PRD."""

        text = value.strip()
        if not text:
            raise ValueError("lesson text must not be blank")
        if len(text.split()) > 60:
            raise ValueError("lesson text must contain at most 60 words")
        return text

    @model_validator(mode="after")
    def validate_typed_provenance(self) -> Lesson:
        """Enforce the typed lesson and provenance contract at every boundary.

        The diagnose/lesson-writer leak guard also checks case values.  This
        model deliberately owns only the schema-level part of that contract:
        tool notes name a tool, domain rules do not, and both carry enough
        provenance to be followed back to a run and generation.
        """

        if self.kind == "tool_note" and not self.tool:
            raise ValueError("tool_note lessons require a non-empty tool")
        if self.kind == "domain_rule" and self.tool is not None:
            raise ValueError("domain_rule lessons must not name a tool")
        _validate_lesson_provenance("evidence", self.evidence)
        _validate_lesson_provenance("born", self.born, require_case_ids=False)
        if self.born["run_id"] != self.evidence["run_id"]:
            raise ValueError("lesson born.run_id must match evidence.run_id")
        if self.born["generation"] != self.evidence["generation"]:
            raise ValueError("lesson born.generation must match evidence.generation")
        return self


class RoleTrace(ContractModel):
    """Per-role accounting and output for one evaluated case.

    ``cost_usd`` is the **displayed** cost and is what ablation's ``cost_share``
    and the metrics strip read: for a granted model it is the list-rate
    equivalent, never the $0 bill (`00 §7`, `01 §5`). Executors must copy it
    from ``CostBreakdown.cost_usd``, the nominal bill from
    ``CostBreakdown.billed_cost_usd``, and the label verbatim from
    ``CostBreakdown.label`` so the TUI can say which one it is showing.
    """

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    billed_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_label: str = Field(default="metered", min_length=1)
    latency_s: float = Field(default=0.0, ge=0.0)
    output: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    cached: bool = False
    error: str | None = Field(default=None, max_length=256)


class CaseResult(ContractModel):
    """The final result and accounting for one case."""

    case_id: str
    answer: str = ""
    passed: bool
    grade_error: str | None = Field(default=None, max_length=256)
    role_error: str | None = Field(default=None, max_length=256)
    sub_results: dict[str, bool] = Field(default_factory=dict)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_s: float = Field(default=0.0, ge=0.0)
    per_role: dict[str, RoleTrace] = Field(default_factory=dict)


class RunResult(ContractModel):
    """Aggregate result for a full, ablated, or baseline variant."""

    architecture_id: str
    variant: str
    results: list[CaseResult] = Field(default_factory=list)
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_s_mean: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)


class ConfidenceInterval(ContractModel):
    """A lower/upper confidence interval for a reported metric."""

    lo: float
    hi: float


class BaselineComparison(ContractModel):
    """Comparison between a generation and its cost-matched baseline."""

    pass_delta: float = Field(ge=-1.0, le=1.0)
    cost_ratio: float = Field(ge=0.0)


class MetricsSnapshot(ContractModel):
    """The complete metrics strip emitted for one generation."""

    generation: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    ci: ConfidenceInterval
    cost_usd: float = Field(ge=0.0)
    latency_s_mean: float = Field(ge=0.0)
    latency_s_p50: float = Field(ge=0.0)
    latency_s_p90: float = Field(ge=0.0)
    tokens: int = Field(ge=0)
    tool_calls_per_case: float = Field(ge=0.0)
    reliability: float = Field(ge=0.0, le=1.0)
    reliability_pass3: float | None = Field(default=None, ge=0.0, le=1.0)
    speed: float = Field(ge=0.0)
    structural_fidelity: float = Field(ge=0.0, le=1.0)
    vs_baseline: BaselineComparison


class SourceSpec(ContractModel):
    """Provenance for a committed or generated task pack."""

    kind: str
    repo: str
    file: str
    license: str


class Task(ContractModel):
    """The YAML manifest describing a task pack."""

    name: str
    domain: str
    goal: str
    answer_format: str
    tools: list[str]
    checker: CheckerName
    examples: int = Field(ge=0)
    memory: str = ""
    source: SourceSpec


TaskPack = Task


class GenerationState(BaseModel):
    """Reducer-owned snapshot for one architecture generation."""

    model_config = ConfigDict(extra="forbid")

    generation: int = Field(ge=0)
    architecture: dict[str, Any] | None = None
    executions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ablation: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    metrics: MetricsSnapshot | None = None
    reliability: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    mutation: dict[str, Any] | None = None
    revert: dict[str, Any] | None = None
    reverted: bool = False


class State(BaseModel):
    """Full run snapshot derived from the append-only event stream."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    last_seq: int = Field(default=-1, ge=-1)
    run_name: str = ""
    memory_ns: str = ""
    lessons_loaded: list[Lesson] = Field(default_factory=list)
    lessons: list[Lesson] = Field(default_factory=list)
    task: dict[str, Any] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    generations: dict[str, GenerationState] = Field(default_factory=dict)
    current_generation: int | None = Field(default=None, ge=0)
    best_generation: int | None = Field(default=None, ge=0)
    completed: bool = False
    summary: dict[str, Any] = Field(default_factory=dict)
    diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    mutations: list[dict[str, Any]] = Field(default_factory=list)
    logs: list[dict[str, Any]] = Field(default_factory=list)


class Event(ContractModel):
    """One line in ``events.jsonl``.

    The envelope is typed here; per-event data is validated against the versioned
    JSON Schema by the store layer before anything is written or consumed.
    """

    ts: datetime
    run_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    type: EventType
    data: dict[str, Any]

    @field_validator("ts", mode="before")
    @classmethod
    def timestamp_must_be_iso(cls, value: Any) -> Any:
        """Require an ISO timestamp input rather than a numeric datetime."""

        if isinstance(value, datetime):
            return value
        if not isinstance(value, str) or ("T" not in value and "t" not in value):
            raise ValueError("ts must be an ISO 8601 datetime string")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError as exc:
            raise ValueError("ts must be an ISO 8601 datetime string") from exc
        return value

    @field_validator("ts")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        """Require RFC 3339 timestamps with an explicit UTC offset."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must include a timezone offset")
        return value


__all__ = [
    "Architecture",
    "Case",
    "CaseResult",
    "CheckerName",
    "ControlMode",
    "ConfidenceInterval",
    "BaselineComparison",
    "Event",
    "EventType",
    "GenerationState",
    "Justification",
    "Lesson",
    "MemoryPolicy",
    "Role",
    "RoleTrace",
    "RunResult",
    "MetricsSnapshot",
    "SourceSpec",
    "State",
    "Task",
    "TaskPack",
    "ToolSpec",
    "Verdict",
]
