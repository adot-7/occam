"""DAG executor for an :class:`Architecture` over a set of evaluation cases.

Implements ``prd/01-ARCHITECTURE.md`` §4.2.  The rules that are load-bearing
elsewhere in the system, and are therefore enforced here rather than in a
prompt:

* Roles are topologically sorted; roles with no mutual dependency run
  concurrently, bounded by a per-model semaphore derived from ``models.yaml``
  rate limits (the client's token bucket still enforces the rpm ceiling).
* ``control="deterministic"`` routes an upstream role's output into downstream
  prompts **by ``output_key`` only** — no model decides the wiring, and a role
  can only see the keys it declared in ``inputs``.  ``control="llm"`` widens
  every role's view to the whole context; ``03``'s finding is that this is a
  bad trade, so it is supported but discouraged.
* A knocked-out role is removed from the DAG and its ``output_key`` renders as
  ``[no input from <role name>]``.  Upstream role prompts are byte-identical to
  the full run, so they are LLM cache hits; only descendants recompute.
* Every role keeps a :class:`RoleTrace` holding **raw tool responses**.  That is
  what lets ``diagnose.py`` see ``requested_date`` differing from ``rate_date``,
  so traces are persisted to ``generations/gNNN/results.jsonl`` verbatim.
* Cost is accounted **per role**; ablation's ``cost_share`` reads the displayed
  equivalent, while each trace also keeps nominal billed cost and its label.
* Any LLM failure that survives the client's retries fails that one case with
  the error recorded in its trace.  A run is never crashed by a single case.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from occam.core.models import (
    Architecture,
    Case,
    CaseResult,
    Role,
    RoleTrace,
    RunResult,
    ToolSpec,
)
from occam.llm.client import LLMClient, LLMError
from occam.llm.config import ConfigurationError
from occam.llm.tracing import is_enabled, set_span_attributes, span, trace_context
from occam.tools.accounting import STATUS_ERROR, STATUS_OK, ToolCall
from occam.tools.registry import ToolBinding, ToolRegistry

#: ``Role.inputs`` entry standing for the raw case text rather than a role id.
TASK_INPUT_KEY = "task"

#: Default number of cases evaluated concurrently.
DEFAULT_CASE_CONCURRENCY = 8

#: Upper bound on in-flight requests for one model lane.
MAX_MODEL_CONCURRENCY = 16

#: Concurrency target: roughly this many seconds of rpm budget in flight.
INFLIGHT_SECONDS = 4.0

#: Fallback lane concurrency when a model declares no rpm.
DEFAULT_MODEL_CONCURRENCY = 4

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VARIANT_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_EMPTY_USER_MESSAGE = "Produce your output now."


class ExecutorError(RuntimeError):
    """Base class for executor-level failures."""


class ArchitectureError(ExecutorError):
    """The architecture cannot be executed as written."""


class CycleError(ArchitectureError):
    """The role graph is not a DAG."""


def sentinel_for(role: Role) -> str:
    """Return the knock-out marker downstream prompts render for ``role``."""

    return f"[no input from {role.name}]"


def normalize_registry(registry: Any) -> dict[str, ToolBinding]:
    """Accept the real WP-04 bindings and small test doubles.

    ``ToolRegistry.bindings()`` returns objects with ``spec`` and ``call``
    attributes.  The tuple and callable forms remain useful for isolated unit
    tests, but the real binding is preserved rather than mistaken for its
    callable wrapper.
    """

    if registry is None:
        return {}
    if isinstance(registry, ToolRegistry):
        registry = registry.bindings()
    if not isinstance(registry, Mapping):
        if hasattr(registry, "items"):
            registry = dict(registry.items())
        else:
            raise ArchitectureError("tool registry must be a mapping of name -> tool")
    bindings: dict[str, ToolBinding] = {}
    for name, entry in registry.items():
        bindings[str(name)] = _binding(str(name), entry)
    return bindings


def _binding(name: str, entry: Any) -> ToolBinding:
    if isinstance(entry, ToolBinding):
        return entry
    if isinstance(entry, tuple) and len(entry) == 2:
        spec, call = entry
    elif hasattr(entry, "spec") and hasattr(entry, "call"):
        spec, call = entry.spec, entry.call
    else:
        spec, call = getattr(entry, "spec", None), entry
    if spec is None:
        raise ArchitectureError(f"tool {name!r} has no ToolSpec")
    if not isinstance(spec, ToolSpec):
        spec = ToolSpec.model_validate(spec if isinstance(spec, Mapping) else _jsonable(spec))
    if not callable(call):
        raise ArchitectureError(f"tool {name!r} is not callable")
    return ToolBinding(spec=spec, call=call, _registry=getattr(entry, "_registry", None))


def role_index(architecture: Architecture) -> dict[str, Role]:
    """Map role id to role, rejecting duplicate ids."""

    index: dict[str, Role] = {}
    for role in architecture.roles:
        if role.id in index:
            raise ArchitectureError(f"duplicate role id {role.id!r}")
        index[role.id] = role
    return index


def validate_architecture(
    architecture: Architecture,
    *,
    tools: Mapping[str, ToolBinding] | None = None,
) -> dict[str, Role]:
    """Check ids, wiring, acyclicity, and tool bindings before spending money."""

    index = role_index(architecture)
    if not index:
        raise ArchitectureError("architecture has no roles")
    if architecture.final_role not in index:
        raise ArchitectureError(f"final_role {architecture.final_role!r} is not a role")
    output_keys: dict[str, str] = {}
    for role in architecture.roles:
        previous_role = output_keys.get(role.output_key)
        if previous_role is not None:
            raise ArchitectureError(
                f"duplicate output_key {role.output_key!r} on roles "
                f"{previous_role!r} and {role.id!r}"
            )
        output_keys[role.output_key] = role.id
        for source in role.inputs:
            if source != TASK_INPUT_KEY and source not in index:
                raise ArchitectureError(f"role {role.id!r} reads unknown input {source!r}")
        if role.id in role.inputs:
            raise CycleError(f"role {role.id!r} depends on itself")
        if tools is not None:
            missing = [name for name in role.tools if name not in tools]
            if missing:
                raise ArchitectureError(
                    f"role {role.id!r} binds tools not in the registry: {', '.join(missing)}"
                )
    topological_levels(architecture)
    return index


def topological_levels(
    architecture: Architecture,
    *,
    exclude: Iterable[str] = (),
) -> list[list[Role]]:
    """Group roles into dependency levels; a level runs concurrently.

    Excluded (knocked-out) roles are removed from the graph entirely, and edges
    into them are dropped: their consumers read a sentinel instead of waiting
    for a producer that will never run.
    """

    index = role_index(architecture)
    removed = {role_id for role_id in exclude if role_id in index}
    pending = {role_id: role for role_id, role in index.items() if role_id not in removed}
    dependencies = {
        role_id: {
            source
            for source in role.inputs
            if source != TASK_INPUT_KEY and source in pending and source != role_id
        }
        for role_id, role in pending.items()
    }
    levels: list[list[Role]] = []
    resolved: set[str] = set()
    while pending:
        ready = sorted(role_id for role_id, needs in dependencies.items() if needs <= resolved)
        ready = [role_id for role_id in ready if role_id not in resolved]
        if not ready:
            remaining = ", ".join(sorted(pending))
            raise CycleError(f"architecture role graph has a cycle among: {remaining}")
        levels.append([pending[role_id] for role_id in ready])
        for role_id in ready:
            resolved.add(role_id)
            pending.pop(role_id)
            dependencies.pop(role_id)
    return levels


def descendants(architecture: Architecture, role_id: str) -> set[str]:
    """Return every role transitively downstream of ``role_id``.

    Ablation uses this to reason about what must recompute; everything else is
    an LLM cache hit.
    """

    index = role_index(architecture)
    if role_id not in index:
        raise ArchitectureError(f"unknown role {role_id!r}")
    consumers: dict[str, set[str]] = {key: set() for key in index}
    for role in architecture.roles:
        for source in role.inputs:
            if source in consumers:
                consumers[source].add(role.id)
    found: set[str] = set()
    frontier = [role_id]
    while frontier:
        current = frontier.pop()
        for consumer in sorted(consumers[current]):
            if consumer not in found:
                found.add(consumer)
                frontier.append(consumer)
    found.discard(role_id)
    return found


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a pass rate; ``(0, 0)`` for an empty set."""

    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _jsonable(value: Any) -> Any:
    """Coerce a tool response to something ``json.dumps`` and pydantic accept."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    return str(value)


def _cost_label(labels: Sequence[str]) -> str:
    """Describe the cost basis of all completions contributing to one role.

    A role normally uses one model/rate basis for every turn. Cache hits are
    also labelled by the client, so a multi-turn trace that mixes a cache hit
    and a live completion must not pretend the whole trace used either basis
    exclusively.
    """

    unique: set[str] = set()
    for label in labels:
        if not label:
            continue
        if label.startswith("mixed (") and label.endswith(")"):
            unique.update(part.strip() for part in label[7:-1].split(",") if part.strip())
        else:
            unique.add(label)
    unique = sorted(unique)
    if not unique:
        return "unavailable"
    if len(unique) == 1:
        return unique[0]
    return "mixed (" + ", ".join(unique) + ")"


def _registry_owner(
    source: Any,
    bindings: Mapping[str, ToolBinding],
) -> ToolRegistry | None:
    """Recover a real registry from either it or its published bindings."""

    if isinstance(source, ToolRegistry):
        return source
    owners = [getattr(binding, "_registry", None) for binding in bindings.values()]
    owners = [owner for owner in owners if owner is not None]
    if not owners:
        return None
    first = owners[0]
    if len(owners) != len(bindings) or any(owner is not first for owner in owners[1:]):
        return None
    return first if isinstance(first, ToolRegistry) else None


def _record_from_registry(
    registry: ToolRegistry | None,
    before: int,
    name: str,
) -> ToolCall | None:
    """Return the one authoritative record emitted by a registry invocation."""

    if registry is None:
        return None
    calls = registry.log.calls
    if len(calls) <= before:
        return None
    record = calls[before]
    return record if record.name == name else None


def _response_bytes(response: Any) -> int:
    """Measure a fallback binding response using the registry's JSON shape."""

    if isinstance(response, str):
        return len(response.encode("utf-8"))
    return len(json.dumps(response, ensure_ascii=False, default=str).encode("utf-8"))


def _http_status(exc: BaseException) -> int | None:
    """Extract an HTTP status from a custom binding exception when available."""

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _grade(grader: Callable[..., Any] | None, answer: str, expected: Any) -> tuple[bool, dict]:
    """Normalise whatever the checker returns into ``(passed, sub_results)``."""

    if grader is None:
        return (False, {})
    outcome = grader(answer, expected)
    if isinstance(outcome, bool):
        return (outcome, {})
    passed = bool(
        outcome.get("passed", False)
        if isinstance(outcome, Mapping)
        else getattr(outcome, "passed", False)
    )
    raw = (
        outcome.get("sub_results", {})
        if isinstance(outcome, Mapping)
        else getattr(outcome, "sub_results", {})
    )
    sub_results = {str(key): bool(value) for key, value in dict(raw or {}).items()}
    return (passed, sub_results)


def resolve_grader(checker: str) -> Callable[..., Any]:
    """Look a task pack's checker up in WP-03's ``occam.tasks.checkers``."""

    try:
        from occam.tasks import checkers
    except ImportError as exc:  # pragma: no cover - depends on WP-03 landing
        raise ExecutorError("occam.tasks.checkers is unavailable") from exc
    grader = getattr(checkers, checker, None)
    if grader is None:
        raise ExecutorError(f"checker {checker!r} is not defined in occam.tasks.checkers")
    return grader


class Executor:
    """Runs one architecture over one case set, live or ablated."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        tools: Any = None,
        grader: Callable[..., Any] | None = None,
        writer: Any = None,
        run_dir: str | Path | None = None,
        run_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        case_concurrency: int = DEFAULT_CASE_CONCURRENCY,
        model_concurrency: int | None = None,
    ) -> None:
        if case_concurrency < 1:
            raise ValueError("case_concurrency must be at least 1")
        if model_concurrency is not None and model_concurrency < 1:
            raise ValueError("model_concurrency must be at least 1")
        self._llm = llm
        self.tools = normalize_registry(tools)
        self._tool_registry = _registry_owner(tools, self.tools)
        self.grader = grader
        self.writer = writer
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.run_name = run_name
        # temperature 0 is determinism hygiene (03 §4.5); the baseline samples
        # at 0.7 through its own path, not through here.
        self.temperature = temperature
        # max_tokens=None defers to models.yaml, which is >= 1024 by
        # construction: GLM-4.7-Flash spends its budget on hidden reasoning
        # first and returns empty content below that (OPEN-QUESTIONS, resolved).
        self.max_tokens = max_tokens
        self.case_concurrency = case_concurrency
        self._model_concurrency = model_concurrency
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._writer_lock: asyncio.Lock | None = None
        self.results_history: list[RunResult] = []

    @property
    def llm(self) -> LLMClient:
        """The shared completion client, built from ``models.yaml`` on demand."""

        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    # -- concurrency ----------------------------------------------------

    def model_concurrency(self, model_key: str) -> int:
        """In-flight cap for one model lane, derived from its ``rpm``."""

        if self._model_concurrency is not None:
            return self._model_concurrency
        config = getattr(self.llm, "configs", {}).get(model_key)
        rpm = getattr(config, "rpm", None)
        if not rpm:
            return DEFAULT_MODEL_CONCURRENCY
        return max(1, min(MAX_MODEL_CONCURRENCY, round(rpm / 60.0 * INFLIGHT_SECONDS)))

    def _semaphore(self, model_key: str) -> asyncio.Semaphore:
        # An asyncio.Semaphore binds to the loop that first awaits it, so an
        # executor reused across calls (full run, then each knockout) has to
        # rebuild its lanes whenever the loop changes.
        loop = asyncio.get_running_loop()
        if loop is not self._semaphore_loop:
            self._semaphores.clear()
            self._semaphore_loop = loop
        semaphore = self._semaphores.get(model_key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.model_concurrency(model_key))
            self._semaphores[model_key] = semaphore
        return semaphore

    # -- public entry points --------------------------------------------

    def execute(
        self,
        architecture: Architecture,
        cases: Sequence[Case],
        *,
        variant: str | None = None,
        ablate_role: str | None = None,
        use_cache: bool = True,
        generation: int = 0,
        grader: Callable[..., Any] | None = None,
    ) -> RunResult:
        """Synchronous wrapper around :meth:`execute_async`."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ExecutorError("execute() cannot be called from a running event loop")
        return asyncio.run(
            self.execute_async(
                architecture,
                cases,
                variant=variant,
                ablate_role=ablate_role,
                use_cache=use_cache,
                generation=generation,
                grader=grader,
            )
        )

    def run_variant(
        self,
        architecture: Architecture,
        cases: Sequence[Case],
        *,
        variant: str,
        ablate_role: str | None = None,
        use_cache: bool = True,
        generation: int = 0,
        grader: Callable[..., Any] | None = None,
    ) -> RunResult:
        """Run the narrow variant protocol consumed by :mod:`ablation`.

        ``execute`` is the executor's primary API; this named adapter keeps
        ablation independent of executor implementation details while carrying
        the generation and cache-bypass controls needed for event ordering and
        the fresh ``full_repeat`` noise-floor pass.
        """

        return self.execute(
            architecture,
            cases,
            variant=variant,
            ablate_role=ablate_role,
            use_cache=use_cache,
            generation=generation,
            grader=grader,
        )

    async def execute_async(
        self,
        architecture: Architecture,
        cases: Sequence[Case],
        *,
        variant: str | None = None,
        ablate_role: str | None = None,
        use_cache: bool = True,
        generation: int = 0,
        grader: Callable[..., Any] | None = None,
    ) -> RunResult:
        """Run every case and emit ``execution.started|case|completed``."""

        index = validate_architecture(architecture, tools=self.tools)
        if ablate_role is not None and ablate_role not in index:
            raise ArchitectureError(f"cannot ablate unknown role {ablate_role!r}")
        excluded = {ablate_role} if ablate_role else set()
        variant = variant or ("full" if not ablate_role else f"ablate:{ablate_role}")
        levels = topological_levels(architecture, exclude=excluded)
        active_grader = grader if grader is not None else self.grader
        cases = list(cases)

        self._writer_lock = asyncio.Lock()
        await self._emit(
            "execution.started",
            {"generation": generation, "variant": variant, "n_cases": len(cases)},
        )

        results: list[CaseResult | None] = [None] * len(cases)
        limiter = asyncio.Semaphore(self.case_concurrency)
        emitted = 0

        async def run_one(position: int, case: Case) -> int:
            async with limiter:
                results[position] = await self._run_case(
                    architecture,
                    index,
                    levels,
                    excluded,
                    case,
                    active_grader,
                    use_cache=use_cache,
                    generation=generation,
                    variant=variant,
                )
            return position

        pending = {
            asyncio.ensure_future(run_one(position, case)) for position, case in enumerate(cases)
        }
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            # Emit in case order regardless of completion order: the event log
            # is the contract and must be reproducible.
            while emitted < len(results) and results[emitted] is not None:
                await self._emit_case(generation, variant, results[emitted])
                emitted += 1

        final = [result for result in results if result is not None]
        passed = sum(1 for result in final if result.passed)
        pass_rate = passed / len(final) if final else 0.0
        cost = sum(result.cost_usd for result in final)
        latency_mean = (sum(result.latency_s for result in final) / len(final)) if final else 0.0
        tokens = sum(result.tokens_in + result.tokens_out for result in final)
        lo, hi = wilson_ci(passed, len(final))
        # Results contain the authoritative per-role traces, including raw
        # tool responses.  Make them durable before advertising completion so
        # a reader that reacts to the event can open the variant immediately.
        await asyncio.to_thread(self._write_results, generation, variant, final)
        await self._emit(
            "execution.completed",
            {
                "generation": generation,
                "variant": variant,
                "pass_rate": pass_rate,
                "cost_usd": cost,
                "latency_s_mean": latency_mean,
                "tokens": tokens,
                "ci": {"lo": lo, "hi": hi},
            },
        )
        run_result = RunResult(
            architecture_id=architecture.id,
            variant=variant,
            results=final,
            pass_rate=pass_rate,
            cost_usd=cost,
            latency_s_mean=latency_mean,
            tokens=tokens,
        )
        self.results_history.append(run_result)
        return run_result

    # -- case and role execution ----------------------------------------

    async def _run_case(
        self,
        architecture: Architecture,
        index: Mapping[str, Role],
        levels: Sequence[Sequence[Role]],
        excluded: set[str],
        case: Case,
        grader: Callable[..., Any] | None,
        use_cache: bool,
        *,
        generation: int = 0,
        variant: str = "full",
    ) -> CaseResult:
        if not is_enabled():
            return await self._run_case_impl(
                architecture,
                index,
                levels,
                excluded,
                case,
                grader,
                use_cache=use_cache,
                generation=generation,
                variant=variant,
                trace_attributes={},
            )
        trace_attributes: dict[str, Any] = {
            "generation": generation,
            "variant": variant,
            "case_id": case.id,
        }
        for name, value in (
            ("run_id", getattr(self.writer, "run_id", None)),
            ("run_name", self.run_name or getattr(self.writer, "run_name", None)),
        ):
            if value:
                trace_attributes[name] = value

        with span(f"case.{case.id}", kind="WORKFLOW", attributes=trace_attributes) as span_object:
            result = await self._run_case_impl(
                architecture,
                index,
                levels,
                excluded,
                case,
                grader,
                use_cache=use_cache,
                generation=generation,
                variant=variant,
                trace_attributes=trace_attributes,
            )
            set_span_attributes(
                span_object,
                {
                    "passed": result.passed,
                    "cost_usd": result.cost_usd,
                    "latency_s": result.latency_s,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                },
            )
            return result

    async def _run_case_impl(
        self,
        architecture: Architecture,
        index: Mapping[str, Role],
        levels: Sequence[Sequence[Role]],
        excluded: set[str],
        case: Case,
        grader: Callable[..., Any] | None,
        *,
        use_cache: bool,
        generation: int,
        variant: str,
        trace_attributes: Mapping[str, Any],
    ) -> CaseResult:
        started = time.perf_counter()
        context: dict[str, str] = {
            index[role_id].output_key: sentinel_for(index[role_id]) for role_id in sorted(excluded)
        }
        traces: dict[str, RoleTrace] = {}
        role_failed = False
        for level in levels:
            level_traces = await asyncio.gather(
                *(
                    self._run_role(
                        role,
                        case,
                        context,
                        index,
                        architecture.control,
                        use_cache=use_cache,
                        generation=generation,
                        variant=variant,
                        trace_attributes=trace_attributes,
                    )
                    for role in level
                )
            )
            for role, trace in zip(level, level_traces, strict=True):
                traces[role.id] = trace
                context[role.output_key] = trace.output
            if any(trace.error for trace in level_traces):
                # An upstream role failed after retries; downstream work would
                # only burn tokens on garbage.  Fail this case, keep the run.
                role_failed = True
                break

        final_role = index[architecture.final_role]
        answer = context.get(final_role.output_key, "").strip()
        passed, sub_results = False, {}
        if grader is not None:
            try:
                passed, sub_results = _grade(grader, answer, case.expected)
            except Exception:  # noqa: BLE001 - a checker must never crash a run
                passed, sub_results = False, {}
        if role_failed:
            passed = False
        return CaseResult(
            case_id=case.id,
            answer=answer,
            passed=passed,
            sub_results=sub_results,
            tokens_in=sum(trace.tokens_in for trace in traces.values()),
            tokens_out=sum(trace.tokens_out for trace in traces.values()),
            cost_usd=sum(trace.cost_usd for trace in traces.values()),
            latency_s=time.perf_counter() - started,
            per_role=traces,
        )

    async def _run_role(
        self,
        role: Role,
        case: Case,
        context: Mapping[str, str],
        index: Mapping[str, Role],
        control: str,
        *,
        use_cache: bool = True,
        generation: int = 0,
        variant: str = "full",
        trace_attributes: Mapping[str, Any] | None = None,
    ) -> RoleTrace:
        """Run one role's bounded tool-call loop and return its trace."""

        if not is_enabled():
            return await self._run_role_impl(
                role,
                case,
                context,
                index,
                control,
                use_cache=use_cache,
            )
        role_attributes = dict(trace_attributes or {})
        role_attributes.update(
            role_id=role.id,
            role_name=role.name,
            justification=role.justification,
            model=role.model,
            generation=generation,
            variant=variant,
            case_id=case.id,
        )
        with trace_context(role_attributes):
            return await self._run_role_impl(
                role,
                case,
                context,
                index,
                control,
                use_cache=use_cache,
                generation=generation,
                variant=variant,
                trace_attributes=role_attributes,
            )

    async def _run_role_impl(
        self,
        role: Role,
        case: Case,
        context: Mapping[str, str],
        index: Mapping[str, Role],
        control: str,
        *,
        use_cache: bool = True,
        generation: int = 0,
        variant: str = "full",
        trace_attributes: Mapping[str, Any] | None = None,
    ) -> RoleTrace:
        """Run one role's bounded tool-call loop without changing its context."""

        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        nested_traces: list[RoleTrace] = []
        nested_trace_guard = threading.Lock()
        role_registry = self._role_registry(
            role,
            case,
            context,
            index,
            control,
            loop,
            nested_traces,
            nested_trace_guard,
            use_cache=use_cache,
            generation=generation,
            variant=variant,
            trace_attributes=trace_attributes,
        )
        role_tools = (
            normalize_registry(role_registry.bindings(role.tools))
            if role_registry is not None
            else {name: self.tools[name] for name in role.tools}
        )
        system, user = self._render_messages(role, case, context, index, control)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        specs = [role_tools[name].spec.model_dump() for name in role.tools]
        tool_calls: list[dict[str, Any]] = []
        tokens_in = tokens_out = 0
        cost = 0.0
        billed_cost = 0.0
        cost_labels: list[str] = []
        cached = True
        output = ""
        error: str | None = None

        for turn in range(1, role.max_turns + 1):
            try:
                completion = await self._complete(
                    role.model, messages, specs or None, use_cache=use_cache
                )
            except (LLMError, ConfigurationError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            tokens_in += completion.tokens_in
            tokens_out += completion.tokens_out
            cost += completion.cost_usd
            billed_cost += completion.billed_cost_usd
            cost_labels.append(completion.cost_label)
            cached = cached and completion.cached
            text = (completion.text or "").strip()
            if not completion.tool_calls:
                output = text
                break
            output = text
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.text or "",
                    "tool_calls": completion.tool_calls,
                }
            )
            for call in completion.tool_calls:
                record, content = await self._invoke_tool(
                    role,
                    turn,
                    call,
                    role_tools,
                    role_registry,
                )
                tool_calls.append(record)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or f"{role.id}-{turn}"),
                        "content": content,
                    }
                )
            if turn == role.max_turns:
                error = f"max_turns ({role.max_turns}) reached before a final answer"

        with nested_trace_guard:
            branches = list(nested_traces)
        branch_tool_calls = [tool_call for branch in branches for tool_call in branch.tool_calls]
        branch_labels = [branch.cost_label for branch in branches]
        branch_cached = all(branch.cached for branch in branches)
        return RoleTrace(
            tokens_in=tokens_in + sum(branch.tokens_in for branch in branches),
            tokens_out=tokens_out + sum(branch.tokens_out for branch in branches),
            cost_usd=cost + sum(branch.cost_usd for branch in branches),
            billed_cost_usd=billed_cost + sum(branch.billed_cost_usd for branch in branches),
            cost_label=_cost_label([*cost_labels, *branch_labels]),
            # ``CaseResult.latency_s`` remains wall-clock latency.  A role
            # trace is cumulative accounting, so retain branch work as well
            # as the calling role's elapsed time.
            latency_s=(
                time.perf_counter() - started + sum(branch.latency_s for branch in branches)
            ),
            output=output,
            tool_calls=[*tool_calls, *branch_tool_calls],
            cached=cached and branch_cached and error is None,
            error=error,
        )

    def _role_registry(
        self,
        role: Role,
        case: Case,
        context: Mapping[str, str],
        index: Mapping[str, Role],
        control: str,
        loop: asyncio.AbstractEventLoop,
        nested_traces: list[RoleTrace],
        nested_trace_guard: threading.Lock,
        use_cache: bool,
        generation: int,
        variant: str,
        trace_attributes: Mapping[str, Any] | None,
    ) -> ToolRegistry | None:
        """Create the registry scoped to ``role`` and its prompt runner."""

        if self._tool_registry is None:
            return None

        def run_subtask(subtask: str) -> str:
            # ``fan_out`` is synchronous and runs this callback in its own
            # worker threads.  Submit the real async role runner back to the
            # executor loop; the parent loop remains free while the tool call
            # itself is offloaded.
            future = asyncio.run_coroutine_threadsafe(
                self._run_role(
                    role,
                    case.model_copy(update={"input": subtask}),
                    context,
                    index,
                    control,
                    use_cache=use_cache,
                    generation=generation,
                    variant=variant,
                    trace_attributes=trace_attributes,
                ),
                loop,
            )
            trace = future.result()
            with nested_trace_guard:
                nested_traces.append(trace)
            if trace.error:
                raise ExecutorError(trace.error)
            return trace.output

        return self._tool_registry.for_role(run_subtask)

    async def _complete(
        self,
        model_key: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        *,
        use_cache: bool,
    ) -> Any:
        payload = [dict(message) for message in messages]
        async with self._semaphore(model_key):
            return await asyncio.to_thread(
                self.llm.complete,
                model_key,
                payload,
                tools,
                None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                use_cache=use_cache,
            )

    async def _invoke_tool(
        self,
        role: Role,
        turn: int,
        call: Mapping[str, Any],
        bindings: Mapping[str, ToolBinding],
        role_registry: ToolRegistry | None,
    ) -> tuple[dict[str, Any], str]:
        """Run one native tool call, recording the response verbatim.

        A tool error is fed back to the model as its own result rather than
        failing the case: recovering from a bad argument is exactly the kind of
        behaviour the tool loop exists for.
        """

        function = dict(call.get("function") or {})
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments", "")
        started = time.perf_counter()
        error: str | None = None
        failure: BaseException | None = None
        response: Any = None
        arguments: dict[str, Any] = {}
        try:
            arguments = _parse_arguments(raw_arguments)
        except ValueError as exc:
            error = f"invalid tool arguments: {exc}"
        if error is None and name not in role.tools:
            error = f"tool {name!r} is not bound to role {role.id!r}"
        binding = bindings.get(name)
        if error is None and binding is None:
            error = f"tool {name!r} is not in the registry"
        before = len(role_registry.log.calls) if role_registry is not None else 0
        if error is None:
            try:
                # Registry handlers are synchronous (including HTTP and
                # subprocess tools).  Never let one hold the event loop while
                # independent roles are awaiting their own tools.
                result = await asyncio.to_thread(binding.call, **arguments)
                if inspect.isawaitable(result):
                    result = await result
                response = _jsonable(result)
            except Exception as exc:  # noqa: BLE001 - surfaced to the model
                failure = exc
                error = f"{type(exc).__name__}: {exc}"
        authoritative = _record_from_registry(role_registry, before, name)
        if authoritative is None or (error is not None and authoritative.error is None):
            authoritative = ToolCall(
                name=name,
                arguments=_jsonable(arguments),
                status=STATUS_ERROR if error is not None else STATUS_OK,
                latency_s=time.perf_counter() - started,
                bytes=0 if error is not None else _response_bytes(response),
                cached=False,
                response=None if error is not None else response,
                http_status=_http_status(failure) if failure is not None else None,
                error=error,
            )
        record = authoritative.as_dict()
        content = (
            authoritative.error
            if authoritative.error is not None
            else json.dumps(authoritative.response, ensure_ascii=False)
        )
        return record, content

    # -- prompt assembly --------------------------------------------------

    def _render_messages(
        self,
        role: Role,
        case: Case,
        context: Mapping[str, str],
        index: Mapping[str, Role],
        control: str,
    ) -> tuple[str, str]:
        """Build ``(system, user)`` for one role from its declared inputs.

        Under deterministic control a role sees exactly the keys it declared.
        A ``{output_key}`` placeholder in the system prompt is substituted in
        place and then omitted from the user message, so an architect can
        choose the layout without the content appearing twice.
        """

        visible = _visible_context(role, case, context, index, control)
        system, used = _render_template(role.system_prompt, visible)
        sections = [f"### {key}\n{value}" for key, value in visible.items() if key not in used]
        user = "\n\n".join(sections).strip() or _EMPTY_USER_MESSAGE
        return system, user

    # -- persistence -------------------------------------------------------

    async def _emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        if self.writer is None:
            return
        event = {"ts": _now_iso(), "type": event_type, "data": dict(data)}
        lock = self._writer_lock
        if lock is None:
            await asyncio.to_thread(self.writer.append, event)
            return
        async with lock:
            await asyncio.to_thread(self.writer.append, event)

    async def _emit_case(self, generation: int, variant: str, result: CaseResult) -> None:
        await self._emit(
            "execution.case",
            {
                "generation": generation,
                "variant": variant,
                "case_id": result.case_id,
                "passed": result.passed,
                "cost_usd": result.cost_usd,
                "latency_s": result.latency_s,
            },
        )

    def _write_results(
        self,
        generation: int,
        variant: str,
        results: Sequence[CaseResult],
    ) -> None:
        """Persist full ``CaseResult``s, including raw tool responses.

        The event schema deliberately keeps ``execution.case`` small, so this
        file is the only home for per-role traces — which ablation needs for
        ``cost_share`` and diagnose needs for tool-response evidence.
        """

        if self.run_dir is None:
            return
        directory = self.run_dir / "generations" / f"g{generation:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        name = "results.jsonl" if variant == "full" else f"results.{_slug(variant)}.jsonl"
        destination = directory / name
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=directory,
                prefix=f".{name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                for result in results:
                    handle.write(
                        json.dumps(
                            result.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
            _fsync_directory(directory)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _fsync_directory(directory: Path) -> None:
    """Flush the rename metadata where the platform permits directory fsync."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _slug(variant: str) -> str:
    return _VARIANT_SAFE.sub("_", variant).strip("_") or "variant"


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("tool arguments must decode to an object")
        return {str(key): value for key, value in parsed.items()}
    raise ValueError(f"unsupported tool argument type {type(raw).__name__}")


def _visible_context(
    role: Role,
    case: Case,
    context: Mapping[str, str],
    index: Mapping[str, Role],
    control: str,
) -> dict[str, str]:
    """Return the ``key -> text`` view one role is allowed to see.

    Deterministic control is the point of the product: a role reads exactly the
    ``output_key``s of the roles it declared, so context isolation is a
    structural property rather than a promise made in a prompt.
    """

    if control == "llm":
        # Discouraged (01 §4.2): the model, not the DAG, decides what matters,
        # so every key produced so far is exposed.
        visible = {TASK_INPUT_KEY: case.input}
        visible.update({key: context[key] for key in sorted(context)})
        return visible
    visible = {}
    if TASK_INPUT_KEY in role.inputs:
        visible[TASK_INPUT_KEY] = case.input
    for source in role.inputs:
        if source == TASK_INPUT_KEY:
            continue
        upstream = index.get(source)
        if upstream is None:
            raise ArchitectureError(f"role {role.id!r} reads unknown input {source!r}")
        visible[upstream.output_key] = context.get(upstream.output_key, "")
    return visible


def _render_template(text: str, values: Mapping[str, str]) -> tuple[str, set[str]]:
    """Substitute ``{key}`` for known keys only, leaving other braces intact."""

    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            used.add(key)
            return values[key]
        return match.group(0)

    return _PLACEHOLDER.sub(replace, text), used


__all__ = [
    "ArchitectureError",
    "CycleError",
    "DEFAULT_CASE_CONCURRENCY",
    "Executor",
    "ExecutorError",
    "TASK_INPUT_KEY",
    "ToolBinding",
    "descendants",
    "normalize_registry",
    "resolve_grader",
    "role_index",
    "sentinel_for",
    "topological_levels",
    "validate_architecture",
    "wilson_ci",
]
