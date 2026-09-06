"""Build and emit the per-generation ``metrics.snapshot`` event (`03 §7`)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from occam.core.models import BaselineComparison, MetricsSnapshot, RunResult
from occam.engine.emit import EventSink
from occam.metrics.aggregate import (
    latency_stats,
    pass_rate,
    pass_rate_ci,
    tool_calls_per_case,
    total_cost_usd,
    total_tokens,
)
from occam.metrics.stats import clamp

# Used when a generation has no cost-matched baseline yet. The loop always runs
# `baseline` before `metrics.snapshot` (`01 §4.7`), so this only shows up in
# partial runs; a neutral ratio is honest there, a zero would read as "free".
NO_BASELINE = BaselineComparison(pass_delta=0.0, cost_ratio=1.0)


class BaselineSummary(BaseModel):
    """The two baseline numbers the metrics strip compares against (`03 §6`)."""

    model_config = ConfigDict(extra="forbid")

    pass_rate: float = Field(ge=0.0, le=1.0)
    cost_usd: float = Field(ge=0.0)


def compare_to_baseline(
    ours_pass_rate: float,
    ours_cost_usd: float,
    baseline: BaselineSummary | None,
) -> BaselineComparison:
    """``pass_delta = ours - baseline``, ``cost_ratio = ours / baseline``."""

    if baseline is None:
        return NO_BASELINE.model_copy()
    ratio = ours_cost_usd / baseline.cost_usd if baseline.cost_usd > 0 else 0.0
    return BaselineComparison(
        pass_delta=clamp(ours_pass_rate - baseline.pass_rate, -1.0, 1.0),
        cost_ratio=max(0.0, ratio),
    )


def build_snapshot(
    *,
    generation: int,
    full: RunResult,
    noise_rate: float,
    structural_fidelity: float,
    baseline: BaselineSummary | None = None,
    reliability_pass3: float | None = None,
) -> MetricsSnapshot:
    """Assemble the metrics strip for one generation.

    ``full`` is the generation's full run over all eval cases. ``noise_rate``
    comes from the ablation noise floor (`03 §4.1`); ``reliability`` is its
    complement — whether the system gives the same answer twice.
    """

    rate = pass_rate(full)
    latency = latency_stats(full)
    cost = total_cost_usd(full)
    return MetricsSnapshot(
        generation=generation,
        pass_rate=clamp(rate),
        ci=pass_rate_ci(full),
        cost_usd=cost,
        latency_s_mean=latency.mean,
        latency_s_p50=latency.p50,
        latency_s_p90=latency.p90,
        tokens=total_tokens(full),
        tool_calls_per_case=tool_calls_per_case(full),
        reliability=clamp(1.0 - noise_rate),
        reliability_pass3=None if reliability_pass3 is None else clamp(reliability_pass3),
        speed=(1.0 / latency.mean) if latency.mean > 0 else 0.0,
        structural_fidelity=clamp(structural_fidelity),
        vs_baseline=compare_to_baseline(rate, cost, baseline),
    )


def snapshot_event_data(snapshot: MetricsSnapshot) -> dict:
    """Serialize a snapshot into the ``metrics.snapshot`` event payload."""

    return snapshot.model_dump(mode="json", exclude_none=False)


def emit_snapshot(sink: EventSink, snapshot: MetricsSnapshot) -> MetricsSnapshot:
    """Emit ``metrics.snapshot`` and return the snapshot for the caller."""

    sink("metrics.snapshot", snapshot_event_data(snapshot))
    return snapshot


__all__ = [
    "NO_BASELINE",
    "BaselineSummary",
    "build_snapshot",
    "compare_to_baseline",
    "emit_snapshot",
    "snapshot_event_data",
]
