"""Pass rate, cost, latency, calls/case, structural fidelity and the CIs.

`prd/03-ABLATION-AND-METRICS.md §7` is the definition of every number here.
"""

from occam.metrics.aggregate import (
    LatencyStats,
    latency_stats,
    pass_rate,
    pass_rate_ci,
    passed_count,
    role_cost_shares,
    role_cost_totals,
    role_token_totals,
    tool_calls_per_case,
    total_cost_usd,
    total_tokens,
)
from occam.metrics.snapshot import (
    BaselineSummary,
    build_snapshot,
    compare_to_baseline,
    emit_snapshot,
    snapshot_event_data,
)
from occam.metrics.stats import (
    DEFAULT_RESAMPLES,
    bootstrap_mean_ci,
    clamp,
    mean,
    percentile,
    seeded_rng,
    wilson_interval,
)

__all__ = [
    "DEFAULT_RESAMPLES",
    "BaselineSummary",
    "LatencyStats",
    "bootstrap_mean_ci",
    "build_snapshot",
    "clamp",
    "compare_to_baseline",
    "emit_snapshot",
    "latency_stats",
    "mean",
    "pass_rate",
    "pass_rate_ci",
    "passed_count",
    "percentile",
    "role_cost_shares",
    "role_cost_totals",
    "role_token_totals",
    "seeded_rng",
    "snapshot_event_data",
    "tool_calls_per_case",
    "total_cost_usd",
    "total_tokens",
    "wilson_interval",
]
