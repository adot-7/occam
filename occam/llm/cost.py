"""Client-side token cost accounting."""

from __future__ import annotations

from dataclasses import dataclass

from occam.llm.config import ModelConfig


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Both the nominal bill and the displayed equivalent cost."""

    cost_usd: float
    billed_cost_usd: float
    label: str
    approximate: bool = False


def calculate_cost(
    config: ModelConfig,
    tokens_in: int,
    tokens_out: int,
    *,
    approximate: bool = False,
) -> CostBreakdown:
    """Calculate cost from usage counts, using grant-equivalent rates visibly."""

    nominal = (tokens_in * config.in_per_m + tokens_out * config.out_per_m) / 1_000_000
    displayed = (
        tokens_in * config.display_rate("input") + tokens_out * config.display_rate("output")
    ) / 1_000_000
    if config.is_granted:
        label = "list-rate-equivalent"
    else:
        label = "approximate" if approximate else "metered"
    return CostBreakdown(
        cost_usd=displayed,
        billed_cost_usd=nominal,
        label=label,
        approximate=approximate,
    )


__all__ = ["CostBreakdown", "calculate_cost"]
