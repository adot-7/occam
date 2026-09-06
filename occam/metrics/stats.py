"""Estimators shared by ablation verdicts and the metrics strip.

Wilson intervals for pass rates (`03 §7`) and a percentile bootstrap over paired
case outcomes for influence (`03 §4.2`). No numpy: the samples are tiny and the
engine must stay dependency-light and deterministic.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence

from occam.core.models import ConfidenceInterval

Z_95 = 1.959963984540054
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0


def seeded_rng(seed: int, *parts: str) -> random.Random:
    """Return an RNG seeded reproducibly across processes and platforms.

    ``random.Random(str)`` is deterministic in CPython but hashing the parts
    explicitly keeps that guarantee independent of the interpreter.
    """

    material = ":".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value into ``[lo, hi]`` to absorb float drift before emission."""

    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""

    if not values:
        return 0.0
    return math.fsum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of ``values`` for ``q`` in ``[0, 1]``."""

    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, n: int, z: float = Z_95) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion (`03 §7`, `pass_rate`)."""

    if successes < 0 or n < 0 or successes > n:
        raise ValueError("successes must be within [0, n]")
    if n == 0:
        return ConfidenceInterval(lo=0.0, hi=1.0)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    return ConfidenceInterval(lo=clamp(centre - half_width), hi=clamp(centre + half_width))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
    label: str = "",
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean of ``values``.

    ``values`` are the *paired* per-case differences, so resampling them
    resamples cases — which is exactly the uncertainty ablation cares about
    (`03 §4.2`).
    """

    if not values:
        raise ValueError("bootstrap over an empty sample")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    point = mean(values)
    if len(set(values)) == 1:
        return ConfidenceInterval(lo=point, hi=point)
    rng = seeded_rng(seed, label)
    n = len(values)
    means = [mean(rng.choices(values, k=n)) for _ in range(resamples)]
    return ConfidenceInterval(
        lo=percentile(means, alpha / 2.0),
        hi=percentile(means, 1.0 - alpha / 2.0),
    )


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "Z_95",
    "bootstrap_mean_ci",
    "clamp",
    "mean",
    "percentile",
    "seeded_rng",
    "wilson_interval",
]
