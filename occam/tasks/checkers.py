"""Task-pack graders.

``fx_total`` grades the answer defined by ``prd/02-DATA-AND-TASKS.md`` §1.2: pass
iff ``|answer - expected.total_inr| <= max(5.00, 0.001 * |expected.total_inr|)``,
with per-invoice correctness recorded in ``sub_results`` at ₹1 or 0.1%.  The
answer itself is the last fenced JSON block (`02 §1.2`).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

TOTAL_ABSOLUTE_TOLERANCE = 5.0
TOTAL_RELATIVE_TOLERANCE = 0.001
INVOICE_ABSOLUTE_TOLERANCE = 1.0
INVOICE_RELATIVE_TOLERANCE = 0.001


@dataclass(frozen=True)
class GradeResult(Mapping[str, Any]):
    """A checker result usable as an object or a small result mapping."""

    passed: bool
    sub_results: dict[str, bool] = field(default_factory=dict)
    answer_total: float | None = None
    expected_total: float | None = None
    total_tolerance: float | None = None
    error: str | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "passed":
            return self.passed
        if key == "sub_results":
            return self.sub_results
        if key == "answer_total":
            return self.answer_total
        if key == "expected_total":
            return self.expected_total
        if key == "total_tolerance":
            return self.total_tolerance
        if key == "error":
            return self.error
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "passed",
                "sub_results",
                "answer_total",
                "expected_total",
                "total_tolerance",
                "error",
            )
        )

    def __len__(self) -> int:
        return 6

    def __bool__(self) -> bool:
        return self.passed


def fx_total(answer: str | Mapping[str, Any], expected: Mapping[str, Any]) -> GradeResult:
    """Grade an answer against the FX total and per-invoice tolerances (`02 §1.2`).

    The contract is the last fenced JSON block, and that is what the packs ask
    for.  When no fence is present the last line that is a JSON object is read
    instead — defensive salvage for a model that drops the fence, not part of
    the contract.  A malformed answer is a normal failed grade so a run can
    continue and record the failure.
    """

    # A malformed *pack* must fail the case, never raise: the grader is on the
    # hot path of a run and may not take it down.
    try:
        expected_total = _number(expected.get("total_inr"))
    except (TypeError, ValueError) as exc:
        return GradeResult(False, error=f"expected total_inr is unusable: {exc}")
    expected_per_invoice = expected.get("per_invoice", {})
    if not isinstance(expected_per_invoice, Mapping):
        return GradeResult(False, error="expected per_invoice is not an object")
    invoice_ids = [str(invoice_id) for invoice_id in expected_per_invoice]
    try:
        candidate = _candidate_json(answer)
        answer_total = _number(candidate.get("total_inr"))
        answer_per_invoice = candidate.get("per_invoice", {})
        if not isinstance(answer_per_invoice, Mapping):
            raise ValueError("per_invoice is not an object")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return GradeResult(
            passed=False,
            sub_results={invoice_id: False for invoice_id in invoice_ids},
            expected_total=expected_total,
            error=str(exc),
        )

    total_tolerance = max(
        TOTAL_ABSOLUTE_TOLERANCE,
        TOTAL_RELATIVE_TOLERANCE * abs(expected_total),
    )
    total_passed = abs(answer_total - expected_total) <= total_tolerance
    sub_results: dict[str, bool] = {}
    for invoice_id, expected_value in expected_per_invoice.items():
        candidate_value = answer_per_invoice.get(invoice_id)
        try:
            expected_number = _number(expected_value)
            candidate_number = _number(candidate_value)
        except (TypeError, ValueError):
            sub_results[str(invoice_id)] = False
            continue
        tolerance = max(
            INVOICE_ABSOLUTE_TOLERANCE,
            INVOICE_RELATIVE_TOLERANCE * abs(expected_number),
        )
        sub_results[str(invoice_id)] = abs(candidate_number - expected_number) <= tolerance

    return GradeResult(
        passed=total_passed,
        sub_results=sub_results,
        answer_total=answer_total,
        expected_total=expected_total,
        total_tolerance=total_tolerance,
    )


def grade_fx_total(answer: str | Mapping[str, Any], expected: Mapping[str, Any]) -> GradeResult:
    """Descriptive alias for callers that prefer a verb phrase."""

    return fx_total(answer, expected)


def check_fx_total(answer: str | Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Return only the pass/fail bit for a lightweight checker interface."""

    return fx_total(answer, expected).passed


def _candidate_json(answer: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(answer, Mapping):
        candidate: Any = answer
    elif isinstance(answer, str):
        blocks = re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        encoded = blocks[-1] if blocks else _last_json_line(answer)
        candidate = json.loads(encoded)
    else:
        raise TypeError("answer must be a mapping or JSON text")
    if not isinstance(candidate, Mapping):
        raise ValueError("answer JSON is not an object")
    return candidate


def _last_json_line(answer: str) -> str:
    """Return the last line that is a JSON object, else the whole answer."""

    for line in reversed(answer.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return answer.strip()


def _number(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("value is not numeric")
    if isinstance(value, str):
        value = value.replace(",", "")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError("value is not finite")
    return number


__all__ = [
    "GradeResult",
    "check_fx_total",
    "fx_total",
    "grade_fx_total",
]
