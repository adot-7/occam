#!/usr/bin/env python3
"""Report how visible each learnable lesson is at the grader's tolerance.

A lesson is only learnable if an agent that never learns it *fails* the case.
The grader passes when ``|answer - expected| <= max(5, 0.001 * |expected|)``
(`02 §1.2`), so for every lesson we compute the total a systematically wrong
agent would report and divide the resulting error by that tolerance.  A margin
below 1.0x means the mistake is invisible in the pass rate and the lesson is
silently dead; `02 §1.3` lists the failure modes this models.

The error models, one deviation from `02 §1.2` each:

L1  every lookup the API resolved backward (a weekend or an ECB holiday) is
    taken from the next business day instead — the "wrong side of the gap"
    mistake an agent makes when it indexes ``fx_series`` by the requested date.
D2  a cross-currency settlement is converted at the *invoice* currency's rate
    rather than the received currency's.
D3  a settled invoice is revalued at the valuation date like an open one, so
    the realised settlement (and its bank fee) is discarded.

D1 is not modelled here: its error is exactly the bank fee, a literal in the
case text, and ``gen_fx_cases`` already asserts that margin at generation time.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from occam.tasks.fx_reference import parse_case_input  # noqa: E402
from occam.tools.fx import FXClient  # noqa: E402

REPORTING = "INR"
LESSONS = ("L1", "D2", "D3")
# The observed floor is 2.99x (L1, fxa_015).  Two is a guard against silent
# regression, not a target: the report prints the real minimum every run.
MINIMUM_MARGIN = 2.0


class _Rates:
    """Cache-backed rate lookups plus the next-business-day rate for L1."""

    def __init__(self, client: FXClient) -> None:
        self._client = client
        self._next_business_day: dict[tuple[str, str], Decimal | None] = {}

    def rate(self, day: str, currency: str) -> Decimal:
        return Decimal(str(self._client.fx_rate(day, currency, REPORTING)["rate"]))

    def resolved_date(self, day: str, currency: str) -> str:
        return str(self._client.fx_rate(day, currency, REPORTING)["rate_date"])

    def naive_rate(self, day: str, currency: str) -> Decimal:
        """Return the next business day's rate, or the correct one if no gap."""

        resolved = self.resolved_date(day, currency)
        if resolved == day:
            return self.rate(day, currency)
        key = (day, currency)
        if key not in self._next_business_day:
            end = (date.fromisoformat(day) + timedelta(days=12)).isoformat()
            series = self._client.fx_series(day, end, currency, REPORTING)["rates"]
            later = sorted(rate_date for rate_date in series if rate_date > resolved)
            self._next_business_day[key] = Decimal(str(series[later[0]])) if later else None
        naive = self._next_business_day[key]
        return naive if naive is not None else self.rate(day, currency)


def _total(invoices: list[dict[str, Any]], valuation: str, rates: _Rates, mode: str) -> Decimal:
    total = Decimal(0)

    def lookup(day: str, currency: str) -> Decimal:
        return rates.naive_rate(day, currency) if mode == "L1" else rates.rate(day, currency)

    for invoice in invoices:
        amount = Decimal(str(invoice["amount"]))
        currency = invoice["currency"]
        booked = amount * lookup(invoice["issue_date"], currency)
        if invoice["status"] == "OPEN" or mode == "D3":
            value = amount * lookup(valuation, currency)
        else:
            received = Decimal(str(invoice["received_amount"]))
            received_currency = invoice["received_currency"]
            settled_in = currency if mode == "D2" else received_currency
            value = received * lookup(invoice["settlement_date"], settled_in)
            value += Decimal(str(invoice["bank_fee_inr"]))
        total += value - booked
    return total


def _applies(invoices: list[dict[str, Any]], valuation: str, rates: _Rates, mode: str) -> bool:
    if mode == "D2":
        return any(
            invoice["status"] == "SETTLED" and invoice["received_currency"] != invoice["currency"]
            for invoice in invoices
        )
    if mode == "D3":
        return any(invoice["status"] == "SETTLED" for invoice in invoices)
    for invoice in invoices:
        days = [invoice["issue_date"]]
        days.append(invoice["settlement_date"] if invoice["status"] == "SETTLED" else valuation)
        if any(rates.resolved_date(day, invoice["currency"]) != day for day in days):
            return True
    return False


def pack_margins(pack_dir: str | Path, client: FXClient) -> list[dict[str, Any]]:
    """Return each case's error/tolerance ratio for every lesson that applies."""

    rates = _Rates(client)
    rows: list[dict[str, Any]] = []
    lines = Path(pack_dir).joinpath("cases.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        case = json.loads(line)
        valuation, invoices = parse_case_input(case["input"])
        expected = Decimal(str(case["expected"]["total_inr"]))
        tolerance = max(Decimal("5"), Decimal("0.001") * abs(expected))
        row: dict[str, Any] = {"id": case["id"], "tolerance": float(tolerance)}
        for mode in LESSONS:
            if not _applies(invoices, valuation, rates, mode):
                row[mode] = None
                continue
            error = abs(_total(invoices, valuation, rates, mode) - expected)
            row[mode] = float(error / tolerance)
        rows.append(row)
    return rows


def minimum_margins(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return the tightest margin per lesson, or None when no case carries it."""

    minimums: dict[str, float | None] = {}
    for mode in LESSONS:
        margins = [row[mode] for row in rows if row[mode] is not None]
        minimums[mode] = min(margins) if margins else None
    return minimums


def _print_pack(pack_dir: Path, rows: list[dict[str, Any]]) -> None:
    print(f"=== {pack_dir.name} ===")
    for row in rows:
        cells = " ".join(
            f"{mode}={'    n/a ' if row[mode] is None else format(row[mode], '8.2f') + 'x'}"
            for mode in LESSONS
        )
        print(f"  {row['id']}  tolerance={row['tolerance']:8.2f}  {cells}")
    for mode, margin in minimum_margins(rows).items():
        carried = sum(1 for row in rows if row[mode] is not None)
        if margin is None:
            print(f"  -> {mode}: no case carries this lesson")
            continue
        verdict = "OK" if margin >= MINIMUM_MARGIN else "TOO TIGHT"
        below = sum(1 for row in rows if row[mode] is not None and row[mode] < 1.0)
        print(
            f"  -> {mode}: {carried} cases, tightest margin {margin:.2f}x, "
            f"{below} invisible at the grader tolerance [{verdict}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", nargs="*", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "fx_cache")
    args = parser.parse_args()
    packs = args.packs or [ROOT / "tasks" / "fx_recon_a", ROOT / "tasks" / "fx_recon_b"]

    with FXClient(cache_dir=args.cache_dir) as client:
        for pack in packs:
            _print_pack(pack, pack_margins(pack, client))
        network = sum(1 for call in client.calls if not call.cached)
    print(f"network calls: {network}")


if __name__ == "__main__":
    main()
