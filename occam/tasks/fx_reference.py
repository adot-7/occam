"""Deterministic FX revaluation reference implementation.

The calculation below is intentionally a direct transcription of
``prd/02-DATA-AND-TASKS.md`` §1.2.  It does not infer dates or currencies: the
rate returned by ``fx_rate`` is used as supplied by the tool.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from occam.tools.fx import FXClient, fx_rate

REPORTING_CURRENCY = "INR"
PAISA = Decimal("0.01")

RateLookup = Callable[[str, str, str], Mapping[str, Any] | float | int | Decimal]


def compute_reference(
    valuation_date: str,
    invoices: Iterable[Mapping[str, Any]],
    *,
    client: FXClient | None = None,
    rate_lookup: RateLookup | None = None,
    reporting_currency: str = REPORTING_CURRENCY,
) -> dict[str, Any]:
    """Compute total and per-invoice FX gain/loss using the v3 formula.

    ``client`` is normally the shared cache-backed ``FXClient``.  ``rate_lookup``
    is provided for hand-computed tests and remains intentionally compatible
    with the three-argument ``fx_rate(date, base, symbol)`` tool shape.
    """

    if client is not None and rate_lookup is not None:
        raise ValueError("pass either client or rate_lookup, not both")
    if client is not None:
        lookup: RateLookup = client.fx_rate
    elif rate_lookup is not None:
        lookup = rate_lookup
    else:
        lookup = fx_rate

    reporting = reporting_currency.strip().upper()
    total = Decimal("0")
    per_invoice: dict[str, float] = {}
    for invoice in invoices:
        invoice_id = _required_text(invoice, "id")
        currency = _required_text(invoice, "currency").upper()
        amount = _decimal(invoice.get("amount"), "amount")
        issue_date = _required_text(invoice, "issue_date")

        booked = amount * _rate(lookup, issue_date, currency, reporting)
        status = _required_text(invoice, "status").upper()
        if status == "OPEN":
            value = amount * _rate(lookup, valuation_date, currency, reporting)
        elif status == "SETTLED":
            settlement_date = _required_text(invoice, "settlement_date")
            received_currency = _required_text(
                invoice,
                "received_currency",
                aliases=("settled_currency", "currency"),
            ).upper()
            received_amount = _decimal(
                invoice.get("received_amount", invoice.get("received")),
                "received_amount",
            )
            bank_fee = _decimal(
                invoice.get("bank_fee_inr", invoice.get("bank_fee", 0)),
                "bank_fee_inr",
            )
            value = (
                received_amount
                * _rate(
                    lookup,
                    settlement_date,
                    received_currency,
                    reporting,
                )
                + bank_fee
            )
        else:
            raise ValueError(f"invoice {invoice_id} has unsupported status {status!r}")

        gain = value - booked
        total += gain
        per_invoice[invoice_id] = _as_number(gain)

    return {"total_inr": _as_number(total), "per_invoice": per_invoice}


def reference_case(
    case: Mapping[str, Any],
    *,
    client: FXClient | None = None,
    rate_lookup: RateLookup | None = None,
) -> dict[str, Any]:
    """Compute a generated case from structured fields or its ledger text."""

    valuation_date = case.get("valuation_date")
    invoices = case.get("invoices")
    if not isinstance(valuation_date, str) or not isinstance(invoices, Iterable):
        text = case.get("input")
        if not isinstance(text, str):
            raise ValueError("case must contain valuation_date/invoices or input")
        valuation_date, invoices = parse_case_input(text)
    return compute_reference(
        valuation_date,
        invoices,
        client=client,
        rate_lookup=rate_lookup,
    )


def parse_case_input(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse the canonical human-readable ledger text emitted by the generator."""

    valuation_match = re.search(r"Valuation date:\s*(\d{4}-\d{2}-\d{2})", text)
    if valuation_match is None:
        raise ValueError("case input has no valuation date")

    invoices: list[dict[str, Any]] = []
    invoice_pattern = re.compile(
        r"^\s*\d+\.\s+(?P<id>\S+)\s+·\s+.+?\s+·\s+"
        r"(?P<currency>[A-Z]{3})\s+(?P<amount>[\d,]+(?:\.\d+)?)\s+·\s+"
        r"issued\s+(?P<issue_date>\d{4}-\d{2}-\d{2})\s+·\s+(?P<tail>.+)$"
    )
    settled_pattern = re.compile(
        r"SETTLED\s+(?P<settlement_date>\d{4}-\d{2}-\d{2}),\s+"
        r"received\s+(?P<received_currency>[A-Z]{3})\s+"
        r"(?P<received_amount>[\d,]+(?:\.\d+)?)"
        r"(?:,\s+net of bank fee INR\s+(?P<bank_fee>[\d,]+(?:\.\d+)?))?$"
    )
    for line in text.splitlines():
        match = invoice_pattern.match(line)
        if match is None:
            continue
        fields = match.groupdict()
        tail = fields.pop("tail")
        if tail == "OPEN":
            invoice = {
                **fields,
                "amount": _parse_number(fields["amount"]),
                "status": "OPEN",
            }
        else:
            settled = settled_pattern.fullmatch(tail)
            if settled is None:
                raise ValueError(f"unrecognised invoice status: {tail!r}")
            invoice = {
                **fields,
                "amount": _parse_number(fields["amount"]),
                "status": "SETTLED",
                "settlement_date": settled["settlement_date"],
                "received_currency": settled["received_currency"],
                "received_amount": _parse_number(settled["received_amount"]),
                "bank_fee_inr": _parse_number(settled["bank_fee"] or "0"),
            }
        invoices.append(invoice)

    if not invoices:
        raise ValueError("case input has no parseable invoices")
    return valuation_match.group(1), invoices


def _rate(lookup: RateLookup, date: str, base: str, symbol: str) -> Decimal:
    response = lookup(date, base, symbol)
    if isinstance(response, Mapping):
        if "rate" not in response:
            raise ValueError("rate lookup response has no rate")
        response = response["rate"]
    return _decimal(response, "rate")


def _required_text(
    invoice: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    value = invoice.get(name)
    if value is None:
        for alias in aliases:
            value = invoice.get(alias)
            if value is not None:
                break
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invoice is missing {name}")
    return value.strip()


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"invoice is missing {field}")
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invoice has invalid {field}: {value!r}") from exc


def _parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def _as_number(value: Decimal) -> float:
    rounded = value.quantize(PAISA, rounding=ROUND_HALF_UP)
    if rounded == 0:
        return 0.0
    return float(rounded)


__all__ = [
    "PAISA",
    "REPORTING_CURRENCY",
    "compute_reference",
    "parse_case_input",
    "reference_case",
]
