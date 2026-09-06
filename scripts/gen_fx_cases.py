#!/usr/bin/env python3
"""Generate a deterministic, cached FX revaluation task pack."""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

# Make the documented ``python scripts/gen_fx_cases.py`` invocation work from
# a checkout without requiring an editable install first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from occam.core import Task  # noqa: E402
from occam.store.schema import validate_task  # noqa: E402
from occam.tasks.fx_reference import compute_reference  # noqa: E402
from occam.tools.fx import MAX_CONCURRENT_REQUESTS, FXClient  # noqa: E402

ISSUE_START = date(2026, 1, 5)
ISSUE_END = date(2026, 4, 20)
HOLIDAY_LOOKUP_DATES = (
    date(2026, 4, 3),
    date(2026, 4, 4),
    date(2026, 4, 5),
    date(2026, 4, 6),
)
CURRENCIES = ("EUR", "USD", "GBP", "JPY", "AUD", "SGD", "CHF", "CAD")
# Approximate INR mid-rates for the generated window, used only to keep a
# cross-currency settlement amount economically plausible.  The rates behind
# ``expected`` always come from the cached Frankfurter client, never from here.
APPROX_INR_RATES = {
    "AUD": 64.69,
    "CAD": 66.96,
    "CHF": 117.46,
    "EUR": 107.48,
    "GBP": 123.20,
    "JPY": 0.5833,
    "SGD": 71.90,
    "USD": 92.15,
}
BANK_FEES = (1500.0, 2400.0, 3200.0, 4500.0)
# A D1 miss must move the total by well more than the grader's tolerance, or an
# agent that never learns the rule still passes the fee cases.
MINIMUM_FEE_HEADROOM = 3.0
TRAILING_INSTRUCTION = (
    "Compute the total FX gain/(loss) in INR as of the valuation date, "
    "and the gain/(loss) per invoice."
)
CUSTOMERS = (
    "Acme GmbH",
    "Kyoto Labs",
    "Harbor Ltd",
    "Saffron Systems",
    "Northstar BV",
    "Orbit Works",
    "Maple Analytics",
    "Lighthouse Pty",
    "Bluebird AG",
    "Cedar Cloud",
)


def generate_pack(
    *,
    seed: int,
    n: int,
    out: str | Path,
    cache_dir: str | Path = "data/fx_cache",
) -> dict[str, Any]:
    """Generate and write one pack, returning its report data."""

    if n < 20:
        raise ValueError("n must be at least 20 to satisfy the v3 pack mix guarantees")
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_name = output_dir.name
    if not pack_name.startswith("fx_recon_"):
        raise ValueError("out directory must be named fx_recon_a or fx_recon_b")
    prefix = pack_name.removeprefix("fx_recon_")
    if not prefix:
        raise ValueError("pack directory has no case prefix")

    rng = random.Random(seed)
    generated = [_make_case(rng, prefix, index) for index in range(1, n + 1)]
    with FXClient(cache_dir=cache_dir, max_concurrency=MAX_CONCURRENT_REQUESTS) as client:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            expected = list(
                executor.map(
                    lambda item: compute_reference(
                        item["valuation_date"],
                        item["invoices"],
                        client=client,
                    ),
                    generated,
                )
            )

    cases: list[dict[str, Any]] = []
    for item, reference in zip(generated, expected, strict=True):
        cases.append(
            {
                "id": item["id"],
                "input": item["input"],
                "expected": reference,
                "meta": item["meta"],
            }
        )

    report = _report(cases, generated)
    _assert_mix(report, n)
    _write_task_yaml(output_dir, pack_name)
    cases_path = output_dir / "cases.jsonl"
    # ``newline=""`` keeps the pack byte-identical on Windows, where the default
    # translation would emit CRLF and break regeneration against the committed
    # LF packs.
    cases_path.write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for case in cases
        ),
        encoding="utf-8",
        newline="",
    )
    return report


def _make_case(rng: random.Random, prefix: str, index: int) -> dict[str, Any]:
    valuation = (
        date(2026, 4, 30) if index <= 6 else date(2026, 3, 31) if index % 2 else date(2026, 4, 30)
    )
    force_fee = index <= 5
    force_cross = index <= 5
    force_jpy = index <= 4
    force_holiday = index <= 6
    n_invoices = rng.randint(6, 10)
    invoices: list[dict[str, Any]] = []
    lines: list[str] = [
        f"Valuation date: {valuation.isoformat()}. Reporting currency: INR.",
        "Ledger:",
    ]

    for invoice_index in range(n_invoices):
        holiday_date = HOLIDAY_LOOKUP_DATES[(index + invoice_index) % len(HOLIDAY_LOOKUP_DATES)]
        invoice = _make_invoice(
            rng,
            prefix,
            index,
            invoice_index,
            valuation,
            force_fee=force_fee and invoice_index == 0,
            force_cross=force_cross and invoice_index == 0,
            force_jpy=force_jpy and invoice_index == 2,
            force_holiday=force_holiday and invoice_index == 1,
            holiday_date=holiday_date,
        )
        invoices.append(invoice)
        lines.append(_invoice_line(invoice, invoice_index + 1))
    lines.append(TRAILING_INSTRUCTION)

    return {
        "id": f"fx{prefix}_{index:03d}",
        "valuation_date": valuation.isoformat(),
        "invoices": invoices,
        "input": "\n".join(lines),
        "meta": {
            "n_invoices": n_invoices,
            "has_weekend_or_holiday": force_holiday,
            "has_bank_fee": any(invoice["bank_fee_inr"] > 0 for invoice in invoices),
            "has_cross_ccy": any(
                invoice["status"] == "SETTLED"
                and invoice["received_currency"] != invoice["currency"]
                for invoice in invoices
            ),
            "has_jpy": any(invoice["currency"] == "JPY" for invoice in invoices),
            "valuation_date": valuation.isoformat(),
        },
    }


def _make_invoice(
    rng: random.Random,
    prefix: str,
    case_index: int,
    invoice_index: int,
    valuation: date,
    *,
    force_fee: bool,
    force_cross: bool,
    force_jpy: bool,
    force_holiday: bool,
    holiday_date: date,
) -> dict[str, Any]:
    currency = "JPY" if force_jpy else rng.choice(CURRENCIES)
    latest_issue = min(ISSUE_END, valuation - timedelta(days=2))
    if force_holiday:
        issue_date = holiday_date
    else:
        issue_date = _random_date(rng, ISSUE_START, latest_issue)
    settled = force_fee or force_cross or rng.random() < 0.62
    if settled:
        settlement_start = max(issue_date + timedelta(days=1), ISSUE_START)
        settlement_date = _random_date(rng, settlement_start, valuation)
        if force_holiday:
            settlement_date = holiday_date
            if settlement_date <= issue_date:
                issue_date = ISSUE_START
        received_currency = currency
        if force_cross:
            received_currency = rng.choice(
                tuple(value for value in CURRENCIES if value != currency)
            )
        elif rng.random() < 0.18:
            received_currency = rng.choice(
                tuple(value for value in CURRENCIES if value != currency)
            )
        amount = _amount(rng, currency)
        received_amount = _received_amount(rng, amount, currency, received_currency)
        bank_fee = float(rng.choice(BANK_FEES)) if force_fee else 0.0
        return {
            "id": _invoice_id(prefix, case_index, invoice_index),
            "customer": CUSTOMERS[(case_index + invoice_index) % len(CUSTOMERS)],
            "currency": currency,
            "amount": amount,
            "issue_date": issue_date.isoformat(),
            "status": "SETTLED",
            "settlement_date": settlement_date.isoformat(),
            "received_currency": received_currency,
            "received_amount": received_amount,
            "bank_fee_inr": bank_fee,
        }

    return {
        "id": _invoice_id(prefix, case_index, invoice_index),
        "customer": CUSTOMERS[(case_index + invoice_index) % len(CUSTOMERS)],
        "currency": currency,
        "amount": _amount(rng, currency),
        "issue_date": issue_date.isoformat(),
        "status": "OPEN",
        "bank_fee_inr": 0.0,
    }


def _invoice_line(invoice: dict[str, Any], number: int) -> str:
    amount = _money(invoice["amount"], invoice["currency"])
    line = (
        f"{number}. {invoice['id']} · {invoice['customer']} · {invoice['currency']} {amount} · "
        f"issued {invoice['issue_date']} · "
    )
    if invoice["status"] == "OPEN":
        return line + "OPEN"
    line += (
        f"SETTLED {invoice['settlement_date']}, received "
        f"{invoice['received_currency']} "
        f"{_money(invoice['received_amount'], invoice['received_currency'])}"
    )
    if invoice["bank_fee_inr"]:
        line += f", net of bank fee INR {_money(invoice['bank_fee_inr'], 'INR')}"
    return line


def _amount(rng: random.Random, currency: str) -> float:
    if currency == "JPY":
        return float(rng.randrange(450_000, 2_200_001, 1000))
    return round(rng.uniform(650.0, 18_500.0), 2)


def _received_amount(
    rng: random.Random,
    amount: float,
    currency: str,
    received_currency: str,
) -> float:
    """Convert at roughly the market rate, so the realised move stays plausible.

    Exactly one draw is taken on every branch: the pack's dates and currencies —
    and therefore its set of cached rate lookups — must not depend on how the
    settled amount is computed.
    """

    jitter = rng.uniform(0.97, 1.03)
    converted = amount * jitter * APPROX_INR_RATES[currency] / APPROX_INR_RATES[received_currency]
    if received_currency == "JPY":
        return float(round(converted))
    return round(converted, 2)


def _money(value: float, currency: str) -> str:
    decimals = 0 if currency == "JPY" else 2
    return f"{value:,.{decimals}f}"


def _invoice_id(prefix: str, case_index: int, invoice_index: int) -> str:
    return f"INV-{prefix.upper()}{case_index:03d}-{invoice_index + 1:02d}"


def _random_date(rng: random.Random, start: date, end: date) -> date:
    if end < start:
        return start
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def _write_task_yaml(output_dir: Path, pack_name: str) -> None:
    task = {
        "name": pack_name,
        "domain": "finance_ops",
        "goal": (
            "Month-end FX revaluation: for a ledger of foreign-currency receivables, compute "
            "the total FX gain/(loss) in INR as of the valuation date and the gain/(loss) per "
            "invoice. Use the exchange rate tools for rates. Follow the company's ledger "
            "conventions as stated in each case."
        ),
        "answer_format": (
            "End your reply with a fenced json code block containing "
            '{"total_inr": <number>, "per_invoice": {"<id>": <number>, ...}}'
        ),
        "tools": ["fx_rate", "fx_series", "python_exec"],
        "checker": "fx_total",
        "examples": 3,
        "memory": "memory/fx_recon",
        "source": {
            "kind": "generated",
            "repo": "occam",
            "file": f"tasks/{pack_name}/cases.jsonl",
            "license": "MIT",
        },
    }
    # Keep manifests human-readable and stable across runs.  The schema and
    # model validation below are the source of truth for the accepted fields.
    task_yaml = yaml.safe_dump(task, sort_keys=False, allow_unicode=True, width=100)
    output_dir.joinpath("task.yaml").write_text(task_yaml, encoding="utf-8", newline="")
    validate_task(task)
    Task.model_validate(task)


def _report(cases: list[dict[str, Any]], generated: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases)
    mix = {
        "holiday_weekend": sum(case["meta"]["has_weekend_or_holiday"] for case in cases),
        "bank_fee": sum(case["meta"]["has_bank_fee"] for case in cases),
        "cross_currency": sum(case["meta"]["has_cross_ccy"] for case in cases),
        "jpy": sum(case["meta"]["has_jpy"] for case in cases),
    }
    absolute_totals = [abs(float(case["expected"]["total_inr"])) for case in cases]
    relative_tolerances = [
        max(5.0, 0.001 * absolute) / absolute for absolute in absolute_totals if absolute
    ]
    # A bank fee smaller than the grader's tolerance would let an agent that
    # ignores D1 pass the total anyway, so the fee cases would teach nothing.
    fee_headrooms = [
        fee / max(5.0, 0.001 * abs(float(case["expected"]["total_inr"])))
        for case, item in zip(cases, generated, strict=True)
        if (fee := sum(invoice["bank_fee_inr"] for invoice in item["invoices"]))
    ]
    return {
        "n": n,
        "mix": mix,
        "min_abs_expected": min(absolute_totals),
        "max_abs_expected": max(absolute_totals),
        "tightest_relative_tolerance": min(relative_tolerances),
        "min_bank_fee_headroom": min(fee_headrooms),
    }


def _assert_mix(report: dict[str, Any], n: int) -> None:
    minimums = {"holiday_weekend": 6, "bank_fee": 5, "cross_currency": 5, "jpy": 4}
    for name, minimum in minimums.items():
        if report["mix"][name] < minimum:
            raise AssertionError(f"{name} mix {report['mix'][name]} is below {minimum}")
    if report["n"] != n:
        raise AssertionError("generator report count does not match requested n")
    if report["min_bank_fee_headroom"] < MINIMUM_FEE_HEADROOM:
        raise AssertionError(
            f"bank fee headroom {report['min_bank_fee_headroom']:.2f}x is below the required "
            f"{MINIMUM_FEE_HEADROOM:.0f}x, so a fee case could pass without applying D1"
        )


def _print_report(report: dict[str, Any], *, seed: int, out: Path) -> None:
    mix = report["mix"]
    print(f"generated {report['n']} cases in {out} (seed={seed})")
    print(
        "mix: "
        f"holiday/weekend={mix['holiday_weekend']}/{report['n']} (>=6), "
        f"bank-fee={mix['bank_fee']}/{report['n']} (>=5), "
        f"cross-currency={mix['cross_currency']}/{report['n']} (>=5), "
        f"JPY={mix['jpy']}/{report['n']} (>=4)"
    )
    print(
        "tolerance sanity: "
        f"min_abs_expected={report['min_abs_expected']:.2f}, "
        f"max_abs_expected={report['max_abs_expected']:.2f}, "
        f"tightest_relative_tolerance={report['tightest_relative_tolerance']:.6f}, "
        f"min_bank_fee_headroom={report['min_bank_fee_headroom']:.2f}x "
        f"(>={MINIMUM_FEE_HEADROOM:.0f}x required)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "fx_cache")
    args = parser.parse_args()
    report = generate_pack(seed=args.seed, n=args.n, out=args.out, cache_dir=args.cache_dir)
    _print_report(report, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
