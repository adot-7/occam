"""Formula, grader, and Frankfurter client regression coverage for WP-03."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from occam.tasks.checkers import fx_total
from occam.tasks.fx_reference import compute_reference
from occam.tools.fx import FXClient, FXProtocolError


def test_open_invoice_matches_hand_computed_paisa() -> None:
    rates = {
        ("2026-03-01", "EUR", "INR"): 89.12,
        ("2026-04-30", "EUR", "INR"): 90.07,
    }

    result = compute_reference(
        "2026-04-30",
        [
            {
                "id": "INV-HAND-01",
                "currency": "EUR",
                "amount": 100.25,
                "issue_date": "2026-03-01",
                "status": "OPEN",
            }
        ],
        rate_lookup=lambda date, base, symbol: {"rate": rates[(date, base, symbol)]},
    )

    assert result == {"total_inr": 95.24, "per_invoice": {"INV-HAND-01": 95.24}}


def test_settled_cross_currency_adds_bank_fee_before_rounding() -> None:
    rates = {
        ("2026-02-10", "EUR", "INR"): 88.45,
        ("2026-04-04", "USD", "INR"): 83.17,
    }

    result = compute_reference(
        "2026-04-30",
        [
            {
                "id": "INV-HAND-02",
                "currency": "EUR",
                "amount": 250.50,
                "issue_date": "2026-02-10",
                "status": "SETTLED",
                "settlement_date": "2026-04-04",
                "received_currency": "USD",
                "received_amount": 300.75,
                "bank_fee_inr": 125.50,
            }
        ],
        rate_lookup=lambda date, base, symbol: {"rate": rates[(date, base, symbol)]},
    )

    assert result == {"total_inr": 2982.15, "per_invoice": {"INV-HAND-02": 2982.15}}


def test_settled_jpy_case_matches_hand_computed_paisa() -> None:
    rates = {
        ("2026-01-20", "JPY", "INR"): 0.5832,
        ("2026-03-31", "JPY", "INR"): 0.5799,
    }

    result = compute_reference(
        "2026-03-31",
        [
            {
                "id": "INV-HAND-03",
                "currency": "JPY",
                "amount": 1_250_000,
                "issue_date": "2026-01-20",
                "status": "SETTLED",
                "settlement_date": "2026-03-31",
                "received_currency": "JPY",
                "received_amount": 1_230_000,
                "bank_fee_inr": 875.25,
            }
        ],
        rate_lookup=lambda date, base, symbol: {"rate": rates[(date, base, symbol)]},
    )

    assert result == {"total_inr": -14847.75, "per_invoice": {"INV-HAND-03": -14847.75}}


def test_fx_total_extracts_last_fenced_json_and_records_sub_results() -> None:
    answer = """```json
    {"total_inr": 1000, "per_invoice": {"INV-1": 999, "INV-2": 123}}
    ```
    ```json
    {"total_inr": 1004, "per_invoice": {"INV-1": 1000, "INV-2": 125}}
    ```"""

    result = fx_total(
        answer,
        {"total_inr": 1000, "per_invoice": {"INV-1": 1000, "INV-2": 125}},
    )

    assert result.passed is True
    assert result.sub_results == {"INV-1": True, "INV-2": True}
    assert result.total_tolerance == 5.0


def test_fx_total_salvages_a_bare_final_json_line_when_the_fence_is_missing() -> None:
    """Defensive only: the contract is the fenced block, but a dropped fence
    must not fail an otherwise correct answer."""

    answer = 'Working: booked 100, revalued 105.\n{"total_inr": 5, "per_invoice": {"INV-1": 5}}'

    result = fx_total(answer, {"total_inr": 5, "per_invoice": {"INV-1": 5}})

    assert result.passed is True
    assert result.sub_results == {"INV-1": True}


def test_fx_total_uses_point_one_percent_for_large_totals() -> None:
    result = fx_total(
        '{"total_inr": 10009, "per_invoice": {"INV-1": 10000}}',
        {"total_inr": 10000, "per_invoice": {"INV-1": 10000}},
    )

    assert result.passed is True
    assert result.total_tolerance == 10.0


def test_fx_client_uses_base_symbols_and_caches_full_request_path(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert set(request.url.params) == {"base", "symbols"}
        return httpx.Response(
            200,
            json={
                "amount": 1.0,
                "base": request.url.params["base"],
                "date": "2026-04-02",
                "rates": {request.url.params["symbols"]: 1.1525},
            },
        )

    with FXClient(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    ) as client:
        first = client.fx_rate("2026-04-04", "EUR", "USD")
        second = client.fx_rate("2026-04-04", "EUR", "USD")

        assert first == {
            "requested_date": "2026-04-04",
            "rate_date": "2026-04-02",
            "base": "EUR",
            "symbol": "USD",
            "rate": 1.1525,
        }
        assert second == first
        assert len(requests) == 1
        assert client.calls[-1].cached is True
        assert client.calls[0].request_path == "/v1/2026-04-04?base=EUR&symbols=USD"
        assert client.cache_path_for(client.calls[0].request_path).is_file()


def test_fx_client_rejects_a_response_with_the_wrong_base(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"base": "USD", "date": "2026-04-02", "rates": {"INR": 85.91}},
        )

    with FXClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FXProtocolError, match="requested 'EUR'"):
            client.fx_rate("2026-04-04", "EUR", "INR")


def test_fx_series_normalizes_daily_rows_and_limits_concurrency(tmp_path: Path) -> None:
    active = 0
    peak = 0
    guard = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005)
        with guard:
            active -= 1
        if ".." in request.url.path:
            return httpx.Response(
                200,
                json={
                    "base": "EUR",
                    "rates": {
                        "2026-04-01": {"USD": 1.10},
                        "2026-04-02": {"USD": 1.11},
                    },
                },
            )
        return httpx.Response(
            200,
            json={"base": "EUR", "date": "2026-04-02", "rates": {"USD": 1.11}},
        )

    with FXClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler)) as client:
        series = client.fx_series("2026-04-01", "2026-04-03", "EUR", "USD")
        assert series == {
            "base": "EUR",
            "symbol": "USD",
            "rates": {"2026-04-01": 1.1, "2026-04-02": 1.11},
        }
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(
                executor.map(
                    lambda index: client.fx_rate(
                        f"2026-04-{index:02d}",
                        "EUR",
                        "USD",
                    ),
                    range(1, 11),
                )
            )
        assert peak <= 5
