"""Adversarial coverage for the WP-03 FX client hardening."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from occam.tools.fx import MAX_CONCURRENT_REQUESTS, FXClient, FXProtocolError


def _daily(
    *,
    base: str = "EUR",
    symbol: str = "USD",
    actual_date: str = "2026-04-02",
    rate: Any = 1.15,
) -> dict[str, Any]:
    return {"base": base, "date": actual_date, "rates": {symbol: rate}}


def _series(
    *,
    base: str = "EUR",
    symbol: str = "USD",
    start_date: str = "2026-04-01",
    end_date: str = "2026-04-02",
    rows: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "base": base,
        "start_date": start_date,
        "end_date": end_date,
        "rates": rows
        if rows is not None
        else {
            "2026-04-01": {symbol: 1.14},
            "2026-04-02": {symbol: 1.15},
        },
    }


def _raising_transport() -> httpx.MockTransport:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid input reached the HTTP transport")

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    "bad_date",
    [
        "2026-2-03",
        "20260203",
        "2026-02-29",
        "2026-02-30",
        "2026-02-03T00:00:00",
        "2026-02-03 ",
        "2026-02-03?base=USD",
        "2026-02-03/../../latest",
        "2026-02-03%3Fbase%3DUSD",
    ],
)
def test_invalid_daily_dates_are_rejected_before_endpoint_construction(
    tmp_path: Path, bad_date: str
) -> None:
    with FXClient(cache_dir=tmp_path, transport=_raising_transport()) as client:
        with pytest.raises(ValueError, match="exact ISO|valid ISO"):
            client.fx_rate(bad_date, "EUR", "USD")


@pytest.mark.parametrize(
    "bad_currency",
    [
        "eur",
        "EU",
        "EURO",
        "E1R",
        "EUR ",
        " EUR",
        "ÉUR",
        "ＥＵＲ",
        "EUR&symbols=GBP",
        "USD?base=EUR",
    ],
)
def test_currency_arguments_must_be_uppercase_ascii_three_letter_codes(
    tmp_path: Path, bad_currency: str
) -> None:
    with FXClient(cache_dir=tmp_path, transport=_raising_transport()) as client:
        with pytest.raises(ValueError, match="uppercase ASCII 3-letter"):
            client.fx_rate("2026-02-03", bad_currency, "USD")
        with pytest.raises(ValueError, match="uppercase ASCII 3-letter"):
            client.fx_rate("2026-02-03", "EUR", bad_currency)


def test_same_currency_and_reversed_or_invalid_ranges_are_rejected_before_http(
    tmp_path: Path,
) -> None:
    with FXClient(cache_dir=tmp_path, transport=_raising_transport()) as client:
        with pytest.raises(ValueError, match="distinct"):
            client.fx_rate("2026-02-03", "EUR", "EUR")
        with pytest.raises(ValueError, match="distinct"):
            client.fx_series("2026-02-03", "2026-02-04", "EUR", "EUR")
        with pytest.raises(ValueError, match="after"):
            client.fx_series("2026-02-04", "2026-02-03", "EUR", "USD")
        with pytest.raises(ValueError, match="exact ISO|valid ISO"):
            client.fx_series("2026-02-03", "2026-2-04", "EUR", "USD")


@pytest.mark.parametrize(
    "payload,match",
    [
        (_daily(base="USD"), "requested 'EUR'"),
        (_daily(actual_date="not-a-date"), "response date"),
        (_daily(actual_date="2026-04-05"), "after requested"),
        ({"base": "EUR", "date": "2026-04-02", "rates": {}}, "no USD rate"),
        (_daily(rate=0), "non-positive"),
        (_daily(rate=-1), "non-positive"),
        (_daily(rate=float("nan")), "non-finite"),
        (_daily(rate=float("inf")), "non-finite"),
        ({**_daily(), "rates": {"USD": 1.15, "GBP": -1}}, "non-positive"),
    ],
)
def test_invalid_daily_network_payload_is_rejected_before_cache_write(
    tmp_path: Path, payload: dict[str, Any], match: str
) -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            content=json.dumps(payload, allow_nan=True).encode("utf-8"),
        )

    with FXClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FXProtocolError, match=match):
            client.fx_rate("2026-04-04", "EUR", "USD")

    assert requests == 1
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_network_json_is_rejected_before_cache_write(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"base":')

    with FXClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FXProtocolError, match="invalid JSON"):
            client.fx_rate("2026-04-04", "EUR", "USD")

    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "payload,match",
    [
        (_series(start_date="2026-04-02"), "after requested start"),
        (_series(end_date="2026-04-03"), "after requested end"),
        (_series(start_date="not-a-date"), "response start_date"),
        (
            _series(start_date="2026-04-02", end_date="2026-04-01"),
            "range is reversed",
        ),
        (
            _series(rows={"2026-04-01": {"USD": 1.14}, "2026-04-03": {"USD": 1.15}}),
            "outside the requested response range",
        ),
        (_series(rows={"2026-04-01": {"USD": 1.14}}), "range does not match"),
        (_series(rows={"2026-04-01": {"EUR": 1.14}, "2026-04-02": {"EUR": 1.15}}), "no USD"),
        (_series(rows={"2026-04-01": {"USD": 0}, "2026-04-02": {"USD": 1.15}}), "non-positive"),
        (
            _series(
                rows={
                    "2026-04-01": {"USD": 1.14, "GBP": float("inf")},
                    "2026-04-02": {"USD": 1.15},
                }
            ),
            "non-finite",
        ),
        (_series(rows={"2026-04-01": 1.14, "2026-04-02": 1.15}), "no USD rate"),
        (
            {"base": "EUR", "start_date": "2026-04-01", "end_date": "2026-04-02", "rates": {}},
            "no daily rates",
        ),
    ],
)
def test_invalid_series_network_payload_is_rejected_before_cache_write(
    tmp_path: Path, payload: dict[str, Any], match: str
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(payload, allow_nan=True).encode("utf-8"),
        )

    with FXClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FXProtocolError, match=match):
            client.fx_series("2026-04-01", "2026-04-02", "EUR", "USD")

    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_existing_cache_is_a_logged_miss_and_is_repaired(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_daily())

    client = FXClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    request_path = client.request_path("/2026-04-04", "EUR", "USD")
    cache_path = client.cache_path_for(request_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not-json", encoding="utf-8")

    caplog.set_level(logging.WARNING, logger="occam.tools.fx")
    with client:
        result = client.fx_rate("2026-04-04", "EUR", "USD")

    assert result["rate"] == 1.15
    assert requests == 1
    assert "invalid FX cache entry; treating it as a miss and refetching" in caplog.text
    assert "{not-json" not in caplog.text
    assert json.loads(cache_path.read_text(encoding="utf-8"))["response"] == _daily()


@pytest.mark.parametrize(
    "response,match",
    [
        (_daily(actual_date="2026-04-05"), "after requested"),
        (_daily(rate=-1), "non-positive"),
    ],
)
def test_semantically_invalid_existing_cache_is_revalidated(
    tmp_path: Path, response: dict[str, Any], match: str, caplog: pytest.LogCaptureFixture
) -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_daily())

    client = FXClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler))
    request_path = client.request_path("/2026-04-04", "EUR", "USD")
    cache_path = client.cache_path_for(request_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "request_path": request_path,
                "status": 200,
                "response": response,
            }
        ),
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING, logger="occam.tools.fx")
    with client:
        result = client.fx_rate("2026-04-04", "EUR", "USD")

    assert result["rate"] == 1.15
    assert requests == 1
    assert "invalid FX cache entry; treating it as a miss and refetching" in caplog.text
    assert match not in caplog.text


def test_two_preopened_clients_share_one_cache_key_lock(tmp_path: Path) -> None:
    requests = 0
    request_guard = threading.Lock()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        with request_guard:
            requests += 1
        time.sleep(0.02)
        return httpx.Response(200, json=_daily())

    transport = httpx.MockTransport(handler)
    first = FXClient(cache_dir=tmp_path, transport=transport)
    second = FXClient(cache_dir=tmp_path, transport=transport)
    with first, second:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda client: client.fx_rate("2026-04-04", "EUR", "USD"),
                    (first, second),
                )
            )

    assert results[0] == results[1]
    assert requests == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_two_preopened_clients_share_the_five_live_request_budget(tmp_path: Path) -> None:
    active = 0
    peak = 0
    state_guard = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        with state_guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with state_guard:
            active -= 1
        requested_date = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=_daily(actual_date=requested_date))

    transport = httpx.MockTransport(handler)
    first = FXClient(cache_dir=tmp_path, transport=transport)
    second = FXClient(cache_dir=tmp_path, transport=transport)
    dates = [f"2026-04-{day:02d}" for day in range(1, 11)]
    with first, second:
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(
                executor.map(
                    lambda item: item[0].fx_rate(item[1], "EUR", "USD"),
                    zip((first, second) * 5, dates, strict=True),
                )
            )

    assert peak <= MAX_CONCURRENT_REQUESTS
    cache_files = sorted(tmp_path.glob("*.json"))
    assert len(cache_files) == len(dates)
    for cache_file in cache_files:
        envelope = json.loads(cache_file.read_text(encoding="utf-8"))
        assert envelope["status"] == 200
        assert envelope["request_path"].startswith("/v1/")
        assert envelope["response"]["rates"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_series_cache_reads_use_the_same_semantic_validation(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_series()))
    with FXClient(cache_dir=tmp_path, transport=transport) as client:
        assert client.fx_series("2026-04-01", "2026-04-02", "EUR", "USD")["rates"]
        request_path = client.request_path("/2026-04-01..2026-04-02", "EUR", "USD")
        cache_path = client.cache_path_for(request_path)
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["response"]["rates"]["2026-04-02"]["USD"] = -1
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

        result = client.fx_series("2026-04-01", "2026-04-02", "EUR", "USD")

        assert result["rates"]["2026-04-02"] == 1.15
        repaired = json.loads(cache_path.read_text(encoding="utf-8"))
        assert repaired["response"]["rates"]["2026-04-02"]["USD"] == 1.15


def test_lookup_metadata_helper_uses_actual_lookup_dates() -> None:
    from scripts.gen_fx_cases import _has_weekend_or_holiday

    assert _has_weekend_or_holiday(
        date(2026, 4, 30),
        [
            {
                "issue_date": "2026-04-01",
                "status": "SETTLED",
                "settlement_date": "2026-04-04",
            }
        ],
    )
    assert not _has_weekend_or_holiday(
        date(2026, 4, 30),
        [
            {
                "issue_date": "2026-04-01",
                "status": "SETTLED",
                "settlement_date": "2026-04-07",
            }
        ],
    )
