"""Small, cache-backed client for the Frankfurter v1 rates API.

The client deliberately returns only the data an agent needs.  Accounting
metadata is available through :attr:`FXClient.calls`, leaving tool-registry
concerns to the later WP-04 integration.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import httpx

from occam.llm.tracing import is_enabled, set_span_attributes, span

FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v1"
DEFAULT_CACHE_DIR = Path("data/fx_cache")
MAX_CONCURRENT_REQUESTS = 5
_EXACT_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")

# FX clients are often opened independently by generator workers, role
# registries, or tests.  These guards therefore live at module scope rather
# than on one client instance.  The cache-path key includes the cache
# directory so unrelated caches do not block one another.
_PROCESS_CACHE_GUARD = threading.Lock()
_PROCESS_CACHE_KEY_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LIVE_REQUESTS = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
_LOGGER = logging.getLogger(__name__)

# These are intentionally plain.  Tool-registry descriptions and learned
# lessons are assembled by later work packages, not by this HTTP client.
FX_RATE_DESCRIPTION = "Get the exchange rate between two currencies on a date."
FX_SERIES_DESCRIPTION = "Get exchange rates between two currencies."


class FXProtocolError(ValueError):
    """Raised when Frankfurter returns a response outside its contract."""


@dataclass(frozen=True)
class FXCall:
    """Accounting information for one request or cache read."""

    request_path: str
    latency_s: float
    response_bytes: int
    status: int
    cached: bool

    @property
    def bytes(self) -> int:
        """Expose the concise field name used by the event contract."""

        return self.response_bytes


class FXClient:
    """Synchronous Frankfurter client with deterministic on-disk caching.

    ``max_concurrency`` is capped at five to keep callers polite to the free
    public API.  A process-global semaphore enforces the same five-request
    ceiling across independently opened clients; the per-client semaphore is
    an additional, caller-selected limit.
    """

    def __init__(
        self,
        *,
        base_url: str = FRANKFURTER_BASE_URL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        timeout: float = 30.0,
        max_concurrency: int = MAX_CONCURRENT_REQUESTS,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if http_client is not None and transport is not None:
            raise ValueError("pass either http_client or transport, not both")

        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_concurrency = min(max_concurrency, MAX_CONCURRENT_REQUESTS)
        self._http = http_client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._owns_http = http_client is None
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._cache_guard = threading.RLock()
        self._calls: list[FXCall] = []
        self._thread_state = threading.local()

    @property
    def calls(self) -> tuple[FXCall, ...]:
        """Return an immutable view of per-request accounting records."""

        with self._cache_guard:
            return tuple(self._calls)

    @property
    def last_call(self) -> FXCall | None:
        """Return the most recent accounting record, if one exists."""

        with self._cache_guard:
            return self._calls[-1] if self._calls else None

    @property
    def thread_last_call(self) -> FXCall | None:
        """Return this thread's most recent accounting record.

        The tool registry attributes latency, bytes, status and ``cached`` to the
        call it just made.  ``last_call`` cannot do that once roles run
        concurrently, because another thread may have recorded in between.
        """

        return getattr(self._thread_state, "last_call", None)

    def close(self) -> None:
        """Close the underlying HTTP client when this instance owns it."""

        if self._owns_http:
            self._http.close()

    def __enter__(self) -> FXClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def cache_path_for(self, request_path: str) -> Path:
        """Return the deterministic cache path for a full request path."""

        encoded = quote(request_path.lstrip("/"), safe="")
        return self.cache_dir / f"{encoded}.json"

    def request_path(self, endpoint: str, base: str, symbol: str) -> str:
        """Build the canonical path used both on the wire and as the cache key."""

        base_path = urlsplit(self.base_url).path.rstrip("/")
        query = urlencode((("base", base), ("symbols", symbol)))
        return f"{base_path}{endpoint}?{query}"

    def fx_rate(self, date: str, base: str, symbol: str) -> dict[str, Any]:
        """Get one exchange rate and retain the API's returned rate date."""

        requested_date = self._input_date(date, "date")
        requested_base = self._currency(base, "base")
        requested_symbol = self._currency(symbol, "symbol")
        self._assert_distinct_currencies(requested_base, requested_symbol)
        endpoint = f"/{date}"
        request_path = self.request_path(endpoint, requested_base, requested_symbol)
        data, _ = self._get_json(
            endpoint,
            requested_base,
            requested_symbol,
            request_path,
            response_kind="daily",
            requested_start=requested_date,
            requested_end=requested_date,
            tool="fx_rate",
            requested_date=date,
        )
        rate_date = data.get("date")
        rate = self._rate_value(data, requested_symbol)
        return {
            "requested_date": date,
            "rate_date": rate_date,
            "base": requested_base,
            "symbol": requested_symbol,
            "rate": rate,
        }

    def fx_series(self, start: str, end: str, base: str, symbol: str) -> dict[str, Any]:
        """Get all returned daily rates for a date interval."""

        requested_start = self._input_date(start, "start")
        requested_end = self._input_date(end, "end")
        if requested_start > requested_end:
            raise ValueError("start date must not be after end date")
        requested_base = self._currency(base, "base")
        requested_symbol = self._currency(symbol, "symbol")
        self._assert_distinct_currencies(requested_base, requested_symbol)
        endpoint = f"/{start}..{end}"
        request_path = self.request_path(endpoint, requested_base, requested_symbol)
        data, _ = self._get_json(
            endpoint,
            requested_base,
            requested_symbol,
            request_path,
            response_kind="series",
            requested_start=requested_start,
            requested_end=requested_end,
            tool="fx_series",
            requested_date=f"{start}..{end}",
        )
        raw_rates = data.get("rates")

        rates: dict[str, float] = {}
        for rate_date in sorted(raw_rates):
            row = raw_rates[rate_date]
            value = row[requested_symbol]
            rates[rate_date] = self._as_float(value, requested_symbol)
        return {"base": requested_base, "symbol": requested_symbol, "rates": rates}

    def _get_json(
        self,
        endpoint: str,
        base: str,
        symbol: str,
        request_path: str,
        *,
        response_kind: str,
        requested_start: date,
        requested_end: date,
        tool: str,
        requested_date: str,
    ) -> tuple[dict[str, Any], bool]:
        cache_path = self.cache_path_for(request_path)
        key_lock = self._lock_for(cache_path)
        with key_lock:
            cached_payload = self._read_cache(
                cache_path,
                request_path,
                base=base,
                symbol=symbol,
                response_kind=response_kind,
                requested_start=requested_start,
                requested_end=requested_end,
            )
            if cached_payload is not None:
                data = cached_payload["response"]
                response_bytes = len(self._json_bytes(data))
                self._record(
                    FXCall(
                        request_path=request_path,
                        latency_s=0.0,
                        response_bytes=response_bytes,
                        status=cached_payload["status"],
                        cached=True,
                    )
                )
                return data, True

            if is_enabled():
                span_attributes = {
                    "tool": tool,
                    "requested_date": requested_date,
                    "cached": False,
                }
                if tool == "fx_series":
                    span_attributes.update(
                        requested_start=requested_start.isoformat(),
                        requested_end=requested_end.isoformat(),
                    )
                with span(f"tool.{tool}", kind="TOOL", attributes=span_attributes) as span_object:
                    return self._request_json(
                        endpoint,
                        base,
                        symbol,
                        request_path,
                        cache_path,
                        response_kind=response_kind,
                        requested_start=requested_start,
                        requested_end=requested_end,
                        tool=tool,
                        span_object=span_object,
                    )
            return self._request_json(
                endpoint,
                base,
                symbol,
                request_path,
                cache_path,
                response_kind=response_kind,
                requested_start=requested_start,
                requested_end=requested_end,
                tool=tool,
                span_object=None,
            )

    def _request_json(
        self,
        endpoint: str,
        base: str,
        symbol: str,
        request_path: str,
        cache_path: Path,
        *,
        response_kind: str,
        requested_start: date,
        requested_end: date,
        tool: str,
        span_object: Any | None,
    ) -> tuple[dict[str, Any], bool]:
        """Fetch one uncached response and finish its optional manual span."""

        started = time.perf_counter()
        params = {"base": base, "symbols": symbol}
        response: httpx.Response | None = None
        response_bytes = 0
        status: int | None = None
        try:
            with self._semaphore, _PROCESS_LIVE_REQUESTS:
                response = self._http.get(f"{self.base_url}{endpoint}", params=params)
            elapsed = time.perf_counter() - started
            status = response.status_code
            response_bytes = len(response.content)
            if span_object is not None:
                set_span_attributes(
                    span_object,
                    {"status": status, "bytes": response_bytes},
                )
            response.raise_for_status()
            try:
                data = response.json()
            except (TypeError, ValueError) as exc:
                raise FXProtocolError("Frankfurter returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise FXProtocolError("Frankfurter returned a non-object JSON response")
            rate_date: str | None = None
            if tool == "fx_rate":
                value = data.get("date")
                rate_date = value if isinstance(value, str) else None
            elif tool == "fx_series":
                response_start = data.get("start_date")
                response_end = data.get("end_date")
                if isinstance(response_start, str) and isinstance(response_end, str):
                    rate_date = (
                        response_start
                        if response_start == response_end
                        else f"{response_start}..{response_end}"
                    )
            if span_object is not None:
                set_span_attributes(span_object, {"rate_date": rate_date})
            self._validate_response(
                data,
                base=base,
                symbol=symbol,
                response_kind=response_kind,
                requested_start=requested_start,
                requested_end=requested_end,
            )
            self._write_cache(
                cache_path,
                request_path,
                response.status_code,
                data,
            )
            self._record(
                FXCall(
                    request_path=request_path,
                    latency_s=elapsed,
                    response_bytes=response_bytes,
                    status=response.status_code,
                    cached=False,
                )
            )
            return data, False
        except Exception as exc:
            if span_object is not None:
                set_span_attributes(
                    span_object,
                    {
                        "status": status,
                        "bytes": response_bytes,
                        "error": type(exc).__name__,
                    },
                )
            raise

    @staticmethod
    def _lock_for(cache_path: Path) -> threading.Lock:
        key = str(cache_path.resolve())
        with _PROCESS_CACHE_GUARD:
            return _PROCESS_CACHE_KEY_LOCKS.setdefault(key, threading.Lock())

    def _record(self, call: FXCall) -> None:
        with self._cache_guard:
            self._calls.append(call)
        self._thread_state.last_call = call

    def _read_cache(
        self,
        path: Path,
        request_path: str,
        *,
        base: str,
        symbol: str,
        response_kind: str,
        requested_start: date,
        requested_end: date,
    ) -> dict[str, Any] | None:
        try:
            if not path.exists():
                return None
            if not path.is_file():
                raise FXProtocolError("cache path is not a file")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, FXProtocolError) as exc:
            _LOGGER.warning(
                "invalid FX cache entry; treating it as a miss and refetching (%s)",
                type(exc).__name__,
            )
            return None
        try:
            if not isinstance(payload, dict):
                raise FXProtocolError("cache envelope is not an object")
            if payload.get("request_path") != request_path:
                raise FXProtocolError("request path does not match")
            response = payload.get("response")
            status = payload.get("status")
            if not isinstance(response, dict):
                raise FXProtocolError("response is not an object")
            if isinstance(status, bool) or not isinstance(status, int) or not 200 <= status < 300:
                raise FXProtocolError("status is not a successful HTTP status")
            self._validate_response(
                response,
                base=base,
                symbol=symbol,
                response_kind=response_kind,
                requested_start=requested_start,
                requested_end=requested_end,
            )
        except FXProtocolError as exc:
            _LOGGER.warning(
                "invalid FX cache entry; treating it as a miss and refetching (%s)",
                type(exc).__name__,
            )
            return None
        return {"response": response, "status": status}

    def _write_cache(
        self,
        path: Path,
        request_path: str,
        status: int,
        response: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "request_path": request_path,
            "response": response,
            "status": status,
        }
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(self._json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (encoded + "\n").encode("utf-8")

    @staticmethod
    def _input_date(value: Any, field: str) -> date:
        return FXClient._parse_iso_date(value, field=field, error_type=ValueError)

    @staticmethod
    def _parse_iso_date(
        value: Any,
        *,
        field: str,
        error_type: type[ValueError],
    ) -> date:
        if not isinstance(value, str) or _EXACT_ISO_DATE.fullmatch(value) is None:
            raise error_type(f"{field} must be an exact ISO calendar date (YYYY-MM-DD)")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise error_type(f"{field} is not a valid ISO calendar date: {value!r}") from exc

    @staticmethod
    def _currency(value: Any, field: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 3
            or any(character < "A" or character > "Z" for character in value)
        ):
            raise ValueError(f"{field} must be an uppercase ASCII 3-letter currency code")
        return value

    @staticmethod
    def _assert_distinct_currencies(base: str, symbol: str) -> None:
        if base == symbol:
            raise ValueError("base and symbol must be distinct currencies")

    @staticmethod
    def _assert_base(data: dict[str, Any], requested_base: str) -> None:
        returned_base = data.get("base")
        if returned_base != requested_base:
            raise FXProtocolError(
                f"Frankfurter returned base {returned_base!r}, requested {requested_base!r}"
            )

    def _validate_response(
        self,
        data: dict[str, Any],
        *,
        base: str,
        symbol: str,
        response_kind: str,
        requested_start: date,
        requested_end: date,
    ) -> None:
        self._assert_base(data, base)
        if response_kind == "daily":
            self._validate_daily_response(data, symbol, requested_end)
        elif response_kind == "series":
            self._validate_series_response(
                data,
                symbol,
                requested_start=requested_start,
                requested_end=requested_end,
            )
        else:  # pragma: no cover - private callers only use the two constants.
            raise ValueError(f"unknown FX response kind: {response_kind}")

    @classmethod
    def _validate_daily_response(
        cls,
        data: dict[str, Any],
        symbol: str,
        requested_end: date,
    ) -> None:
        actual_date = cls._parse_iso_date(
            data.get("date"),
            field="response date",
            error_type=FXProtocolError,
        )
        if actual_date > requested_end:
            raise FXProtocolError(
                f"Frankfurter response date {actual_date.isoformat()} is after "
                f"requested date {requested_end.isoformat()}"
            )
        rates = data.get("rates")
        if not isinstance(rates, dict) or symbol not in rates:
            raise FXProtocolError(f"Frankfurter response has no {symbol} rate")
        for returned_symbol, value in rates.items():
            cls._as_float(value, str(returned_symbol))

    @classmethod
    def _validate_series_response(
        cls,
        data: dict[str, Any],
        symbol: str,
        *,
        requested_start: date,
        requested_end: date,
    ) -> None:
        actual_start = cls._parse_iso_date(
            data.get("start_date"),
            field="response start_date",
            error_type=FXProtocolError,
        )
        actual_end = cls._parse_iso_date(
            data.get("end_date"),
            field="response end_date",
            error_type=FXProtocolError,
        )
        if actual_start > actual_end:
            raise FXProtocolError("Frankfurter response range is reversed")
        # Frankfurter may resolve a weekend/holiday start backward to the last
        # available business day, but it must never invent data after the
        # requested upper bound or silently omit the requested lower bound.
        if actual_start > requested_start:
            raise FXProtocolError(
                f"Frankfurter response starts at {actual_start.isoformat()}, after "
                f"requested start {requested_start.isoformat()}"
            )
        if actual_end > requested_end:
            raise FXProtocolError(
                f"Frankfurter response ends at {actual_end.isoformat()}, after "
                f"requested end {requested_end.isoformat()}"
            )

        raw_rates = data.get("rates")
        if not isinstance(raw_rates, dict) or not raw_rates:
            raise FXProtocolError("Frankfurter series response has no daily rates")
        row_dates: list[date] = []
        for raw_date, row in raw_rates.items():
            row_date = cls._parse_iso_date(
                raw_date,
                field="series rate date",
                error_type=FXProtocolError,
            )
            if row_date < actual_start or row_date > actual_end or row_date > requested_end:
                raise FXProtocolError(
                    f"Frankfurter series row {row_date.isoformat()} is outside the "
                    "requested response range"
                )
            if not isinstance(row, dict) or symbol not in row:
                raise FXProtocolError(
                    f"Frankfurter series response has no {symbol} rate on {raw_date}"
                )
            for returned_symbol, value in row.items():
                cls._as_float(value, str(returned_symbol))
            row_dates.append(row_date)

        if min(row_dates) != actual_start or max(row_dates) != actual_end:
            raise FXProtocolError("Frankfurter series response range does not match its daily rows")

    @classmethod
    def _rate_value(cls, data: dict[str, Any], symbol: str) -> float:
        rates = data["rates"]
        return cls._as_float(rates[symbol], symbol)

    @staticmethod
    def _as_float(value: Any, symbol: str) -> float:
        if isinstance(value, bool):
            raise FXProtocolError(f"Frankfurter returned a boolean {symbol} rate")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise FXProtocolError(f"Frankfurter returned an invalid {symbol} rate") from exc
        if not math.isfinite(number):
            raise FXProtocolError(f"Frankfurter returned a non-finite {symbol} rate")
        if number <= 0:
            raise FXProtocolError(f"Frankfurter returned a non-positive {symbol} rate")
        return number


_default_client: FXClient | None = None
_default_guard = threading.Lock()


def get_default_client() -> FXClient:
    """Return the process-level client used by the reference implementation."""

    global _default_client
    with _default_guard:
        if _default_client is None:
            _default_client = FXClient()
        return _default_client


def close_default_client() -> None:
    """Close and discard the process-level client."""

    global _default_client
    with _default_guard:
        if _default_client is not None:
            _default_client.close()
            _default_client = None


def fx_rate(date: str, base: str, symbol: str) -> dict[str, Any]:
    """Get a single exchange rate using the default cache-backed client."""

    return get_default_client().fx_rate(date, base, symbol)


def fx_series(start: str, end: str, base: str, symbol: str) -> dict[str, Any]:
    """Get exchange rates for an interval using the default client."""

    return get_default_client().fx_series(start, end, base, symbol)


__all__ = [
    "DEFAULT_CACHE_DIR",
    "FXCall",
    "FXClient",
    "FXProtocolError",
    "FX_RATE_DESCRIPTION",
    "FX_SERIES_DESCRIPTION",
    "FRANKFURTER_BASE_URL",
    "MAX_CONCURRENT_REQUESTS",
    "close_default_client",
    "fx_rate",
    "fx_series",
    "get_default_client",
]
