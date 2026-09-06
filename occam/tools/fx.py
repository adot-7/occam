"""Small, cache-backed client for the Frankfurter v1 rates API.

The client deliberately returns only the data an agent needs.  Accounting
metadata is available through :attr:`FXClient.calls`, leaving tool-registry
concerns to the later WP-04 integration.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import httpx

FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v1"
DEFAULT_CACHE_DIR = Path("data/fx_cache")
MAX_CONCURRENT_REQUESTS = 5

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
    public API.  The semaphore is shared by all requests made through one
    client, including calls made by multiple generator worker threads.
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
        self._key_locks: dict[str, threading.Lock] = {}
        self._calls: list[FXCall] = []

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

        requested_base = self._currency(base)
        requested_symbol = self._currency(symbol)
        endpoint = f"/{date}"
        request_path = self.request_path(endpoint, requested_base, requested_symbol)
        data, _ = self._get_json(
            endpoint,
            requested_base,
            requested_symbol,
            request_path,
        )
        rate_date = data.get("date")
        if not isinstance(rate_date, str) or not rate_date:
            raise FXProtocolError("Frankfurter rate response has no date")
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

        requested_base = self._currency(base)
        requested_symbol = self._currency(symbol)
        endpoint = f"/{start}..{end}"
        request_path = self.request_path(endpoint, requested_base, requested_symbol)
        data, _ = self._get_json(
            endpoint,
            requested_base,
            requested_symbol,
            request_path,
        )
        raw_rates = data.get("rates")
        if not isinstance(raw_rates, dict):
            raise FXProtocolError("Frankfurter series response has no rates object")

        rates: dict[str, float] = {}
        for rate_date in sorted(raw_rates):
            row = raw_rates[rate_date]
            if isinstance(row, dict):
                if requested_symbol not in row:
                    raise FXProtocolError(
                        f"Frankfurter series response has no {requested_symbol} rate on {rate_date}"
                    )
                value = row[requested_symbol]
            else:
                value = row
            rates[rate_date] = self._as_float(value, requested_symbol)
        return {"base": requested_base, "symbol": requested_symbol, "rates": rates}

    def _get_json(
        self,
        endpoint: str,
        base: str,
        symbol: str,
        request_path: str,
    ) -> tuple[dict[str, Any], bool]:
        cache_path = self.cache_path_for(request_path)
        key_lock = self._lock_for(request_path)
        with self._semaphore, key_lock:
            cached_payload = self._read_cache(cache_path, request_path)
            if cached_payload is not None:
                data = cached_payload["response"]
                self._assert_base(data, base)
                self._assert_symbol(data, symbol)
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

            started = time.perf_counter()
            params = {"base": base, "symbols": symbol}
            response = self._http.get(f"{self.base_url}{endpoint}", params=params)
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            try:
                data = response.json()
            except (TypeError, ValueError) as exc:
                raise FXProtocolError("Frankfurter returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise FXProtocolError("Frankfurter returned a non-object JSON response")
            self._assert_base(data, base)
            self._assert_symbol(data, symbol)
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
                    response_bytes=len(response.content),
                    status=response.status_code,
                    cached=False,
                )
            )
            return data, False

    def _lock_for(self, request_path: str) -> threading.Lock:
        with self._cache_guard:
            return self._key_locks.setdefault(request_path, threading.Lock())

    def _record(self, call: FXCall) -> None:
        with self._cache_guard:
            self._calls.append(call)

    def _read_cache(self, path: Path, request_path: str) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("request_path") != request_path:
            return None
        response = payload.get("response")
        status = payload.get("status", 200)
        if not isinstance(response, dict) or not isinstance(status, int):
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
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(self._json_bytes(payload))
        temporary.replace(path)

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (encoded + "\n").encode("utf-8")

    @staticmethod
    def _currency(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("currency must be a non-empty string")
        return value.strip().upper()

    @staticmethod
    def _assert_base(data: dict[str, Any], requested_base: str) -> None:
        returned_base = data.get("base")
        if returned_base != requested_base:
            raise FXProtocolError(
                f"Frankfurter returned base {returned_base!r}, requested {requested_base!r}"
            )

    @staticmethod
    def _assert_symbol(data: dict[str, Any], symbol: str) -> None:
        rates = data.get("rates")
        if not isinstance(rates, dict):
            raise FXProtocolError("Frankfurter response has no rates object")
        if rates and all(isinstance(value, dict) for value in rates.values()):
            if any(symbol not in value for value in rates.values()):
                raise FXProtocolError(f"Frankfurter response has no {symbol} rate")
        elif symbol not in rates:
            raise FXProtocolError(f"Frankfurter response has no {symbol} rate")

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
        if number != number or number in (float("inf"), float("-inf")):
            raise FXProtocolError(f"Frankfurter returned a non-finite {symbol} rate")
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
