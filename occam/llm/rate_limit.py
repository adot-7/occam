"""Thread-safe token buckets used to cap requests per model lane."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock


class TokenBucket:
    """A continuously refilling token bucket.

    The model table expresses ``rpm`` (requests per minute), so the client
    acquires one bucket token per provider attempt.  ``tokens`` is exposed for
    future TPM limiting and makes the class useful independently in tests.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if capacity <= 0 or refill_rate <= 0:
            raise ValueError("token bucket capacity and refill_rate must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._clock = clock
        self._sleep = sleeper
        self._tokens = self.capacity
        self._updated_at = clock()
        self._lock = Lock()

    @property
    def available(self) -> float:
        with self._lock:
            self._refill(self._clock())
            return self._tokens

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._updated_at = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until tokens are available and return the waited seconds."""

        if tokens <= 0 or tokens > self.capacity:
            raise ValueError("requested tokens must be positive and fit bucket capacity")
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._refill(now)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                delay = (tokens - self._tokens) / self.refill_rate
            self._sleep(delay)
            waited += delay


class PerModelRateLimiter:
    """Lazily creates one request bucket for every model key."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleeper
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def acquire(self, model_key: str, rpm: float | None) -> float:
        """Acquire one request token for ``model_key`` when ``rpm`` is set."""

        if rpm is None:
            return 0.0
        with self._lock:
            bucket = self._buckets.get(model_key)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=max(1.0, float(rpm)),
                    refill_rate=float(rpm) / 60.0,
                    clock=self._clock,
                    sleeper=self._sleep,
                )
                self._buckets[model_key] = bucket
        return bucket.acquire()

    def bucket(self, model_key: str) -> TokenBucket | None:
        """Expose a bucket for diagnostics and deterministic tests."""

        with self._lock:
            return self._buckets.get(model_key)


__all__ = ["PerModelRateLimiter", "TokenBucket"]
