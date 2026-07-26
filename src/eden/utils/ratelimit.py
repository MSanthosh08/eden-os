"""Client-side rate limiting.

A token bucket smooths bursts against a per-minute quota. Enforcing the limit
locally is cheaper and kinder than discovering it from a vendor's ``429``, and
it gives the router an honest signal about remaining capacity.
"""

from __future__ import annotations

import asyncio

from eden.utils.clock import Clock, SystemClock

_SECONDS_PER_MINUTE = 60.0


class TokenBucket:
    """A refilling token bucket guarding a per-minute request quota.

    Example:
        >>> bucket = TokenBucket(requests_per_minute=0)
        >>> bucket.unlimited
        True
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        clock: Clock | None = None,
        burst: int | None = None,
    ) -> None:
        """Initialise the bucket.

        Args:
            requests_per_minute: Sustained quota. ``0`` disables limiting.
            clock: Time source, injected for deterministic tests.
            burst: Bucket capacity. Defaults to the per-minute quota.

        Raises:
            ValueError: If ``requests_per_minute`` is negative.
        """
        if requests_per_minute < 0:
            message = "requests_per_minute must not be negative."
            raise ValueError(message)
        self._rate_per_second = requests_per_minute / _SECONDS_PER_MINUTE
        self._capacity = float(burst if burst is not None else max(requests_per_minute, 1))
        self._tokens = self._capacity
        self._clock = clock or SystemClock()
        self._updated_at = self._clock.monotonic()
        self._lock = asyncio.Lock()
        self._unlimited = requests_per_minute == 0

    @property
    def unlimited(self) -> bool:
        """Return whether rate limiting is disabled."""
        return self._unlimited

    @property
    def available(self) -> float:
        """Return the currently available token count without consuming."""
        if self._unlimited:
            return self._capacity
        return min(self._capacity, self._tokens + self._earned())

    def _earned(self) -> float:
        """Return tokens accrued since the last refill."""
        return max(0.0, self._clock.monotonic() - self._updated_at) * self._rate_per_second

    def _refill(self) -> None:
        """Credit accrued tokens and reset the refill marker."""
        now = self._clock.monotonic()
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
        self._updated_at = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """Consume ``tokens``, waiting for refill when necessary.

        Args:
            tokens: Cost of the operation being admitted.

        Returns:
            Seconds spent waiting, useful for telemetry.
        """
        if self._unlimited:
            return 0.0
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                delay = deficit / self._rate_per_second if self._rate_per_second > 0 else 0.0
                waited += delay
                await self._clock.sleep(delay)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Consume ``tokens`` without waiting.

        Args:
            tokens: Cost of the operation being admitted.

        Returns:
            ``True`` if the tokens were available and consumed.
        """
        if self._unlimited:
            return True
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False
