"""Time abstraction.

Anything that decides *when* — retry backoff, circuit-breaker cool-down, rate
limiting — takes a :class:`Clock` by injection instead of calling
:mod:`time` directly. That single indentation makes those components testable
without ``sleep`` and keeps the test suite fast and deterministic.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """A source of time and delay."""

    def monotonic(self) -> float:
        """Return a monotonically increasing time in seconds."""
        ...

    def now(self) -> datetime:
        """Return the current UTC wall-clock time."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the current task for ``seconds``."""
        ...


class SystemClock:
    """The real clock, backed by :mod:`time` and :mod:`asyncio`."""

    def monotonic(self) -> float:
        """Return :func:`time.monotonic`."""
        return time.monotonic()

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(tz=UTC)

    async def sleep(self, seconds: float) -> None:
        """Await :func:`asyncio.sleep`."""
        if seconds > 0:
            await asyncio.sleep(seconds)


class ManualClock:
    """A controllable clock for tests.

    ``sleep`` advances the virtual time instantly and records the requested
    delay, so a test can assert on backoff behaviour without waiting.
    """

    def __init__(self, start: float = 0.0) -> None:
        """Initialise the clock at ``start`` seconds."""
        self._monotonic = start
        self._slept: list[float] = []

    @property
    def slept(self) -> tuple[float, ...]:
        """Return every delay requested so far, in order."""
        return tuple(self._slept)

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self._monotonic += seconds

    def monotonic(self) -> float:
        """Return the virtual monotonic time."""
        return self._monotonic

    def now(self) -> datetime:
        """Return the epoch offset by the virtual time."""
        return datetime.fromtimestamp(self._monotonic, tz=UTC)

    async def sleep(self, seconds: float) -> None:
        """Record the delay, advance virtual time, and yield to the event loop.

        The yield matters. A background loop shaped like
        ``while True: await clock.sleep(interval)`` would otherwise never give
        control back, and a test that drives such a loop would hang rather than
        run fast. Virtual time is instant; cooperation still is not optional.
        """
        self._slept.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)
