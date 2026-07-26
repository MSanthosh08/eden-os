"""Reusable asynchronous primitives.

Retry, timeout and concurrency limiting are needed by every provider and will be
needed by hardware drivers and agents later. They live here once so no subsystem
grows its own subtly different version.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Iterable, Sequence

from eden.config.schema import RetryConfig
from eden.errors import EdenError
from eden.utils.clock import Clock, SystemClock

_JITTER_FLOOR = 0.0


def compute_backoff(
    attempt: int,
    policy: RetryConfig,
    *,
    rng: random.Random | None = None,
) -> float:
    """Return the delay in seconds before ``attempt``.

    Args:
        attempt: One-based index of the attempt about to be made. Attempt 1
            never waits.
        policy: Backoff parameters.
        rng: Random source. Injected so tests can pin the jitter.

    Returns:
        Seconds to wait, jittered and clamped to the policy ceiling.

    Example:
        >>> compute_backoff(1, RetryConfig())
        0.0
    """
    if attempt <= 1:
        return 0.0
    raw = policy.initial_backoff_seconds * (policy.multiplier ** (attempt - 2))
    capped = min(raw, policy.max_backoff_seconds)
    if policy.jitter_ratio <= 0:
        return capped
    source = rng or random
    spread = capped * policy.jitter_ratio
    jittered = capped - spread + (source.random() * 2 * spread)
    return max(_JITTER_FLOOR, min(jittered, policy.max_backoff_seconds))


def is_retryable(error: BaseException) -> bool:
    """Return whether ``error`` is worth retrying.

    EDEN errors declare this themselves; unknown exceptions are treated as
    non-retryable so that a genuine bug is not hidden behind repeated attempts.
    """
    if isinstance(error, EdenError):
        return error.retryable
    return isinstance(error, TimeoutError | ConnectionError)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryConfig,
    *,
    logger: logging.Logger,
    operation_name: str,
    clock: Clock | None = None,
    retryable: Callable[[BaseException], bool] = is_retryable,
    rng: random.Random | None = None,
) -> T:
    """Invoke ``operation`` with exponential backoff on retryable failures.

    Args:
        operation: Zero-argument coroutine factory. It is called afresh each
            attempt, so callers must not pass an already-awaited coroutine.
        policy: Backoff parameters.
        logger: Logger receiving one structured record per failed attempt.
        operation_name: Name recorded in log fields.
        clock: Time source. Defaults to the real clock.
        retryable: Predicate deciding whether a failure justifies another try.
        rng: Random source for jitter.

    Returns:
        The result of the first successful attempt.

    Raises:
        BaseException: The final failure, re-raised once attempts are exhausted.
    """
    active_clock = clock or SystemClock()
    last_error: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        delay = compute_backoff(attempt, policy, rng=rng)
        if delay > 0:
            await active_clock.sleep(delay)
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            exhausted = attempt >= policy.max_attempts
            if exhausted or not retryable(exc):
                raise
            logger.warning(
                "Attempt failed; retrying.",
                extra={
                    "operation": operation_name,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "error_type": type(exc).__name__,
                },
            )

    raise last_error if last_error is not None else RuntimeError("retry loop exited unexpectedly")


async def with_timeout[T](
    awaitable: Awaitable[T],
    seconds: float,
    *,
    on_timeout: Callable[[], BaseException] | None = None,
) -> T:
    """Await ``awaitable`` under a wall-clock budget.

    Args:
        awaitable: Work to perform.
        seconds: Budget in seconds.
        on_timeout: Factory producing the exception to raise instead of the
            built-in :class:`TimeoutError`.

    Returns:
        The awaited result.

    Raises:
        BaseException: ``on_timeout()`` if supplied, otherwise ``TimeoutError``.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except TimeoutError as exc:
        if on_timeout is not None:
            raise on_timeout() from exc
        raise


async def gather_limited[T](
    factories: Sequence[Callable[[], Awaitable[T]]],
    *,
    limit: int,
) -> list[T | BaseException]:
    """Run ``factories`` concurrently with a ceiling of ``limit`` in flight.

    Args:
        factories: Zero-argument coroutine factories.
        limit: Maximum simultaneous tasks. Must be positive.

    Returns:
        Results in input order; failures are returned rather than raised so a
        single bad member cannot abort the batch.

    Raises:
        ValueError: If ``limit`` is not positive.
    """
    if limit <= 0:
        message = "Concurrency limit must be positive."
        raise ValueError(message)
    semaphore = asyncio.Semaphore(limit)

    async def guarded(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    tasks: Iterable[Awaitable[T]] = [guarded(factory) for factory in factories]
    return list(await asyncio.gather(*tasks, return_exceptions=True))
