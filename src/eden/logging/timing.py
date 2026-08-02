"""Execution-time instrumentation.

Every module is required to log how long its work took. Rather than scattering
``time.perf_counter()`` calls, a single decorator and context manager are
provided here; both emit a structured ``duration_ms`` field and both record the
duration on the failure path as well as the success path.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, cast

from eden.errors import EdenError

P = ParamSpec("P")
R = TypeVar("R")

_MS_PER_SECOND = 1000.0


class Stopwatch:
    """Measures elapsed wall-clock time using a monotonic source."""

    __slots__ = ("_elapsed_ms", "_start")

    def __init__(self) -> None:
        """Create a stopped stopwatch."""
        self._start: float | None = None
        self._elapsed_ms: float = 0.0

    def start(self) -> Stopwatch:
        """Start or restart the stopwatch and return it."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the stopwatch and return the elapsed milliseconds."""
        if self._start is not None:
            self._elapsed_ms = (time.perf_counter() - self._start) * _MS_PER_SECOND
            self._start = None
        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds, whether running or stopped."""
        if self._start is None:
            return self._elapsed_ms
        return (time.perf_counter() - self._start) * _MS_PER_SECOND

    def __enter__(self) -> Stopwatch:
        """Start timing on block entry."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop timing on block exit."""
        self.stop()


@contextmanager
def timed_block(
    logger: logging.Logger,
    operation: str,
    *,
    level: int = logging.DEBUG,
    **fields: Any,  # noqa: ANN401 - arbitrary structured fields
) -> Iterator[Stopwatch]:
    """Log the duration of the enclosed block.

    Args:
        logger: Logger that receives the record.
        operation: Short name of the work being measured.
        level: Severity used on the success path. Failures log at ``ERROR``.
        **fields: Extra structured fields attached to the record.

    Yields:
        The running stopwatch, in case the caller wants the duration itself.

    Example:
        >>> import logging
        >>> with timed_block(logging.getLogger("eden.demo"), "warmup") as watch:
        ...     pass
        >>> watch.elapsed_ms >= 0
        True
    """
    watch = Stopwatch().start()
    try:
        yield watch
    except Exception as exc:
        duration = watch.stop()
        # Only pass exc_info when we actually want a traceback attached.
        # Stdlib logging converts a *truthy* exc_info into a proper
        # (type, value, tb) tuple, but leaves a falsy value — e.g. False —
        # stored verbatim on the record. A formatter that then checks
        # `record.exc_info is not None` sees False and tries to subscript it.
        include_traceback = not isinstance(exc, EdenError)
        logger.error(
            "Operation failed.",
            extra=_fields(operation, duration, outcome="error", error=exc, extra=fields),
            **({"exc_info": True} if include_traceback else {}),
        )
        raise
    else:
        duration = watch.stop()
        logger.log(
            level,
            "Operation completed.",
            extra=_fields(operation, duration, outcome="success", error=None, extra=fields),
        )


def _fields(
    operation: str,
    duration_ms: float,
    *,
    outcome: str,
    error: BaseException | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the structured payload for a timing record."""
    payload: dict[str, Any] = {
        "operation": operation,
        "duration_ms": round(duration_ms, 3),
        "outcome": outcome,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        if isinstance(error, EdenError):
            payload["error_code"] = error.code
    payload.update(extra)
    return payload


def timed(
    logger: logging.Logger,
    operation: str | None = None,
    *,
    level: int = logging.DEBUG,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a callable so that its execution time is logged.

    Works transparently on both ``def`` and ``async def`` functions.

    Args:
        logger: Logger that receives the record.
        operation: Override for the logged operation name. Defaults to the
            qualified function name.
        level: Severity used on the success path.

    Returns:
        A decorator preserving the wrapped signature.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        name = operation or func.__qualname__

        if inspect.iscoroutinefunction(func):
            async_func = cast("Callable[P, Coroutine[Any, Any, Any]]", func)

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:  # noqa: ANN401
                with timed_block(logger, name, level=level):
                    return await async_func(*args, **kwargs)

            return cast("Callable[P, R]", async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with timed_block(logger, name, level=level):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator
