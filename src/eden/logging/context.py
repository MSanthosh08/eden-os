"""Log context propagation.

A correlation identifier is attached to every log record emitted while handling
one request, which is what makes a distributed trace readable once EDEN spreads
across processes. :class:`contextvars.ContextVar` is used rather than a global
so the value follows the async task, not the thread.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from eden.utils.ids import new_id

_UNSET = "-"

_correlation_id: ContextVar[str] = ContextVar("eden_correlation_id", default=_UNSET)
_component: ContextVar[str] = ContextVar("eden_component", default=_UNSET)


def get_correlation_id() -> str:
    """Return the correlation id bound to the current context."""
    return _correlation_id.get()


def get_component() -> str:
    """Return the component name bound to the current context."""
    return _component.get()


@contextmanager
def correlation_scope(correlation_id: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of the block.

    Args:
        correlation_id: Identifier to bind. A new one is generated when omitted.

    Yields:
        The bound correlation id.

    Example:
        >>> with correlation_scope("req-1") as cid:
        ...     get_correlation_id() == cid
        True
    """
    value = correlation_id or new_id("req")
    token: Token[str] = _correlation_id.set(value)
    try:
        yield value
    finally:
        _correlation_id.reset(token)


@contextmanager
def component_scope(component: str) -> Iterator[str]:
    """Bind a component name for the duration of the block.

    Args:
        component: Logical subsystem name, e.g. ``"gateway"``.

    Yields:
        The bound component name.
    """
    token: Token[str] = _component.set(component)
    try:
        yield component
    finally:
        _component.reset(token)
