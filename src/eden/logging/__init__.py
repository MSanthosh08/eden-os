"""Central logging subsystem.

Modules obtain a logger with :func:`get_logger` and never configure handlers
themselves. The kernel calls :func:`configure_logging` exactly once at startup.
"""

from __future__ import annotations

from eden.logging.context import (
    component_scope,
    correlation_scope,
    get_component,
    get_correlation_id,
)
from eden.logging.formatters import ConsoleFormatter, JsonFormatter, RedactionFilter
from eden.logging.setup import ROOT_LOGGER_NAME, configure_logging, get_logger, reset_logging
from eden.logging.timing import Stopwatch, timed, timed_block

__all__ = [
    "ROOT_LOGGER_NAME",
    "ConsoleFormatter",
    "JsonFormatter",
    "RedactionFilter",
    "Stopwatch",
    "component_scope",
    "configure_logging",
    "correlation_scope",
    "get_component",
    "get_correlation_id",
    "get_logger",
    "reset_logging",
    "timed",
    "timed_block",
]
