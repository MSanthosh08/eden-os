"""Log record rendering and sanitisation.

Two formatters are provided: a compact human-readable one for a developer's
terminal and a single-line JSON one for log aggregators. Both draw structured
fields from the ``extra`` mapping supplied at the call site, and both run behind
:class:`RedactionFilter`, which guarantees that credentials never reach a sink.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Final

from eden.config.secrets import REDACTED
from eden.errors import EdenError
from eden.logging.context import get_component, get_correlation_id
from eden.utils.redaction import (
    DEFAULT_SENSITIVE_KEYS,
    is_sensitive_key,
    redact_text,
    redact_value,
)

EXTRA_FIELD: Final[str] = "eden_fields"

_RESERVED_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class RedactionFilter(logging.Filter):
    """Strips credentials from the message and structured fields of a record.

    Implemented as a filter rather than inside a formatter so that it applies
    once, regardless of how many handlers are attached.
    """

    def __init__(self, sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS) -> None:
        """Initialise the filter.

        Args:
            sensitive_keys: Field names whose values must be masked.
        """
        super().__init__(name="eden.redaction")
        self._sensitive_keys = frozenset(key.lower() for key in sensitive_keys)

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitise ``record`` in place and always allow it through."""
        try:
            record.msg = redact_text(str(record.msg))
            if record.args:
                record.msg = record.getMessage()
                record.args = ()
                record.msg = redact_text(record.msg)
            for key, value in list(record.__dict__.items()):
                if key in _RESERVED_ATTRIBUTES or key.startswith("_"):
                    continue
                # A sensitive name at the top level matters as much as one
                # nested inside a mapping: `extra={"api_key": ...}` lands here
                # as a plain record attribute.
                if is_sensitive_key(key, self._sensitive_keys):
                    record.__dict__[key] = REDACTED
                    continue
                record.__dict__[key] = redact_value(value, sensitive_keys=self._sensitive_keys)
        except Exception:  # noqa: BLE001 - a logging failure must never propagate
            record.msg = "<redaction failed; message suppressed>"
            record.args = ()
        return True


def _structured_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Extract user-supplied structured fields from a record."""
    fields: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _RESERVED_ATTRIBUTES or key.startswith("_"):
            continue
        if key == EXTRA_FIELD and isinstance(value, dict):
            fields.update(value)
            continue
        fields[key] = value
    return fields


class JsonFormatter(logging.Formatter):
    """Renders each record as one JSON object, suitable for log shipping."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a single-line JSON document."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "component": get_component(),
        }
        payload.update(_structured_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
            error = record.exc_info[1]
            if isinstance(error, EdenError):
                payload["error"] = error.to_dict()
        try:
            return json.dumps(payload, default=str, separators=(",", ":"))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return json.dumps({"level": record.levelname, "message": "<unserialisable log record>"})


class ConsoleFormatter(logging.Formatter):
    """Renders a compact, aligned line for interactive terminals."""

    _TEMPLATE = "{timestamp} {level:<8} {logger:<28} {correlation} {message}{fields}"

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a human-readable line."""
        fields = _structured_fields(record)
        rendered_fields = ""
        if fields:
            rendered_fields = " | " + " ".join(f"{key}={value}" for key, value in fields.items())
        line = self._TEMPLATE.format(
            timestamp=datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S.%f")[:-3],
            level=record.levelname,
            logger=record.name,
            correlation=get_correlation_id(),
            message=record.getMessage(),
            fields=rendered_fields,
        )
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line
