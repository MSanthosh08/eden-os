"""Redaction helpers.

Two defences run in series: a *key* filter that hides values whose field name
looks sensitive, and a *pattern* filter that catches credentials pasted into
free text. Both are applied by the logging layer before a record is emitted, so
no call site has to remember to sanitise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

from eden.config.secrets import REDACTED, SecretStr

DEFAULT_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "token",
        "x-api-key",
    }
)

# Common vendor credential shapes: sk-..., gsk_..., AIza..., Bearer <token>.
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\b(?:gsk|xai|hf)_[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
)

_MAX_DEPTH = 6


def is_sensitive_key(key: str, sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS) -> bool:
    """Return whether ``key`` names a sensitive field.

    Args:
        key: Field name to test.
        sensitive_keys: Lower-case field names considered sensitive.

    Returns:
        ``True`` when the key contains any sensitive token.
    """
    lowered = key.lower()
    return any(token in lowered for token in sensitive_keys)


def redact_text(text: str) -> str:
    """Replace credential-shaped substrings in ``text`` with a marker.

    Args:
        text: Arbitrary free text, typically a log message.

    Returns:
        The text with recognisable credentials masked.

    Example:
        >>> redact_text("using sk-abcdefghijklmnopqrstuvwx now")
        'using ***redacted*** now'
    """
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result


def redact_value(
    value: Any,  # noqa: ANN401 - arbitrary log payload
    *,
    sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS,
    _depth: int = 0,
) -> Any:  # noqa: ANN401 - mirrors the input shape
    """Return a redacted copy of an arbitrary structure.

    Mappings, sequences and :class:`~eden.config.secrets.SecretStr` values are
    handled; anything else is stringified only if it is not a primitive.

    Args:
        value: Structure to sanitise.
        sensitive_keys: Field names whose values must be masked.
        _depth: Internal recursion guard.

    Returns:
        A sanitised copy safe to serialise into a log record.
    """
    if isinstance(value, SecretStr):
        return REDACTED
    if _depth >= _MAX_DEPTH:
        return "<max-depth>"
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if is_sensitive_key(str(key), sensitive_keys)
                else redact_value(item, sensitive_keys=sensitive_keys, _depth=_depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [
            redact_value(item, sensitive_keys=sensitive_keys, _depth=_depth + 1) for item in value
        ]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bool | int | float) or value is None:
        return value
    return redact_text(str(value))
