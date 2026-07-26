"""Identifier generation.

Identifiers are short, prefixed and sortable-enough for logs. They are not
security tokens; nothing in EDEN may use them for authorisation.
"""

from __future__ import annotations

import uuid

_DEFAULT_LENGTH = 12


def new_id(prefix: str = "", *, length: int = _DEFAULT_LENGTH) -> str:
    """Return a new random identifier.

    Args:
        prefix: Optional short tag prepended with a hyphen, e.g. ``"req"``.
        length: Number of hexadecimal characters after the prefix.

    Returns:
        An identifier such as ``"req-8f21c0a9b3d4"``.

    Example:
        >>> value = new_id("req")
        >>> value.startswith("req-")
        True
    """
    body = uuid.uuid4().hex[:length]
    return f"{prefix}-{body}" if prefix else body
