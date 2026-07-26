"""Secret handling primitives.

EDEN never stores credentials in configuration files. Configuration declares
the *name of the environment variable* that holds a credential; the value is
resolved lazily at use time and wrapped in :class:`SecretStr`, whose ``repr``
and ``str`` are redacted so that an accidental log or traceback cannot leak it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

from eden.errors import SecretResolutionError

REDACTED: Final[str] = "***redacted***"


class SecretStr:
    """A string whose value is hidden from ``repr``, ``str`` and formatting.

    The plaintext is only reachable through :meth:`reveal`, which makes leaks
    greppable in code review.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        """Wrap ``value`` so that it is never rendered accidentally."""
        self._value = value

    def reveal(self) -> str:
        """Return the plaintext secret.

        Returns:
            The wrapped value. Call sites should pass the result straight to a
            transport header and never store or log it.
        """
        return self._value

    def __bool__(self) -> bool:
        """Return ``True`` when a non-empty secret is present."""
        return bool(self._value)

    def __len__(self) -> int:
        """Return the length of the secret without revealing it."""
        return len(self._value)

    def __eq__(self, other: object) -> bool:
        """Compare two secrets by value in constant-ish time."""
        if not isinstance(other, SecretStr):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        """Hash by the wrapped value so secrets can key a dictionary."""
        return hash(self._value)

    def __str__(self) -> str:
        """Return the redaction marker instead of the secret."""
        return REDACTED

    def __repr__(self) -> str:
        """Return the redaction marker instead of the secret."""
        return f"SecretStr({REDACTED})"


class SecretResolver:
    """Resolves secret references against an environment mapping.

    The mapping is injected rather than read from :data:`os.environ` directly,
    which keeps the resolver deterministic under test.
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        """Initialise the resolver.

        Args:
            environ: Source of environment variables. Defaults to the process
                environment.
        """
        self._environ: Mapping[str, str] = os.environ if environ is None else environ

    def resolve(self, variable_name: str, *, required: bool = True) -> SecretStr | None:
        """Resolve the secret stored in ``variable_name``.

        Args:
            variable_name: Name of the environment variable to read.
            required: When ``True`` a missing or empty variable is an error.

        Returns:
            The wrapped secret, or ``None`` when absent and not required.

        Raises:
            SecretResolutionError: If the variable is required but unavailable.
        """
        raw = self._environ.get(variable_name, "").strip()
        if raw:
            return SecretStr(raw)
        if required:
            raise SecretResolutionError(
                "Required credential is not present in the environment.",
                context={"variable": variable_name},
            )
        return None

    def has(self, variable_name: str) -> bool:
        """Return whether ``variable_name`` holds a non-empty value."""
        return bool(self._environ.get(variable_name, "").strip())
