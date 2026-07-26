"""Configuration sources.

A source produces a plain nested ``dict``. The loader merges sources in
precedence order and only then constructs the frozen schema objects. Keeping
*acquisition* separate from *validation* means a new source (Consul, Vault, a
database) is an additive change that touches nothing else.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from eden.errors import ConfigurationError

ENV_PREFIX = "EDEN__"
ENV_NESTING_SEPARATOR = "__"


@runtime_checkable
class ConfigSource(Protocol):
    """A provider of raw configuration fragments."""

    @property
    def name(self) -> str:
        """Return a short identifier used in diagnostics."""
        ...

    def load(self) -> dict[str, Any]:
        """Return this source's contribution as a nested mapping."""
        ...


class MappingSource:
    """Wraps an in-memory mapping, used for defaults and runtime overrides."""

    def __init__(self, data: Mapping[str, Any], *, name: str = "mapping") -> None:
        """Initialise the source.

        Args:
            data: Nested configuration fragment.
            name: Identifier reported in diagnostics.
        """
        self._data = _deep_copy(data)
        self._name = name

    @property
    def name(self) -> str:
        """Return the source identifier."""
        return self._name

    def load(self) -> dict[str, Any]:
        """Return a defensive copy of the wrapped mapping."""
        return _deep_copy(self._data)


class TomlFileSource:
    """Reads a TOML file from disk.

    A missing file is not an error when ``required`` is ``False``; that is the
    normal case for an operator who configures everything through environment
    variables.
    """

    def __init__(self, path: Path, *, required: bool = False) -> None:
        """Initialise the source.

        Args:
            path: Location of the TOML document.
            required: Whether absence should raise.
        """
        self._path = path
        self._required = required

    @property
    def name(self) -> str:
        """Return the source identifier including the file path."""
        return f"toml:{self._path}"

    def load(self) -> dict[str, Any]:
        """Parse the TOML document.

        Returns:
            The parsed mapping, or an empty mapping when optional and absent.

        Raises:
            ConfigurationError: If the file is required and missing, or malformed.
        """
        if not self._path.is_file():
            if self._required:
                raise ConfigurationError(
                    "Required configuration file was not found.",
                    context={"path": str(self._path)},
                )
            return {}
        try:
            with self._path.open("rb") as handle:
                return tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigurationError(
                "Configuration file is not valid TOML.",
                context={"path": str(self._path)},
                cause=exc,
            ) from exc
        except OSError as exc:
            raise ConfigurationError(
                "Configuration file could not be read.",
                context={"path": str(self._path)},
                cause=exc,
            ) from exc


class EnvironmentSource:
    """Projects ``EDEN__``-prefixed environment variables into a nested mapping.

    ``EDEN__LOGGING__LEVEL=DEBUG`` becomes ``{"logging": {"level": "DEBUG"}}``.
    Values stay as strings; the loader coerces them against the schema, so the
    environment never needs to encode types.
    """

    def __init__(self, environ: Mapping[str, str], *, prefix: str = ENV_PREFIX) -> None:
        """Initialise the source.

        Args:
            environ: Environment mapping to read.
            prefix: Variable prefix that marks EDEN settings.
        """
        self._environ = environ
        self._prefix = prefix

    @property
    def name(self) -> str:
        """Return the source identifier."""
        return f"env:{self._prefix}*"

    def load(self) -> dict[str, Any]:
        """Return the nested mapping described by the environment.

        Raises:
            ConfigurationError: If a variable name collides with a nested branch.
        """
        result: dict[str, Any] = {}
        for raw_key, raw_value in sorted(self._environ.items()):
            if not raw_key.startswith(self._prefix):
                continue
            path = [
                part.lower()
                for part in raw_key[len(self._prefix) :].split(ENV_NESTING_SEPARATOR)
                if part
            ]
            if not path:
                continue
            self._assign(result, path, raw_value, raw_key)
        return result

    @staticmethod
    def _assign(target: dict[str, Any], path: list[str], value: str, origin: str) -> None:
        """Write ``value`` into ``target`` at the nested ``path``.

        Raises:
            ConfigurationError: If an intermediate key is already a scalar.
        """
        cursor = target
        for part in path[:-1]:
            existing = cursor.get(part)
            if existing is None:
                nested: dict[str, Any] = {}
                cursor[part] = nested
                cursor = nested
            elif isinstance(existing, dict):
                cursor = existing
            else:
                raise ConfigurationError(
                    "Environment variable conflicts with an existing scalar key.",
                    context={"variable": origin, "conflict_at": part},
                )
        cursor[path[-1]] = value


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` on top of ``base``.

    Mappings merge key-by-key; every other type — including lists — is replaced
    wholesale, so an operator can override a provider catalogue without having
    to reason about element-wise merge semantics.

    Args:
        base: Lower-precedence mapping.
        overlay: Higher-precedence mapping.

    Returns:
        A new merged mapping. Neither argument is mutated.
    """
    merged: dict[str, Any] = _deep_copy(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = _deep_copy_value(value)
    return merged


def _deep_copy(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a nested mapping."""
    return {key: _deep_copy_value(value) for key, value in data.items()}


def _deep_copy_value(value: Any) -> Any:  # noqa: ANN401 - arbitrary config payloads
    """Return a deep copy of an arbitrary configuration value."""
    if isinstance(value, Mapping):
        return _deep_copy(value)
    if isinstance(value, list | tuple):
        return [_deep_copy_value(item) for item in value]
    return value
