"""Generic name-to-factory registry.

Providers, agents, hardware drivers and automations all need "look up an
implementation by a name that came from configuration". That pattern is written
once here and parameterised, rather than reimplemented per subsystem.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from eden.errors import RegistryError


class Registry[T]:
    """An ordered, case-insensitive registry of named factories.

    Example:
        >>> registry: Registry[int] = Registry("demo")
        >>> registry.register("answer", lambda: 42)
        >>> registry.create("answer")
        42
    """

    def __init__(self, label: str) -> None:
        """Initialise an empty registry.

        Args:
            label: Human-readable name used in error messages, e.g. ``"provider"``.
        """
        self._label = label
        self._factories: dict[str, Callable[[], T]] = {}

    @property
    def label(self) -> str:
        """Return the registry label."""
        return self._label

    def register(
        self,
        name: str,
        factory: Callable[[], T],
        *,
        replace: bool = False,
    ) -> None:
        """Register ``factory`` under ``name``.

        Args:
            name: Lookup key. Compared case-insensitively.
            factory: Zero-argument callable producing the instance.
            replace: Permit overwriting an existing entry.

        Raises:
            RegistryError: If the name is empty, or already taken and
                ``replace`` is ``False``.
        """
        key = name.strip().lower()
        if not key:
            raise RegistryError(
                "Registry keys must not be empty.",
                context={"registry": self._label},
            )
        if key in self._factories and not replace:
            raise RegistryError(
                "An entry with this name is already registered.",
                context={"registry": self._label, "name": name},
            )
        self._factories[key] = factory

    def unregister(self, name: str) -> None:
        """Remove ``name`` from the registry.

        Raises:
            RegistryError: If the name is not registered.
        """
        key = name.strip().lower()
        if key not in self._factories:
            raise RegistryError(
                "No entry is registered under this name.",
                context={"registry": self._label, "name": name},
            )
        del self._factories[key]

    def create(self, name: str) -> T:
        """Instantiate the entry registered under ``name``.

        Raises:
            RegistryError: If the name is unknown.
        """
        key = name.strip().lower()
        factory = self._factories.get(key)
        if factory is None:
            raise RegistryError(
                "No entry is registered under this name.",
                context={
                    "registry": self._label,
                    "name": name,
                    "known": sorted(self._factories),
                },
            )
        return factory()

    def create_all(self) -> Mapping[str, T]:
        """Instantiate every registered entry, preserving insertion order."""
        return {name: factory() for name, factory in self._factories.items()}

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is registered."""
        return isinstance(name, str) and name.strip().lower() in self._factories

    def __len__(self) -> int:
        """Return the number of registered entries."""
        return len(self._factories)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names in insertion order."""
        return iter(self._factories)

    def names(self) -> tuple[str, ...]:
        """Return every registered name in insertion order."""
        return tuple(self._factories)
