"""Dependency-injection container.

Components declare what they need in their constructor and are wired together
here, at the composition root. No module ever reaches out to fetch a
collaborator, which is what keeps subsystems independently testable and keeps
import cycles structurally impossible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum, unique
from typing import Any, TypeVar, cast

from eden.errors import DependencyResolutionError

T = TypeVar("T")


@unique
class Scope(StrEnum):
    """Lifetime of a registered binding."""

    SINGLETON = "singleton"
    TRANSIENT = "transient"


class _Binding:
    """Internal record describing how to build one dependency."""

    __slots__ = ("factory", "instance", "scope")

    def __init__(self, factory: Callable[[Container], Any], scope: Scope) -> None:
        """Store the factory and its lifetime."""
        self.factory = factory
        self.scope = scope
        self.instance: Any | None = None


class Container:
    """Resolves dependencies by type, with singleton and transient lifetimes.

    Example:
        >>> class Engine:
        ...     pass
        >>> container = Container()
        >>> container.register(Engine, lambda c: Engine())
        >>> isinstance(container.resolve(Engine), Engine)
        True
    """

    def __init__(self) -> None:
        """Create an empty container."""
        self._bindings: dict[str, _Binding] = {}
        self._resolving: list[str] = []

    @staticmethod
    def _key(target: type[Any], name: str | None) -> str:
        """Return the internal lookup key for a type and optional qualifier."""
        base = f"{target.__module__}.{target.__qualname__}"
        return base if name is None else f"{base}#{name}"

    def register(
        self,
        target: type[T],
        factory: Callable[[Container], T],
        *,
        scope: Scope = Scope.SINGLETON,
        name: str | None = None,
        replace: bool = False,
    ) -> None:
        """Bind ``target`` to a factory.

        Args:
            target: Type or protocol used as the lookup key.
            factory: Callable receiving the container and returning an instance.
            scope: Lifetime of instances produced by the factory.
            name: Optional qualifier allowing several bindings per type.
            replace: Permit overwriting an existing binding.

        Raises:
            DependencyResolutionError: If the binding already exists and
                ``replace`` is ``False``.
        """
        key = self._key(target, name)
        if key in self._bindings and not replace:
            raise DependencyResolutionError(
                "A binding for this type is already registered.",
                context={"binding": key},
            )
        self._bindings[key] = _Binding(factory, scope)

    def register_instance(
        self,
        target: type[T],
        instance: T,
        *,
        name: str | None = None,
        replace: bool = False,
    ) -> None:
        """Bind ``target`` to an already-constructed ``instance``."""
        self.register(
            target,
            lambda _: instance,
            scope=Scope.SINGLETON,
            name=name,
            replace=replace,
        )

    def has(self, target: type[Any], *, name: str | None = None) -> bool:
        """Return whether a binding exists for ``target``."""
        return self._key(target, name) in self._bindings

    def resolve(self, target: type[T], *, name: str | None = None) -> T:
        """Return an instance satisfying ``target``.

        Args:
            target: Type or protocol to resolve.
            name: Optional qualifier used at registration time.

        Returns:
            The resolved instance, cached when the binding is a singleton.

        Raises:
            DependencyResolutionError: If no binding exists, a cycle is
                detected, or the factory itself fails.
        """
        key = self._key(target, name)
        binding = self._bindings.get(key)
        if binding is None:
            raise DependencyResolutionError(
                "No binding is registered for this type.",
                context={"binding": key, "known": sorted(self._bindings)},
            )
        if binding.scope is Scope.SINGLETON and binding.instance is not None:
            return cast("T", binding.instance)

        if key in self._resolving:
            raise DependencyResolutionError(
                "Circular dependency detected.",
                context={"chain": [*self._resolving, key]},
            )

        with self._tracking(key):
            try:
                instance = binding.factory(self)
            except DependencyResolutionError:
                raise
            except Exception as exc:
                raise DependencyResolutionError(
                    "Factory raised while constructing a dependency.",
                    context={"binding": key},
                    cause=exc,
                ) from exc

        if binding.scope is Scope.SINGLETON:
            binding.instance = instance
        return cast("T", instance)

    @contextmanager
    def _tracking(self, key: str) -> Iterator[None]:
        """Push ``key`` onto the resolution stack for cycle detection."""
        self._resolving.append(key)
        try:
            yield
        finally:
            self._resolving.pop()

    def singletons(self) -> tuple[Any, ...]:
        """Return every instantiated singleton, in registration order."""
        return tuple(
            binding.instance
            for binding in self._bindings.values()
            if binding.scope is Scope.SINGLETON and binding.instance is not None
        )

    def clear(self) -> None:
        """Drop every binding and cached instance."""
        self._bindings.clear()
        self._resolving.clear()
