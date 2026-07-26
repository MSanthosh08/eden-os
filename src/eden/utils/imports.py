"""Dynamic import helpers.

Configuration names implementations by import path — ``"my_pkg.drivers:Arm"`` —
which is what allows a third party to add a provider, agent or hardware driver
without EDEN's source ever mentioning it. Loading is lazy, so an optional
dependency only costs anything if it is actually used.
"""

from __future__ import annotations

import importlib
from typing import Any

from eden.errors import PluginLoadError

_PATH_SEPARATOR = ":"


def import_object(path: str) -> Any:  # noqa: ANN401 - the caller narrows the type
    """Import and return the object named by ``path``.

    Args:
        path: Either ``"package.module:attribute"`` or ``"package.module.attribute"``.

    Returns:
        The imported attribute.

    Raises:
        PluginLoadError: If the path is malformed, the module is missing or the
            attribute does not exist.

    Example:
        >>> import_object("math:sqrt")(9.0)
        3.0
    """
    cleaned = path.strip()
    if not cleaned:
        raise PluginLoadError("Import path must not be empty.", context={"path": path})

    if _PATH_SEPARATOR in cleaned:
        module_name, _, attribute = cleaned.partition(_PATH_SEPARATOR)
    else:
        module_name, _, attribute = cleaned.rpartition(".")
    if not module_name or not attribute:
        raise PluginLoadError(
            "Import path must reference a module and an attribute.",
            context={"path": path},
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PluginLoadError(
            "Module could not be imported.",
            context={"path": path, "module": module_name},
            cause=exc,
        ) from exc

    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise PluginLoadError(
            "Module does not define the requested attribute.",
            context={"path": path, "module": module_name, "attribute": attribute},
            cause=exc,
        ) from exc


def import_subclass[T](path: str, expected: type[T]) -> type[T]:
    """Import a class and assert that it derives from ``expected``.

    Args:
        path: Import path of the class.
        expected: Base class the target must implement.

    Returns:
        The imported class.

    Raises:
        PluginLoadError: If the target is not a class, or not a subclass.
    """
    target = import_object(path)
    if not isinstance(target, type):
        raise PluginLoadError(
            "Import path does not reference a class.",
            context={"path": path, "actual_type": type(target).__name__},
        )
    if not issubclass(target, expected):
        raise PluginLoadError(
            "Imported class does not implement the required base class.",
            context={"path": path, "expected": expected.__name__},
        )
    return target
