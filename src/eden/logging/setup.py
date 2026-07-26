"""Central logging configuration.

:func:`configure_logging` is the only place in EDEN that touches handlers on the
root logger. Every module obtains its logger through :func:`get_logger` and
never configures anything itself, which keeps behaviour predictable when EDEN is
embedded inside a host application.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Final

from eden.config.enums import LogFormat
from eden.config.schema import LoggingConfig, PathsConfig
from eden.logging.formatters import ConsoleFormatter, JsonFormatter, RedactionFilter

ROOT_LOGGER_NAME: Final[str] = "eden"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return the EDEN logger for ``name``.

    Args:
        name: Usually ``__name__``. A leading ``eden.`` is added when absent so
            that every EDEN record sits under one configurable root.

    Returns:
        A standard library logger.

    Example:
        >>> get_logger("eden.gateway").name
        'eden.gateway'
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    config: LoggingConfig,
    paths: PathsConfig | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Install EDEN's handlers on the ``eden`` logger.

    Calling this twice is a no-op unless ``force`` is set, so importing a module
    that configures logging cannot clobber a host application's setup.

    Args:
        config: Logging behaviour to apply.
        paths: Filesystem layout, required when ``config.to_file`` is set.
        force: Reconfigure even if setup already ran.

    Returns:
        The configured ``eden`` root logger.

    Raises:
        OSError: If the log directory cannot be created.
    """
    global _configured  # noqa: PLW0603 - single, deliberate module-level latch
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter = (
        JsonFormatter() if config.format is LogFormat.JSON else ConsoleFormatter()
    )
    redaction = RedactionFilter(config.redact_keys)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(redaction)
    root.addHandler(stream_handler)

    if config.to_file:
        if paths is None:
            message = "File logging requires a PathsConfig."
            raise ValueError(message)
        paths.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=paths.log_dir / config.file_name,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction)
        root.addHandler(file_handler)

    root.setLevel(config.level.value)
    root.propagate = False
    _configured = True
    return root


def reset_logging() -> None:
    """Detach every EDEN handler and clear the configured latch.

    Intended for test teardown and for hosts that reconfigure logging at runtime.
    """
    global _configured  # noqa: PLW0603 - paired with configure_logging
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    _configured = False
