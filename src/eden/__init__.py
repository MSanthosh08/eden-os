"""EDEN — a modular AI Operating System.

EDEN is organised in strict dependency layers. Each layer may import from the
layers beneath it and never from those above::

    config  ->  utils  ->  logging  ->  errors  ->  core  ->  transport
            ->  gateway  ->  (memory, agents, execution, hardware, ...)

The public entry point is :class:`~eden.core.kernel.EdenKernel`, which is the
composition root: it reads configuration, wires every subsystem through the
dependency-injection container, and owns ordered startup and shutdown.

Example:
    >>> import asyncio
    >>> from eden import EdenConfig, EdenKernel
    >>> async def main() -> str:
    ...     async with EdenKernel(EdenConfig()).session() as kernel:
    ...         return kernel.config.app_name
    >>> asyncio.run(main())
    'eden'
"""

from __future__ import annotations

from eden.config import ConfigLoader, EdenConfig, load_config
from eden.core.container import Container, Scope
from eden.core.kernel import EdenKernel
from eden.core.types import ChatRequest, ChatResponse, Message, StreamChunk, Usage
from eden.errors import EdenError
from eden.logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ConfigLoader",
    "Container",
    "EdenConfig",
    "EdenError",
    "EdenKernel",
    "Message",
    "Scope",
    "StreamChunk",
    "Usage",
    "__version__",
    "configure_logging",
    "get_logger",
    "load_config",
]
