"""Core kernel primitives.

This package holds the composition root, the dependency-injection container,
the generic registry, the structural contracts every subsystem implements, and
the provider-neutral domain types that flow between them.
"""

from __future__ import annotations

from eden.core.container import Container, Scope
from eden.core.interfaces import (
    HealthReport,
    Lifecycle,
    LLMProvider,
    ProviderSelector,
)
from eden.core.kernel import EdenKernel
from eden.core.registry import Registry
from eden.core.types import ChatRequest, ChatResponse, Message, StreamChunk, Usage

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Container",
    "EdenKernel",
    "HealthReport",
    "LLMProvider",
    "Lifecycle",
    "Message",
    "ProviderSelector",
    "Registry",
    "Scope",
    "StreamChunk",
    "Usage",
]
