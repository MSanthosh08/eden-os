"""Structural contracts.

These are :class:`typing.Protocol` definitions rather than abstract base
classes, so a component satisfies a contract by *shape*. That keeps the
dependency arrow pointing one way — high-level policy depends on these
abstractions, and concrete adapters depend on them too, but never on each other.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from eden.config.enums import Capability, HealthState, PrivacyTier
from eden.core.types import ChatRequest, ChatResponse, StreamChunk


@runtime_checkable
class Lifecycle(Protocol):
    """A component the kernel starts and stops."""

    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        ...

    async def start(self) -> None:
        """Acquire resources. Must be idempotent."""
        ...

    async def stop(self) -> None:
        """Release resources. Must be idempotent and must not raise."""
        ...


@runtime_checkable
class HealthReport(Protocol):
    """A snapshot of a component's health."""

    @property
    def state(self) -> HealthState:
        """Return the observed health state."""
        ...

    @property
    def detail(self) -> str:
        """Return a short human-readable explanation."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """An adapter to one AI model provider.

    Implementations translate between EDEN's neutral types and a vendor's wire
    format. They never make routing decisions and never read configuration
    themselves; both are supplied by the layer above.
    """

    @property
    def name(self) -> str:
        """Return the logical provider name from configuration."""
        ...

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Return every capability this provider can serve."""
        ...

    @property
    def privacy_tier(self) -> PrivacyTier:
        """Return the data-residency guarantee of this provider."""
        ...

    def supports(self, request: ChatRequest) -> bool:
        """Return whether this provider can serve ``request`` at all."""
        ...

    def estimate_cost(self, request: ChatRequest) -> float:
        """Return the estimated spend for ``request``."""
        ...

    def expected_latency_ms(self, request: ChatRequest) -> float:
        """Return the expected wall-clock latency for ``request``."""
        ...

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Perform a single non-streamed generation."""
        ...

    def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Perform a streamed generation."""
        ...

    async def health_check(self) -> HealthState:
        """Probe the provider and return its observed health."""
        ...

    async def aclose(self) -> None:
        """Release any transport resources held by this provider."""
        ...


@runtime_checkable
class ProviderSelector(Protocol):
    """Chooses which providers should serve a request, in preference order."""

    def select(self, request: ChatRequest, providers: Sequence[LLMProvider]) -> list[LLMProvider]:
        """Return an ordered shortlist of candidates for ``request``.

        Returns:
            Candidates ordered best-first. An empty list means nothing is
            eligible, which the caller must translate into a clear error.
        """
        ...
