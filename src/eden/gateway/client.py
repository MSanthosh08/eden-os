"""The AI Gateway façade.

Everything above this line — agents, memory, automation, the interface layer —
talks to :class:`GatewayClient` and to nothing else in the gateway. That single
seam is what makes the promise "swap OpenAI for Ollama without touching business
logic" true rather than aspirational.

The client owns the *execution* half of routing: it walks the shortlist the
router produced, feeds every outcome back into the health tracker, and gives up
with a precise error rather than a generic one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from eden.config.enums import Capability, HealthState
from eden.config.schema import GatewayConfig
from eden.core.types import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
    StreamChunk,
)
from eden.errors import (
    EdenError,
    EmbeddingNotSupportedError,
    GatewayError,
    NoProviderAvailableError,
    ProviderError,
)
from eden.gateway.health import HealthTracker
from eden.gateway.provider import BaseProvider
from eden.gateway.router.omni_router import OmniRouter
from eden.logging import get_logger, timed_block
from eden.utils.async_tools import gather_limited

_LOGGER = get_logger("gateway.client")

COMPONENT_NAME = "ai-gateway"


@dataclass(frozen=True, slots=True)
class ProviderHealthSummary:
    """A redaction-safe health line for diagnostics.

    Attributes:
        provider: Logical provider name.
        state: Coarse health state.
        success_rate: Observed success ratio.
        latency_ms: Smoothed observed latency.
        circuit: Circuit-breaker state.
    """

    provider: str
    state: HealthState
    success_rate: float
    latency_ms: float
    circuit: str


class GatewayClient:
    """Executes generation requests across a fleet of providers.

    Example:
        >>> from eden.config.schema import GatewayConfig
        >>> from eden.gateway.health import HealthTracker
        >>> tracker = HealthTracker(GatewayConfig().router.circuit_breaker)
        >>> client = GatewayClient(GatewayConfig(), [], OmniRouter(
        ...     GatewayConfig().router, tracker), tracker)
        >>> client.component_name
        'ai-gateway'
    """

    def __init__(
        self,
        config: GatewayConfig,
        providers: Sequence[BaseProvider],
        router: OmniRouter,
        health: HealthTracker,
    ) -> None:
        """Initialise the client.

        Args:
            config: Gateway configuration.
            providers: Constructed provider adapters.
            router: Selection policy.
            health: Shared health tracker.
        """
        self._config = config
        self._providers = list(providers)
        self._router = router
        self._health = health
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return COMPONENT_NAME

    async def start(self) -> None:
        """Log the fleet composition. Idempotent."""
        if self._started:
            return
        self._started = True
        _LOGGER.info(
            "AI gateway started.",
            extra={
                "providers": [provider.name for provider in self._providers],
                "strategy": self._router.strategy_name,
            },
        )

    async def stop(self) -> None:
        """Close every provider. Idempotent and never raises."""
        if not self._started:
            return
        self._started = False
        for provider in self._providers:
            try:
                await provider.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                _LOGGER.warning(
                    "Provider failed to close cleanly.",
                    extra={"provider": provider.name, "error_type": type(exc).__name__},
                )
        _LOGGER.info("AI gateway stopped.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def providers(self) -> tuple[BaseProvider, ...]:
        """Return every registered provider."""
        return tuple(self._providers)

    def provider(self, name: str) -> BaseProvider | None:
        """Return the provider registered under ``name``, if any."""
        for provider in self._providers:
            if provider.name == name:
                return provider
        return None

    def health_summary(self) -> list[ProviderHealthSummary]:
        """Return a health line per provider for diagnostics endpoints."""
        summaries: list[ProviderHealthSummary] = []
        for provider in self._providers:
            record = self._health.record(provider.name)
            summaries.append(
                ProviderHealthSummary(
                    provider=provider.name,
                    state=record.state,
                    success_rate=round(record.success_rate, 4),
                    latency_ms=round(record.ewma_latency_ms, 2),
                    circuit=str(record.circuit),
                )
            )
        return summaries

    async def probe_all(self) -> dict[str, HealthState]:
        """Probe every provider concurrently and record the outcomes.

        Returns:
            Mapping of provider name to observed health state.
        """
        if not self._providers:
            return {}
        results = await gather_limited(
            [self._probe_factory(provider) for provider in self._providers],
            limit=min(len(self._providers), 8),
        )
        outcomes: dict[str, HealthState] = {}
        for provider, result in zip(self._providers, results, strict=True):
            state = result if isinstance(result, HealthState) else HealthState.UNHEALTHY
            outcomes[provider.name] = state
            if state is HealthState.HEALTHY:
                self._health.observe_success(provider.name, record_latency(provider))
            else:
                self._health.observe_failure(provider.name, error_code="health.probe")
        return outcomes

    @staticmethod
    def _probe_factory(provider: BaseProvider) -> _ProbeFactory:
        """Return a zero-argument coroutine factory probing ``provider``."""
        return _ProbeFactory(provider)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Serve ``request``, failing over across the router's shortlist.

        Args:
            request: The generation request.

        Returns:
            The first successful response, annotated with the attempt count.

        Raises:
            NoProviderAvailableError: If nothing is eligible, or every eligible
                provider failed. The final provider error is chained as the
                cause so the root problem is never lost.
        """
        decision = self._router.decide(request, self._providers)
        if not decision.has_candidates:
            raise NoProviderAvailableError(
                "No provider satisfies this request.",
                context={
                    "required_capabilities": sorted(
                        capability.value for capability in request.required_capabilities
                    ),
                    "excluded": decision.excluded,
                },
            )

        router_config = self._config.router
        limit = 1 + router_config.max_failovers if router_config.failover_enabled else 1
        shortlist = [scored.provider for scored in decision.ranked][:limit]

        last_error: EdenError | None = None
        with timed_block(_LOGGER, "gateway.chat", candidates=len(shortlist)):
            for attempt, provider in enumerate(shortlist, start=1):
                try:
                    response = await provider.chat(request)
                except ProviderError as exc:
                    last_error = exc
                    self._health.observe_failure(provider.name, error_code=exc.code)
                    _LOGGER.warning(
                        "Provider failed; considering failover.",
                        extra={
                            "provider": provider.name,
                            "attempt": attempt,
                            "remaining": len(shortlist) - attempt,
                            "error_code": exc.code,
                        },
                    )
                    continue
                except EdenError as exc:
                    last_error = exc
                    self._health.observe_failure(provider.name, error_code=exc.code)
                    continue
                self._health.observe_success(provider.name, response.latency_ms)
                return ChatResponse(
                    content=response.content,
                    model=response.model,
                    provider=response.provider,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    latency_ms=response.latency_ms,
                    cost=response.cost,
                    attempts=attempt,
                    metadata=response.metadata,
                )

        raise NoProviderAvailableError(
            "Every eligible provider failed to serve this request.",
            context={
                "attempted": [provider.name for provider in shortlist],
                "excluded": decision.excluded,
            },
            cause=last_error,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed a batch of texts, failing over across capable providers.

        Routing reuses the chat machinery by expressing the embedding demand as
        a capability requirement, so cost, health and privacy filtering apply
        unchanged.

        Args:
            request: The embedding request.

        Returns:
            The first successful response.

        Raises:
            EmbeddingNotSupportedError: If no provider advertises embeddings.
            NoProviderAvailableError: If every capable provider failed.
        """
        probe = ChatRequest(
            messages=[Message.user(request.texts[0][:64] or "embed")],
            model=request.model,
            required_capabilities=frozenset({Capability.EMBEDDING}),
        )
        decision = self._router.decide(probe, self._providers)
        if not decision.has_candidates:
            raise EmbeddingNotSupportedError(
                "No configured provider advertises embeddings.",
                context={"excluded": decision.excluded},
            )

        router_config = self._config.router
        limit = 1 + router_config.max_failovers if router_config.failover_enabled else 1
        shortlist = [scored.provider for scored in decision.ranked][:limit]

        last_error: EdenError | None = None
        with timed_block(_LOGGER, "gateway.embed", batch=len(request.texts)):
            for provider in shortlist:
                try:
                    response = await provider.embed(request)
                except EdenError as exc:
                    last_error = exc
                    self._health.observe_failure(provider.name, error_code=exc.code)
                    continue
                self._health.observe_success(provider.name, response.latency_ms)
                return response

        raise NoProviderAvailableError(
            "Every capable provider failed to embed this batch.",
            context={"attempted": [provider.name for provider in shortlist]},
            cause=last_error,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Serve ``request`` as a stream from the single best provider.

        Failover is not attempted mid-stream: once bytes have reached the
        caller, silently switching providers would splice two different
        generations together. A failure *before* the first chunk still fails
        over, which is where retrying is actually safe.

        Args:
            request: The generation request.

        Returns:
            An async iterator of chunks.

        Raises:
            NoProviderAvailableError: If no provider can open a stream.
        """
        streaming_request = request if request.stream else _as_streaming(request)
        decision = self._router.decide(streaming_request, self._providers)
        if not decision.has_candidates:
            raise NoProviderAvailableError(
                "No provider satisfies this streaming request.",
                context={"excluded": decision.excluded},
            )

        last_error: EdenError | None = None
        for scored in decision.ranked:
            provider = scored.provider
            try:
                iterator = await provider.stream(streaming_request)
                first = await _first_chunk(iterator)
            except EdenError as exc:
                last_error = exc
                self._health.observe_failure(provider.name, error_code=exc.code)
                _LOGGER.warning(
                    "Stream failed to open; trying the next provider.",
                    extra={"provider": provider.name, "error_code": exc.code},
                )
                continue
            if first is None:
                self._health.observe_success(provider.name, 0.0)
                return _empty_stream()
            self._health.observe_success(provider.name, 0.0)
            return _replay(first, iterator)

        raise NoProviderAvailableError(
            "No provider could open a stream for this request.",
            context={"excluded": decision.excluded},
            cause=last_error,
        )


class _ProbeFactory:
    """Callable wrapper turning a provider probe into a zero-argument factory."""

    __slots__ = ("_provider",)

    def __init__(self, provider: BaseProvider) -> None:
        """Store the provider to probe."""
        self._provider = provider

    async def __call__(self) -> HealthState:
        """Run the probe and return the observed state."""
        return await self._provider.health_check()


def record_latency(provider: BaseProvider) -> float:
    """Return the configured latency prior for a probe result.

    Probes are deliberately cheap and unrepresentative of real traffic, so their
    duration is not fed into the latency average; the configured prior is used
    instead.
    """
    if provider.config.models:
        return provider.config.models[0].expected_latency_ms
    return 0.0


def _as_streaming(request: ChatRequest) -> ChatRequest:
    """Return a copy of ``request`` with streaming enabled."""
    from dataclasses import replace  # noqa: PLC0415 - local to keep the import graph flat

    return replace(request, stream=True)


async def _first_chunk(iterator: AsyncIterator[StreamChunk]) -> StreamChunk | None:
    """Pull the first chunk so that open failures surface before yielding."""
    async for chunk in iterator:
        return chunk
    return None


async def _replay(
    first: StreamChunk,
    rest: AsyncIterator[StreamChunk],
) -> AsyncIterator[StreamChunk]:
    """Yield the already-consumed first chunk, then the remainder."""
    yield first
    async for chunk in rest:
        yield chunk


async def _empty_stream() -> AsyncIterator[StreamChunk]:
    """Yield nothing; used when a provider opens an empty stream."""
    empty: tuple[StreamChunk, ...] = ()
    for chunk in empty:  # pragma: no cover - intentionally never iterates
        yield chunk


__all__ = ["COMPONENT_NAME", "GatewayClient", "GatewayError", "ProviderHealthSummary"]
