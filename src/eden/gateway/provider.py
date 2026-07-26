"""Provider base class.

:class:`BaseProvider` is a Template Method. It owns everything that is identical
for every vendor — rate limiting, timeout enforcement, retry, timing, structured
logging and error translation — and delegates only the two genuinely
vendor-specific steps to subclasses: build the request payload, parse the
response. A new provider is therefore roughly fifty lines, and improvements to
the shared machinery benefit every vendor at once.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Mapping
from typing import Any

from eden.config.enums import Capability, HealthState, PrivacyTier
from eden.config.schema import ProviderConfig
from eden.config.secrets import SecretResolver, SecretStr
from eden.core.types import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Usage,
)
from eden.errors import (
    ModelNotSupportedError,
    ProviderError,
    ProviderTimeoutError,
    TransportError,
)
from eden.logging import get_logger, timed_block
from eden.utils.async_tools import retry_async, with_timeout
from eden.utils.clock import Clock, SystemClock
from eden.utils.ratelimit import TokenBucket

_TOKENS_PER_THOUSAND = 1000.0


class BaseProvider(abc.ABC):
    """Common behaviour for every AI provider adapter.

    Subclasses implement :meth:`_perform_chat` and, when they advertise
    :attr:`~eden.config.enums.Capability.STREAMING`, :meth:`_perform_stream`.
    They must not implement retry, timing or logging themselves.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        secrets: SecretResolver | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the adapter from its configuration slice.

        Args:
            config: Declarative description of this provider.
            secrets: Resolver used to fetch the credential lazily.
            clock: Time source used by the rate limiter and retry backoff.
        """
        self._config = config
        self._secrets = secrets or SecretResolver()
        self._clock = clock or SystemClock()
        self._logger = get_logger(f"gateway.provider.{config.name}")
        self._bucket = TokenBucket(
            requests_per_minute=config.requests_per_minute,
            clock=self._clock,
        )
        self._capabilities = self._resolve_capabilities()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Return the logical provider name from configuration."""
        return self._config.name

    @property
    def config(self) -> ProviderConfig:
        """Return this provider's configuration slice."""
        return self._config

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Return every capability advertised by this provider."""
        return self._capabilities

    @property
    def privacy_tier(self) -> PrivacyTier:
        """Return the configured data-residency guarantee."""
        return self._config.privacy_tier

    def _resolve_capabilities(self) -> frozenset[Capability]:
        """Return the union of catalogue capabilities, defaulting to chat."""
        declared = self._config.capabilities()
        return declared or frozenset({Capability.CHAT})

    # ------------------------------------------------------------------
    # Selection support
    # ------------------------------------------------------------------
    def resolve_model(self, request: ChatRequest) -> str:
        """Return the model that will serve ``request``.

        Args:
            request: The pending request.

        Returns:
            The explicitly requested model, the configured default, or the
            first catalogue entry.

        Raises:
            ModelNotSupportedError: If no model can be determined.
        """
        if request.model:
            return request.model
        if self._config.default_model:
            return self._config.default_model
        if self._config.models:
            return self._config.models[0].name
        raise ModelNotSupportedError(
            "Provider has no model configured and the request did not name one.",
            provider=self.name,
        )

    def supports(self, request: ChatRequest) -> bool:
        """Return whether this provider can serve ``request``.

        Checks capability coverage, model availability and the request's
        privacy floor. Health is deliberately *not* considered here; that is
        the router's responsibility.
        """
        if not request.required_capabilities <= self._capabilities:
            return False
        floor = request.minimum_privacy_tier
        if floor is not None and self._config.privacy_tier < floor:
            return False
        if not request.model:
            return bool(self._config.default_model or self._config.models)
        if not self._config.models:
            return True
        return self._config.model(request.model) is not None

    def estimate_cost(self, request: ChatRequest) -> float:
        """Return the estimated spend for ``request`` in configured units."""
        try:
            model_name = self.resolve_model(request)
        except ModelNotSupportedError:
            return 0.0
        model = self._config.model(model_name)
        if model is None:
            return 0.0
        prompt_tokens = request.estimated_prompt_tokens
        completion_tokens = request.max_tokens or prompt_tokens
        return (
            prompt_tokens * model.input_cost_per_1k + completion_tokens * model.output_cost_per_1k
        ) / _TOKENS_PER_THOUSAND

    def expected_latency_ms(self, request: ChatRequest) -> float:
        """Return the configured latency prior for ``request``."""
        try:
            model_name = self.resolve_model(request)
        except ModelNotSupportedError:
            return float("inf")
        model = self._config.model(model_name)
        return model.expected_latency_ms if model is not None else 1000.0

    def actual_cost(self, model_name: str, usage: Usage) -> float:
        """Return the realised spend for a completed call."""
        model = self._config.model(model_name)
        if model is None:
            return 0.0
        return (
            usage.prompt_tokens * model.input_cost_per_1k
            + usage.completion_tokens * model.output_cost_per_1k
        ) / _TOKENS_PER_THOUSAND

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------
    def credential(self, *, required: bool = True) -> SecretStr | None:
        """Resolve this provider's credential from the environment.

        Args:
            required: Whether a missing credential is an error.

        Returns:
            The wrapped secret, or ``None`` when absent and optional.

        Raises:
            SecretResolutionError: If the credential is required and missing.
        """
        if not self._config.api_key_env:
            return None
        return self._secrets.resolve(self._config.api_key_env, required=required)

    def base_headers(self) -> dict[str, str]:
        """Return static headers declared in configuration."""
        return dict(self._config.headers)

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Perform one non-streamed generation with the full safety envelope.

        Args:
            request: The generation request.

        Returns:
            The parsed response, annotated with measured latency and cost.

        Raises:
            ProviderError: Translated from any vendor or transport failure.
        """
        model_name = self.resolve_model(request)
        await self._bucket.acquire()

        async def attempt() -> ChatResponse:
            return await with_timeout(
                self._perform_chat(request, model_name),
                self._config.timeout_seconds,
                on_timeout=lambda: ProviderTimeoutError(
                    "Provider exceeded its configured timeout.",
                    provider=self.name,
                    context={"timeout_seconds": self._config.timeout_seconds},
                ),
            )

        with timed_block(
            self._logger,
            "provider.chat",
            provider=self.name,
            model=model_name,
        ) as watch:
            try:
                response = await retry_async(
                    attempt,
                    self._config.retry,
                    logger=self._logger,
                    operation_name=f"{self.name}.chat",
                    clock=self._clock,
                )
            except ProviderError:
                raise
            except TransportError as exc:
                raise self._translate(exc, model_name) from exc
            except Exception as exc:
                raise ProviderError(
                    "Provider call failed unexpectedly.",
                    provider=self.name,
                    context={"model": model_name},
                    cause=exc,
                ) from exc

        latency_ms = watch.elapsed_ms
        return ChatResponse(
            content=response.content,
            model=response.model or model_name,
            provider=self.name,
            usage=response.usage,
            finish_reason=response.finish_reason,
            latency_ms=latency_ms,
            cost=self.actual_cost(response.model or model_name, response.usage),
            metadata=response.metadata,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Perform one streamed generation.

        Args:
            request: The generation request.

        Returns:
            An async iterator of incremental chunks.

        Raises:
            ModelNotSupportedError: If this provider does not stream.
        """
        if Capability.STREAMING not in self._capabilities:
            raise ModelNotSupportedError(
                "Provider does not advertise streaming.",
                provider=self.name,
            )
        model_name = self.resolve_model(request)
        await self._bucket.acquire()
        self._logger.debug(
            "Opening provider stream.",
            extra={"provider": self.name, "model": model_name},
        )
        return self._perform_stream(request, model_name)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed a batch of texts with the full safety envelope.

        Args:
            request: The embedding request.

        Returns:
            One vector per input text, in order.

        Raises:
            ModelNotSupportedError: If this provider does not embed.
            ProviderError: Translated from any vendor or transport failure.
        """
        if Capability.EMBEDDING not in self._capabilities:
            raise ModelNotSupportedError(
                "Provider does not advertise embeddings.",
                provider=self.name,
            )
        model_name = request.model or self._embedding_model()
        await self._bucket.acquire()

        async def attempt() -> EmbeddingResponse:
            return await with_timeout(
                self._perform_embed(request, model_name),
                self._config.timeout_seconds,
                on_timeout=lambda: ProviderTimeoutError(
                    "Provider exceeded its configured timeout.",
                    provider=self.name,
                    context={"timeout_seconds": self._config.timeout_seconds},
                ),
            )

        with timed_block(
            self._logger,
            "provider.embed",
            provider=self.name,
            model=model_name,
            batch=len(request.texts),
        ) as watch:
            try:
                response = await retry_async(
                    attempt,
                    self._config.retry,
                    logger=self._logger,
                    operation_name=f"{self.name}.embed",
                    clock=self._clock,
                )
            except ProviderError:
                raise
            except TransportError as exc:
                raise self._translate(exc, model_name) from exc
            except Exception as exc:
                raise ProviderError(
                    "Provider embedding call failed unexpectedly.",
                    provider=self.name,
                    context={"model": model_name},
                    cause=exc,
                ) from exc

        return EmbeddingResponse(
            vectors=response.vectors,
            model=response.model or model_name,
            provider=self.name,
            usage=response.usage,
            latency_ms=watch.elapsed_ms,
            cost=self.actual_cost(response.model or model_name, response.usage),
        )

    def _embedding_model(self) -> str:
        """Return the first catalogue model advertising embeddings.

        Raises:
            ModelNotSupportedError: If no such model is configured.
        """
        for model in self._config.models:
            if Capability.EMBEDDING in model.capabilities:
                return model.name
        raise ModelNotSupportedError(
            "Provider has no embedding model configured.",
            provider=self.name,
        )

    async def _perform_embed(
        self,
        request: EmbeddingRequest,
        model_name: str,
    ) -> EmbeddingResponse:
        """Execute one vendor embedding call.

        The default implementation refuses. A provider that advertises
        :attr:`~eden.config.enums.Capability.EMBEDDING` must override it.

        Raises:
            ModelNotSupportedError: Always, unless overridden.
        """
        raise ModelNotSupportedError(
            "Provider advertises embeddings but does not implement them.",
            provider=self.name,
            context={"model": model_name, "batch": len(request.texts)},
        )

    async def health_check(self) -> HealthState:
        """Probe the provider with a minimal request.

        Returns:
            :attr:`HealthState.HEALTHY` when the probe succeeds, otherwise
            :attr:`HealthState.UNHEALTHY`. Never raises.
        """
        try:
            await self._perform_health_check()
        except Exception as exc:  # noqa: BLE001 - a probe must never propagate
            self._logger.warning(
                "Health probe failed.",
                extra={"provider": self.name, "error_type": type(exc).__name__},
            )
            return HealthState.UNHEALTHY
        return HealthState.HEALTHY

    async def aclose(self) -> None:
        """Release provider resources. Base implementation holds none."""
        return

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def _perform_chat(self, request: ChatRequest, model_name: str) -> ChatResponse:
        """Execute one vendor call and parse the result.

        Implementations return a response whose ``latency_ms`` and ``cost`` may
        be zero; the base class fills those in.
        """

    async def _perform_stream(
        self,
        request: ChatRequest,
        model_name: str,
    ) -> AsyncIterator[StreamChunk]:
        """Execute one streamed vendor call.

        The default implementation degrades gracefully by emitting the
        non-streamed result as a single terminal chunk, so a provider without
        native streaming still satisfies streaming callers correctly.
        """
        response = await self._perform_chat(request, model_name)
        yield StreamChunk(
            delta=response.content,
            provider=self.name,
            model=response.model or model_name,
            finish_reason=response.finish_reason,
            index=0,
        )

    async def _perform_health_check(self) -> None:
        """Probe the vendor. Default probe issues a one-token generation."""
        from eden.core.types import Message  # noqa: PLC0415 - avoids an import cycle

        probe = ChatRequest(
            messages=[Message.user("ping")],
            max_tokens=1,
            required_capabilities=frozenset({Capability.CHAT}),
        )
        await self._perform_chat(probe, self.resolve_model(probe))

    def _translate(self, error: TransportError, model_name: str) -> ProviderError:
        """Convert a transport failure into a provider-scoped error."""
        return ProviderError(
            "Transport failure while calling the provider.",
            provider=self.name,
            context={"model": model_name, **error.context},
            cause=error,
        )

    def describe(self) -> Mapping[str, Any]:
        """Return a redaction-safe description for diagnostics endpoints."""
        return {
            "name": self.name,
            "kind": str(self._config.kind),
            "privacy_tier": self._config.privacy_tier.name.lower(),
            "capabilities": sorted(capability.value for capability in self._capabilities),
            "models": [model.name for model in self._config.models],
            "default_model": self._config.default_model,
            "requests_per_minute": self._config.requests_per_minute,
        }
