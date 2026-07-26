"""Deterministic in-process provider.

This is not a stub. It is a first-class provider used for three real purposes:
offline development with no credentials, deterministic integration tests, and a
last-resort fallback that lets an operator prove the routing path is healthy
even when every vendor is down.

Its behaviour is fully controllable through configuration and constructor
arguments, so a test can make it fail on demand without monkey-patching.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Callable

from eden.config.enums import FinishReason
from eden.config.schema import ProviderConfig
from eden.config.secrets import SecretResolver
from eden.core.types import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Usage,
)
from eden.errors import ProviderUnavailableError
from eden.gateway.provider import BaseProvider
from eden.utils.clock import Clock

DEFAULT_TEMPLATE = "[{provider}:{model}] {prompt}"
DEFAULT_DIMENSIONS = 64
_CHARS_PER_TOKEN = 4


def hash_vector(text: str, dimensions: int) -> tuple[float, ...]:
    """Return a deterministic unit vector for ``text``.

    Each whitespace token contributes to a fixed bucket chosen by its digest, so
    documents sharing vocabulary end up close together under cosine similarity.
    This is a genuine offline embedder, not a stub: it is the default used when
    no provider advertises embeddings.

    Args:
        text: Input to embed.
        dimensions: Width of the returned vector. Must be positive.

    Returns:
        A unit-length vector of length ``dimensions``.

    Raises:
        ValueError: If ``dimensions`` is not positive.

    Example:
        >>> len(hash_vector("hello world", 8))
        8
    """
    if dimensions <= 0:
        message = "dimensions must be positive."
        raise ValueError(message)
    buckets = [0.0] * dimensions
    tokens = text.lower().split() or [text.lower()]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        buckets[index] += sign
    norm = math.sqrt(sum(value * value for value in buckets))
    if norm == 0.0:
        return tuple(buckets)
    return tuple(value / norm for value in buckets)


class MockProvider(BaseProvider):
    """Echoes the final user turn, optionally failing on demand."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        secrets: SecretResolver | None = None,
        clock: Clock | None = None,
        template: str = DEFAULT_TEMPLATE,
        failure: Callable[[], BaseException] | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Initialise the provider.

        Args:
            config: Provider configuration slice.
            secrets: Credential resolver, unused but accepted for symmetry.
            clock: Time source for rate limiting and simulated latency.
            template: Format string applied to the echoed prompt.
            failure: When supplied, every call raises this exception, which is
                how failover tests drive the router deterministically.
            latency_ms: Artificial delay applied before responding.
        """
        super().__init__(config, secrets=secrets, clock=clock)
        self._template = template
        self._failure = failure
        self._latency_ms = latency_ms
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Return how many generations this provider has served."""
        return self._call_count

    def set_failure(self, failure: Callable[[], BaseException] | None) -> None:
        """Install or clear the failure injector."""
        self._failure = failure

    async def _perform_chat(self, request: ChatRequest, model_name: str) -> ChatResponse:
        """Return a deterministic echo of the last message.

        Raises:
            BaseException: Whatever the injected failure factory produces.
            ProviderUnavailableError: If the request carries no message.
        """
        self._call_count += 1
        if self._latency_ms > 0:
            await self._clock.sleep(self._latency_ms / 1000.0)
        if self._failure is not None:
            raise self._failure()
        if not request.messages:
            raise ProviderUnavailableError(
                "Mock provider received an empty request.",
                provider=self.name,
            )
        prompt = request.messages[-1].content
        content = self._template.format(provider=self.name, model=model_name, prompt=prompt)
        return ChatResponse(
            content=content,
            model=model_name,
            provider=self.name,
            usage=Usage(
                prompt_tokens=max(1, len(prompt) // _CHARS_PER_TOKEN),
                completion_tokens=max(1, len(content) // _CHARS_PER_TOKEN),
            ),
            finish_reason=FinishReason.STOP,
        )

    async def _perform_stream(
        self,
        request: ChatRequest,
        model_name: str,
    ) -> AsyncIterator[StreamChunk]:
        """Yield the echoed response one word at a time."""
        response = await self._perform_chat(request, model_name)
        words = response.content.split(" ")
        for index, word in enumerate(words):
            is_last = index == len(words) - 1
            yield StreamChunk(
                delta=word if is_last else f"{word} ",
                provider=self.name,
                model=model_name,
                finish_reason=FinishReason.STOP if is_last else None,
                index=index,
            )

    async def _perform_embed(
        self,
        request: EmbeddingRequest,
        model_name: str,
    ) -> EmbeddingResponse:
        """Return deterministic unit vectors derived from the input text.

        The same text always yields the same vector and similar texts share
        tokens, so cosine similarity behaves sensibly enough to test retrieval
        without any network or model.

        Raises:
            BaseException: Whatever the injected failure factory produces.
        """
        self._call_count += 1
        if self._failure is not None:
            raise self._failure()
        width = request.dimensions or DEFAULT_DIMENSIONS
        return EmbeddingResponse(
            vectors=tuple(hash_vector(text, width) for text in request.texts),
            model=model_name,
            provider=self.name,
            usage=Usage(prompt_tokens=request.estimated_tokens),
        )

    async def _perform_health_check(self) -> None:
        """Succeed unless a failure has been injected.

        Raises:
            BaseException: Whatever the injected failure factory produces.
        """
        if self._failure is not None:
            raise self._failure()
