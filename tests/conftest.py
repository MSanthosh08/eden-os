"""Shared test fixtures.

The fake transport is the reason the provider suite needs no network access:
adapters depend on the ``HttpTransport`` protocol, so a dictionary of canned
responses is a complete substitute for the internet.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import pytest

from eden.config.enums import Capability, PrivacyTier, ProviderKind
from eden.config.schema import (
    CircuitBreakerConfig,
    GatewayConfig,
    ModelConfig,
    ProviderConfig,
    RetryConfig,
    RouterConfig,
)
from eden.config.secrets import SecretResolver
from eden.gateway.health import HealthTracker
from eden.gateway.providers.mock import MockProvider
from eden.logging import reset_logging
from eden.transport.base import HttpResponse
from eden.utils.clock import ManualClock


class FakeTransport:
    """An in-memory ``HttpTransport`` returning scripted responses."""

    def __init__(self) -> None:
        self.responses: list[HttpResponse] = []
        self.stream_lines: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def queue_json(self, payload: Mapping[str, Any], *, status_code: int = 200) -> None:
        """Queue a JSON response."""
        self.responses.append(
            HttpResponse(
                status_code=status_code,
                headers={"content-type": "application/json"},
                body=json.dumps(payload).encode(),
            )
        )

    def queue_error(self, status_code: int, detail: str = "boom") -> None:
        """Queue an error response."""
        self.responses.append(HttpResponse(status_code=status_code, body=detail.encode()))

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse:
        """Record the request and pop the next scripted response."""
        self.requests.append(
            {"url": url, "headers": dict(headers), "payload": dict(payload), "timeout": timeout}
        )
        if not self.responses:
            return HttpResponse(status_code=500, body=b"no scripted response")
        return self.responses.pop(0)

    async def stream_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> AsyncIterator[str]:
        """Record the request and replay the scripted stream lines."""
        self.requests.append(
            {"url": url, "headers": dict(headers), "payload": dict(payload), "timeout": timeout}
        )
        for line in self.stream_lines:
            yield line

    async def aclose(self) -> None:
        """Mark the transport closed."""
        self.closed = True


def make_model(
    name: str = "test-model",
    *,
    input_cost: float = 1.0,
    output_cost: float = 2.0,
    latency_ms: float = 500.0,
    capabilities: frozenset[Capability] | None = None,
) -> ModelConfig:
    """Build a model catalogue entry for tests."""
    return ModelConfig(
        name=name,
        capabilities=capabilities or frozenset({Capability.CHAT, Capability.STREAMING}),
        input_cost_per_1k=input_cost,
        output_cost_per_1k=output_cost,
        expected_latency_ms=latency_ms,
    )


def make_provider_config(
    name: str,
    *,
    kind: ProviderKind = ProviderKind.MOCK,
    base_url: str = "",
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC_CLOUD,
    input_cost: float = 1.0,
    latency_ms: float = 500.0,
    weight: float = 1.0,
    capabilities: frozenset[Capability] | None = None,
    max_attempts: int = 1,
) -> ProviderConfig:
    """Build a provider declaration for tests."""
    model = make_model(
        f"{name}-model",
        input_cost=input_cost,
        latency_ms=latency_ms,
        capabilities=capabilities,
    )
    return ProviderConfig(
        name=name,
        kind=kind,
        base_url=base_url,
        default_model=model.name,
        models=(model,),
        privacy_tier=privacy_tier,
        weight=weight,
        retry=RetryConfig(max_attempts=max_attempts, initial_backoff_seconds=0.0),
    )


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Detach EDEN log handlers between tests."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def clock() -> ManualClock:
    """Return a controllable clock."""
    return ManualClock()


@pytest.fixture
def transport() -> FakeTransport:
    """Return a fresh fake transport."""
    return FakeTransport()


@pytest.fixture
def secrets() -> SecretResolver:
    """Return a resolver backed by a fixed environment."""
    return SecretResolver({"TEST_KEY": "sk-unit-test-credential-value"})


@pytest.fixture
def tracker(clock: ManualClock) -> HealthTracker:
    """Return a health tracker with a low failure threshold."""
    return HealthTracker(
        config=CircuitBreakerConfig(failure_threshold=2, reset_seconds=10.0),
        clock=clock,
    )


@pytest.fixture
def gateway_config() -> GatewayConfig:
    """Return a gateway configuration with two mock providers."""
    return GatewayConfig(
        providers=(
            make_provider_config("cheap", input_cost=0.5, latency_ms=900.0),
            make_provider_config("fast", input_cost=5.0, latency_ms=100.0),
        ),
        router=RouterConfig(circuit_breaker=CircuitBreakerConfig(failure_threshold=2)),
    )


def make_mock_provider(
    name: str,
    *,
    clock: ManualClock | None = None,
    **kwargs: Any,
) -> MockProvider:
    """Build a ``MockProvider`` from a generated configuration."""
    config = make_provider_config(name, **kwargs)
    return MockProvider(config, clock=clock or ManualClock())
