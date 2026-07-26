"""Unit tests for the AI Gateway: providers, health, routing and construction."""

from __future__ import annotations

import pytest

from eden.config.enums import (
    Capability,
    CircuitState,
    FinishReason,
    HealthState,
    PrivacyTier,
    ProviderKind,
    RoutingStrategyName,
)
from eden.config.schema import (
    CircuitBreakerConfig,
    ProviderConfig,
    RouterConfig,
    RouterWeights,
)
from eden.config.secrets import SecretResolver
from eden.core.types import ChatRequest, Message
from eden.errors import (
    InvalidConfigError,
    ModelNotSupportedError,
    PluginLoadError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from eden.gateway.factory import ProviderFactory
from eden.gateway.health import HealthTracker
from eden.gateway.providers.anthropic import AnthropicProvider
from eden.gateway.providers.gemini import GeminiProvider
from eden.gateway.providers.mock import MockProvider
from eden.gateway.providers.openai_compatible import OpenAICompatibleProvider
from eden.gateway.router.omni_router import OmniRouter
from eden.gateway.router.strategy import build_strategy
from eden.utils.clock import ManualClock
from tests.conftest import FakeTransport, make_provider_config


def request_for(text: str = "hello", **kwargs: object) -> ChatRequest:
    """Build a simple chat request."""
    return ChatRequest(messages=[Message.user(text)], **kwargs)  # type: ignore[arg-type]


class TestMockProvider:
    async def test_echoes_the_last_turn(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("m"), clock=clock)
        response = await provider.chat(request_for("ping"))
        assert "ping" in response.content
        assert response.provider == "m"
        assert response.finish_reason is FinishReason.STOP

    async def test_records_usage_and_cost(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("m", input_cost=1000.0), clock=clock)
        response = await provider.chat(request_for("a longer prompt here"))
        assert response.usage.total_tokens > 0
        assert response.cost > 0

    async def test_injected_failure_propagates(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("m"), clock=clock)
        provider.set_failure(lambda: ProviderUnavailableError("down", provider="m"))
        with pytest.raises(ProviderUnavailableError):
            await provider.chat(request_for())

    async def test_stream_terminates_with_a_finish_reason(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("m"), clock=clock)
        chunks = [chunk async for chunk in await provider.stream(request_for("a b c"))]
        assert chunks[-1].is_final
        assert "".join(chunk.delta for chunk in chunks).endswith("a b c")

    async def test_health_check_never_raises(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("m"), clock=clock)
        provider.set_failure(lambda: RuntimeError("explode"))
        assert await provider.health_check() is HealthState.UNHEALTHY

    def test_model_resolution_order(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("m"), clock=clock)
        assert provider.resolve_model(request_for()) == "m-model"
        assert provider.resolve_model(request_for(model="explicit")) == "explicit"

    def test_missing_model_raises(self, clock: ManualClock) -> None:
        provider = MockProvider(ProviderConfig(name="bare", kind=ProviderKind.MOCK), clock=clock)
        with pytest.raises(ModelNotSupportedError):
            provider.resolve_model(request_for())

    def test_capability_and_privacy_filters(self, clock: ManualClock) -> None:
        provider = MockProvider(
            make_provider_config(
                "m",
                privacy_tier=PrivacyTier.PUBLIC_CLOUD,
                capabilities=frozenset({Capability.CHAT}),
            ),
            clock=clock,
        )
        assert provider.supports(request_for()) is True
        assert (
            provider.supports(request_for(required_capabilities=frozenset({Capability.VISION})))
            is False
        )
        assert provider.supports(request_for(minimum_privacy_tier=PrivacyTier.LOCAL_ONLY)) is False

    def test_describe_leaks_nothing_sensitive(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("m"), clock=clock)
        described = provider.describe()
        assert "api_key" not in described
        assert described["name"] == "m"


class TestOpenAICompatibleProvider:
    def build(
        self, transport: FakeTransport, secrets: SecretResolver, clock: ManualClock
    ) -> OpenAICompatibleProvider:
        config = make_provider_config(
            "oai", kind=ProviderKind.OPENAI_COMPATIBLE, base_url="https://api.test/v1"
        )
        config = ProviderConfig(
            name=config.name,
            kind=config.kind,
            base_url=config.base_url,
            api_key_env="TEST_KEY",
            default_model=config.default_model,
            models=config.models,
            retry=config.retry,
        )
        return OpenAICompatibleProvider(config, transport, secrets=secrets, clock=clock)

    async def test_parses_a_completion(
        self, transport: FakeTransport, secrets: SecretResolver, clock: ManualClock
    ) -> None:
        transport.queue_json(
            {
                "id": "cmpl-1",
                "model": "oai-model",
                "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )
        provider = self.build(transport, secrets, clock)
        response = await provider.chat(request_for())
        assert response.content == "hi there"
        assert response.usage.total_tokens == 5
        assert transport.requests[0]["url"] == "https://api.test/v1/chat/completions"

    async def test_sends_bearer_credential_without_logging_it(
        self, transport: FakeTransport, secrets: SecretResolver, clock: ManualClock
    ) -> None:
        transport.queue_json({"choices": [{"message": {"content": "x"}}]})
        await self.build(transport, secrets, clock).chat(request_for())
        assert transport.requests[0]["headers"]["authorization"].startswith("Bearer sk-")

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, ProviderAuthenticationError),
            (403, ProviderAuthenticationError),
            (429, ProviderRateLimitError),
            (500, ProviderUnavailableError),
            (400, ProviderResponseError),
        ],
    )
    async def test_status_codes_map_to_precise_errors(
        self,
        status: int,
        expected: type[Exception],
        transport: FakeTransport,
        secrets: SecretResolver,
        clock: ManualClock,
    ) -> None:
        transport.queue_error(status)
        with pytest.raises(expected):
            await self.build(transport, secrets, clock).chat(request_for())

    async def test_malformed_payload_is_rejected(
        self, transport: FakeTransport, secrets: SecretResolver, clock: ManualClock
    ) -> None:
        transport.queue_json({"choices": []})
        with pytest.raises(ProviderResponseError):
            await self.build(transport, secrets, clock).chat(request_for())

    async def test_stream_parses_server_sent_events(
        self, transport: FakeTransport, secrets: SecretResolver, clock: ManualClock
    ) -> None:
        transport.stream_lines = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        provider = self.build(transport, secrets, clock)
        chunks = [c async for c in await provider.stream(request_for(stream=True))]
        assert "".join(chunk.delta for chunk in chunks) == "Hello"
        assert chunks[-1].is_final


class TestAnthropicProvider:
    async def test_hoists_system_prompt_and_parses_blocks(
        self, transport: FakeTransport, secrets: SecretResolver, clock: ManualClock
    ) -> None:
        transport.queue_json(
            {
                "id": "msg_1",
                "model": "claude-test",
                "content": [{"type": "text", "text": "answer"}],
                "usage": {"input_tokens": 4, "output_tokens": 1},
                "stop_reason": "end_turn",
            }
        )
        config = make_provider_config(
            "anthropic", kind=ProviderKind.ANTHROPIC, base_url="https://api.anthropic.test"
        )
        provider = AnthropicProvider(config, transport, secrets=secrets, clock=clock)
        request = ChatRequest(
            messages=[Message.system("be brief"), Message.user("question")],
            max_tokens=64,
        )
        response = await provider.chat(request)
        payload = transport.requests[0]["payload"]
        assert payload["system"] == "be brief"
        assert len(payload["messages"]) == 1
        assert response.content == "answer"
        assert response.usage.prompt_tokens == 4


class TestGeminiProvider:
    async def test_maps_assistant_role_and_parts(
        self, transport: FakeTransport, secrets: SecretResolver, clock: ManualClock
    ) -> None:
        transport.queue_json(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "result"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
            }
        )
        config = make_provider_config(
            "gemini", kind=ProviderKind.GEMINI, base_url="https://gen.test"
        )
        provider = GeminiProvider(config, transport, secrets=secrets, clock=clock)
        request = ChatRequest(
            messages=[Message.user("q"), Message.assistant("a"), Message.user("q2")]
        )
        response = await provider.chat(request)
        roles = [turn["role"] for turn in transport.requests[0]["payload"]["contents"]]
        assert roles == ["user", "model", "user"]
        assert response.content == "result"


class TestHealthTracker:
    def test_starts_unknown_and_optimistic(self, tracker: HealthTracker) -> None:
        record = tracker.record("p")
        assert record.state is HealthState.UNKNOWN
        assert record.success_rate == 1.0

    def test_circuit_opens_after_threshold_failures(self, tracker: HealthTracker) -> None:
        tracker.observe_failure("p", error_code="x")
        assert tracker.is_available("p") is True
        tracker.observe_failure("p", error_code="x")
        assert tracker.record("p").circuit is CircuitState.OPEN
        assert tracker.is_available("p") is False

    def test_circuit_half_opens_after_cooldown_then_closes(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        tracker.observe_failure("p")
        tracker.observe_failure("p")
        clock.advance(11.0)
        assert tracker.is_available("p") is True
        assert tracker.record("p").circuit is CircuitState.HALF_OPEN
        tracker.observe_success("p", 100.0)
        tracker.observe_success("p", 100.0)
        assert tracker.record("p").circuit is CircuitState.CLOSED

    def test_latency_is_smoothed_not_replaced(self, tracker: HealthTracker) -> None:
        tracker.observe_success("p", 100.0)
        tracker.observe_success("p", 1000.0)
        smoothed = tracker.record("p").ewma_latency_ms
        assert 100.0 < smoothed < 1000.0

    def test_open_circuit_scores_zero(self, tracker: HealthTracker) -> None:
        tracker.observe_failure("p")
        tracker.observe_failure("p")
        assert tracker.record("p").score() == 0.0

    def test_reset_clears_state(self, tracker: HealthTracker) -> None:
        tracker.observe_failure("p")
        tracker.reset("p")
        assert tracker.record("p").failures == 0


class TestRouter:
    def providers(self, clock: ManualClock) -> list[MockProvider]:
        return [
            MockProvider(
                make_provider_config("cheap", input_cost=0.1, latency_ms=2000.0), clock=clock
            ),
            MockProvider(
                make_provider_config("fast", input_cost=10.0, latency_ms=50.0), clock=clock
            ),
            MockProvider(
                make_provider_config(
                    "local",
                    input_cost=0.0,
                    latency_ms=800.0,
                    privacy_tier=PrivacyTier.LOCAL_ONLY,
                ),
                clock=clock,
            ),
        ]

    def router(self, tracker: HealthTracker, **kwargs: object) -> OmniRouter:
        config = RouterConfig(**kwargs)  # type: ignore[arg-type]
        return OmniRouter(config, tracker)

    def test_cheapest_strategy_prefers_cost(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        router = self.router(tracker, strategy=RoutingStrategyName.CHEAPEST)
        ranked = router.decide(request_for(), self.providers(clock))
        assert ranked.provider_names()[0] in {"local", "cheap"}

    def test_fastest_strategy_prefers_latency(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        router = self.router(tracker, strategy=RoutingStrategyName.FASTEST)
        assert router.decide(request_for(), self.providers(clock)).provider_names()[0] == "fast"

    def test_privacy_first_strategy_prefers_local(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        router = self.router(tracker, strategy=RoutingStrategyName.PRIVACY_FIRST)
        assert router.decide(request_for(), self.providers(clock)).provider_names()[0] == "local"

    def test_privacy_floor_is_a_hard_filter_not_a_preference(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        router = self.router(tracker, strategy=RoutingStrategyName.CHEAPEST)
        decision = router.decide(
            request_for(minimum_privacy_tier=PrivacyTier.LOCAL_ONLY), self.providers(clock)
        )
        assert decision.provider_names() == ("local",)
        assert decision.excluded["cheap"] == "privacy_tier_below_floor"

    def test_missing_capability_excludes_with_a_reason(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        decision = self.router(tracker).decide(
            request_for(required_capabilities=frozenset({Capability.VISION})),
            self.providers(clock),
        )
        assert not decision.has_candidates
        assert all("missing_capabilities" in reason for reason in decision.excluded.values())

    def test_open_circuit_removes_a_provider(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        tracker.observe_failure("fast")
        tracker.observe_failure("fast")
        decision = self.router(tracker).decide(request_for(), self.providers(clock))
        assert decision.excluded["fast"] == "circuit_open"

    def test_preferred_provider_wins_when_healthy(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        router = self.router(tracker, weights=RouterWeights(preference=20.0))
        decision = router.decide(request_for(preferred_provider="cheap"), self.providers(clock))
        assert decision.provider_names()[0] == "cheap"

    def test_shortlist_is_truncated_by_max_failovers(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        router = self.router(tracker, max_failovers=1)
        assert len(router.select(request_for(), self.providers(clock))) == 2

    def test_failover_disabled_yields_one_candidate(
        self, tracker: HealthTracker, clock: ManualClock
    ) -> None:
        router = self.router(tracker, failover_enabled=False, max_failovers=5)
        assert len(router.select(request_for(), self.providers(clock))) == 1

    def test_round_robin_rotates(self, tracker: HealthTracker, clock: ManualClock) -> None:
        router = self.router(tracker, strategy=RoutingStrategyName.ROUND_ROBIN)
        providers = self.providers(clock)
        first = router.decide(request_for(), providers).provider_names()
        second = router.decide(request_for(), providers).provider_names()
        assert first != second

    def test_ranking_is_deterministic(self, tracker: HealthTracker, clock: ManualClock) -> None:
        router = self.router(tracker)
        providers = self.providers(clock)
        assert (
            router.decide(request_for(), providers).provider_names()
            == router.decide(request_for(), providers).provider_names()
        )

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            build_strategy("not-a-strategy", RouterWeights())  # type: ignore[arg-type]


class TestProviderFactory:
    def test_builds_each_known_kind(self, transport: FakeTransport, clock: ManualClock) -> None:
        factory = ProviderFactory(transport=transport, clock=clock)
        for kind, url in (
            (ProviderKind.OPENAI_COMPATIBLE, "https://a.test/v1"),
            (ProviderKind.ANTHROPIC, "https://b.test"),
            (ProviderKind.GEMINI, "https://c.test"),
            (ProviderKind.MOCK, ""),
        ):
            config = make_provider_config(f"p-{kind.value}", kind=kind, base_url=url)
            assert factory.create(config).name == f"p-{kind.value}"

    def test_network_provider_without_transport_is_rejected(self, clock: ManualClock) -> None:
        factory = ProviderFactory(clock=clock)
        config = make_provider_config("x", kind=ProviderKind.ANTHROPIC, base_url="https://b.test")
        with pytest.raises(InvalidConfigError):
            factory.create(config)

    def test_custom_kind_loads_by_import_path(self, clock: ManualClock) -> None:
        config = ProviderConfig(
            name="custom",
            kind=ProviderKind.CUSTOM,
            implementation="eden.gateway.providers.mock:MockProvider",
            default_model="m",
        )
        provider = ProviderFactory(clock=clock).create(config)
        assert isinstance(provider, MockProvider)

    def test_custom_kind_rejects_a_non_provider(self, clock: ManualClock) -> None:
        config = ProviderConfig(
            name="bad", kind=ProviderKind.CUSTOM, implementation="decimal:Decimal"
        )
        with pytest.raises(PluginLoadError):
            ProviderFactory(clock=clock).create(config)

    def test_a_broken_provider_does_not_abort_the_fleet(self, clock: ManualClock) -> None:
        good = make_provider_config("good")
        bad = ProviderConfig(name="bad", kind=ProviderKind.CUSTOM, implementation="nowhere:Absent")
        providers = ProviderFactory(clock=clock).create_all((good, bad))
        assert [provider.name for provider in providers] == ["good"]

    def test_new_protocol_family_can_be_registered(self, clock: ManualClock) -> None:
        factory = ProviderFactory(clock=clock)
        with pytest.raises(InvalidConfigError):
            factory.register_builder(ProviderKind.MOCK, lambda ctx: MockProvider(ctx.config))
        factory.register_builder(
            ProviderKind.MOCK, lambda ctx: MockProvider(ctx.config), replace=True
        )
        assert factory.create(make_provider_config("m")).name == "m"


class TestCircuitBreakerConfigValidation:
    def test_non_positive_threshold_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            CircuitBreakerConfig(failure_threshold=0)
