"""Integration tests exercising more than one subsystem at a time.

The headline test is :meth:`TestProviderSwapping.test_swapping_vendors_is_a_config_change`:
it asserts the architectural promise that business logic never changes when the
underlying vendor does.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from eden.agents.types import Task
from eden.automation.scheduler import Rule, every
from eden.config.enums import (
    ActionKind,
    AutomationStatus,
    Capability,
    Environment,
    ExecutionStatus,
    MemoryKind,
    PrivacyTier,
    ProviderKind,
    RoutingStrategyName,
)
from eden.config.loader import ConfigLoader
from eden.config.schema import (
    AgentConfig,
    AutomationConfig,
    CircuitBreakerConfig,
    DeviceConfig,
    EdenConfig,
    ExecutionConfig,
    GatewayConfig,
    HardwareConfig,
    InterfaceConfig,
    LoggingConfig,
    MemoryConfig,
    PathsConfig,
    RouterConfig,
)
from eden.core.container import Container
from eden.core.kernel import EdenKernel
from eden.core.types import ChatRequest, EmbeddingRequest, Message
from eden.errors import (
    EmbeddingNotSupportedError,
    LifecycleError,
    NoProviderAvailableError,
    ProviderUnavailableError,
)
from eden.execution.permissions import AlwaysApproveGate, CallbackGate
from eden.execution.types import Action, Verdict
from eden.gateway.client import GatewayClient
from eden.gateway.health import HealthTracker
from eden.gateway.providers.mock import MockProvider
from eden.gateway.router.omni_router import OmniRouter
from eden.interface.api import build_router
from eden.interface.server import HttpServer, WebApprovalGate
from eden.memory.types import MemoryQuery
from eden.utils.clock import ManualClock
from tests.conftest import FakeTransport, make_provider_config

pytestmark = pytest.mark.integration


def build_client(
    providers: list[MockProvider],
    *,
    clock: ManualClock,
    strategy: RoutingStrategyName = RoutingStrategyName.BALANCED,
    max_failovers: int = 2,
    failover_enabled: bool = True,
) -> GatewayClient:
    """Assemble a gateway client around ready-made providers."""
    router_config = RouterConfig(
        strategy=strategy,
        max_failovers=max_failovers,
        failover_enabled=failover_enabled,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=2, reset_seconds=5.0),
    )
    tracker = HealthTracker(config=router_config.circuit_breaker, clock=clock)
    config = GatewayConfig(
        providers=tuple(provider.config for provider in providers),
        router=router_config,
    )
    return GatewayClient(config, providers, OmniRouter(router_config, tracker), tracker)


def request_for(text: str = "hello") -> ChatRequest:
    """Build a simple chat request."""
    return ChatRequest(messages=[Message.user(text)])


def _failure_for(name: str) -> Callable[[], BaseException]:
    """Return a failure factory bound to one provider name."""

    def factory() -> BaseException:
        return ProviderUnavailableError("down", provider=name)

    return factory


class TestGatewayFailover:
    async def test_falls_over_to_the_next_provider(self, clock: ManualClock) -> None:
        primary = MockProvider(make_provider_config("primary", latency_ms=10.0), clock=clock)
        backup = MockProvider(make_provider_config("backup", latency_ms=20.0), clock=clock)
        primary.set_failure(lambda: ProviderUnavailableError("down", provider="primary"))

        client = build_client([primary, backup], clock=clock)
        response = await client.chat(request_for("ping"))

        assert response.provider == "backup"
        assert response.attempts == 2
        assert "ping" in response.content

    async def test_exhaustion_raises_with_the_root_cause_attached(self, clock: ManualClock) -> None:
        providers = [MockProvider(make_provider_config(f"p{i}"), clock=clock) for i in range(3)]
        for provider in providers:
            provider.set_failure(_failure_for(provider.name))

        client = build_client(providers, clock=clock)
        with pytest.raises(NoProviderAvailableError) as caught:
            await client.chat(request_for())

        assert isinstance(caught.value.cause, ProviderUnavailableError)
        assert "attempted" in caught.value.context

    async def test_failover_disabled_means_a_single_attempt(self, clock: ManualClock) -> None:
        primary = MockProvider(make_provider_config("primary", latency_ms=10.0), clock=clock)
        backup = MockProvider(make_provider_config("backup", latency_ms=999.0), clock=clock)
        primary.set_failure(lambda: ProviderUnavailableError("down", provider="primary"))

        client = build_client(
            [primary, backup],
            clock=clock,
            failover_enabled=False,
            strategy=RoutingStrategyName.FASTEST,
        )
        with pytest.raises(NoProviderAvailableError):
            await client.chat(request_for())
        assert backup.call_count == 0

    async def test_repeated_failures_open_the_circuit_and_stop_traffic(
        self, clock: ManualClock
    ) -> None:
        broken = MockProvider(make_provider_config("broken"), clock=clock)
        healthy = MockProvider(make_provider_config("healthy"), clock=clock)
        broken.set_failure(lambda: ProviderUnavailableError("down", provider="broken"))

        client = build_client([broken, healthy], clock=clock)
        for _ in range(3):
            await client.chat(request_for())

        calls_before = broken.call_count
        await client.chat(request_for())
        assert broken.call_count == calls_before

    async def test_no_eligible_provider_reports_why(self, clock: ManualClock) -> None:
        cloud = MockProvider(
            make_provider_config("cloud", privacy_tier=PrivacyTier.PUBLIC_CLOUD), clock=clock
        )
        client = build_client([cloud], clock=clock)
        request = ChatRequest(
            messages=[Message.user("secret")],
            minimum_privacy_tier=PrivacyTier.LOCAL_ONLY,
        )
        with pytest.raises(NoProviderAvailableError) as caught:
            await client.chat(request)
        assert caught.value.context["excluded"]["cloud"] == "privacy_tier_below_floor"


class TestGatewayStreaming:
    async def test_streams_from_the_selected_provider(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("s"), clock=clock)
        client = build_client([provider], clock=clock)
        chunks = [chunk async for chunk in await client.stream(request_for("one two"))]
        assert "".join(chunk.delta for chunk in chunks).endswith("one two")
        assert chunks[-1].is_final

    async def test_open_failure_falls_over_before_any_bytes(self, clock: ManualClock) -> None:
        broken = MockProvider(make_provider_config("broken", latency_ms=1.0), clock=clock)
        good = MockProvider(make_provider_config("good", latency_ms=2.0), clock=clock)
        broken.set_failure(lambda: ProviderUnavailableError("down", provider="broken"))

        client = build_client([broken, good], clock=clock, strategy=RoutingStrategyName.FASTEST)
        chunks = [chunk async for chunk in await client.stream(request_for("x"))]
        assert all(chunk.provider == "good" for chunk in chunks)


class TestGatewayObservability:
    async def test_health_summary_reflects_outcomes(self, clock: ManualClock) -> None:
        good = MockProvider(make_provider_config("good"), clock=clock)
        bad = MockProvider(make_provider_config("bad"), clock=clock)
        bad.set_failure(lambda: ProviderUnavailableError("down", provider="bad"))

        client = build_client([good, bad], clock=clock)
        await client.probe_all()

        summary = {line.provider: line for line in client.health_summary()}
        assert summary["bad"].success_rate < 1.0
        assert summary["good"].success_rate == 1.0

    async def test_lookup_by_name(self, clock: ManualClock) -> None:
        provider = MockProvider(make_provider_config("named"), clock=clock)
        client = build_client([provider], clock=clock)
        assert client.provider("named") is provider
        assert client.provider("absent") is None


class TestProviderSwapping:
    """The architectural promise: swapping vendors is a configuration change."""

    async def business_logic(self, gateway: GatewayClient) -> str:
        """Application code that must remain identical across every vendor."""
        response = await gateway.chat(
            ChatRequest(messages=[Message.system("be brief"), Message.user("summarise")])
        )
        return response.content

    @pytest.mark.parametrize(
        ("kind", "base_url"),
        [
            (ProviderKind.OPENAI_COMPATIBLE, "https://api.openai.test/v1"),
            (ProviderKind.OPENAI_COMPATIBLE, "https://api.groq.test/openai/v1"),
            (ProviderKind.OPENAI_COMPATIBLE, "http://localhost:11434/v1"),
            (ProviderKind.ANTHROPIC, "https://api.anthropic.test"),
            (ProviderKind.GEMINI, "https://gen.googleapis.test"),
        ],
    )
    async def test_swapping_vendors_is_a_config_change(
        self,
        kind: ProviderKind,
        base_url: str,
        transport: FakeTransport,
        clock: ManualClock,
    ) -> None:
        if kind is ProviderKind.ANTHROPIC:
            transport.queue_json({"content": [{"type": "text", "text": "done"}]})
        elif kind is ProviderKind.GEMINI:
            transport.queue_json({"candidates": [{"content": {"parts": [{"text": "done"}]}}]})
        else:
            transport.queue_json({"choices": [{"message": {"content": "done"}}]})

        config = EdenConfig(
            environment=Environment.TESTING,
            logging=LoggingConfig(),
            gateway=GatewayConfig(
                providers=(make_provider_config("vendor", kind=kind, base_url=base_url),),
            ),
        )
        kernel = EdenKernel(config, transport=transport, clock=clock)
        async with kernel.session() as running:
            assert await self.business_logic(running.gateway) == "done"


class TestKernelLifecycle:
    async def test_boots_and_shuts_down_cleanly(self, clock: ManualClock) -> None:
        config = EdenConfig(
            gateway=GatewayConfig(providers=(make_provider_config("m"),)),
        )
        kernel = EdenKernel(config, clock=clock)
        assert kernel.is_started is False
        async with kernel.session() as running:
            assert running.is_started is True
            assert len(running.gateway.providers) == 1
        assert kernel.is_started is False

    async def test_gateway_is_unavailable_before_start(self, clock: ManualClock) -> None:
        kernel = EdenKernel(EdenConfig(), clock=clock)
        with pytest.raises(LifecycleError):
            _ = kernel.gateway

    async def test_start_is_idempotent(self, clock: ManualClock) -> None:
        kernel = EdenKernel(EdenConfig(), clock=clock)
        await kernel.start()
        first = kernel.gateway
        await kernel.start()
        assert kernel.gateway is first
        await kernel.stop()
        await kernel.stop()

    async def test_host_may_override_any_binding(self, clock: ManualClock) -> None:
        container = Container()
        sentinel = MockProvider(make_provider_config("host-supplied"), clock=clock)
        router_config = RouterConfig()
        tracker = HealthTracker(config=router_config.circuit_breaker, clock=clock)
        container.register_instance(
            GatewayClient,
            GatewayClient(
                GatewayConfig(providers=(sentinel.config,)),
                [sentinel],
                OmniRouter(router_config, tracker),
                tracker,
            ),
        )
        kernel = EdenKernel(EdenConfig(), container=container, clock=clock)
        async with kernel.session() as running:
            assert running.gateway.provider("host-supplied") is sentinel

    async def test_no_transport_is_created_for_local_only_fleets(self, clock: ManualClock) -> None:
        config = EdenConfig(gateway=GatewayConfig(providers=(make_provider_config("m"),)))
        kernel = EdenKernel(config, clock=clock)
        async with kernel.session() as running:
            assert running.gateway.providers[0].name == "m"

    async def test_directories_are_created_on_boot(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        paths = PathsConfig(
            root=tmp_path / "eden",
            data_dir=tmp_path / "eden" / "data",
            cache_dir=tmp_path / "eden" / "cache",
            log_dir=tmp_path / "eden" / "logs",
            plugin_dir=tmp_path / "eden" / "plugins",
        )
        kernel = EdenKernel(EdenConfig(paths=paths), clock=clock)
        async with kernel.session():
            assert all(directory.is_dir() for directory in paths.all_directories())


class TestEndToEndConfiguration:
    async def test_a_toml_file_boots_a_working_system(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config_file = tmp_path / "eden.toml"
        config_file.write_text(
            "\n".join(
                [
                    'app_name = "eden-integration"',
                    'environment = "testing"',
                    "[paths]",
                    f'root = "{(tmp_path / "state").as_posix()}"',
                    "[gateway.router]",
                    'strategy = "privacy_first"',
                    "[[gateway.providers]]",
                    'name = "local"',
                    'kind = "mock"',
                    'default_model = "local-model"',
                    'privacy_tier = "local_only"',
                    "[[gateway.providers.models]]",
                    'name = "local-model"',
                    'capabilities = ["chat", "streaming"]',
                ]
            ),
            encoding="utf-8",
        )
        config = ConfigLoader().with_toml(config_file, required=True).build()
        assert config.gateway.router.strategy is RoutingStrategyName.PRIVACY_FIRST

        async with EdenKernel(config, clock=clock).session() as kernel:
            response = await kernel.gateway.chat(request_for("integration"))
            assert response.provider == "local"
            assert "integration" in response.content


class TestMemoryIntegration:
    """Memory reached through a booted kernel, wired by the composition root."""

    def config_with_memory(self, tmp_path: Path) -> EdenConfig:
        """Return a config with mock inference and durable memory."""
        return EdenConfig(
            environment=Environment.TESTING,
            paths=PathsConfig(
                root=tmp_path / "eden",
                data_dir=tmp_path / "eden" / "data",
                cache_dir=tmp_path / "eden" / "cache",
                log_dir=tmp_path / "eden" / "logs",
                plugin_dir=tmp_path / "eden" / "plugins",
            ),
            gateway=GatewayConfig(providers=(make_provider_config("local"),)),
            memory=MemoryConfig(persist=True, vector_dimensions=32),
        )

    async def test_kernel_starts_and_stops_memory(self, tmp_path: Path, clock: ManualClock) -> None:
        kernel = EdenKernel(self.config_with_memory(tmp_path), clock=clock)
        async with kernel.session() as running:
            assert len(running.memory.stores) == 5

    async def test_memory_survives_a_restart(self, tmp_path: Path, clock: ManualClock) -> None:
        config = self.config_with_memory(tmp_path)
        async with EdenKernel(config, clock=clock).session() as kernel:
            await kernel.memory.remember(
                "the deploy command is make ship",
                kind=MemoryKind.LONG_TERM,
                namespace="eden",
            )
        async with EdenKernel(config, clock=clock).session() as kernel:
            hits = await kernel.memory.recall(MemoryQuery(text="deploy command", namespace="eden"))
            assert any("make ship" in hit.record.content for hit in hits)

    async def test_disabled_memory_is_reported_clearly(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config = replace(self.config_with_memory(tmp_path), memory=MemoryConfig(enabled=False))
        async with EdenKernel(config, clock=clock).session() as kernel:
            with pytest.raises(LifecycleError):
                _ = kernel.memory

    async def test_embeddings_route_through_the_gateway(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config = replace(
            self.config_with_memory(tmp_path),
            gateway=GatewayConfig(
                providers=(
                    make_provider_config(
                        "embedder",
                        capabilities=frozenset(
                            {Capability.CHAT, Capability.STREAMING, Capability.EMBEDDING}
                        ),
                    ),
                )
            ),
        )
        async with EdenKernel(config, clock=clock).session() as kernel:
            response = await kernel.gateway.embed(EmbeddingRequest(texts=["alpha", "beta"]))
            assert len(response.vectors) == 2
            assert response.provider == "embedder"

    async def test_no_embedding_provider_is_reported_not_crashed(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        async with EdenKernel(self.config_with_memory(tmp_path), clock=clock).session() as k:
            with pytest.raises(EmbeddingNotSupportedError):
                await k.gateway.embed(EmbeddingRequest(texts=["x"]))

    async def test_vector_memory_degrades_when_embeddings_are_unavailable(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """No embedding provider must not mean no semantic memory."""
        async with EdenKernel(self.config_with_memory(tmp_path), clock=clock).session() as k:
            stored = await k.memory.remember(
                "rollback the payment migration",
                kind=MemoryKind.VECTOR,
                namespace="eden",
            )
            assert stored.embedding is not None
            hits = await k.memory.recall(
                MemoryQuery(text="rollback payment", namespace="eden"),
                kinds=frozenset({MemoryKind.VECTOR}),
            )
            assert hits[0].record.id == stored.id

    async def test_conversation_window_feeds_the_gateway(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """The end-to-end loop: remember turns, window them, generate."""
        async with EdenKernel(self.config_with_memory(tmp_path), clock=clock).session() as k:
            await k.memory.observe(Message.system("be brief"), namespace="chat-1")
            await k.memory.observe(Message.user("what did I ask?"), namespace="chat-1")
            window = await k.memory.conversation.window("chat-1")
            response = await k.gateway.chat(ChatRequest(messages=window))
            assert response.provider == "local"
            assert window[0].role.value == "system"


class TestExecutionIntegration:
    """The execution pipeline reached through a booted kernel."""

    def config_with_execution(self, tmp_path: Path, **overrides: object) -> EdenConfig:
        """Return a config whose workspace lives under ``tmp_path``."""
        base: dict[str, object] = {"workspace_root": tmp_path / "workspace"}
        base.update(overrides)
        return EdenConfig(
            environment=Environment.TESTING,
            paths=PathsConfig(
                root=tmp_path / "eden",
                data_dir=tmp_path / "eden" / "data",
                cache_dir=tmp_path / "eden" / "cache",
                log_dir=tmp_path / "eden" / "logs",
                plugin_dir=tmp_path / "eden" / "plugins",
            ),
            gateway=GatewayConfig(providers=(make_provider_config("local"),)),
            memory=MemoryConfig(persist=False),
            execution=ExecutionConfig(**base),  # type: ignore[arg-type]
        )

    async def test_kernel_starts_all_three_subsystems(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(self.config_with_execution(tmp_path), clock=clock)
        async with kernel.session() as running:
            assert running.gateway is not None
            assert len(running.memory.stores) == 5
            assert running.execution.component_name == "execution"

    async def test_unattended_kernel_refuses_actions_needing_approval(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """No approver wired in means silence is refusal, never consent."""
        config = self.config_with_execution(tmp_path)
        target = tmp_path / "workspace" / "existing.txt"
        target.parent.mkdir(parents=True)
        target.write_text("untouched", encoding="utf-8")

        async with EdenKernel(config, clock=clock).session() as kernel:
            record = await kernel.execution.submit(
                Action(
                    kind=ActionKind.FILE_WRITE,
                    summary="Overwrite an existing file",
                    parameters={"path": "existing.txt", "content": "changed"},
                )
            )
        assert record.status is ExecutionStatus.DENIED
        assert target.read_text(encoding="utf-8") == "untouched"

    async def test_an_injected_gate_enables_confirmed_actions(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        seen: list[str] = []

        async def approve(action: Action, verdict: Verdict) -> bool:
            seen.append(f"{action.summary}:{verdict.risk.name.lower()}")
            return True

        config = self.config_with_execution(tmp_path)
        target = tmp_path / "workspace" / "existing.txt"
        target.parent.mkdir(parents=True)
        target.write_text("v1", encoding="utf-8")

        kernel = EdenKernel(config, clock=clock, approval_gate=CallbackGate(approve))
        async with kernel.session() as running:
            record = await running.execution.submit(
                Action(
                    kind=ActionKind.FILE_WRITE,
                    summary="Overwrite an existing file",
                    parameters={"path": "existing.txt", "content": "v2"},
                )
            )
        assert record.succeeded is True
        assert target.read_text(encoding="utf-8") == "v2"
        assert seen == ["Overwrite an existing file:moderate"]

    async def test_the_journal_is_written_under_the_data_directory(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config = self.config_with_execution(tmp_path)
        async with EdenKernel(config, clock=clock).session() as kernel:
            await kernel.execution.submit(Action(kind=ActionKind.NOOP, summary="audited"))
        journal = tmp_path / "eden" / "data" / "execution" / "execution.jsonl"
        assert journal.is_file()
        assert "audited" in journal.read_text(encoding="utf-8")

    async def test_disabled_execution_is_reported_clearly(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config = replace(
            self.config_with_execution(tmp_path), execution=ExecutionConfig(enabled=False)
        )
        async with EdenKernel(config, clock=clock).session() as kernel:
            with pytest.raises(LifecycleError):
                _ = kernel.execution

    async def test_generate_then_review_then_execute(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """The loop Phase 4 agents will use: model proposes, pipeline disposes.

        The model's output becomes an *Action* — inert data — which is reviewed
        and only then submitted. At no point can generated text reach the
        filesystem without passing verification and permission.
        """
        config = self.config_with_execution(tmp_path)
        kernel = EdenKernel(config, clock=clock, approval_gate=AlwaysApproveGate())
        async with kernel.session() as running:
            generated = await running.gateway.chat(
                ChatRequest(messages=[Message.user("draft release notes")])
            )
            proposal = Action(
                kind=ActionKind.FILE_WRITE,
                summary="Write the drafted release notes",
                parameters={"path": "notes.md", "content": generated.content},
                actor="drafting-agent",
            )

            verdict, _ = await running.execution.review(proposal)
            assert verdict.blocked is False
            assert verdict.reversible is True
            assert not (tmp_path / "workspace" / "notes.md").exists()

            record = await running.execution.submit(proposal)
            assert record.succeeded is True
            written = (tmp_path / "workspace" / "notes.md").read_text(encoding="utf-8")
            assert "draft release notes" in written

    async def test_a_model_proposing_an_escape_is_stopped(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """Even a fully approving gate cannot clear a blocking finding."""
        config = self.config_with_execution(tmp_path)
        kernel = EdenKernel(config, clock=clock, approval_gate=AlwaysApproveGate())
        async with kernel.session() as running:
            record = await running.execution.submit(
                Action(
                    kind=ActionKind.FILE_WRITE,
                    summary="Innocuous-sounding summary",
                    parameters={"path": "../../.ssh/authorized_keys", "content": "key"},
                    actor="hostile-agent",
                )
            )
        assert record.status is ExecutionStatus.REJECTED
        assert record.executed is False


class TestAgentIntegration:
    """Agents reached through a fully booted kernel."""

    def full_config(self, tmp_path: Path, **agent_overrides: object) -> EdenConfig:
        """Return a config with all four subsystems enabled."""
        return EdenConfig(
            environment=Environment.TESTING,
            paths=PathsConfig(
                root=tmp_path / "eden",
                data_dir=tmp_path / "eden" / "data",
                cache_dir=tmp_path / "eden" / "cache",
                log_dir=tmp_path / "eden" / "logs",
                plugin_dir=tmp_path / "eden" / "plugins",
            ),
            gateway=GatewayConfig(providers=(make_provider_config("local"),)),
            memory=MemoryConfig(persist=True, vector_dimensions=32),
            execution=ExecutionConfig(workspace_root=tmp_path / "workspace"),
            agents=AgentConfig(**agent_overrides),  # type: ignore[arg-type]
        )

    async def test_all_four_subsystems_start_in_dependency_order(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(self.full_config(tmp_path), clock=clock)
        async with kernel.session() as running:
            names = [component.component_name for component in running._components]
        assert names == ["ai-gateway", "memory", "execution", "agents"]

    async def test_a_question_is_routed_to_the_conversation_agent(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        async with EdenKernel(self.full_config(tmp_path), clock=clock).session() as kernel:
            report = await kernel.agents.dispatch(
                Task(goal="explain dependency injection", namespace="chat")
            )
        assert report.agent == "conversation"
        assert report.succeeded is True

    async def test_a_file_task_is_routed_and_executed(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as running:
            report = await running.agents.dispatch(
                Task(
                    goal="write a summary file",
                    context={"path": "summary.md", "content": "# Summary"},
                )
            )
        assert report.agent == "filetask"
        assert report.succeeded is True
        assert (tmp_path / "workspace" / "summary.md").exists()

    async def test_an_agent_cannot_escape_the_workspace(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """Phase 3's guarantees are inherited, not re-implemented."""
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as running:
            report = await running.agents.dispatch(
                Task(
                    goal="write a file",
                    context={"path": "../../pwned.txt", "content": "x"},
                )
            )
        assert report.succeeded is False
        assert not (tmp_path.parent / "pwned.txt").exists()

    async def test_an_unattended_kernel_agent_cannot_change_anything(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "existing.txt").write_text("original", encoding="utf-8")
        async with EdenKernel(self.full_config(tmp_path), clock=clock).session() as kernel:
            report = await kernel.agents.dispatch(
                Task(
                    goal="update the file",
                    context={"path": "existing.txt", "content": "changed"},
                )
            )
        assert report.succeeded is False
        assert (workspace / "existing.txt").read_text(encoding="utf-8") == "original"

    async def test_conversation_survives_a_restart_through_memory(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config = self.full_config(tmp_path)
        async with EdenKernel(config, clock=clock).session() as kernel:
            await kernel.agents.dispatch(
                Task(goal="remember the launch is in March", namespace="project-x")
            )
        async with EdenKernel(config, clock=clock).session() as kernel:
            hits = await kernel.memory.recall(
                MemoryQuery(text="launch March", namespace="project-x")
            )
        assert any("March" in hit.record.content for hit in hits)

    async def test_agent_work_is_journalled_by_the_execution_layer(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as running:
            await running.agents.dispatch(
                Task(goal="write notes", context={"path": "n.md", "content": "x"})
            )
        journal = tmp_path / "eden" / "data" / "execution" / "execution.jsonl"
        assert "filetask" in journal.read_text(encoding="utf-8")

    async def test_disabled_agents_are_reported_clearly(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config = replace(self.full_config(tmp_path), agents=AgentConfig(enabled=False))
        async with EdenKernel(config, clock=clock).session() as kernel:
            with pytest.raises(LifecycleError):
                _ = kernel.agents

    async def test_agents_run_with_execution_disabled(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """A read-only EDEN is a valid deployment, not a broken one."""
        config = replace(self.full_config(tmp_path), execution=ExecutionConfig(enabled=False))
        async with EdenKernel(config, clock=clock).session() as kernel:
            report = await kernel.agents.dispatch(Task(goal="explain something"))
            assert report.succeeded is True
            decision = kernel.agents.route(Task(goal="write a file", context={"path": "a.txt"}))
            assert decision.winner == "conversation"

    async def test_a_batch_of_tasks_runs_in_order(self, tmp_path: Path, clock: ManualClock) -> None:
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as running:
            reports = await running.agents.dispatch_all(
                [
                    Task(goal="write one", context={"path": "one.txt", "content": "1"}),
                    Task(goal="write two", context={"path": "two.txt", "content": "2"}),
                    Task(goal="explain the result"),
                ]
            )
        assert [report.succeeded for report in reports] == [True, True, True]
        assert (tmp_path / "workspace" / "one.txt").exists()
        assert (tmp_path / "workspace" / "two.txt").exists()


class TestPhase5Integration:
    """Hardware, automation and the interface reached through a booted kernel."""

    def full_config(self, tmp_path: Path, **overrides: object) -> EdenConfig:
        """Return a config with every subsystem switched on."""
        base = EdenConfig(
            environment=Environment.TESTING,
            paths=PathsConfig(
                root=tmp_path / "eden",
                data_dir=tmp_path / "eden" / "data",
                cache_dir=tmp_path / "eden" / "cache",
                log_dir=tmp_path / "eden" / "logs",
                plugin_dir=tmp_path / "eden" / "plugins",
            ),
            gateway=GatewayConfig(providers=(make_provider_config("local"),)),
            memory=MemoryConfig(persist=False),
            execution=ExecutionConfig(workspace_root=tmp_path / "workspace"),
            hardware=HardwareConfig(
                enabled=True,
                devices=(
                    DeviceConfig(
                        name="rig",
                        channels=("servo",),
                        limits=(("servo", 0.0, 90.0),),
                    ),
                ),
            ),
            automation=AutomationConfig(enabled=True),
            interface=InterfaceConfig(enabled=True, port=0),
        )
        return replace(base, **overrides) if overrides else base  # type: ignore[arg-type]

    async def test_all_six_subsystems_boot_and_shut_down(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(self.full_config(tmp_path), clock=clock)
        async with kernel.session() as k:
            assert k.gateway is not None
            assert len(k.memory.stores) == 5
            assert k.execution.component_name == "execution"
            assert k.agents is not None
            assert len(k.hardware.devices) == 1
            assert k.automation.component_name == "automation"
        assert kernel.is_started is False

    async def test_device_actuation_is_journalled_like_any_other_action(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as k:
            record = await k.execution.submit(
                Action(
                    kind=ActionKind.DEVICE_COMMAND,
                    summary="Move the servo to 45 degrees",
                    parameters={"device": "rig", "channel": "servo", "value": 45.0},
                )
            )
            assert record.succeeded is True
            assert (await k.hardware.read("rig", "servo")).value == 45.0
            entries = await k.execution.journal.read()
            assert entries[-1]["action"]["kind"] == "device_command"

    async def test_an_unsafe_device_command_is_refused_end_to_end(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """Even a fully approving gate cannot push a servo past its limit."""
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as k:
            before = (await k.hardware.read("rig", "servo")).value
            record = await k.execution.submit(
                Action(
                    kind=ActionKind.DEVICE_COMMAND,
                    summary="Slam the servo to 400 degrees",
                    parameters={"device": "rig", "channel": "servo", "value": 400.0},
                )
            )
            assert record.succeeded is False
            assert (await k.hardware.read("rig", "servo")).value == before

    async def test_an_automation_rule_drives_an_agent(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as k:
            k.automation.register(
                Rule(
                    name="nightly-notes",
                    trigger=every(3600),
                    task=Task(goal="write a file of notes", context={"path": "auto.md"}),
                    description="Drafts notes on a schedule.",
                )
            )
            runs = await k.automation.tick()
            assert runs[0].status is AutomationStatus.SUCCEEDED
            assert (tmp_path / "workspace" / "auto.md").exists()

    async def test_an_automation_rule_drives_a_device(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(
            self.full_config(tmp_path), clock=clock, approval_gate=AlwaysApproveGate()
        )
        async with kernel.session() as k:
            k.automation.register(
                Rule(
                    name="park-servo",
                    trigger=every(60),
                    action=Action(
                        kind=ActionKind.DEVICE_COMMAND,
                        summary="Park the servo",
                        parameters={"device": "rig", "channel": "servo", "value": 0.0},
                    ),
                )
            )
            await k.automation.tick()
            assert (await k.hardware.read("rig", "servo")).value == 0.0

    async def test_the_web_console_serves_and_reports(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        kernel = EdenKernel(self.full_config(tmp_path), clock=clock)
        async with kernel.session() as k:
            gate = WebApprovalGate(k.config.interface)
            server = HttpServer(k.config.interface, build_router(k, gate))
            await server.start()
            try:
                index = await _http_get(server.port, "/")
                status = json.loads(await _http_get(server.port, "/api/status"))
                devices = json.loads(await _http_get(server.port, "/api/devices"))
                missing = await _http_get(server.port, "/api/nope")
            finally:
                await server.stop()

        assert "EDEN" in index
        assert status["subsystems"]["hardware"] is True
        assert devices[0]["name"] == "rig"
        assert "404" in missing or "No route" in missing

    async def test_observe_only_mode_refuses_mutation_but_allows_reads(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        config = replace(
            self.full_config(tmp_path),
            interface=InterfaceConfig(enabled=True, port=0, allow_actions=False),
        )
        kernel = EdenKernel(config, clock=clock)
        async with kernel.session() as k:
            server = HttpServer(k.config.interface, build_router(k, None))
            await server.start()
            try:
                blocked = json.loads(
                    await _http_post(server.port, "/api/task", {"goal": "do a thing"})
                )
                reading = json.loads(
                    await _http_get(server.port, "/api/device/read?device=rig&channel=servo")
                )
            finally:
                await server.stop()

        assert "observe-only" in blocked["error"]
        assert reading["channel"] == "servo"


async def _http_get(port: int, path: str) -> str:
    """Perform a bare HTTP GET against the local console."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    await writer.drain()
    raw = await reader.read()
    writer.close()
    return raw.decode("utf-8", errors="replace").split("\r\n\r\n", 1)[-1]


async def _http_post(port: int, path: str, payload: dict[str, object]) -> str:
    """Perform a bare HTTP POST against the local console."""
    body = json.dumps(payload).encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
    )
    await writer.drain()
    raw = await reader.read()
    writer.close()
    return raw.decode("utf-8", errors="replace").split("\r\n\r\n", 1)[-1]
