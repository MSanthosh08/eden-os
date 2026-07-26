"""The EDEN kernel.

The kernel is the composition root: the one place that knows how every
subsystem is wired together. Everything else receives its collaborators and
stays ignorant of how they were built.

Startup order follows the dependency order of the system — configuration,
logging, transport, gateway — and shutdown runs strictly in reverse, which is
the only ordering that guarantees a component's dependencies still exist while
it is releasing its own resources.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from types import TracebackType

from eden.agents.orchestrator import AgentOrchestrator, build_orchestrator
from eden.agents.types import Task
from eden.automation.scheduler import AutomationScheduler, build_scheduler
from eden.config.enums import ProviderKind
from eden.config.schema import EdenConfig, ExecutionConfig
from eden.config.secrets import SecretResolver
from eden.core.container import Container, Scope
from eden.core.interfaces import Lifecycle
from eden.errors import LifecycleError
from eden.execution.engine import ExecutionEngine
from eden.execution.handlers import ActionHandler, default_handlers
from eden.execution.journal import build_journal
from eden.execution.permissions import ApprovalGate, PolicyEngine
from eden.execution.types import Action
from eden.gateway.client import GatewayClient
from eden.gateway.factory import ProviderFactory
from eden.gateway.health import HealthTracker
from eden.gateway.router.omni_router import OmniRouter
from eden.hardware.manager import (
    DeviceCommandHandler,
    DeviceManager,
    build_device_manager,
)
from eden.logging import configure_logging, get_logger
from eden.memory.manager import MemoryManager, build_memory_manager
from eden.transport.base import HttpTransport
from eden.utils.clock import Clock, SystemClock

_LOGGER = get_logger("core.kernel")

_NETWORK_KINDS = frozenset(
    {
        ProviderKind.OPENAI_COMPATIBLE,
        ProviderKind.ANTHROPIC,
        ProviderKind.GEMINI,
    }
)


class EdenKernel:
    """Boots, owns and shuts down every EDEN subsystem.

    Example:
        >>> import asyncio
        >>> from eden.config.schema import EdenConfig
        >>> async def main() -> str:
        ...     async with EdenKernel(EdenConfig()).session() as kernel:
        ...         return kernel.config.app_name
        >>> asyncio.run(main())
        'eden'
    """

    def __init__(
        self,
        config: EdenConfig,
        *,
        container: Container | None = None,
        transport: HttpTransport | None = None,
        clock: Clock | None = None,
        secrets: SecretResolver | None = None,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        """Initialise the kernel.

        Args:
            config: Validated configuration tree.
            container: Pre-populated DI container, allowing a host application
                to override any binding before startup.
            transport: HTTP transport. Constructed on demand when omitted and
                at least one network provider is configured.
            clock: Time source shared by every timing-sensitive component.
            secrets: Credential resolver.
            approval_gate: Mechanism consulted when an action needs approval.
                Omitting it means confirmation-requiring actions are refused,
                which is the safe default for an unattended process.
        """
        self._config = config
        self._container = container or Container()
        self._transport = transport
        self._clock = clock or SystemClock()
        self._secrets = secrets or SecretResolver()
        self._approval_gate = approval_gate
        self._components: list[Lifecycle] = []
        self._started = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def config(self) -> EdenConfig:
        """Return the configuration this kernel was built from."""
        return self._config

    @property
    def container(self) -> Container:
        """Return the dependency-injection container."""
        return self._container

    @property
    def is_started(self) -> bool:
        """Return whether startup has completed."""
        return self._started

    @property
    def gateway(self) -> GatewayClient:
        """Return the AI Gateway façade.

        Raises:
            LifecycleError: If the kernel has not been started.
        """
        self._require_started()
        return self._container.resolve(GatewayClient)

    @property
    def memory(self) -> MemoryManager:
        """Return the memory subsystem.

        Raises:
            LifecycleError: If the kernel has not been started, or memory is
                disabled in configuration.
        """
        self._require_started()
        if not self._config.memory.enabled:
            raise LifecycleError(
                "The memory subsystem is disabled in configuration.",
                context={"key": "memory.enabled"},
            )
        return self._container.resolve(MemoryManager)

    @property
    def execution(self) -> ExecutionEngine:
        """Return the execution engine.

        Raises:
            LifecycleError: If the kernel has not been started, or execution is
                disabled in configuration.
        """
        self._require_started()
        if not self._config.execution.enabled:
            raise LifecycleError(
                "The execution subsystem is disabled in configuration.",
                context={"key": "execution.enabled"},
            )
        return self._container.resolve(ExecutionEngine)

    @property
    def agents(self) -> AgentOrchestrator:
        """Return the agent orchestrator.

        Raises:
            LifecycleError: If the kernel has not been started, or agents are
                disabled in configuration.
        """
        self._require_started()
        if not self._config.agents.enabled:
            raise LifecycleError(
                "The agent subsystem is disabled in configuration.",
                context={"key": "agents.enabled"},
            )
        return self._container.resolve(AgentOrchestrator)

    @property
    def hardware(self) -> DeviceManager:
        """Return the device fleet.

        Raises:
            LifecycleError: If the kernel has not been started, or hardware is
                disabled in configuration.
        """
        self._require_started()
        if not self._config.hardware.enabled:
            raise LifecycleError(
                "The hardware subsystem is disabled in configuration.",
                context={"key": "hardware.enabled"},
            )
        return self._container.resolve(DeviceManager)

    @property
    def automation(self) -> AutomationScheduler:
        """Return the automation scheduler.

        Raises:
            LifecycleError: If the kernel has not been started, or automation is
                disabled in configuration.
        """
        self._require_started()
        if not self._config.automation.enabled:
            raise LifecycleError(
                "The automation subsystem is disabled in configuration.",
                context={"key": "automation.enabled"},
            )
        return self._container.resolve(AutomationScheduler)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> EdenKernel:
        """Wire and start every subsystem.

        Returns:
            This kernel, so the call can be chained.

        Raises:
            LifecycleError: If any component fails to start. Components already
                started are stopped again before the error propagates, so a
                failed boot never leaves resources dangling.
        """
        if self._started:
            return self

        configure_logging(self._config.logging, self._config.paths)
        self._config.paths.ensure()
        _LOGGER.info(
            "Starting EDEN.",
            extra={
                "app": self._config.app_name,
                "environment": str(self._config.environment),
                "version": self._config.version,
            },
        )

        self._register_bindings()
        # Startup order mirrors the dependency order: memory may embed through
        # the gateway, so the gateway comes first and is stopped last.
        self._components = [self._container.resolve(GatewayClient)]
        if self._config.memory.enabled:
            self._components.append(self._container.resolve(MemoryManager))
        if self._config.hardware.enabled:
            self._components.append(self._container.resolve(DeviceManager))
        if self._config.execution.enabled:
            self._components.append(self._container.resolve(ExecutionEngine))
        if self._config.agents.enabled:
            self._components.append(self._container.resolve(AgentOrchestrator))
        if self._config.automation.enabled:
            self._components.append(self._container.resolve(AutomationScheduler))

        started: list[Lifecycle] = []
        try:
            for component in self._components:
                await component.start()
                started.append(component)
                _LOGGER.debug("Component started.", extra={"component": component.component_name})
        except Exception as exc:
            await self._stop_components(reversed(started))
            raise LifecycleError(
                "EDEN failed to start.",
                context={"started": [item.component_name for item in started]},
                cause=exc,
            ) from exc

        self._started = True
        _LOGGER.info(
            "EDEN started.",
            extra={"components": [item.component_name for item in self._components]},
        )
        return self

    async def stop(self) -> None:
        """Stop every subsystem in reverse order. Idempotent and never raises."""
        if not self._started:
            return
        _LOGGER.info("Stopping EDEN.")
        await self._stop_components(reversed(self._components))
        if self._transport is not None:
            try:
                await self._transport.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                _LOGGER.warning(
                    "Transport failed to close cleanly.",
                    extra={"error_type": type(exc).__name__},
                )
        self._started = False
        _LOGGER.info("EDEN stopped.")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[EdenKernel]:
        """Start the kernel, yield it, and guarantee shutdown.

        Yields:
            The started kernel.
        """
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def __aenter__(self) -> EdenKernel:
        """Start the kernel on context entry."""
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the kernel on context exit."""
        await self.stop()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _register_bindings(self) -> None:
        """Register every binding the kernel owns, without overriding a host's."""
        container = self._container

        if not container.has(EdenConfig):
            container.register_instance(EdenConfig, self._config)
        if not container.has(SecretResolver):
            container.register_instance(SecretResolver, self._secrets)

        if not container.has(HealthTracker):
            container.register(
                HealthTracker,
                lambda _: HealthTracker(
                    config=self._config.gateway.router.circuit_breaker,
                    clock=self._clock,
                ),
                scope=Scope.SINGLETON,
            )

        if not container.has(ProviderFactory):
            container.register(
                ProviderFactory,
                lambda _: ProviderFactory(
                    transport=self._resolve_transport(),
                    secrets=self._secrets,
                    clock=self._clock,
                ),
                scope=Scope.SINGLETON,
            )

        if not container.has(OmniRouter):
            container.register(
                OmniRouter,
                lambda c: OmniRouter(
                    self._config.gateway.router,
                    c.resolve(HealthTracker),
                ),
                scope=Scope.SINGLETON,
            )

        if not container.has(GatewayClient):
            container.register(
                GatewayClient,
                lambda c: GatewayClient(
                    self._config.gateway,
                    c.resolve(ProviderFactory).create_all(self._config.gateway.enabled_providers()),
                    c.resolve(OmniRouter),
                    c.resolve(HealthTracker),
                ),
                scope=Scope.SINGLETON,
            )

        self._register_memory_bindings()
        self._register_hardware_bindings()
        self._register_execution_bindings()
        self._register_agent_bindings()
        self._register_automation_bindings()

    def _register_memory_bindings(self) -> None:
        """Register the memory subsystem, unless a host already supplied one."""
        if not self._config.memory.enabled or self._container.has(MemoryManager):
            return
        self._container.register(
            MemoryManager,
            lambda c: build_memory_manager(
                self._config.memory,
                self._config.paths,
                gateway=c.resolve(GatewayClient),
            ),
            scope=Scope.SINGLETON,
        )

    def _register_execution_bindings(self) -> None:
        """Register the execution subsystem, unless a host already supplied one."""
        if not self._config.execution.enabled or self._container.has(ExecutionEngine):
            return
        execution = self._config.execution
        self._container.register(
            ExecutionEngine,
            lambda c: ExecutionEngine(
                execution,
                handlers=self._execution_handlers(execution, c),
                policy=PolicyEngine(execution, self._approval_gate),
                journal=build_journal(
                    enabled=execution.journal_enabled,
                    directory=self._config.paths.data_dir / "execution",
                ),
                clock=self._clock,
            ),
            scope=Scope.SINGLETON,
        )

    def _execution_handlers(
        self,
        execution: ExecutionConfig,
        container: Container,
    ) -> list[ActionHandler]:
        """Return the handler set, adding device actuation when hardware is on.

        The device handler is registered here rather than inside the hardware
        package, so that moving a servo is structurally just another action the
        Phase 3 pipeline verifies, authorises, journals and can roll back.
        """
        handlers = default_handlers(execution)
        if self._config.hardware.enabled:
            handlers.append(DeviceCommandHandler(execution, container.resolve(DeviceManager)))
        return handlers

    def _register_agent_bindings(self) -> None:
        """Register the agent subsystem, unless a host already supplied one.

        Agents are wired last and started last, because they depend on every
        other subsystem and must not outlive any of them.
        """
        if not self._config.agents.enabled or self._container.has(AgentOrchestrator):
            return
        self._container.register(
            AgentOrchestrator,
            lambda c: build_orchestrator(
                self._config.agents,
                c.resolve(GatewayClient),
                memory=(c.resolve(MemoryManager) if self._config.memory.enabled else None),
                execution=(c.resolve(ExecutionEngine) if self._config.execution.enabled else None),
            ),
            scope=Scope.SINGLETON,
        )

    def _register_hardware_bindings(self) -> None:
        """Register the device fleet, unless a host already supplied one."""
        if not self._config.hardware.enabled or self._container.has(DeviceManager):
            return
        self._container.register(
            DeviceManager,
            lambda _: build_device_manager(
                self._config.hardware,
                transport=self._resolve_transport(),
                clock=self._clock,
            ),
            scope=Scope.SINGLETON,
        )

    def _register_automation_bindings(self) -> None:
        """Register the scheduler, unless a host already supplied one.

        The scheduler is handed *callables*, not subsystems, so it stays
        decoupled from what it drives and cannot reach past their façades.
        """
        if not self._config.automation.enabled or self._container.has(AutomationScheduler):
            return

        async def run_task(task: Task) -> object:
            return await self._container.resolve(AgentOrchestrator).dispatch(task)

        async def run_action(action: Action) -> object:
            return await self._container.resolve(ExecutionEngine).submit(action)

        self._container.register(
            AutomationScheduler,
            lambda _: build_scheduler(
                self._config.automation,
                task_runner=run_task if self._config.agents.enabled else None,
                action_runner=run_action if self._config.execution.enabled else None,
                clock=self._clock,
            ),
            scope=Scope.SINGLETON,
        )

    def _resolve_transport(self) -> HttpTransport | None:
        """Return the HTTP transport, constructing it only when needed.

        A deployment running purely local or mocked providers never pays the
        cost of importing or opening a connection pool.
        """
        if self._transport is not None:
            return self._transport
        needs_network = any(
            provider.kind in _NETWORK_KINDS for provider in self._config.gateway.enabled_providers()
        )
        if not needs_network:
            return None
        from eden.transport.httpx_transport import (  # noqa: PLC0415 - deliberate lazy import
            HttpxTransport,
        )

        self._transport = HttpxTransport(
            default_timeout=self._config.gateway.request_timeout_seconds
        )
        return self._transport

    @staticmethod
    async def _stop_components(components: Iterable[Lifecycle]) -> None:
        """Stop each component, logging and swallowing individual failures."""
        for component in components:
            try:
                await component.stop()
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                _LOGGER.warning(
                    "Component failed to stop cleanly.",
                    extra={
                        "component": component.component_name,
                        "error_type": type(exc).__name__,
                    },
                )

    def _require_started(self) -> None:
        """Raise if the kernel has not been started.

        Raises:
            LifecycleError: If startup has not run.
        """
        if not self._started:
            raise LifecycleError("The kernel has not been started.")
