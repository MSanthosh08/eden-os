"""Agent orchestration.

The orchestrator answers one question: *which agent should take this task?* It
does so by asking every registered agent through ``can_handle`` and ranking the
answers, which is the same shape as provider selection in ADR-0002 and memory
recall in ADR-0003 — ask everyone, score, explain.

Routing is never hardcoded. An agent becomes preferred by returning a higher
score, not by being registered earlier or named in a conditional.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from eden.agents.base import BaseAgent
from eden.agents.context import AgentContext
from eden.agents.types import AgentReport, Suitability, Task
from eden.config.enums import TaskStatus
from eden.config.schema import AgentConfig
from eden.core.registry import Registry
from eden.errors import NoSuitableAgentError, RegistryError
from eden.execution.engine import ExecutionEngine
from eden.gateway.client import GatewayClient
from eden.logging import get_logger, timed_block
from eden.memory.manager import MemoryManager

_LOGGER = get_logger("agents.orchestrator")

COMPONENT_NAME = "agents"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The explainable outcome of choosing an agent.

    Attributes:
        ranked: Every willing agent with its score, best first.
        declined: Agent name mapped to why it refused.
    """

    ranked: tuple[tuple[str, Suitability], ...] = ()
    declined: dict[str, str] = field(default_factory=dict)

    @property
    def has_candidate(self) -> bool:
        """Return whether any agent was willing."""
        return bool(self.ranked)

    @property
    def winner(self) -> str:
        """Return the name of the best-scoring agent, or an empty string."""
        return self.ranked[0][0] if self.ranked else ""


class AgentOrchestrator:
    """Registers agents, routes tasks to them and owns their lifecycle."""

    def __init__(
        self,
        config: AgentConfig,
        context: AgentContext,
        agents: Sequence[BaseAgent] = (),
    ) -> None:
        """Initialise the orchestrator.

        Args:
            config: Constraints applied to every agent.
            context: The capability surface shared by all agents.
            agents: Agents to register at construction.
        """
        self._config = config
        self._context = context
        self._agents: Registry[BaseAgent] = Registry("agent")
        self._started = False
        for agent in agents:
            self.register(agent)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return COMPONENT_NAME

    async def start(self) -> None:
        """Log the registered roster. Idempotent."""
        if self._started:
            return
        self._started = True
        _LOGGER.info(
            "Agent subsystem started.",
            extra={
                "agents": list(self._agents.names()),
                "can_act": self._context.has_execution,
                "has_memory": self._context.has_memory,
            },
        )

    async def stop(self) -> None:
        """Release the subsystem. Idempotent and never raises."""
        if not self._started:
            return
        self._started = False
        _LOGGER.info("Agent subsystem stopped.")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, agent: BaseAgent, *, replace: bool = False) -> None:
        """Register ``agent`` under its name.

        Raises:
            RegistryError: If the name is taken and ``replace`` is ``False``.
        """
        self._agents.register(agent.name, lambda: agent, replace=replace)

    def unregister(self, name: str) -> None:
        """Remove an agent from the roster.

        Raises:
            RegistryError: If the name is not registered.
        """
        self._agents.unregister(name)

    @property
    def agents(self) -> tuple[BaseAgent, ...]:
        """Return every registered agent."""
        return tuple(self._agents.create_all().values())

    def agent(self, name: str) -> BaseAgent:
        """Return the agent registered under ``name``.

        Raises:
            RegistryError: If the name is unknown.
        """
        return self._agents.create(name)

    def roster(self) -> dict[str, str]:
        """Return each agent's name mapped to its description."""
        return {agent.name: agent.description for agent in self.agents}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def route(self, task: Task) -> RoutingDecision:
        """Return the ranked shortlist of agents willing to take ``task``.

        ``can_handle`` is documented as cheap and side-effect free, but an agent
        that raises here is treated as declining rather than being allowed to
        break routing for everybody else.
        """
        ranked: list[tuple[str, Suitability]] = []
        declined: dict[str, str] = {}

        for agent in self.agents:
            try:
                suitability = agent.can_handle(task)
            except Exception as exc:  # noqa: BLE001 - one bad agent must not block routing
                declined[agent.name] = f"can_handle raised {type(exc).__name__}"
                _LOGGER.warning(
                    "Agent raised during routing; treating it as declining.",
                    extra={"agent": agent.name, "error_type": type(exc).__name__},
                )
                continue
            if suitability.can_handle and suitability.score >= self._config.min_suitability:
                ranked.append((agent.name, suitability))
            else:
                declined[agent.name] = suitability.reason or "below the suitability floor"

        ranked.sort(key=lambda entry: (-entry[1].score, entry[0]))
        decision = RoutingDecision(ranked=tuple(ranked), declined=declined)
        _LOGGER.debug(
            "Task routed.",
            extra={
                "task": task.id,
                "winner": decision.winner,
                "candidates": [name for name, _ in ranked],
                "declined": declined,
            },
        )
        return decision

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    async def dispatch(self, task: Task, *, agent: str = "") -> AgentReport:
        """Route ``task`` to an agent and run it.

        Args:
            task: The goal to attempt.
            agent: Force a specific agent by name, bypassing routing.

        Returns:
            The agent's report.

        Raises:
            NoSuitableAgentError: If nothing is willing to take the task.
            RegistryError: If ``agent`` names an unknown agent.
        """
        with timed_block(_LOGGER, "agents.dispatch", task=task.id):
            if agent:
                return await self._agents.create(agent).run(task)

            decision = self.route(task)
            if not decision.has_candidate:
                raise NoSuitableAgentError(
                    "No registered agent is willing to take this task.",
                    context={
                        "task": task.id,
                        "goal": task.goal[:200],
                        "declined": decision.declined,
                    },
                )
            chosen = self._agents.create(decision.winner)
            _LOGGER.info(
                "Task dispatched.",
                extra={
                    "task": task.id,
                    "agent": chosen.name,
                    "score": round(decision.ranked[0][1].score, 3),
                },
            )
            return await chosen.run(task)

    async def dispatch_all(self, tasks: Sequence[Task]) -> list[AgentReport]:
        """Run several tasks in order, continuing past failures.

        Sequential rather than concurrent by design: agents share one workspace
        and one memory namespace, so running them in parallel would let two of
        them race on the same file with no coordination. Concurrency belongs
        after the execution layer can lock, not before.
        """
        reports: list[AgentReport] = []
        for task in tasks:
            try:
                reports.append(await self.dispatch(task))
            except NoSuitableAgentError as exc:
                _LOGGER.warning(
                    "Skipping a task nothing would take.",
                    extra={"task": task.id, "error_code": exc.code},
                )
                reports.append(
                    AgentReport(
                        task=task,
                        agent="",
                        status=TaskStatus.REJECTED,
                        suitability=Suitability.no(exc.message),
                        summary=exc.message,
                    )
                )
        return reports


def build_orchestrator(
    config: AgentConfig,
    gateway: GatewayClient,
    *,
    memory: MemoryManager | None = None,
    execution: ExecutionEngine | None = None,
    agents: Sequence[BaseAgent] | None = None,
) -> AgentOrchestrator:
    """Construct a fully wired agent subsystem.

    Args:
        config: Agent constraints.
        gateway: AI Gateway façade.
        memory: Memory subsystem, or ``None`` when disabled.
        execution: Execution engine, or ``None`` when disabled.
        agents: Explicit roster. Defaults to the built-in agents.

    Returns:
        A ready-to-start orchestrator.
    """
    from eden.agents.builtin import (  # noqa: PLC0415 - avoids an import cycle
        ConversationAgent,
        EchoAgent,
        FileTaskAgent,
    )

    context = AgentContext(config, gateway, memory=memory, execution=execution)
    roster = (
        list(agents)
        if agents is not None
        else [
            ConversationAgent(context),
            FileTaskAgent(context),
            EchoAgent(context),
        ]
    )
    return AgentOrchestrator(config, context, roster)


__all__ = [
    "COMPONENT_NAME",
    "AgentOrchestrator",
    "RegistryError",
    "RoutingDecision",
    "build_orchestrator",
]
