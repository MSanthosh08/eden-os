"""The agent capability surface.

:class:`AgentContext` is the complete set of things an agent can do. It is
handed in at construction, and an agent has no other route to the outside world:
it cannot build a gateway client, reach a memory store directly, or — critically
— obtain an execution handler.

That last point is what makes "an agent cannot act unsupervised" structural
rather than a rule people remember. :meth:`AgentContext.act` goes through
:class:`~eden.execution.engine.ExecutionEngine`, and nothing in this class
exposes anything beneath it.

Subsystems are optional. An EDEN with execution disabled still runs agents; they
simply find that :meth:`act` refuses, which is the honest outcome rather than a
crash on startup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from eden.config.enums import MemoryKind
from eden.config.schema import AgentConfig
from eden.core.types import ChatRequest, ChatResponse, Message
from eden.errors import AgentCapabilityError
from eden.execution.engine import ExecutionEngine
from eden.execution.types import Action, ExecutionRecord, Preparation
from eden.gateway.client import GatewayClient
from eden.logging import get_logger
from eden.memory.manager import MemoryManager
from eden.memory.types import MemoryQuery, MemoryRecord, SearchHit

_LOGGER = get_logger("agents.context")


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """The result of performing one action, with its undo state retained.

    The preparation is kept because an agent's own verification runs *after*
    execution and may decide the goal was not met. Compensating then needs the
    state captured before the action ran, which no longer exists anywhere else.

    Attributes:
        record: The execution audit record.
        preparation: Undo state captured before the action ran.
    """

    record: ExecutionRecord
    preparation: Preparation

    @property
    def succeeded(self) -> bool:
        """Return whether the action completed successfully."""
        return self.record.succeeded


class AgentContext:
    """Everything an agent is permitted to do."""

    def __init__(
        self,
        config: AgentConfig,
        gateway: GatewayClient,
        *,
        memory: MemoryManager | None = None,
        execution: ExecutionEngine | None = None,
    ) -> None:
        """Initialise the context.

        Args:
            config: Constraints applied to every agent.
            gateway: AI Gateway façade. Always required — an agent that cannot
                think is not an agent.
            memory: Memory subsystem, or ``None`` when disabled.
            execution: Execution engine, or ``None`` when disabled.
        """
        self._config = config
        self._gateway = gateway
        self._memory = memory
        self._execution = execution

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    @property
    def config(self) -> AgentConfig:
        """Return the agent constraints."""
        return self._config

    @property
    def has_memory(self) -> bool:
        """Return whether memory is available."""
        return self._memory is not None

    @property
    def has_execution(self) -> bool:
        """Return whether this context can change the world."""
        return self._execution is not None

    @property
    def memory(self) -> MemoryManager:
        """Return the memory subsystem.

        Raises:
            AgentCapabilityError: If memory is disabled.
        """
        if self._memory is None:
            raise AgentCapabilityError(
                "This agent needs memory, but the memory subsystem is disabled.",
                context={"key": "memory.enabled"},
            )
        return self._memory

    # ------------------------------------------------------------------
    # Thinking
    # ------------------------------------------------------------------
    async def think(
        self,
        prompt: str,
        *,
        system: str = "",
        history: Sequence[Message] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Ask a model a question and return its text.

        Args:
            prompt: The question.
            system: Optional instruction prepended to the conversation.
            history: Prior turns to include before the prompt.
            temperature: Sampling temperature. Defaults to the planning value.
            max_tokens: Output ceiling.

        Returns:
            The generated text, stripped.
        """
        response = await self.generate(
            prompt,
            system=system,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content.strip()

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        history: Sequence[Message] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Ask a model a question and return the full response.

        Use this rather than :meth:`think` when the agent needs the provider
        name, token usage or cost for its report.
        """
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.extend(history)
        messages.append(Message.user(prompt))
        return await self._gateway.chat(
            ChatRequest(
                messages=messages,
                model=self._config.planning_model,
                temperature=(
                    self._config.planning_temperature if temperature is None else temperature
                ),
                max_tokens=max_tokens,
            )
        )

    # ------------------------------------------------------------------
    # Remembering
    # ------------------------------------------------------------------
    async def recall(
        self,
        text: str,
        *,
        namespace: str,
        limit: int | None = None,
    ) -> list[SearchHit]:
        """Search memory, returning an empty list when memory is disabled.

        Recall degrades silently because an agent that cannot remember should
        still be able to answer; a missing memory is not a failed task.
        """
        if self._memory is None:
            return []
        return await self._memory.recall(
            MemoryQuery(
                text=text,
                namespace=namespace,
                limit=limit or self._config.recall_limit,
            )
        )

    async def remember(
        self,
        content: str,
        *,
        namespace: str,
        kind: MemoryKind = MemoryKind.LONG_TERM,
        importance: float = 0.5,
        tags: frozenset[str] = frozenset(),
    ) -> MemoryRecord | None:
        """Store a memory, returning ``None`` when memory is disabled."""
        if self._memory is None:
            return None
        return await self._memory.remember(
            content,
            kind=kind,
            namespace=namespace,
            importance=importance,
            tags=tags,
        )

    async def observe(self, message: Message, *, namespace: str) -> None:
        """Record a conversation turn and consolidate if it has grown long."""
        if self._memory is None:
            return
        await self._memory.observe_and_consolidate(message, namespace=namespace)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    async def act(self, action: Action) -> ActionOutcome:
        """Perform ``action`` through the execution pipeline.

        This is the only way an agent changes anything. The action is prepared,
        verified, permitted and only then executed; the agent has no say in any
        of those phases.

        Args:
            action: The intended effect.

        Returns:
            The audit record together with the undo state captured beforehand.

        Raises:
            AgentCapabilityError: If execution is disabled.
        """
        engine = self._require_execution()
        # Review first so the undo state survives for post-hoc compensation.
        # Preparation is read-only by contract, so running it twice is safe.
        _, preparation = await engine.review(action)
        record = await engine.submit(action)
        return ActionOutcome(record=record, preparation=preparation)

    async def preview(self, action: Action) -> str:
        """Return a human-readable assessment of ``action`` without running it.

        Raises:
            AgentCapabilityError: If execution is disabled.
        """
        engine = self._require_execution()
        verdict, _ = await engine.review(action)
        lines = [
            f"{action.summary}",
            f"  risk: {verdict.risk.name.lower()}",
            f"  reversible: {verdict.reversible}",
        ]
        lines.extend(f"  - {finding.message}" for finding in verdict.findings)
        return "\n".join(lines)

    async def undo(self, outcome: ActionOutcome) -> bool:
        """Compensate a completed action. Returns whether it worked.

        Raises:
            AgentCapabilityError: If execution is disabled.
        """
        engine = self._require_execution()
        if not outcome.preparation.reversible:
            _LOGGER.warning(
                "Agent asked to undo an irreversible action.",
                extra={"action": outcome.record.action.id},
            )
            return False
        rolled = await engine.rollback(outcome.record, outcome.preparation)
        return rolled.rollback_applied

    def _require_execution(self) -> ExecutionEngine:
        """Return the execution engine.

        Raises:
            AgentCapabilityError: If execution is disabled.
        """
        if self._execution is None:
            raise AgentCapabilityError(
                "This agent needs to act, but the execution subsystem is disabled.",
                context={"key": "execution.enabled"},
            )
        return self._execution
