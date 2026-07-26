"""Agent subsystem.

Every agent inherits :class:`~eden.agents.base.BaseAgent` and has the five
methods: ``can_handle``, ``plan``, ``execute``, ``verify``, ``report``.
:meth:`BaseAgent.run` calls all five in that order, every time, so an agent
cannot skip verification — it does not control the sequence.

An agent's entire capability surface is :class:`~eden.agents.context.AgentContext`.
It cannot construct a gateway, reach a memory store directly, or obtain an
execution handler; ``AgentContext.act`` goes through the Phase 3 pipeline and
nothing exposes anything beneath it. "An agent cannot act unsupervised" is
therefore structural rather than a convention.
"""

from __future__ import annotations

from eden.agents.base import BaseAgent
from eden.agents.builtin import ConversationAgent, EchoAgent, FileTaskAgent
from eden.agents.context import ActionOutcome, AgentContext
from eden.agents.orchestrator import (
    COMPONENT_NAME,
    AgentOrchestrator,
    RoutingDecision,
    build_orchestrator,
)
from eden.agents.types import (
    AgentReport,
    Plan,
    PlanStep,
    StepOutcome,
    Suitability,
    Task,
    Verification,
)

__all__ = [
    "COMPONENT_NAME",
    "ActionOutcome",
    "AgentContext",
    "AgentOrchestrator",
    "AgentReport",
    "BaseAgent",
    "ConversationAgent",
    "EchoAgent",
    "FileTaskAgent",
    "Plan",
    "PlanStep",
    "RoutingDecision",
    "StepOutcome",
    "Suitability",
    "Task",
    "Verification",
    "build_orchestrator",
]
