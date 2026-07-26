"""Agent domain types.

A :class:`Plan` is the agent-level equivalent of an
:class:`~eden.execution.types.Action`: inert data describing intent. Planning
produces one; nothing in it executes. A step that changes the world carries an
``Action``, which still has to pass the whole execution pipeline before anything
happens — so an agent inherits Phase 3's guarantees rather than being trusted to
reimplement them.

That is why plans are worth showing to a human, storing, replaying and refusing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from eden.config.enums import StepStatus, TaskStatus
from eden.errors import ValidationError
from eden.execution.types import Action, ExecutionRecord, Preparation
from eden.utils.ids import new_id

DEFAULT_NAMESPACE = "default"
SYSTEM_ACTOR = "system"


@dataclass(frozen=True, slots=True)
class Task:
    """A goal handed to the agent subsystem.

    Attributes:
        goal: What the requester wants, in their own words.
        id: Stable identifier, generated when omitted.
        actor: Who asked — a user, another agent, or ``"system"``.
        namespace: Isolation boundary for memory and journalling.
        context: Structured inputs the requester already knows.
        constraints: Free-form restrictions an agent must respect, surfaced
            into planning prompts and into reports.
        created_at: UTC creation time.
        metadata: Non-executable annotations.
    """

    goal: str
    id: str = ""
    actor: str = SYSTEM_ACTOR
    namespace: str = DEFAULT_NAMESPACE
    context: Mapping[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and fill in defaults.

        Raises:
            ValidationError: If the task is unusable.
        """
        if not self.goal.strip():
            raise ValidationError("Task goal must not be empty.")
        if not self.namespace.strip():
            raise ValidationError("Task namespace must not be empty.")
        if not self.id:
            object.__setattr__(self, "id", new_id("task"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "goal": self.goal,
            "actor": self.actor,
            "namespace": self.namespace,
            "constraints": list(self.constraints),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class Suitability:
    """One agent's assessment of whether a task is its business.

    A score rather than a boolean, so that several willing agents can be ranked
    without the router hardcoding a preference order — the same reasoning as
    provider selection in ADR-0002.

    Attributes:
        score: Confidence in ``[0, 1]``. Zero means "not mine".
        reason: Why, recorded in the report when a task is refused.
    """

    score: float
    reason: str = ""

    def __post_init__(self) -> None:
        """Clamp the score rather than rejecting an out-of-range value."""
        object.__setattr__(self, "score", max(0.0, min(1.0, self.score)))

    @property
    def can_handle(self) -> bool:
        """Return whether the agent is willing to take the task at all."""
        return self.score > 0.0

    @classmethod
    def no(cls, reason: str) -> Suitability:
        """Return a refusal."""
        return cls(score=0.0, reason=reason)

    @classmethod
    def certain(cls, reason: str = "") -> Suitability:
        """Return full confidence."""
        return cls(score=1.0, reason=reason)


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One unit of work within a plan.

    A step is either an *effect* — it carries an :class:`Action`, which will go
    through the execution pipeline — or a *thought*, which carries a prompt and
    only consults a model. Nothing else is possible, which is what keeps the set
    of things an agent can do enumerable.

    Attributes:
        description: What this step is for, shown to an approver.
        action: The effect, when the step changes the world.
        prompt: The question, when the step only reasons.
        id: Stable identifier, generated when omitted.
        optional: Whether failure should abort the plan.
    """

    description: str
    action: Action | None = None
    prompt: str = ""
    id: str = ""
    optional: bool = False

    def __post_init__(self) -> None:
        """Validate and fill in defaults.

        Raises:
            ValidationError: If the step is empty or ambiguous.
        """
        if not self.description.strip():
            raise ValidationError("Plan step description must not be empty.")
        if self.action is None and not self.prompt.strip():
            raise ValidationError(
                "A plan step must either carry an action or ask something.",
                context={"description": self.description},
            )
        if self.action is not None and self.prompt.strip():
            raise ValidationError(
                "A plan step must not both act and reason; split it in two.",
                context={"description": self.description},
            )
        if not self.id:
            object.__setattr__(self, "id", new_id("step"))

    @property
    def changes_the_world(self) -> bool:
        """Return whether this step has an effect outside the process."""
        return self.action is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "description": self.description,
            "optional": self.optional,
            "action": self.action.to_dict() if self.action else None,
            "prompt": self.prompt,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """An ordered sequence of steps intended to satisfy a task.

    Attributes:
        task_id: Task this plan addresses.
        steps: Work to perform, in order.
        rationale: Why the agent believes this achieves the goal.
    """

    task_id: str
    steps: tuple[PlanStep, ...] = ()
    rationale: str = ""

    @property
    def changes_the_world(self) -> bool:
        """Return whether any step has an effect outside the process."""
        return any(step.changes_the_world for step in self.steps)

    @property
    def effect_count(self) -> int:
        """Return how many steps would change the world."""
        return sum(1 for step in self.steps if step.changes_the_world)

    def describe(self) -> str:
        """Return a human-readable rendering, for approval prompts and logs."""
        lines = [f"Plan for {self.task_id}: {self.rationale}".rstrip(": ")]
        for index, step in enumerate(self.steps, start=1):
            marker = "act " if step.changes_the_world else "think"
            lines.append(f"  {index}. [{marker}] {step.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "task_id": self.task_id,
            "rationale": self.rationale,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """What happened when a step ran.

    Attributes:
        step: The step attempted.
        status: How it ended.
        output: Text produced, whether generated or reported by a handler.
        record: Execution audit record, present for effect steps that ran.
        preparation: Undo state captured for the effect, retained so that the
            agent can compensate later if its own verification fails.
        error: Serialised failure, when one occurred.
    """

    step: PlanStep
    status: StepStatus
    output: str = ""
    record: ExecutionRecord | None = None
    preparation: Preparation | None = None
    error: Mapping[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the step completed successfully."""
        return self.status is StepStatus.SUCCEEDED

    @property
    def reversible(self) -> bool:
        """Return whether this step's effect can still be undone."""
        return self.succeeded and self.preparation is not None and self.preparation.reversible

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "step": self.step.to_dict(),
            "status": self.status.value,
            "output": self.output[:512],
            "record": self.record.to_dict() if self.record else None,
            "error": dict(self.error) if self.error else None,
        }


@dataclass(frozen=True, slots=True)
class Verification:
    """An agent's own judgement of whether it achieved the goal.

    Distinct from execution verification, which asks *may this action run?*
    before the fact. This asks *did the work accomplish what was asked?* after
    it, and can require the agent's completed work to be undone.

    Attributes:
        satisfied: Whether the agent believes the goal was met.
        confidence: How sure it is, in ``[0, 1]``.
        findings: Observations supporting the judgement.
        should_rollback: Whether completed effects must be compensated.
    """

    satisfied: bool
    confidence: float = 1.0
    findings: tuple[str, ...] = ()
    should_rollback: bool = False

    def __post_init__(self) -> None:
        """Clamp the confidence into range."""
        object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "satisfied": self.satisfied,
            "confidence": round(self.confidence, 4),
            "findings": list(self.findings),
            "should_rollback": self.should_rollback,
        }


@dataclass(frozen=True, slots=True)
class AgentReport:
    """The complete account of one task.

    Every task produces exactly one report, including tasks that were refused
    before planning. A refusal is an outcome, not a gap.

    Attributes:
        task: The task attempted.
        agent: Name of the agent that handled it.
        status: Where it ended up.
        suitability: The agent's initial assessment.
        plan: What it intended to do, absent if it never planned.
        outcomes: What happened, step by step.
        verification: The agent's own post-execution judgement.
        summary: One-paragraph human-readable account.
        started_at: When the task began.
        finished_at: When the task ended.
        duration_ms: Measured wall-clock duration.
    """

    task: Task
    agent: str
    status: TaskStatus
    suitability: Suitability
    plan: Plan | None = None
    outcomes: tuple[StepOutcome, ...] = ()
    verification: Verification | None = None
    summary: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None
    duration_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        """Return whether the task completed successfully."""
        return self.status is TaskStatus.COMPLETED

    @property
    def outputs(self) -> tuple[str, ...]:
        """Return the non-empty output of every step, in order."""
        return tuple(outcome.output for outcome in self.outcomes if outcome.output)

    @property
    def final_output(self) -> str:
        """Return the last non-empty step output, or the summary."""
        outputs = self.outputs
        return outputs[-1] if outputs else self.summary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "task": self.task.to_dict(),
            "agent": self.agent,
            "status": self.status.value,
            "suitability": {
                "score": round(self.suitability.score, 4),
                "reason": self.suitability.reason,
            },
            "plan": self.plan.to_dict() if self.plan else None,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "verification": self.verification.to_dict() if self.verification else None,
            "summary": self.summary,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": round(self.duration_ms, 3),
        }


def summarise_outcomes(outcomes: Sequence[StepOutcome]) -> str:
    """Return a one-line tally of step outcomes.

    Example:
        >>> summarise_outcomes(())
        'no steps were run'
    """
    if not outcomes:
        return "no steps were run"
    tally: dict[str, int] = {}
    for outcome in outcomes:
        tally[outcome.status.value] = tally.get(outcome.status.value, 0) + 1
    return ", ".join(f"{count} {status}" for status, count in sorted(tally.items()))
