"""The agent base class.

Every agent has the five methods you specified — ``can_handle``, ``plan``,
``execute``, ``verify``, ``report`` — and :meth:`BaseAgent.run` calls all five,
in that order, every time. An agent cannot skip verification because it does not
control the sequence.

One deliberate deviation is worth stating plainly. Only ``can_handle`` and
``plan`` are abstract. ``execute``, ``verify`` and ``report`` ship with complete,
overridable implementations, because their logic — walk the steps, dispatch
effects through the pipeline, tally what happened, assemble the account — is
identical for every agent. Making all five abstract would have satisfied the
letter of "every agent implements all five" while violating the first rule on
the list, *never duplicate code*, in every agent ever written. The contract is
kept where it matters: all five exist on every agent, and all five run.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from datetime import UTC, datetime

from eden.agents.context import ActionOutcome, AgentContext
from eden.agents.types import (
    AgentReport,
    Plan,
    PlanStep,
    StepOutcome,
    Suitability,
    Task,
    Verification,
    summarise_outcomes,
)
from eden.config.enums import StepStatus, TaskStatus
from eden.errors import AgentError, EdenError, InvalidPlanError
from eden.logging import get_logger, timed_block
from eden.utils.async_tools import with_timeout

_MS_PER_SECOND = 1000.0


class BaseAgent(abc.ABC):
    """Base class for every EDEN agent.

    Example:
        >>> issubclass(BaseAgent, object)
        True
    """

    def __init__(self, context: AgentContext, *, name: str = "") -> None:
        """Initialise the agent.

        Args:
            context: The complete set of capabilities available to this agent.
            name: Logical name. Defaults to the class name, lower-cased.
        """
        self._context = context
        self._name = name or type(self).__name__.removesuffix("Agent").lower()
        self._logger = get_logger(f"agents.{self._name}")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Return the logical agent name."""
        return self._name

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Return a one-line description, shown when routing is explained."""

    @property
    def context(self) -> AgentContext:
        """Return this agent's capability surface."""
        return self._context

    # ------------------------------------------------------------------
    # The five methods
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def can_handle(self, task: Task) -> Suitability:
        """Return how well this agent suits ``task``.

        Must be cheap and free of side effects: the router calls it on every
        registered agent for every task.
        """

    @abc.abstractmethod
    async def plan(self, task: Task) -> Plan:
        """Produce an ordered plan for ``task``.

        Planning must not change anything. A step that has an effect carries an
        :class:`~eden.execution.types.Action`, which is inert until executed.

        Raises:
            PlanningError: If no usable plan can be produced.
        """

    async def execute(self, task: Task, plan: Plan) -> list[StepOutcome]:
        """Run every step in order, stopping at the first required failure.

        Effect steps go through :meth:`AgentContext.act` and therefore through
        the full execution pipeline. Reasoning steps consult a model. There is
        no third possibility.
        """
        outcomes: list[StepOutcome] = []
        for step in plan.steps:
            outcome = await self._run_step(task, step)
            outcomes.append(outcome)
            if outcome.succeeded or step.optional:
                continue
            self._logger.warning(
                "Required step failed; abandoning the rest of the plan.",
                extra={
                    "task": task.id,
                    "step": step.id,
                    "remaining": len(plan.steps) - len(outcomes),
                },
            )
            outcomes.extend(
                StepOutcome(step=remaining, status=StepStatus.SKIPPED)
                for remaining in plan.steps[len(outcomes) :]
            )
            break
        return outcomes

    async def verify(
        self,
        task: Task,
        plan: Plan,
        outcomes: Sequence[StepOutcome],
    ) -> Verification:
        """Judge whether the work achieved the goal.

        This runs *after* execution and asks a different question from the
        execution pipeline's verification, which runs before and asks whether an
        action may proceed at all.

        The default judgement is structural — did every required step succeed?
        Agents that can check the goal itself should override this.
        """
        del task
        required = [
            outcome
            for outcome in outcomes
            if not outcome.step.optional and outcome.status is not StepStatus.SKIPPED
        ]
        failures = [outcome for outcome in outcomes if outcome.status is StepStatus.FAILED]
        skipped = [outcome for outcome in outcomes if outcome.status is StepStatus.SKIPPED]

        satisfied = (
            not failures
            and not skipped
            and len(required) == len([step for step in plan.steps if not step.optional])
        )
        findings: list[str] = []
        findings.extend(f"step '{outcome.step.description}' failed" for outcome in failures)
        if skipped:
            findings.append(f"{len(skipped)} step(s) were never attempted")

        return Verification(
            satisfied=satisfied,
            confidence=1.0 if satisfied else 0.0,
            findings=tuple(findings),
            should_rollback=(
                not satisfied and self._context.config.rollback_on_verification_failure
            ),
        )

    async def report(
        self,
        task: Task,
        suitability: Suitability,
        plan: Plan | None,
        outcomes: Sequence[StepOutcome],
        verification: Verification | None,
        *,
        status: TaskStatus,
        started_at: datetime,
        duration_ms: float,
    ) -> AgentReport:
        """Assemble the account of what happened.

        Override to add domain-specific narrative; the structural fields are
        assembled here so every agent's report has the same shape.
        """
        summary = await self._summarise(task, plan, outcomes, verification, status)
        return AgentReport(
            task=task,
            agent=self._name,
            status=status,
            suitability=suitability,
            plan=plan,
            outcomes=tuple(outcomes),
            verification=verification,
            summary=summary,
            started_at=started_at,
            finished_at=datetime.now(tz=UTC),
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # The lifecycle, which agents do not control
    # ------------------------------------------------------------------
    async def run(self, task: Task) -> AgentReport:
        """Take ``task`` through the full agent lifecycle.

        The order is fixed: ``can_handle`` → ``plan`` → ``execute`` → ``verify``
        → ``report``. A failure at any stage still produces a report, because a
        task that went wrong is exactly the case somebody needs an account of.

        Args:
            task: The goal to attempt.

        Returns:
            The complete report.
        """
        started = datetime.now(tz=UTC)
        suitability = self.can_handle(task)

        if not suitability.can_handle:
            return await self.report(
                task,
                suitability,
                None,
                (),
                None,
                status=TaskStatus.REJECTED,
                started_at=started,
                duration_ms=0.0,
            )

        with timed_block(self._logger, "agent.run", task=task.id, agent=self._name) as watch:
            try:
                return await with_timeout(
                    self._run_lifecycle(task, suitability, started),
                    self._context.config.task_timeout_seconds,
                    on_timeout=lambda: AgentError(
                        "Task exceeded its time budget.",
                        context={
                            "task": task.id,
                            "timeout_seconds": self._context.config.task_timeout_seconds,
                        },
                    ),
                )
            except EdenError as exc:
                self._logger.warning(
                    "Task failed.",
                    extra={"task": task.id, "error_code": exc.code},
                )
                return await self.report(
                    task,
                    suitability,
                    None,
                    (),
                    Verification(satisfied=False, confidence=0.0, findings=(exc.message,)),
                    status=TaskStatus.FAILED,
                    started_at=started,
                    duration_ms=watch.elapsed_ms,
                )

    async def _run_lifecycle(
        self,
        task: Task,
        suitability: Suitability,
        started: datetime,
    ) -> AgentReport:
        """Run planning, execution and verification under the time budget."""
        plan = await self.plan(task)
        self._validate(plan, task)
        self._logger.info(
            "Plan produced.",
            extra={
                "task": task.id,
                "steps": len(plan.steps),
                "effects": plan.effect_count,
            },
        )

        outcomes = await self.execute(task, plan)
        verification = await self.verify(task, plan, outcomes)

        status = TaskStatus.COMPLETED if verification.satisfied else TaskStatus.FAILED
        if verification.should_rollback:
            compensated = await self._compensate(outcomes)
            if compensated:
                status = TaskStatus.ROLLED_BACK
                outcomes = [
                    (
                        StepOutcome(
                            step=outcome.step,
                            status=StepStatus.ROLLED_BACK,
                            output=outcome.output,
                            record=outcome.record,
                            preparation=outcome.preparation,
                        )
                        if outcome.reversible
                        else outcome
                    )
                    for outcome in outcomes
                ]

        elapsed = (datetime.now(tz=UTC) - started).total_seconds() * _MS_PER_SECOND
        return await self.report(
            task,
            suitability,
            plan,
            outcomes,
            verification,
            status=status,
            started_at=started,
            duration_ms=elapsed,
        )

    def _validate(self, plan: Plan, task: Task) -> None:
        """Check the plan against configured constraints.

        Raises:
            InvalidPlanError: If the plan is empty, mismatched or too long.
        """
        if plan.task_id != task.id:
            raise InvalidPlanError(
                "Plan does not belong to this task.",
                context={"task": task.id, "plan_task": plan.task_id},
            )
        if not plan.steps:
            raise InvalidPlanError("Plan contains no steps.", context={"task": task.id})
        limit = self._context.config.max_plan_steps
        if len(plan.steps) > limit:
            raise InvalidPlanError(
                "Plan exceeds the configured step ceiling.",
                context={"task": task.id, "steps": len(plan.steps), "limit": limit},
            )

    async def _run_step(self, task: Task, step: PlanStep) -> StepOutcome:
        """Run one step, converting any failure into an outcome."""
        try:
            if step.action is not None:
                return await self._run_effect(step, step.action)
            return await self._run_thought(task, step)
        except EdenError as exc:
            self._logger.warning(
                "Step failed.",
                extra={"task": task.id, "step": step.id, "error_code": exc.code},
            )
            return StepOutcome(step=step, status=StepStatus.FAILED, error=exc.to_dict())
        except Exception as exc:  # noqa: BLE001 - a step bug must not kill the task
            return StepOutcome(
                step=step,
                status=StepStatus.FAILED,
                error={"type": type(exc).__name__, "message": str(exc)},
            )

    async def _run_effect(self, step: PlanStep, action: object) -> StepOutcome:
        """Perform an effect step through the execution pipeline."""
        from eden.execution.types import Action  # noqa: PLC0415 - flat import graph

        assert isinstance(action, Action)  # noqa: S101 - guarded by PlanStep validation
        outcome: ActionOutcome = await self._context.act(action)
        return StepOutcome(
            step=step,
            status=StepStatus.SUCCEEDED if outcome.succeeded else StepStatus.FAILED,
            output=outcome.record.result.output if outcome.record.result else "",
            record=outcome.record,
            preparation=outcome.preparation,
            error=outcome.record.error,
        )

    async def _run_thought(self, task: Task, step: PlanStep) -> StepOutcome:
        """Perform a reasoning step by consulting a model."""
        text = await self._context.think(
            step.prompt,
            system=self._thinking_prompt(task),
        )
        return StepOutcome(step=step, status=StepStatus.SUCCEEDED, output=text)

    def _thinking_prompt(self, task: Task) -> str:
        """Return the system prompt used for this agent's reasoning steps."""
        parts = [
            f"You are {self._name}, a component of the EDEN system. {self.description}",
            f"You are working on this goal: {task.goal}",
        ]
        if task.constraints:
            parts.append("Constraints: " + "; ".join(task.constraints))
        return "\n".join(parts)

    async def _compensate(self, outcomes: Sequence[StepOutcome]) -> bool:
        """Undo completed effects in reverse order. Returns whether all worked."""
        reversible = [outcome for outcome in outcomes if outcome.reversible]
        if not reversible:
            return False
        all_undone = True
        for outcome in reversed(reversible):
            if outcome.preparation is None or outcome.record is None:
                continue
            undone = await self._context.undo(
                ActionOutcome(record=outcome.record, preparation=outcome.preparation)
            )
            all_undone = all_undone and undone
        self._logger.info(
            "Agent work compensated after failed verification.",
            extra={"steps": len(reversible), "complete": all_undone},
        )
        return all_undone

    async def _summarise(
        self,
        task: Task,
        plan: Plan | None,
        outcomes: Sequence[StepOutcome],
        verification: Verification | None,
        status: TaskStatus,
    ) -> str:
        """Return the human-readable account placed in the report."""
        del plan
        if status is TaskStatus.REJECTED:
            return f"{self._name} declined '{task.goal}'."
        tally = summarise_outcomes(outcomes)
        verdict = "goal met" if verification and verification.satisfied else "goal not met"
        return f"{self._name} ran {tally}; {verdict}."
