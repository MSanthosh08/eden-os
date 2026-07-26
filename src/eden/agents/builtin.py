"""Built-in agents.

Two are shipped, chosen to span the interesting axis rather than to be
exhaustive.

:class:`ConversationAgent` never changes anything. It reads memory, thinks, and
writes back what it learned. It is the shape most agents take.

:class:`FileTaskAgent` changes the world. Every effect it proposes is an
``Action`` that goes through the execution pipeline, so it demonstrates that an
agent gains nothing by wanting to act — permission still comes from policy, not
from the agent's own confidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from eden.agents.base import BaseAgent
from eden.agents.types import Plan, PlanStep, StepOutcome, Suitability, Task, Verification
from eden.config.enums import ActionKind, MemoryKind, StepStatus
from eden.core.types import Message
from eden.errors import PlanningError
from eden.execution.types import Action

GOAL_PARAMETER = "path"
CONTENT_PARAMETER = "content"

_FILE_KEYWORDS = frozenset(
    {"file", "write", "save", "create", "note", "document", "draft", "record"}
)
_DELETE_KEYWORDS = frozenset({"delete", "remove", "erase"})


class ConversationAgent(BaseAgent):
    """Answers questions using memory and a model, without side effects.

    Deliberately the default: most requests want an answer, not an action, and
    an agent that cannot act cannot act wrongly.
    """

    @property
    def description(self) -> str:
        """Return a one-line description."""
        return "Answers questions and holds conversations using recalled context."

    def can_handle(self, task: Task) -> Suitability:
        """Accept anything, but weakly, so specialists outrank it.

        A general agent should be the fallback rather than the favourite, which
        is expressible as a score and would not be expressible as a boolean.
        """
        if task.context.get("no_conversation"):
            return Suitability.no("Task explicitly excludes conversational handling.")
        return Suitability(score=0.3, reason="Can answer any question conversationally.")

    async def plan(self, task: Task) -> Plan:
        """Produce a single reasoning step informed by recalled memory."""
        hits = await self._context.recall(task.goal, namespace=task.namespace)
        recalled = "\n".join(f"- {hit.record.content}" for hit in hits)
        prompt = (
            task.goal
            if not recalled
            else (f"Relevant things you already know:\n{recalled}\n\nNow answer: {task.goal}")
        )
        return Plan(
            task_id=task.id,
            steps=(
                PlanStep(
                    description="Answer the question using recalled context.",
                    prompt=prompt,
                ),
            ),
            rationale=(
                f"Answer directly, informed by {len(hits)} recalled memories."
                if hits
                else "Answer directly; nothing relevant was recalled."
            ),
        )

    async def verify(
        self,
        task: Task,
        plan: Plan,
        outcomes: Sequence[StepOutcome],
    ) -> Verification:
        """Check that an answer was actually produced.

        A model that returns an empty string has technically succeeded and
        substantively failed, which the structural default cannot see.
        """
        base = await super().verify(task, plan, outcomes)
        if not base.satisfied:
            return base
        answered = any(outcome.output.strip() for outcome in outcomes)
        if answered:
            return base
        return Verification(
            satisfied=False,
            confidence=0.0,
            findings=("the model returned no content",),
            should_rollback=False,
        )

    async def _run_thought(self, task: Task, step: PlanStep) -> StepOutcome:
        """Answer, then record both sides of the exchange in memory."""
        outcome = await super()._run_thought(task, step)
        if outcome.succeeded and outcome.output:
            await self._context.observe(Message.user(task.goal), namespace=task.namespace)
            await self._context.observe(Message.assistant(outcome.output), namespace=task.namespace)
        return outcome


class FileTaskAgent(BaseAgent):
    """Writes and deletes files, always through the execution pipeline.

    Notice what this class does *not* contain: any check on paths, any notion of
    what is dangerous, any decision about whether it is allowed. All of that
    lives in :mod:`eden.execution`, which the agent cannot influence.
    """

    @property
    def description(self) -> str:
        """Return a one-line description."""
        return "Creates, updates and removes files inside the EDEN workspace."

    def can_handle(self, task: Task) -> Suitability:
        """Score by keyword overlap and by whether a target was supplied."""
        if not self._context.has_execution:
            return Suitability.no("The execution subsystem is disabled.")
        if not task.context.get(GOAL_PARAMETER):
            return Suitability.no("No target path was supplied in the task context.")
        words = frozenset(task.goal.lower().split())
        overlap = words & (_FILE_KEYWORDS | _DELETE_KEYWORDS)
        if not overlap:
            return Suitability(
                score=0.2, reason="A path was supplied but the goal does not mention files."
            )
        return Suitability(
            score=0.9,
            reason=f"Goal names a file operation ({', '.join(sorted(overlap))}).",
        )

    async def plan(self, task: Task) -> Plan:
        """Draft the content, then propose the write as an inert action.

        Drafting is a separate step from writing on purpose. The plan can be
        shown to a human between the two, and the write is refused by the
        pipeline if the drafted content or the path is unacceptable.

        Raises:
            PlanningError: If the task supplies no usable target path.
        """
        path = str(task.context.get(GOAL_PARAMETER) or "").strip()
        if not path:
            raise PlanningError(
                "A file task requires a 'path' in its context.",
                context={"task": task.id},
            )

        words = frozenset(task.goal.lower().split())
        if words & _DELETE_KEYWORDS:
            return Plan(
                task_id=task.id,
                steps=(
                    PlanStep(
                        description=f"Delete {path}.",
                        action=Action(
                            kind=ActionKind.FILE_DELETE,
                            summary=f"Delete {path} for task {task.id}.",
                            parameters={GOAL_PARAMETER: path},
                            actor=self.name,
                            namespace=task.namespace,
                        ),
                    ),
                ),
                rationale="The goal asks for removal, so a single delete suffices.",
            )

        supplied = task.context.get(CONTENT_PARAMETER)
        if isinstance(supplied, str) and supplied.strip():
            content = supplied
            steps: tuple[PlanStep, ...] = ()
            rationale = "Content was supplied, so only the write is needed."
        else:
            content = await self._context.think(
                f"Write the full contents of the file '{path}'. Goal: {task.goal}. "
                "Output only the file contents, with no commentary or code fences.",
                system=self._thinking_prompt(task),
            )
            if not content.strip():
                raise PlanningError(
                    "The model produced no content to write.",
                    context={"task": task.id, "path": path},
                )
            steps = (
                PlanStep(
                    description=f"Drafted {len(content)} characters for {path}.",
                    prompt=f"Confirm the draft for {path} is complete. Reply 'ready'.",
                    optional=True,
                ),
            )
            rationale = "Draft the content, then write it as a single reviewable action."

        return Plan(
            task_id=task.id,
            steps=(
                *steps,
                PlanStep(
                    description=f"Write {len(content)} characters to {path}.",
                    action=Action(
                        kind=ActionKind.FILE_WRITE,
                        summary=f"Write {path} for task {task.id}.",
                        parameters={GOAL_PARAMETER: path, CONTENT_PARAMETER: content},
                        actor=self.name,
                        namespace=task.namespace,
                    ),
                ),
            ),
            rationale=rationale,
        )

    async def verify(
        self,
        task: Task,
        plan: Plan,
        outcomes: Sequence[StepOutcome],
    ) -> Verification:
        """Confirm every effect step actually reached the world."""
        base = await super().verify(task, plan, outcomes)
        if not base.satisfied:
            return base
        effects = [outcome for outcome in outcomes if outcome.step.changes_the_world]
        landed = [
            outcome
            for outcome in effects
            if outcome.record is not None and outcome.record.succeeded
        ]
        if len(landed) == len(effects) and effects:
            await self._context.remember(
                f"Completed file task: {task.goal}",
                namespace=task.namespace,
                kind=MemoryKind.PROJECT,
                importance=0.6,
                tags=frozenset({"file-task"}),
            )
            return base
        return Verification(
            satisfied=False,
            confidence=0.0,
            findings=(
                f"{len(effects) - len(landed)} of {len(effects)} file operations "
                "did not take effect",
            ),
            should_rollback=self._context.config.rollback_on_verification_failure,
        )


class EchoAgent(BaseAgent):
    """Returns the goal unchanged.

    Exists so that the orchestrator, routing and reporting can be exercised
    without a model or any subsystem at all. It is a real agent, not a stub:
    it satisfies the full contract and appears in the registry like any other.
    """

    @property
    def description(self) -> str:
        """Return a one-line description."""
        return "Echoes the goal back, for diagnostics and routing tests."

    def can_handle(self, task: Task) -> Suitability:
        """Accept only when explicitly asked for."""
        if task.context.get("agent") == self.name:
            return Suitability.certain("Explicitly requested.")
        return Suitability.no("Only handles explicitly routed diagnostic tasks.")

    async def plan(self, task: Task) -> Plan:
        """Return a single trivial step."""
        return Plan(
            task_id=task.id,
            steps=(PlanStep(description="Echo the goal.", prompt=task.goal),),
            rationale="Diagnostics only.",
        )

    async def _run_thought(self, task: Task, step: PlanStep) -> StepOutcome:
        """Return the goal without consulting a model."""
        del task
        return StepOutcome(step=step, status=StepStatus.SUCCEEDED, output=step.prompt)
