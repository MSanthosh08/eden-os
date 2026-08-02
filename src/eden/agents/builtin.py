"""Built-in agents.

Three are shipped, chosen to span the interesting axes rather than to be
exhaustive.

:class:`ConversationAgent` never changes anything. It reads memory, thinks, and
writes back what it learned. It is the shape most agents take.

:class:`FileTaskAgent` changes the world. Every effect it proposes is an
``Action`` that goes through the execution pipeline, so it demonstrates that an
agent gains nothing by wanting to act — permission still comes from policy, not
from the agent's own confidence.

:class:`SearchAgent` reads the world without a model in the loop at all. It
answers "find X" deterministically through
:meth:`~eden.agents.context.AgentContext.search_files`, which is the honest fix
for what a purely conversational agent used to do with such a request: describe
a shell command for the person to run themselves, rather than actually looking.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from eden.agents.base import BaseAgent
from eden.agents.context import FileHit
from eden.agents.types import Plan, PlanStep, StepOutcome, Suitability, Task, Verification
from eden.config.enums import ActionKind, MemoryKind, StepStatus
from eden.core.types import Message
from eden.errors import AgentCapabilityError, PlanningError
from eden.execution.types import Action

GOAL_PARAMETER = "path"
CONTENT_PARAMETER = "content"

_FILE_KEYWORDS = frozenset(
    {"file", "write", "save", "create", "note", "document", "draft", "record"}
)
_DELETE_KEYWORDS = frozenset({"delete", "remove", "erase"})
_SEARCH_KEYWORDS = frozenset({"search", "find", "locate", "list"})
_FILE_NOUNS = frozenset({"file", "files", "document", "documents"})

_MAX_LISTED_RESULTS = 200

# A handful of common extensions a goal might name in plain English, so "list
# all the python files" needs no explicit pattern in the task context.
_EXTENSION_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bpython\b", re.IGNORECASE), "*.py"),
    (re.compile(r"\bjavascript\b", re.IGNORECASE), "*.js"),
    (re.compile(r"\btypescript\b", re.IGNORECASE), "*.ts"),
    (re.compile(r"\bmarkdown\b", re.IGNORECASE), "*.md"),
    (re.compile(r"\bpdf(s)?\b", re.IGNORECASE), "*.pdf"),
    (re.compile(r"\bimage(s)?\b", re.IGNORECASE), "*.png"),
    (re.compile(r"\bconfig(uration)?\s+files?\b", re.IGNORECASE), "*.toml"),
)


def _infer_pattern(goal: str) -> str:
    """Return a glob pattern inferred from plain-English wording in ``goal``."""
    for regex, pattern in _EXTENSION_HINTS:
        if regex.search(goal):
            return pattern
    return "*"


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


class SearchAgent(BaseAgent):
    """Finds files by name pattern, with no model in the loop.

    A search is a read: it has no effect to verify, permit or roll back, so —
    unlike :class:`FileTaskAgent` — its work never touches the execution
    pipeline. Its answer is exactly what
    :meth:`~eden.agents.context.AgentContext.search_files` found, not a
    paraphrase of it, and where it is allowed to look is bounded by
    ``agents.search_roots`` rather than by the agent's own judgement.
    """

    @property
    def description(self) -> str:
        """Return a one-line description."""
        return "Finds files by name or extension within configured search roots."

    def can_handle(self, task: Task) -> Suitability:
        """Score by an explicit pattern, or by search-and-file wording."""
        if not self._context.has_search:
            return Suitability.no(
                "No search roots are configured (agents.search_roots is empty "
                "and execution is disabled)."
            )
        explicit = bool(task.context.get("pattern")) or bool(task.context.get("search"))
        words = frozenset(task.goal.lower().split())
        mentions_search = bool(words & _SEARCH_KEYWORDS)
        names_a_file_type = bool(words & _FILE_NOUNS) or _infer_pattern(task.goal) != "*"
        if explicit or (mentions_search and names_a_file_type):
            return Suitability(score=0.95, reason="Goal asks to find or list files.")
        if mentions_search:
            return Suitability(score=0.3, reason="Mentions searching, but no file type was named.")
        return Suitability.no("Goal does not ask to find or list files.")

    async def plan(self, task: Task) -> Plan:
        """Produce a single deterministic search step; there is nothing to draft."""
        pattern = self._pattern_for(task)
        return Plan(
            task_id=task.id,
            steps=(
                PlanStep(
                    description=f"Search for files matching '{pattern}'.",
                    prompt=pattern,
                ),
            ),
            rationale=f"Deterministic file search for '{pattern}'; no model call is needed.",
        )

    async def _run_thought(self, task: Task, step: PlanStep) -> StepOutcome:
        """Run the search directly rather than consulting a model.

        A factual listing of what exists on disk is not something a model
        should be asked to produce or paraphrase — it is looked up, exactly
        the way :class:`~eden.agents.builtin.EchoAgent` looks nothing up.
        """
        pattern = self._pattern_for(task)
        roots = self._roots_for(task)
        try:
            hits = await self._context.search_files(pattern, roots=roots)
        except AgentCapabilityError as exc:
            return StepOutcome(step=step, status=StepStatus.FAILED, error=exc.to_dict())
        return StepOutcome(
            step=step,
            status=StepStatus.SUCCEEDED,
            output=self._format(hits, pattern),
        )

    @staticmethod
    def _pattern_for(task: Task) -> str:
        """Return an explicit pattern from context, or one inferred from the goal."""
        explicit = task.context.get("pattern")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return _infer_pattern(task.goal)

    @staticmethod
    def _roots_for(task: Task) -> tuple[str, ...]:
        """Return any roots the task requested, narrowing rather than widening."""
        raw = task.context.get("roots")
        if isinstance(raw, list | tuple):
            return tuple(str(item) for item in raw)
        return ()

    @staticmethod
    def _format(hits: Sequence[FileHit], pattern: str) -> str:
        """Render results as a neat, bounded listing."""
        if not hits:
            return (
                f"No files matching '{pattern}' were found within the " "configured search roots."
            )
        shown = hits[:_MAX_LISTED_RESULTS]
        lines = [f"Found {len(hits)} file(s) matching '{pattern}':", ""]
        lines.extend(f"  {hit.path}" for hit in shown)
        if len(hits) > len(shown):
            lines.append(f"  ... and {len(hits) - len(shown)} more.")
        return "\n".join(lines)


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
