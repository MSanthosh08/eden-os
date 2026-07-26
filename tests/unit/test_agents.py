"""Unit tests for the agent subsystem and memory consolidation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from eden.agents.base import BaseAgent
from eden.agents.builtin import ConversationAgent, EchoAgent, FileTaskAgent
from eden.agents.context import AgentContext
from eden.agents.orchestrator import AgentOrchestrator, build_orchestrator
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
from eden.config.enums import ActionKind, MemoryKind, StepStatus, TaskStatus
from eden.config.schema import (
    AgentConfig,
    ExecutionConfig,
    GatewayConfig,
    MemoryConfig,
    PathsConfig,
    RouterConfig,
)
from eden.core.types import Message
from eden.errors import (
    AgentCapabilityError,
    InvalidPlanError,
    NoSuitableAgentError,
    PlanningError,
    RegistryError,
    ValidationError,
)
from eden.execution.engine import ExecutionEngine
from eden.execution.permissions import AlwaysApproveGate, DenyingGate, PolicyEngine
from eden.execution.types import Action
from eden.gateway.client import GatewayClient
from eden.gateway.health import HealthTracker
from eden.gateway.providers.mock import MockProvider
from eden.gateway.router.omni_router import OmniRouter
from eden.memory.consolidation import (
    CONSOLIDATED_TAG,
    Consolidator,
    ExtractiveSummariser,
    GatewaySummariser,
)
from eden.memory.manager import MemoryManager, build_memory_manager
from eden.memory.types import MemoryQuery
from eden.memory.vector import HashEmbedder
from eden.utils.clock import ManualClock
from tests.conftest import make_provider_config


def build_gateway(clock: ManualClock, *, failing: bool = False) -> GatewayClient:
    """Return a gateway backed by one deterministic provider."""
    provider = MockProvider(make_provider_config("local"), clock=clock)
    if failing:
        from eden.errors import ProviderUnavailableError

        provider.set_failure(lambda: ProviderUnavailableError("down", provider="local"))
    router_config = RouterConfig()
    tracker = HealthTracker(config=router_config.circuit_breaker, clock=clock)
    return GatewayClient(
        GatewayConfig(providers=(provider.config,)),
        [provider],
        OmniRouter(router_config, tracker),
        tracker,
    )


def build_context(
    tmp_path: Path,
    clock: ManualClock,
    *,
    with_memory: bool = True,
    with_execution: bool = True,
    approve: bool = True,
    agent_config: AgentConfig | None = None,
) -> AgentContext:
    """Return a context with the requested subsystems attached."""
    config = agent_config or AgentConfig()
    gateway = build_gateway(clock)
    memory = (
        build_memory_manager(
            MemoryConfig(persist=False),
            PathsConfig(root=tmp_path),
            embedder=HashEmbedder(32),
        )
        if with_memory
        else None
    )
    execution = None
    if with_execution:
        execution_config = ExecutionConfig(workspace_root=tmp_path / "workspace")
        execution = ExecutionEngine(
            execution_config,
            policy=PolicyEngine(
                execution_config, AlwaysApproveGate() if approve else DenyingGate()
            ),
        )
    return AgentContext(config, gateway, memory=memory, execution=execution)


class TestAgentTypes:
    def test_task_generates_an_identifier(self) -> None:
        assert Task(goal="do a thing").id.startswith("task-")

    def test_empty_goal_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(goal="  ")

    def test_step_must_either_act_or_reason(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(description="empty")

    def test_step_must_not_do_both(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(
                description="ambiguous",
                prompt="think",
                action=Action(kind=ActionKind.NOOP, summary="act"),
            )

    def test_suitability_is_clamped(self) -> None:
        assert Suitability(score=5.0).score == 1.0
        assert Suitability(score=-2.0).score == 0.0

    def test_refusal_cannot_handle(self) -> None:
        assert Suitability.no("not mine").can_handle is False

    def test_plan_reports_its_effects(self) -> None:
        plan = Plan(
            task_id="t",
            steps=(
                PlanStep(description="think", prompt="q"),
                PlanStep(
                    description="act",
                    action=Action(kind=ActionKind.NOOP, summary="s"),
                ),
            ),
        )
        assert plan.changes_the_world is True
        assert plan.effect_count == 1
        assert "[act " in plan.describe()

    def test_plan_with_no_effects_is_inert(self) -> None:
        plan = Plan(task_id="t", steps=(PlanStep(description="think", prompt="q"),))
        assert plan.changes_the_world is False

    def test_a_plan_cannot_execute_itself(self) -> None:
        """The Phase 3 discipline, inherited: plans describe, they do not do."""
        plan = Plan(task_id="t")
        for attribute in ("run", "execute", "apply", "perform", "__call__"):
            assert not hasattr(plan, attribute)

    def test_outcome_tally(self) -> None:
        assert summarise_outcomes(()) == "no steps were run"


class TestBaseAgentContract:
    def test_every_agent_has_all_five_methods(self) -> None:
        """The contract from the specification, asserted rather than assumed."""
        for agent_class in (ConversationAgent, FileTaskAgent, EchoAgent):
            for method in ("can_handle", "plan", "execute", "verify", "report"):
                assert callable(
                    getattr(agent_class, method)
                ), f"{agent_class.__name__} is missing {method}"

    def test_planning_and_suitability_are_abstract(self) -> None:
        assert BaseAgent.__abstractmethods__ >= {"can_handle", "plan", "description"}

    async def test_run_invokes_all_five_in_order(self, tmp_path: Path, clock: ManualClock) -> None:
        calls: list[str] = []

        class Tracing(EchoAgent):
            def can_handle(self, task: Task) -> Suitability:
                calls.append("can_handle")
                return Suitability.certain()

            async def plan(self, task: Task) -> Plan:
                calls.append("plan")
                return await super().plan(task)

            async def execute(self, task: Task, plan: Plan) -> list[StepOutcome]:
                calls.append("execute")
                return await super().execute(task, plan)

            async def verify(
                self, task: Task, plan: Plan, outcomes: Sequence[StepOutcome]
            ) -> Verification:
                calls.append("verify")
                return await super().verify(task, plan, outcomes)

            async def report(self, *args: object, **kwargs: object) -> AgentReport:
                calls.append("report")
                return await super().report(*args, **kwargs)  # type: ignore[arg-type]

        agent = Tracing(build_context(tmp_path, clock))
        await agent.run(Task(goal="hello"))
        assert calls == ["can_handle", "plan", "execute", "verify", "report"]

    async def test_a_refusal_still_produces_a_report(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        agent = EchoAgent(build_context(tmp_path, clock))
        report = await agent.run(Task(goal="not for you"))
        assert report.status is TaskStatus.REJECTED
        assert report.plan is None
        assert "declined" in report.summary

    async def test_planning_failure_still_produces_a_report(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        class Unplannable(EchoAgent):
            async def plan(self, task: Task) -> Plan:
                raise PlanningError("cannot plan this")

        agent = Unplannable(build_context(tmp_path, clock))
        report = await agent.run(Task(goal="x", context={"agent": "unplannable"}))
        assert report.status is TaskStatus.FAILED
        assert report.verification is not None
        assert "cannot plan" in report.verification.findings[0]

    async def test_empty_plan_is_refused(self, tmp_path: Path, clock: ManualClock) -> None:
        class Emptyhanded(EchoAgent):
            async def plan(self, task: Task) -> Plan:
                return Plan(task_id=task.id)

        agent = Emptyhanded(build_context(tmp_path, clock))
        report = await agent.run(Task(goal="x", context={"agent": "emptyhanded"}))
        assert report.status is TaskStatus.FAILED

    async def test_oversized_plan_is_refused(self, tmp_path: Path, clock: ManualClock) -> None:
        class Verbose(EchoAgent):
            async def plan(self, task: Task) -> Plan:
                return Plan(
                    task_id=task.id,
                    steps=tuple(PlanStep(description=f"step {i}", prompt="q") for i in range(30)),
                )

        context = build_context(tmp_path, clock, agent_config=AgentConfig(max_plan_steps=5))
        report = await Verbose(context).run(Task(goal="x", context={"agent": "verbose"}))
        assert report.status is TaskStatus.FAILED

    async def test_plan_for_the_wrong_task_is_refused(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        class Confused(EchoAgent):
            async def plan(self, task: Task) -> Plan:
                return Plan(
                    task_id="somebody-elses-task",
                    steps=(PlanStep(description="s", prompt="q"),),
                )

        with pytest.raises(InvalidPlanError):
            Confused(build_context(tmp_path, clock))._validate(
                Plan(task_id="other", steps=(PlanStep(description="s", prompt="q"),)),
                Task(goal="x"),
            )

    async def test_a_failed_required_step_skips_the_rest(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        class Doomed(EchoAgent):
            async def plan(self, task: Task) -> Plan:
                return Plan(
                    task_id=task.id,
                    steps=(
                        PlanStep(
                            description="will fail",
                            action=Action(
                                kind=ActionKind.FILE_WRITE,
                                summary="escape",
                                parameters={"path": "../out.txt", "content": "x"},
                            ),
                        ),
                        PlanStep(description="never reached", prompt="q"),
                    ),
                )

        agent = Doomed(build_context(tmp_path, clock))
        report = await agent.run(Task(goal="x", context={"agent": "doomed"}))
        assert report.outcomes[0].status is StepStatus.FAILED
        assert report.outcomes[1].status is StepStatus.SKIPPED

    async def test_an_optional_step_failure_does_not_abort(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        class Tolerant(EchoAgent):
            async def plan(self, task: Task) -> Plan:
                return Plan(
                    task_id=task.id,
                    steps=(
                        PlanStep(
                            description="optional failure",
                            optional=True,
                            action=Action(
                                kind=ActionKind.FILE_WRITE,
                                summary="escape",
                                parameters={"path": "/etc/nope", "content": "x"},
                            ),
                        ),
                        PlanStep(description="still runs", prompt="q"),
                    ),
                )

        agent = Tolerant(build_context(tmp_path, clock))
        report = await agent.run(Task(goal="x", context={"agent": "tolerant"}))
        assert report.outcomes[1].status is StepStatus.SUCCEEDED

    async def test_failed_verification_rolls_the_agent_work_back(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """Agent-level verification can undo work the pipeline happily allowed."""

        class SecondGuessing(EchoAgent):
            async def plan(self, task: Task) -> Plan:
                return Plan(
                    task_id=task.id,
                    steps=(
                        PlanStep(
                            description="write a file",
                            action=Action(
                                kind=ActionKind.FILE_WRITE,
                                summary="write",
                                parameters={"path": "regret.txt", "content": "oops"},
                            ),
                        ),
                    ),
                )

            async def verify(
                self, task: Task, plan: Plan, outcomes: Sequence[StepOutcome]
            ) -> Verification:
                return Verification(
                    satisfied=False,
                    findings=("changed my mind",),
                    should_rollback=True,
                )

        agent = SecondGuessing(build_context(tmp_path, clock))
        report = await agent.run(Task(goal="x", context={"agent": "secondguessing"}))
        assert report.status is TaskStatus.ROLLED_BACK
        assert not (tmp_path / "workspace" / "regret.txt").exists()


class TestAgentContext:
    async def test_act_requires_the_execution_subsystem(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        context = build_context(tmp_path, clock, with_execution=False)
        with pytest.raises(AgentCapabilityError):
            await context.act(Action(kind=ActionKind.NOOP, summary="x"))

    async def test_memory_access_reports_clearly_when_disabled(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        context = build_context(tmp_path, clock, with_memory=False)
        with pytest.raises(AgentCapabilityError):
            _ = context.memory

    async def test_recall_degrades_silently_without_memory(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        context = build_context(tmp_path, clock, with_memory=False)
        assert await context.recall("anything", namespace="n") == []
        assert await context.remember("x", namespace="n") is None

    async def test_context_exposes_no_route_around_the_pipeline(self) -> None:
        """The structural claim: no handler is reachable from an agent."""
        surface = {name for name in dir(AgentContext) if not name.startswith("_")}
        assert "handler_for" not in surface
        assert "execution" not in surface
        for forbidden in ("handlers", "engine", "submit"):
            assert forbidden not in surface

    async def test_act_retains_undo_state(self, tmp_path: Path, clock: ManualClock) -> None:
        context = build_context(tmp_path, clock)
        outcome = await context.act(
            Action(
                kind=ActionKind.FILE_WRITE,
                summary="write",
                parameters={"path": "a.txt", "content": "one"},
            )
        )
        assert outcome.succeeded is True
        assert outcome.preparation.reversible is True
        assert await context.undo(outcome) is True
        assert not (tmp_path / "workspace" / "a.txt").exists()

    async def test_preview_describes_without_acting(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        context = build_context(tmp_path, clock)
        text = await context.preview(
            Action(
                kind=ActionKind.FILE_WRITE,
                summary="write a file",
                parameters={"path": "b.txt", "content": "x"},
            )
        )
        assert "risk:" in text
        assert not (tmp_path / "workspace" / "b.txt").exists()


class TestBuiltinAgents:
    async def test_conversation_agent_answers_and_remembers(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        context = build_context(tmp_path, clock)
        agent = ConversationAgent(context)
        report = await agent.run(Task(goal="what is EDEN?", namespace="chat"))
        assert report.succeeded is True
        assert "what is EDEN?" in report.final_output
        history = await context.memory.conversation.history("chat")
        assert len(history) == 2

    async def test_conversation_agent_recalls_prior_context(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        context = build_context(tmp_path, clock)
        await context.remember(
            "the deploy command is make ship",
            namespace="chat",
            kind=MemoryKind.LONG_TERM,
        )
        report = await ConversationAgent(context).run(Task(goal="deploy command", namespace="chat"))
        assert "make ship" in report.final_output

    async def test_conversation_agent_is_a_weak_generalist(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        suitability = ConversationAgent(build_context(tmp_path, clock)).can_handle(
            Task(goal="anything")
        )
        assert suitability.can_handle is True
        assert suitability.score < 0.5

    async def test_file_agent_declines_without_a_path(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        agent = FileTaskAgent(build_context(tmp_path, clock))
        assert agent.can_handle(Task(goal="write a file")).can_handle is False

    async def test_file_agent_declines_without_execution(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        agent = FileTaskAgent(build_context(tmp_path, clock, with_execution=False))
        task = Task(goal="write a file", context={"path": "a.txt"})
        assert agent.can_handle(task).can_handle is False

    async def test_file_agent_writes_supplied_content(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        agent = FileTaskAgent(build_context(tmp_path, clock))
        report = await agent.run(
            Task(
                goal="write a note",
                context={"path": "note.md", "content": "# Hello"},
            )
        )
        assert report.succeeded is True
        assert (tmp_path / "workspace" / "note.md").read_text(encoding="utf-8") == "# Hello"

    async def test_file_agent_drafts_content_when_none_supplied(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        agent = FileTaskAgent(build_context(tmp_path, clock))
        report = await agent.run(Task(goal="write release notes", context={"path": "notes.md"}))
        assert report.succeeded is True
        assert report.plan is not None
        assert report.plan.effect_count == 1
        assert (tmp_path / "workspace" / "notes.md").exists()

    async def test_file_agent_cannot_escape_the_workspace(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        """The agent proposes; the pipeline disposes."""
        agent = FileTaskAgent(build_context(tmp_path, clock))
        report = await agent.run(
            Task(
                goal="write a file",
                context={"path": "../../escaped.txt", "content": "x"},
            )
        )
        assert report.succeeded is False
        assert not (tmp_path.parent / "escaped.txt").exists()

    async def test_file_agent_is_denied_without_an_approver(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "existing.txt").write_text("original", encoding="utf-8")
        agent = FileTaskAgent(build_context(tmp_path, clock, approve=False))
        report = await agent.run(
            Task(
                goal="update the file",
                context={"path": "existing.txt", "content": "changed"},
            )
        )
        assert report.succeeded is False
        assert (workspace / "existing.txt").read_text(encoding="utf-8") == "original"

    async def test_file_agent_plans_a_delete(self, tmp_path: Path, clock: ManualClock) -> None:
        agent = FileTaskAgent(build_context(tmp_path, clock))
        plan = await agent.plan(Task(goal="delete the old file", context={"path": "old.txt"}))
        assert plan.steps[0].action is not None
        assert plan.steps[0].action.kind is ActionKind.FILE_DELETE


class TestOrchestrator:
    def orchestrator(self, tmp_path: Path, clock: ManualClock) -> AgentOrchestrator:
        return build_orchestrator(
            AgentConfig(),
            build_gateway(clock),
            memory=build_memory_manager(
                MemoryConfig(persist=False),
                PathsConfig(root=tmp_path),
                embedder=HashEmbedder(32),
            ),
            execution=ExecutionEngine(
                ExecutionConfig(workspace_root=tmp_path / "workspace"),
                policy=PolicyEngine(
                    ExecutionConfig(workspace_root=tmp_path / "workspace"),
                    AlwaysApproveGate(),
                ),
            ),
        )

    async def test_default_roster_is_registered(self, tmp_path: Path, clock: ManualClock) -> None:
        roster = self.orchestrator(tmp_path, clock).roster()
        assert set(roster) == {"conversation", "filetask", "echo"}

    async def test_routing_prefers_the_higher_score(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        decision = self.orchestrator(tmp_path, clock).route(
            Task(goal="write a file", context={"path": "a.txt"})
        )
        assert decision.winner == "filetask"

    async def test_generalist_wins_when_no_specialist_applies(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        decision = self.orchestrator(tmp_path, clock).route(Task(goal="explain gravity"))
        assert decision.winner == "conversation"

    async def test_declines_are_explained(self, tmp_path: Path, clock: ManualClock) -> None:
        decision = self.orchestrator(tmp_path, clock).route(Task(goal="explain gravity"))
        assert "echo" in decision.declined
        assert decision.declined["echo"]

    async def test_no_candidate_raises_with_the_reasons(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        orchestrator = self.orchestrator(tmp_path, clock)
        orchestrator.unregister("conversation")
        orchestrator.unregister("filetask")
        with pytest.raises(NoSuitableAgentError) as caught:
            await orchestrator.dispatch(Task(goal="anything"))
        assert "declined" in caught.value.context

    async def test_an_agent_raising_in_can_handle_is_skipped(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        orchestrator = self.orchestrator(tmp_path, clock)

        class Hostile(EchoAgent):
            def can_handle(self, task: Task) -> Suitability:
                message = "routing is broken"
                raise RuntimeError(message)

        orchestrator.register(Hostile(orchestrator.agents[0].context, name="hostile"))
        decision = orchestrator.route(Task(goal="explain gravity"))
        assert decision.winner == "conversation"
        assert "hostile" in decision.declined

    async def test_explicit_agent_bypasses_routing(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        report = await self.orchestrator(tmp_path, clock).dispatch(
            Task(goal="diagnostics", context={"agent": "echo"}), agent="echo"
        )
        assert report.agent == "echo"

    async def test_unknown_agent_name_raises(self, tmp_path: Path, clock: ManualClock) -> None:
        with pytest.raises(RegistryError):
            await self.orchestrator(tmp_path, clock).dispatch(Task(goal="x"), agent="nonexistent")

    async def test_duplicate_registration_raises(self, tmp_path: Path, clock: ManualClock) -> None:
        orchestrator = self.orchestrator(tmp_path, clock)
        with pytest.raises(RegistryError):
            orchestrator.register(ConversationAgent(orchestrator.agents[0].context))

    async def test_batch_dispatch_continues_past_an_untakeable_task(
        self, tmp_path: Path, clock: ManualClock
    ) -> None:
        orchestrator = self.orchestrator(tmp_path, clock)
        orchestrator.unregister("conversation")
        orchestrator.unregister("filetask")
        reports = await orchestrator.dispatch_all(
            [Task(goal="nobody wants this"), Task(goal="d", context={"agent": "echo"})]
        )
        assert reports[0].status is TaskStatus.REJECTED
        assert reports[1].succeeded is True

    async def test_lifecycle_is_idempotent(self, tmp_path: Path, clock: ManualClock) -> None:
        orchestrator = self.orchestrator(tmp_path, clock)
        await orchestrator.start()
        await orchestrator.start()
        await orchestrator.stop()
        await orchestrator.stop()


class TestConsolidation:
    def manager(self, tmp_path: Path, **overrides: object) -> MemoryManager:
        config = MemoryConfig(persist=False, **overrides)  # type: ignore[arg-type]
        return build_memory_manager(config, PathsConfig(root=tmp_path), embedder=HashEmbedder(32))

    async def test_extractive_summariser_respects_the_budget(self) -> None:
        summary = await ExtractiveSummariser().summarise(
            ["word " * 200, "other " * 200], target_words=10
        )
        assert len(summary.split()) <= 11

    async def test_extractive_summariser_handles_empty_input(self) -> None:
        assert await ExtractiveSummariser().summarise([], target_words=10) == ""

    async def test_gateway_summariser_falls_back_when_unavailable(self, clock: ManualClock) -> None:
        summariser = GatewaySummariser(build_gateway(clock, failing=True))
        summary = await summariser.summarise(["something to compress"], target_words=20)
        assert "something to compress" in summary

    async def test_nothing_happens_below_the_threshold(self, tmp_path: Path) -> None:
        manager = self.manager(tmp_path, consolidate_after_turns=50)
        await manager.observe(Message.user("one"))
        assert await manager.consolidate() is None

    async def test_old_turns_are_summarised_and_pruned(self, tmp_path: Path) -> None:
        manager = self.manager(tmp_path, consolidate_after_turns=6, consolidation_keep_turns=2)
        for index in range(10):
            await manager.observe(Message.user(f"turn {index}"))

        record = await manager.consolidate()
        assert record is not None
        assert CONSOLIDATED_TAG in record.tags
        assert await manager.conversation.count("default") == 2

    async def test_the_summary_is_recallable_afterwards(self, tmp_path: Path) -> None:
        manager = self.manager(tmp_path, consolidate_after_turns=4, consolidation_keep_turns=1)
        for index in range(8):
            await manager.observe(Message.user(f"discussed topic {index}"))
        await manager.consolidate()

        hits = await manager.recall(
            MemoryQuery(text="discussed topic", kinds=frozenset({MemoryKind.LONG_TERM}))
        )
        assert any(CONSOLIDATED_TAG in hit.record.tags for hit in hits)

    async def test_system_turns_are_never_consolidated(self, tmp_path: Path) -> None:
        manager = self.manager(tmp_path, consolidate_after_turns=4, consolidation_keep_turns=1)
        await manager.observe(Message.system("always keep me"))
        for index in range(8):
            await manager.observe(Message.user(f"turn {index}"))
        await manager.consolidate()

        remaining = await manager.conversation.history()
        assert any(message.content == "always keep me" for message in remaining)

    async def test_observe_and_consolidate_bounds_history_automatically(
        self, tmp_path: Path
    ) -> None:
        manager = self.manager(tmp_path, consolidate_after_turns=5, consolidation_keep_turns=2)
        for index in range(30):
            await manager.observe_and_consolidate(Message.user(f"turn {index}"))
        assert await manager.conversation.count("default") <= 6

    async def test_consolidator_without_a_manager_is_optional(self, tmp_path: Path) -> None:
        from eden.memory.manager import MemoryManager

        assert await MemoryManager(MemoryConfig(), []).consolidate() is None

    async def test_forcing_below_the_threshold_still_compresses(self, tmp_path: Path) -> None:
        manager = self.manager(tmp_path, consolidate_after_turns=100, consolidation_keep_turns=1)
        for index in range(5):
            await manager.observe(Message.user(f"turn {index}"))
        assert await manager.consolidate(force=True) is not None

    async def test_consolidator_is_directly_constructible(self, tmp_path: Path) -> None:
        manager = self.manager(tmp_path)
        consolidator = Consolidator(
            MemoryConfig(consolidate_after_turns=2, consolidation_keep_turns=1),
            manager.conversation,
            manager.store(MemoryKind.LONG_TERM),
            ExtractiveSummariser(),
        )
        for index in range(5):
            await manager.observe(Message.user(f"turn {index}"))
        assert await consolidator.due("default") is True
        assert await consolidator.consolidate("default") is not None
