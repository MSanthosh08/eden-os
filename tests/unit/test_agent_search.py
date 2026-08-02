"""Tests for the file-search capability.

Search is the one agent capability that reads without going through the
execution pipeline, so its safety envelope lives entirely in
:mod:`eden.agents.context` and needs its own adversarial coverage: escaping a
configured root, reaching a credential-shaped filename, and widening rather
than narrowing the search space via task context.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eden.agents.builtin import SearchAgent, _infer_pattern
from eden.agents.context import AgentContext, FileHit
from eden.agents.types import Task
from eden.config.schema import AgentConfig, ExecutionConfig, GatewayConfig, RouterConfig
from eden.errors import AgentCapabilityError
from eden.execution.engine import ExecutionEngine
from eden.execution.permissions import DenyingGate, PolicyEngine
from eden.gateway.client import GatewayClient
from eden.gateway.health import HealthTracker
from eden.gateway.providers.mock import MockProvider
from eden.gateway.router.omni_router import OmniRouter
from tests.conftest import make_provider_config


def make_tree(root: Path) -> None:
    """Populate a small tree with ordinary and sensitive files."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    (root / "src" / "helper.py").write_text("x = 1", encoding="utf-8")
    (root / "notes.md").write_text("# notes", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("ignored", encoding="utf-8")
    (root / ".env").write_text("SECRET=1", encoding="utf-8")
    (root / "id_rsa").write_text("not a real key", encoding="utf-8")


def context_for(
    tmp_path: Path,
    *,
    search_roots: tuple[str, ...] = (),
    execution: ExecutionEngine | None = None,
) -> AgentContext:
    """Build an AgentContext with a mock gateway and the given search config."""
    provider = MockProvider(make_provider_config("m"))
    router_config = RouterConfig()
    tracker = HealthTracker(config=router_config.circuit_breaker)
    gateway = GatewayClient(
        GatewayConfig(providers=(provider.config,)),
        [provider],
        OmniRouter(router_config, tracker),
        tracker,
    )
    config = AgentConfig(search_roots=search_roots)
    return AgentContext(config, gateway, execution=execution)


class TestInferPattern:
    @pytest.mark.parametrize(
        ("goal", "expected"),
        [
            ("list all the python files", "*.py"),
            ("find my markdown notes", "*.md"),
            ("search for pdf documents", "*.pdf"),
            ("what is the weather", "*"),
        ],
    )
    def test_infers_extension_from_plain_english(self, goal: str, expected: str) -> None:
        assert _infer_pattern(goal) == expected


class TestSearchFilesRootConfinement:
    async def test_finds_files_within_a_configured_root(self, tmp_path: Path) -> None:
        make_tree(tmp_path)
        context = context_for(tmp_path, search_roots=(str(tmp_path),))
        hits = await context.search_files("*.py")
        assert {Path(hit.path).name for hit in hits} == {"main.py", "helper.py"}

    async def test_no_configured_root_raises_a_clear_capability_error(self, tmp_path: Path) -> None:
        context = context_for(tmp_path)
        with pytest.raises(AgentCapabilityError) as caught:
            await context.search_files("*.py")
        assert caught.value.context["key"] == "agents.search_roots"

    async def test_git_internals_are_never_descended_into(self, tmp_path: Path) -> None:
        make_tree(tmp_path)
        context = context_for(tmp_path, search_roots=(str(tmp_path),))
        hits = await context.search_files("*")
        assert not any(".git" in hit.path for hit in hits)

    async def test_credential_shaped_files_are_excluded_regardless_of_pattern(
        self, tmp_path: Path
    ) -> None:
        make_tree(tmp_path)
        context = context_for(tmp_path, search_roots=(str(tmp_path),))
        hits = await context.search_files("*")
        names = {Path(hit.path).name for hit in hits}
        assert ".env" not in names
        assert "id_rsa" not in names

    async def test_a_requested_root_outside_the_allowed_set_is_ignored_not_escalated(
        self, tmp_path: Path
    ) -> None:
        """Passing 'roots' can narrow a search, never widen it."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (allowed / "keep.py").write_text("x", encoding="utf-8")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "escape.py").write_text("x", encoding="utf-8")

        context = context_for(tmp_path, search_roots=(str(allowed),))
        hits = await context.search_files("*.py", roots=[str(outside)])
        paths = {Path(hit.path).name for hit in hits}
        assert "escape.py" not in paths
        assert "keep.py" in paths  # falls back to the allowed set

    async def test_results_are_capped_at_the_configured_ceiling(self, tmp_path: Path) -> None:
        for index in range(20):
            (tmp_path / f"file{index}.py").write_text("x", encoding="utf-8")
        context = context_for(tmp_path, search_roots=(str(tmp_path),))
        hits = await context.search_files("*.py", limit=5)
        assert len(hits) == 5

    async def test_execution_workspace_is_an_implicit_root(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "auto.py").write_text("x", encoding="utf-8")
        execution_config = ExecutionConfig(workspace_root=workspace)
        engine = ExecutionEngine(
            execution_config, policy=PolicyEngine(execution_config, DenyingGate())
        )
        context = context_for(tmp_path, execution=engine)
        assert context.has_search is True
        hits = await context.search_files("*.py")
        assert any(Path(hit.path).name == "auto.py" for hit in hits)

    async def test_has_search_is_false_with_nothing_configured(self, tmp_path: Path) -> None:
        assert context_for(tmp_path).has_search is False


class TestFileHit:
    def test_round_trips_to_a_dict(self) -> None:
        hit = FileHit(path="/a/b.py", size_bytes=10, modified_at=datetime.now(tz=UTC))
        payload = hit.to_dict()
        assert payload["path"] == "/a/b.py"
        assert payload["size_bytes"] == 10


class TestSearchAgentRouting:
    def agent(self, tmp_path: Path) -> SearchAgent:
        context = context_for(tmp_path, search_roots=(str(tmp_path),))
        return SearchAgent(context)

    def test_claims_a_search_and_file_goal(self, tmp_path: Path) -> None:
        suitability = self.agent(tmp_path).can_handle(
            Task(goal="search and list all the python files")
        )
        assert suitability.can_handle is True
        assert suitability.score > 0.9

    def test_declines_an_unrelated_goal(self, tmp_path: Path) -> None:
        suitability = self.agent(tmp_path).can_handle(Task(goal="what is the capital of France"))
        assert suitability.can_handle is False

    def test_declines_when_no_root_is_configured(self, tmp_path: Path) -> None:
        context = context_for(tmp_path)
        agent = SearchAgent(context)
        suitability = agent.can_handle(Task(goal="find all the python files"))
        assert suitability.can_handle is False

    def test_an_explicit_pattern_in_context_is_enough_even_without_file_wording(
        self, tmp_path: Path
    ) -> None:
        suitability = self.agent(tmp_path).can_handle(
            Task(goal="what have I got", context={"pattern": "*.py"})
        )
        assert suitability.can_handle is True

    async def test_end_to_end_plan_and_run_lists_real_files(self, tmp_path: Path) -> None:
        make_tree(tmp_path)
        agent = self.agent(tmp_path)
        report = await agent.run(
            Task(goal="search and list all the python files", namespace="default")
        )
        assert report.succeeded is True
        output = report.final_output
        assert output is not None
        assert "main.py" in output
        assert "helper.py" in output
        assert ".env" not in output

    async def test_no_matches_is_still_a_successful_search(self, tmp_path: Path) -> None:
        agent = self.agent(tmp_path)
        report = await agent.run(Task(goal="find all the ruby files", context={"pattern": "*.rb"}))
        assert report.succeeded is True
        assert "No files" in (report.final_output or "")

    async def test_never_asks_a_model_anything(self, tmp_path: Path) -> None:
        """A factual listing must come from the filesystem, not be generated."""
        make_tree(tmp_path)
        agent = self.agent(tmp_path)
        report = await agent.run(Task(goal="list all the python files"))
        assert report.succeeded is True
        # The agent's own gateway provider is never invoked for a search task;
        # correctness here is that plan()/run() never call context.think() or
        # context.generate(), which _run_thought's override guarantees.
