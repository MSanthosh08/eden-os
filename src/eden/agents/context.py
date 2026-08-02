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

:meth:`AgentContext.search_files` is the one capability here that is *not*
gated by the execution pipeline, on the same reasoning ADR-0006 applied to
hardware: a search only reads, it has no effect to verify or roll back. What it
does need — and what a hardware read does not — is a boundary on *where* it may
read, since a file's path can itself be sensitive. That boundary is
``agents.search_roots``, empty by default: reading the whole filesystem is not
something an agent gets by installing EDEN, the same way running a shell
command is not until it is named in ``execution.allowed_commands``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

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

# Pruned during traversal without ever being descended into. This is a
# performance and relevance filter, not a security boundary — the security
# boundary is root confinement plus _SENSITIVE_NAME_GLOBS below.
_SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
    }
)

# Files matching any of these are never returned, regardless of the requested
# pattern. A search for "*" must not be a way to enumerate credential names.
_SENSITIVE_NAME_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "credentials*",
    "*.token",
)


@dataclass(frozen=True, slots=True)
class FileHit:
    """One file found by :meth:`AgentContext.search_files`.

    Attributes:
        path: Absolute path.
        size_bytes: File size at the time of the search.
        modified_at: Last-modified time.
    """

    path: str
    size_bytes: int
    modified_at: datetime

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at.isoformat(),
        }


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
    def has_search(self) -> bool:
        """Return whether at least one root is available to search.

        True when an operator has named a root in ``agents.search_roots``, or
        when execution is enabled (its workspace is always an implicit root).
        False means an agent should decline rather than search nothing and
        call that an answer.
        """
        return bool(self._config.search_roots) or self._execution is not None

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

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------
    async def search_files(
        self,
        pattern: str = "*",
        *,
        roots: Sequence[str] = (),
        limit: int | None = None,
    ) -> list[FileHit]:
        """Find files matching ``pattern`` under the configured search roots.

        This is a read, not an effect: it has nothing to verify, permit or roll
        back, so — unlike a write — it does not go through the execution
        pipeline. What it does enforce is a *location* boundary, since a path
        can itself be sensitive: only configured roots are ever descended into,
        every candidate is resolved through symlinks before being checked, and
        credential-shaped filenames are excluded regardless of ``pattern``.

        Args:
            pattern: A glob matched against the filename, e.g. ``"*.py"``.
            roots: Additional roots to search, beyond the configured ones. Each
                is still required to resolve inside a configured root — this
                narrows where a search may look, it can never widen it. A
                caller cannot use this to reach outside what an operator
                already allowed.
            limit: Ceiling on results. Defaults to ``agents.search_max_results``.

        Returns:
            Matching files, sorted by path, truncated to the limit.

        Raises:
            AgentCapabilityError: If no search root is configured at all.
        """
        allowed = self._search_roots()
        if not allowed:
            raise AgentCapabilityError(
                "No search roots are configured. Add paths to "
                "agents.search_roots in eden.toml, or enable execution so its "
                "workspace becomes searchable.",
                context={"key": "agents.search_roots"},
            )
        targets = self._narrow_roots(roots, allowed) if roots else allowed
        ceiling = limit or self._config.search_max_results
        return await asyncio.to_thread(self._walk, targets, pattern, ceiling)

    def _search_roots(self) -> list[Path]:
        """Return every configured root, resolved and de-duplicated."""
        candidates = list(self._config.search_roots)
        if self._execution is not None:
            candidates.append(str(self._execution.config.workspace_root))
        resolved = {_resolve(Path(candidate)) for candidate in candidates}
        return sorted(resolved)

    @staticmethod
    def _narrow_roots(requested: Sequence[str], allowed: Sequence[Path]) -> list[Path]:
        """Return the requested roots that resolve inside an allowed root.

        A root outside every allowed root is silently dropped rather than
        raised: an agent narrowing its own search to somewhere disallowed is a
        no-op, not an escalation, and should not fail the whole search.
        """
        narrowed: list[Path] = []
        for raw in requested:
            candidate = _resolve(Path(raw))
            if any(_is_within(candidate, root) for root in allowed):
                narrowed.append(candidate)
            else:
                _LOGGER.warning(
                    "Ignoring a requested search root outside the allowed set.",
                    extra={"requested": raw},
                )
        return narrowed or list(allowed)

    @staticmethod
    def _walk(roots: Sequence[Path], pattern: str, limit: int) -> list[FileHit]:
        """Walk every root, collecting matches up to ``limit``.

        Runs off the event loop via ``asyncio.to_thread``: a large tree walk is
        blocking I/O, and nothing here awaits anything.
        """
        hits: list[FileHit] = []
        seen: set[Path] = set()
        for root in roots:
            if not root.is_dir():
                continue
            for current_dir, subdirs, filenames in os.walk(root):
                subdirs[:] = [name for name in subdirs if name not in _SKIPPED_DIRECTORY_NAMES]
                for filename in filenames:
                    if len(hits) >= limit:
                        return hits
                    if not fnmatch(filename, pattern):
                        continue
                    if any(fnmatch(filename, sensitive) for sensitive in _SENSITIVE_NAME_GLOBS):
                        continue
                    path = Path(current_dir) / filename
                    if path in seen:
                        continue
                    seen.add(path)
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    hits.append(
                        FileHit(
                            path=str(path),
                            size_bytes=stat.st_size,
                            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                        )
                    )
        hits.sort(key=lambda hit: hit.path)
        return hits


def _resolve(path: Path) -> Path:
    """Return an absolute, symlink-resolved path even if it does not exist."""
    return Path(os.path.realpath(path.expanduser()))


def _is_within(candidate: Path, root: Path) -> bool:
    """Return whether ``candidate`` lies inside or at ``root``."""
    return candidate == root or root in candidate.parents
