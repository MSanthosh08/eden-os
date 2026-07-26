"""Conversation and project memory.

Both are specialisations rather than new machinery.

:class:`ConversationMemory` adds the one thing a chat history needs that
generic memory does not: a *window* that fits a token budget. It keeps system
turns unconditionally — dropping the instructions that shape behaviour to save
tokens is never the right trade — and fills the remaining budget with the most
recent turns.

:class:`ProjectMemory` is durable memory with a fact layer on top. It composes
a store rather than subclassing one, because "remember that the deploy command
is X" is a key-value operation, not a new retention policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from eden.config.enums import MemoryKind, Role
from eden.config.schema import MemoryConfig
from eden.core.types import Message
from eden.memory.base import BaseMemoryStore, MemoryStore
from eden.memory.repository import RecordRepository
from eden.memory.stores import LongTermMemory
from eden.memory.types import DEFAULT_NAMESPACE, MemoryQuery, MemoryRecord

FACT_TAG = "fact"
_FACT_KEY = "fact_key"


class ConversationMemory(BaseMemoryStore):
    """Turn history for one or more conversations, with budgeted windowing.

    Example:
        >>> import asyncio
        >>> from eden.config.schema import MemoryConfig
        >>> from eden.core.types import Message
        >>> async def demo() -> int:
        ...     store = ConversationMemory(MemoryConfig())
        ...     await store.append_turn(Message.system("be brief"))
        ...     await store.append_turn(Message.user("hello"))
        ...     return len(await store.window())
        >>> asyncio.run(demo())
        2
    """

    def __init__(
        self,
        config: MemoryConfig,
        repository: RecordRepository | None = None,
        *,
        name: str = "conversation",
    ) -> None:
        """Initialise the store.

        Args:
            config: Retention policy supplying the turn limit and token budget.
            repository: Optional durable backend. When omitted, history is
                volatile, which is the right default for an ephemeral chat.
            name: Logical store name.
        """
        super().__init__(MemoryKind.CONVERSATION, name=name)
        self._config = config
        self._repository = repository
        self._turns: dict[str, list[MemoryRecord]] = {}

    # ------------------------------------------------------------------
    # Turn-oriented API
    # ------------------------------------------------------------------
    async def append_turn(
        self,
        message: Message,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        importance: float = 0.5,
    ) -> MemoryRecord:
        """Record one conversation turn.

        Args:
            message: The turn to record.
            namespace: Conversation identifier.
            importance: Salience used for eviction and scoring.

        Returns:
            The stored record.
        """
        return await self.add(
            MemoryRecord.from_turn(
                message.role,
                message.content,
                namespace=namespace,
                importance=importance,
            )
        )

    async def append_turns(
        self,
        messages: Sequence[Message],
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> list[MemoryRecord]:
        """Record several turns in order."""
        return await self.add_many(
            [
                MemoryRecord.from_turn(message.role, message.content, namespace=namespace)
                for message in messages
            ]
        )

    async def history(self, namespace: str = DEFAULT_NAMESPACE) -> list[Message]:
        """Return the full turn history, oldest first."""
        return [_to_message(record) for record in await self._load(namespace)]

    async def search_records(self, namespace: str = DEFAULT_NAMESPACE) -> list[MemoryRecord]:
        """Return the raw turn records, oldest first.

        Exposed for consolidation, which needs record identifiers in order to
        prune, not the reconstructed :class:`Message` view.
        """
        return await self._load(namespace)

    async def window(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        *,
        token_budget: int | None = None,
    ) -> list[Message]:
        """Return the most recent turns that fit within a token budget.

        System turns are always included, regardless of budget: they define how
        the model behaves, and silently dropping them changes the semantics of
        the conversation rather than merely shortening it.

        Args:
            namespace: Conversation identifier.
            token_budget: Ceiling in estimated tokens. Defaults to the
                configured budget.

        Returns:
            Messages in chronological order.
        """
        budget = token_budget or self._config.conversation_token_budget
        records = await self._load(namespace)

        system = [record for record in records if _role_of(record) is Role.SYSTEM]
        remaining = budget - sum(record.estimated_tokens for record in system)

        selected: list[MemoryRecord] = []
        for record in reversed(records):
            if _role_of(record) is Role.SYSTEM:
                continue
            cost = record.estimated_tokens
            if cost > remaining:
                break
            remaining -= cost
            selected.append(record)
        selected.reverse()

        ordered = sorted(system + selected, key=lambda record: record.created_at)
        return [_to_message(record) for record in ordered]

    # ------------------------------------------------------------------
    # Persistence hooks
    # ------------------------------------------------------------------
    async def _persist(self, record: MemoryRecord) -> None:
        """Append the turn and trim to the configured limit."""
        turns = self._turns.setdefault(record.namespace, [])
        turns.append(record)
        overflow = len(turns) - self._config.conversation_turn_limit
        if overflow > 0:
            del turns[:overflow]
        if self._repository is not None:
            await self._repository.append(record)

    async def _load(self, namespace: str) -> list[MemoryRecord]:
        """Return the turn history, hydrating from the repository if needed."""
        cached = self._turns.get(namespace)
        if cached is not None:
            return list(cached)
        if self._repository is None:
            return []
        stored = await self._repository.read(namespace)
        self._turns[namespace] = list(stored)
        return list(stored)

    async def _remove(self, record_id: str, namespace: str) -> bool:
        """Delete one turn. Returns whether it existed."""
        turns = self._turns.get(namespace, [])
        remaining = [record for record in turns if record.id != record_id]
        changed = len(remaining) != len(turns)
        self._turns[namespace] = remaining
        if self._repository is not None:
            changed = await self._repository.remove(record_id, namespace) or changed
        return changed

    async def _purge(self, namespace: str) -> int:
        """Delete the whole conversation."""
        removed = len(self._turns.pop(namespace, []))
        if self._repository is not None:
            removed = max(removed, await self._repository.purge(namespace))
        return removed


class ProjectMemory:
    """Durable knowledge scoped to one project, with a fact layer.

    Composes a :class:`~eden.memory.base.MemoryStore` rather than extending
    one: facts are an access pattern over durable memory, not a sixth kind.
    """

    def __init__(
        self,
        config: MemoryConfig,
        repository: RecordRepository,
        *,
        name: str = "project",
    ) -> None:
        """Initialise project memory.

        Args:
            config: Retention policy.
            repository: Injected persistence backend.
            name: Logical store name.
        """
        self._store = LongTermMemory(
            config,
            repository,
            kind=MemoryKind.PROJECT,
            name=name,
        )

    @property
    def store(self) -> MemoryStore:
        """Return the underlying store, for use by the manager."""
        return self._store

    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return self._store.component_name

    async def start(self) -> None:
        """Start the underlying store."""
        await self._store.start()

    async def stop(self) -> None:
        """Stop the underlying store."""
        await self._store.stop()

    async def remember_fact(
        self,
        key: str,
        value: str,
        *,
        project: str,
        importance: float = 0.8,
    ) -> MemoryRecord:
        """Record a durable fact, replacing any previous value for ``key``.

        Args:
            key: Fact identifier, e.g. ``"deploy_command"``.
            value: Fact content.
            project: Project namespace.
            importance: Salience. Facts default high because they are asserted
                deliberately rather than observed incidentally.

        Returns:
            The stored record.
        """
        existing = await self._find_fact(key, project)
        if existing is not None:
            await self._store.delete(existing.id, namespace=project)
        return await self._store.add(
            MemoryRecord(
                content=value,
                kind=MemoryKind.PROJECT,
                namespace=project,
                importance=importance,
                tags=frozenset({FACT_TAG}),
                metadata={_FACT_KEY: key},
            )
        )

    async def fact(self, key: str, *, project: str) -> str | None:
        """Return the current value of ``key``, or ``None``."""
        record = await self._find_fact(key, project)
        return record.content if record is not None else None

    async def facts(self, *, project: str) -> dict[str, str]:
        """Return every fact in ``project`` as a mapping."""
        hits = await self._store.search(
            MemoryQuery(namespace=project, tags=frozenset({FACT_TAG}), limit=10_000)
        )
        return {
            str(hit.record.metadata[_FACT_KEY]): hit.record.content
            for hit in hits
            if _FACT_KEY in hit.record.metadata
        }

    async def forget_fact(self, key: str, *, project: str) -> bool:
        """Delete the fact stored under ``key``. Returns whether it existed."""
        record = await self._find_fact(key, project)
        if record is None:
            return False
        return await self._store.delete(record.id, namespace=project)

    async def _find_fact(self, key: str, project: str) -> MemoryRecord | None:
        """Return the record holding ``key``, or ``None``."""
        hits = await self._store.search(
            MemoryQuery(namespace=project, tags=frozenset({FACT_TAG}), limit=10_000)
        )
        for hit in hits:
            if hit.record.metadata.get(_FACT_KEY) == key:
                return hit.record
        return None


def _role_of(record: MemoryRecord) -> Role:
    """Return the conversation role stored on ``record``."""
    raw = record.metadata.get("role")
    if isinstance(raw, str):
        try:
            return Role(raw)
        except ValueError:
            return Role.USER
    return Role.USER


def _to_message(record: MemoryRecord) -> Message:
    """Rebuild a :class:`Message` from a stored conversation turn."""
    return Message(role=_role_of(record), content=record.content)
