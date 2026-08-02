"""Memory subsystem façade.

Agents should ask "what do I know about X?", not "which of five stores might
hold this?". :class:`MemoryManager` answers the first question by fanning one
query across every store concurrently and merging the results into a single
ranked list.

Merging is the interesting part. Scores from different stores are computed by
different signals, so a raw comparison would be meaningless. Each store's hits
are therefore normalised within that store before merging, and a per-store
weight expresses editorial preference — semantic hits from vector memory
outrank incidental lexical overlap in a chat log.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from eden.config.enums import MemoryKind, Role
from eden.config.schema import MemoryConfig, PathsConfig
from eden.core.types import Message
from eden.errors import MemorySubsystemError
from eden.gateway.client import GatewayClient
from eden.logging import get_logger, timed_block
from eden.memory.base import MemoryStore
from eden.memory.consolidation import (
    Consolidator,
    ExtractiveSummariser,
    GatewaySummariser,
    Summariser,
)
from eden.memory.conversation import ConversationMemory, ProjectMemory
from eden.memory.facts import FactExtractor, HeuristicFactExtractor
from eden.memory.repository import (
    InMemoryRecordRepository,
    JsonlRecordRepository,
    RecordRepository,
)
from eden.memory.stores import LongTermMemory, ShortTermMemory
from eden.memory.types import DEFAULT_NAMESPACE, MemoryQuery, MemoryRecord, SearchHit
from eden.memory.vector import Embedder, GatewayEmbedder, HashEmbedder, VectorMemory
from eden.utils.async_tools import gather_limited

_LOGGER = get_logger("memory.manager")

COMPONENT_NAME = "memory"

DEFAULT_STORE_WEIGHTS: Mapping[MemoryKind, float] = {
    MemoryKind.VECTOR: 1.0,
    MemoryKind.PROJECT: 0.9,
    MemoryKind.LONG_TERM: 0.8,
    MemoryKind.SHORT_TERM: 0.7,
    MemoryKind.CONVERSATION: 0.5,
}

_MAX_PARALLEL_STORES = 8


class MemoryManager:
    """Owns every memory store and provides one recall entry point."""

    def __init__(
        self,
        config: MemoryConfig,
        stores: Sequence[MemoryStore],
        *,
        conversation: ConversationMemory | None = None,
        project: ProjectMemory | None = None,
        weights: Mapping[MemoryKind, float] | None = None,
        consolidator: Consolidator | None = None,
        fact_extractor: FactExtractor | None = None,
    ) -> None:
        """Initialise the manager.

        Args:
            config: Retention policy.
            stores: Every store participating in recall.
            conversation: Conversation store, exposed for turn-level access.
            project: Project memory, exposed for fact-level access.
            weights: Per-kind editorial weights applied when merging.
            consolidator: Compresses old conversation turns into long-term
                memory. Absent means conversations grow without bound.
            fact_extractor: Recognises durable facts — a name, a location — in
                each user turn and promotes them into project memory, where
                they survive regardless of the conversation's token budget.
                Absent means nothing is extracted automatically.
        """
        self._config = config
        self._stores = list(stores)
        self._conversation = conversation
        self._project = project
        self._weights = dict(weights or DEFAULT_STORE_WEIGHTS)
        self._consolidator = consolidator
        self._fact_extractor = fact_extractor
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return COMPONENT_NAME

    async def start(self) -> None:
        """Start every store. Idempotent."""
        if self._started:
            return
        for store in self._stores:
            await store.start()
        self._started = True
        _LOGGER.info(
            "Memory subsystem started.",
            extra={"stores": [store.kind.value for store in self._stores]},
        )

    async def stop(self) -> None:
        """Stop every store in reverse order. Idempotent and never raises."""
        if not self._started:
            return
        self._started = False
        for store in reversed(self._stores):
            await store.stop()
        _LOGGER.info("Memory subsystem stopped.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def stores(self) -> tuple[MemoryStore, ...]:
        """Return every registered store."""
        return tuple(self._stores)

    def store(self, kind: MemoryKind) -> MemoryStore:
        """Return the store implementing ``kind``.

        Raises:
            MemorySubsystemError: If no store implements that kind.
        """
        for candidate in self._stores:
            if candidate.kind is kind:
                return candidate
        raise MemorySubsystemError(
            "No store is registered for this memory kind.",
            context={
                "kind": kind.value,
                "available": [store.kind.value for store in self._stores],
            },
        )

    @property
    def conversation(self) -> ConversationMemory:
        """Return conversation memory.

        Raises:
            MemorySubsystemError: If conversation memory is not configured.
        """
        if self._conversation is None:
            raise MemorySubsystemError("Conversation memory is not configured.")
        return self._conversation

    @property
    def project(self) -> ProjectMemory:
        """Return project memory.

        Raises:
            MemorySubsystemError: If project memory is not configured.
        """
        if self._project is None:
            raise MemorySubsystemError("Project memory is not configured.")
        return self._project

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    async def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.LONG_TERM,
        namespace: str = DEFAULT_NAMESPACE,
        importance: float = 0.5,
        tags: frozenset[str] = frozenset(),
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord:
        """Store one memory in the store owning ``kind``.

        Args:
            content: Text to remember.
            kind: Retention policy that should own it.
            namespace: Isolation boundary.
            importance: Salience in ``[0, 1]``.
            tags: Labels usable as a hard search filter.
            metadata: Structured annotations.

        Returns:
            The stored record.

        Raises:
            MemorySubsystemError: If no store implements ``kind``.
        """
        record = MemoryRecord(
            content=content,
            kind=kind,
            namespace=namespace,
            importance=importance,
            tags=tags,
            metadata=dict(metadata or {}),
        )
        return await self.store(kind).add(record)

    async def observe(
        self,
        message: Message,
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> MemoryRecord:
        """Record a conversation turn.

        User turns are additionally scanned for durable facts — a stated name,
        location or occupation — which are promoted into project memory. That
        promotion is what lets a fact outlive the conversation's token budget:
        the raw turn can age out of the window while the fact remains.

        A failure during extraction or promotion is logged and swallowed
        rather than raised: losing a nice-to-have fact must never cost the
        conversation turn itself.

        Raises:
            MemorySubsystemError: If conversation memory is not configured.
        """
        record = await self.conversation.append_turn(message, namespace=namespace)
        if self._fact_extractor is not None and message.role is Role.USER:
            await self._extract_facts(message.content, namespace)
        return record

    async def _extract_facts(self, text: str, namespace: str) -> None:
        """Recognise and durably store any facts found in ``text``."""
        if self._fact_extractor is None or self._project is None:
            return
        try:
            facts = self._fact_extractor.extract(text)
        except Exception as exc:  # noqa: BLE001 - extraction must never break chat
            _LOGGER.warning(
                "Fact extraction failed; continuing without it.",
                extra={"namespace": namespace, "error_type": type(exc).__name__},
            )
            return
        for fact in facts:
            try:
                await self._project.remember_fact(fact.key, fact.value, project=namespace)
            except MemorySubsystemError as exc:
                _LOGGER.warning(
                    "Could not store an extracted fact.",
                    extra={
                        "namespace": namespace,
                        "key": fact.key,
                        "error_code": exc.code,
                    },
                )
            else:
                _LOGGER.debug(
                    "Extracted a durable fact from conversation.",
                    extra={"namespace": namespace, "key": fact.key},
                )

    async def facts_message(self, namespace: str) -> Message | None:
        """Return a system message summarising known facts, or ``None``.

        Prepending this to a prompt is what makes a fact like a name available
        regardless of how long the conversation has grown — it does not
        compete with ordinary turns for the token budget the same way, since it
        is looked up directly rather than carried in the rolling window.

        Args:
            namespace: Project namespace to read facts from.

        Returns:
            A system message listing known facts, or ``None`` when project
            memory is not configured or holds nothing yet.
        """
        if self._project is None:
            return None
        facts = await self._project.facts(project=namespace)
        if not facts:
            return None
        lines = "; ".join(f"{key}: {value}" for key, value in sorted(facts.items()))
        return Message.system(f"Known facts about this person: {lines}.")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    async def recall(
        self,
        query: MemoryQuery,
        *,
        kinds: frozenset[MemoryKind] | None = None,
    ) -> list[SearchHit]:
        """Search every store concurrently and return one merged ranking.

        Args:
            query: The recall request.
            kinds: Restrict the fan-out to these stores. ``None`` searches all
                stores whose kind the query admits.

        Returns:
            Merged hits, best-first, truncated to ``query.limit``.
        """
        targets = [
            store
            for store in self._stores
            if (kinds is None or store.kind in kinds) and query.matches_kind(store.kind)
        ]
        if not targets:
            return []

        with timed_block(_LOGGER, "memory.recall", stores=len(targets), namespace=query.namespace):
            results = await gather_limited(
                [_SearchTask(store, query) for store in targets],
                limit=min(len(targets), _MAX_PARALLEL_STORES),
            )

        merged: list[SearchHit] = []
        for store, result in zip(targets, results, strict=True):
            if isinstance(result, BaseException):
                _LOGGER.warning(
                    "Memory store failed during recall; continuing without it.",
                    extra={"store": store.kind.value, "error_type": type(result).__name__},
                )
                continue
            merged.extend(self._normalise(store.kind, result))

        merged.sort(key=lambda hit: -hit.score)
        return merged[: query.limit]

    def _normalise(self, kind: MemoryKind, hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Rescale one store's hits to ``[0, 1]`` and apply its weight.

        Without per-store normalisation, a store whose scorer happens to
        produce larger numbers would dominate purely by scale.
        """
        if not hits:
            return []
        weight = self._weights.get(kind, 1.0)
        highest = max(hit.score for hit in hits)
        if highest <= 0.0:
            return []
        return [
            SearchHit(record=hit.record, score=(hit.score / highest) * weight, store=hit.store)
            for hit in hits
        ]

    async def consolidate(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        *,
        force: bool = False,
    ) -> MemoryRecord | None:
        """Compress old conversation turns into a long-term summary.

        Args:
            namespace: Conversation to compress.
            force: Compress even below the configured threshold.

        Returns:
            The stored summary, or ``None`` when nothing needed compressing or
            consolidation is not configured.
        """
        if self._consolidator is None:
            return None
        return await self._consolidator.consolidate(namespace, force=force)

    async def observe_and_consolidate(
        self,
        message: Message,
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> MemoryRecord:
        """Record a turn and compress the history if it has grown too long.

        This is the call a long-running agent makes on every turn, so that
        bounded history is the default rather than something to remember.
        """
        record = await self.observe(message, namespace=namespace)
        await self.consolidate(namespace)
        return record

    async def forget(self, namespace: str, *, kind: MemoryKind | None = None) -> int:
        """Delete records in ``namespace``.

        Args:
            namespace: Isolation boundary to clear.
            kind: Restrict to one store. ``None`` clears every store.

        Returns:
            The total number of records removed.
        """
        targets = self._stores if kind is None else [self.store(kind)]
        removed = 0
        for store in targets:
            removed += await store.clear(namespace)
        return removed


class _SearchTask:
    """A zero-argument coroutine factory searching one store."""

    __slots__ = ("_query", "_store")

    def __init__(self, store: MemoryStore, query: MemoryQuery) -> None:
        """Store the target and the query."""
        self._store = store
        self._query = query

    async def __call__(self) -> list[SearchHit]:
        """Run the search."""
        return await self._store.search(self._query)


def build_memory_manager(
    config: MemoryConfig,
    paths: PathsConfig,
    *,
    gateway: GatewayClient | None = None,
    embedder: Embedder | None = None,
    summariser: Summariser | None = None,
    fact_extractor: FactExtractor | None = None,
) -> MemoryManager:
    """Construct a fully wired memory subsystem.

    Persistence and the embedder are both chosen here, at the composition root
    of the subsystem, so that no store has to decide for itself.

    Args:
        config: Retention policy.
        paths: Filesystem layout supplying the durable storage root.
        gateway: Gateway used for embeddings. When omitted, the offline hash
            embedder is used, which keeps vector memory working with no
            credentials and no network.
        embedder: Explicit embedder override, primarily for tests.
        summariser: Explicit summariser override, primarily for tests.
        fact_extractor: Explicit extractor override. Defaults to
            :class:`~eden.memory.facts.HeuristicFactExtractor`, so a name or
            location stated in chat is durable without any configuration.

    Returns:
        A ready-to-start manager.
    """
    root: Path = paths.data_dir / "memory"

    def repository(subdirectory: str) -> RecordRepository:
        if not config.persist:
            return InMemoryRecordRepository()
        return JsonlRecordRepository(root / subdirectory)

    resolved_embedder: Embedder
    if embedder is not None:
        resolved_embedder = embedder
    elif gateway is not None:
        resolved_embedder = GatewayEmbedder(gateway, config)
    else:
        resolved_embedder = HashEmbedder(config.vector_dimensions)

    short_term = ShortTermMemory(config)
    long_term = LongTermMemory(config, repository("long_term"))
    vector = VectorMemory(config, repository("vector"), resolved_embedder)
    conversation = ConversationMemory(
        config,
        repository("conversation") if config.persist else None,
    )
    project = ProjectMemory(config, repository("project"))

    resolved_summariser: Summariser
    if summariser is not None:
        resolved_summariser = summariser
    elif gateway is not None:
        resolved_summariser = GatewaySummariser(gateway)
    else:
        resolved_summariser = ExtractiveSummariser()

    resolved_extractor: FactExtractor = (
        fact_extractor if fact_extractor is not None else HeuristicFactExtractor()
    )

    return MemoryManager(
        config,
        [short_term, long_term, vector, conversation, project.store],
        conversation=conversation,
        project=project,
        consolidator=Consolidator(config, conversation, long_term, resolved_summariser),
        fact_extractor=resolved_extractor,
    )
