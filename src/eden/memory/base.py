"""Memory store contract and shared base.

:class:`MemoryStore` is the structural contract every store satisfies.
:class:`BaseMemoryStore` is a Template Method carrying everything identical
across stores — namespace and tag filtering, recency blending, ordering,
validation, logging and timing — so a concrete store implements only its own
persistence and its own relevance signal.

The relevance signal is the single hook that distinguishes the stores: lexical
overlap for the non-vector ones, cosine similarity for vector memory.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from eden.config.enums import MemoryKind
from eden.logging import get_logger, timed_block
from eden.memory.types import MemoryQuery, MemoryRecord, SearchHit, keyword_score

RECENCY_WEIGHT = 0.2
_RECENCY_HALF_LIFE_SECONDS = 86_400.0


@runtime_checkable
class MemoryStore(Protocol):
    """A store of memory records with one retention policy."""

    @property
    def kind(self) -> MemoryKind:
        """Return the retention policy this store implements."""
        ...

    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        ...

    async def start(self) -> None:
        """Acquire resources. Must be idempotent."""
        ...

    async def stop(self) -> None:
        """Release resources. Must be idempotent and must not raise."""
        ...

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """Store ``record`` and return it as persisted."""
        ...

    async def add_many(self, records: Sequence[MemoryRecord]) -> list[MemoryRecord]:
        """Store several records and return them as persisted."""
        ...

    async def get(self, record_id: str, *, namespace: str) -> MemoryRecord | None:
        """Return the record with ``record_id``, or ``None``."""
        ...

    async def search(self, query: MemoryQuery) -> list[SearchHit]:
        """Return the records matching ``query``, best-first."""
        ...

    async def delete(self, record_id: str, *, namespace: str) -> bool:
        """Delete one record. Returns whether it existed."""
        ...

    async def clear(self, namespace: str) -> int:
        """Delete every record in ``namespace``. Returns the count removed."""
        ...

    async def count(self, namespace: str) -> int:
        """Return the number of records held in ``namespace``."""
        ...


class BaseMemoryStore(abc.ABC):
    """Common behaviour for every memory store.

    Subclasses implement persistence — :meth:`_persist`, :meth:`_load`,
    :meth:`_remove`, :meth:`_purge` — and optionally override
    :meth:`_relevance` to supply a different scoring signal.
    """

    def __init__(self, kind: MemoryKind, *, name: str = "") -> None:
        """Initialise the store.

        Args:
            kind: Retention policy this store implements.
            name: Logical name used in logs and in :attr:`SearchHit.store`.
                Defaults to the kind's value.
        """
        self._kind = kind
        self._name = name or kind.value
        self._logger = get_logger(f"memory.{self._name}")
        self._started = False

    # ------------------------------------------------------------------
    # Identity and lifecycle
    # ------------------------------------------------------------------
    @property
    def kind(self) -> MemoryKind:
        """Return the retention policy this store implements."""
        return self._kind

    @property
    def name(self) -> str:
        """Return the logical store name."""
        return self._name

    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return f"memory:{self._name}"

    async def start(self) -> None:
        """Prepare the store. Idempotent."""
        if self._started:
            return
        await self._on_start()
        self._started = True
        self._logger.info("Memory store started.", extra={"kind": self._kind.value})

    async def stop(self) -> None:
        """Release the store. Idempotent and never raises."""
        if not self._started:
            return
        self._started = False
        try:
            await self._on_stop()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            self._logger.warning(
                "Memory store failed to stop cleanly.",
                extra={"kind": self._kind.value, "error_type": type(exc).__name__},
            )
        else:
            self._logger.info("Memory store stopped.", extra={"kind": self._kind.value})

    async def _on_start(self) -> None:
        """Hook for subclasses needing startup work. Default does nothing."""
        return

    async def _on_stop(self) -> None:
        """Hook for subclasses needing shutdown work. Default does nothing."""
        return

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------
    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """Store ``record``, stamping it with this store's kind.

        Args:
            record: The record to persist.

        Returns:
            The record as persisted, which may differ from the input — vector
            memory attaches an embedding, for instance.
        """
        prepared = await self._prepare(record)
        await self._persist(prepared)
        self._logger.debug(
            "Memory stored.",
            extra={
                "id": prepared.id,
                "namespace": prepared.namespace,
                "kind": prepared.kind.value,
            },
        )
        return prepared

    async def add_many(self, records: Sequence[MemoryRecord]) -> list[MemoryRecord]:
        """Store several records.

        Subclasses with a cheaper batch path override :meth:`_prepare_many`.
        """
        if not records:
            return []
        prepared = await self._prepare_many(records)
        for record in prepared:
            await self._persist(record)
        return prepared

    async def get(self, record_id: str, *, namespace: str) -> MemoryRecord | None:
        """Return the record with ``record_id`` in ``namespace``."""
        for record in await self._load(namespace):
            if record.id == record_id:
                return record
        return None

    async def search(self, query: MemoryQuery) -> list[SearchHit]:
        """Return the records matching ``query``, best-first.

        Filtering, recency blending, thresholding and ordering are handled
        here; subclasses contribute only the relevance signal.
        """
        if not query.matches_kind(self._kind):
            return []
        with timed_block(
            self._logger,
            "memory.search",
            store=self._name,
            namespace=query.namespace,
        ):
            candidates = await self._load(query.namespace)
            scored = await self._score_all(query, candidates)
            hits = [hit for hit in scored if hit.score >= query.min_score]
            hits.sort(key=lambda hit: (-hit.score, -hit.record.created_at.timestamp()))
            return hits[: query.limit]

    async def delete(self, record_id: str, *, namespace: str) -> bool:
        """Delete one record. Returns whether it existed."""
        removed = await self._remove(record_id, namespace)
        if removed:
            self._logger.debug("Memory deleted.", extra={"id": record_id, "namespace": namespace})
        return removed

    async def clear(self, namespace: str) -> int:
        """Delete every record in ``namespace``. Returns the count removed."""
        removed = await self._purge(namespace)
        self._logger.info(
            "Memory namespace cleared.",
            extra={"namespace": namespace, "removed": removed, "store": self._name},
        )
        return removed

    async def count(self, namespace: str) -> int:
        """Return the number of records held in ``namespace``."""
        return len(await self._load(namespace))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    async def _score_all(
        self,
        query: MemoryQuery,
        candidates: Iterable[MemoryRecord],
    ) -> list[SearchHit]:
        """Filter and score every candidate."""
        now = datetime.now(tz=UTC)
        relevance = await self._relevance(query, list(candidates))
        hits: list[SearchHit] = []
        for record in candidates:
            if not self._passes_filters(query, record):
                continue
            base = relevance.get(record.id, 0.0)
            hits.append(
                SearchHit(
                    record=record,
                    score=self._blend(base, record, now),
                    store=self._name,
                )
            )
        return hits

    @staticmethod
    def _passes_filters(query: MemoryQuery, record: MemoryRecord) -> bool:
        """Return whether ``record`` satisfies the query's hard filters."""
        if not query.matches_kind(record.kind):
            return False
        if query.tags and not query.tags <= record.tags:
            return False
        return not (query.since is not None and record.created_at < query.since)

    @staticmethod
    def _blend(relevance: float, record: MemoryRecord, now: datetime) -> float:
        """Blend relevance with recency and importance.

        Recency decays exponentially with a one-day half-life, so a highly
        relevant old memory still outranks a fresh irrelevant one — the decay
        breaks ties, it does not dominate.

        Zero relevance short-circuits to zero. A record excluded by a hard
        signal — no lexical overlap at all, or a cosine below the configured
        floor — must not be resurrected by being recent. This mirrors the
        router's filter/rank split in ADR-0002: a hard constraint is not a
        preference that a strong enough soft signal can outweigh.
        """
        if relevance <= 0.0:
            return 0.0
        age = record.age_seconds(now)
        # `float ** float` is typed as returning Any because a negative base can
        # yield a complex result; math.pow is unambiguously float.
        recency: float = math.pow(0.5, age / _RECENCY_HALF_LIFE_SECONDS)
        blended = (1.0 - RECENCY_WEIGHT) * relevance + RECENCY_WEIGHT * recency
        return blended * (0.5 + 0.5 * record.importance)

    async def _relevance(
        self,
        query: MemoryQuery,
        candidates: Sequence[MemoryRecord],
    ) -> dict[str, float]:
        """Return a relevance score per record id.

        The default is lexical overlap. Vector memory overrides this.
        """
        tokens = query.tokens
        return {record.id: keyword_score(tokens, record.content) for record in candidates}

    async def _prepare(self, record: MemoryRecord) -> MemoryRecord:
        """Return ``record`` adjusted for this store before persistence."""
        prepared = await self._prepare_many([record])
        return prepared[0]

    async def _prepare_many(self, records: Sequence[MemoryRecord]) -> list[MemoryRecord]:
        """Return ``records`` stamped with this store's kind."""
        from dataclasses import replace  # noqa: PLC0415 - keeps the import graph flat

        return [
            record if record.kind is self._kind else replace(record, kind=self._kind)
            for record in records
        ]

    # ------------------------------------------------------------------
    # Persistence hooks
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def _persist(self, record: MemoryRecord) -> None:
        """Write one record to the backing store."""

    @abc.abstractmethod
    async def _load(self, namespace: str) -> list[MemoryRecord]:
        """Return every record held in ``namespace``."""

    @abc.abstractmethod
    async def _remove(self, record_id: str, namespace: str) -> bool:
        """Delete one record. Returns whether it existed."""

    @abc.abstractmethod
    async def _purge(self, namespace: str) -> int:
        """Delete every record in ``namespace``. Returns the count removed."""
