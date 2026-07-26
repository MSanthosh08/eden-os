"""Short-term and long-term stores.

These two differ in exactly two ways: where they persist, and how they forget.
Short-term memory is volatile, bounded and time-limited. Long-term memory is
durable and bounded only by a large ceiling. Everything else — filtering,
scoring, ordering — is inherited from :class:`~eden.memory.base.BaseMemoryStore`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from eden.config.enums import MemoryKind
from eden.config.schema import MemoryConfig
from eden.errors import MemoryCapacityError
from eden.memory.base import BaseMemoryStore
from eden.memory.repository import RecordRepository
from eden.memory.types import MemoryRecord


class ShortTermMemory(BaseMemoryStore):
    """Volatile, bounded, time-limited working memory.

    Eviction is by *importance then age*, not age alone: a buffer full of
    trivia should surrender the trivia, not the one thing worth keeping.

    Example:
        >>> import asyncio
        >>> from eden.config.schema import MemoryConfig
        >>> from eden.memory.types import MemoryRecord
        >>> async def demo() -> int:
        ...     store = ShortTermMemory(MemoryConfig(short_term_capacity=2))
        ...     for text in ("a", "b", "c"):
        ...         await store.add(MemoryRecord(content=text))
        ...     return await store.count("default")
        >>> asyncio.run(demo())
        2
    """

    def __init__(self, config: MemoryConfig, *, name: str = "short_term") -> None:
        """Initialise the store.

        Args:
            config: Retention policy supplying capacity and time-to-live.
            name: Logical store name.
        """
        super().__init__(MemoryKind.SHORT_TERM, name=name)
        self._config = config
        self._buffers: dict[str, list[MemoryRecord]] = {}
        self._lock = asyncio.Lock()

    async def _persist(self, record: MemoryRecord) -> None:
        """Append the record and evict down to capacity."""
        async with self._lock:
            buffer = self._buffers.setdefault(record.namespace, [])
            buffer.append(record)
            self._expire(buffer)
            self._evict(buffer)

    async def _load(self, namespace: str) -> list[MemoryRecord]:
        """Return live records, dropping any that have expired."""
        async with self._lock:
            buffer = self._buffers.get(namespace)
            if buffer is None:
                return []
            self._expire(buffer)
            return list(buffer)

    async def _remove(self, record_id: str, namespace: str) -> bool:
        """Delete one record. Returns whether it existed."""
        async with self._lock:
            buffer = self._buffers.get(namespace)
            if not buffer:
                return False
            remaining = [record for record in buffer if record.id != record_id]
            if len(remaining) == len(buffer):
                return False
            self._buffers[namespace] = remaining
            return True

    async def _purge(self, namespace: str) -> int:
        """Drop the whole namespace buffer."""
        async with self._lock:
            return len(self._buffers.pop(namespace, []))

    def _expire(self, buffer: list[MemoryRecord]) -> None:
        """Remove records older than the configured time-to-live, in place."""
        now = datetime.now(tz=UTC)
        ttl = self._config.short_term_ttl_seconds
        buffer[:] = [record for record in buffer if record.age_seconds(now) < ttl]

    def _evict(self, buffer: list[MemoryRecord]) -> None:
        """Trim the buffer to capacity, in place."""
        overflow = len(buffer) - self._config.short_term_capacity
        if overflow <= 0:
            return
        # Sort a copy to find the weakest records, then filter the original so
        # the buffer keeps its chronological order.
        weakest = sorted(
            buffer,
            key=lambda record: (record.importance, record.created_at),
        )[:overflow]
        doomed = {record.id for record in weakest}
        buffer[:] = [record for record in buffer if record.id not in doomed]
        self._logger.debug("Evicted from short-term memory.", extra={"count": overflow})


class LongTermMemory(BaseMemoryStore):
    """Durable memory delegating persistence to a repository.

    The store is deliberately ignorant of the storage medium; swapping JSONL
    for a database is a constructor argument.
    """

    def __init__(
        self,
        config: MemoryConfig,
        repository: RecordRepository,
        *,
        kind: MemoryKind = MemoryKind.LONG_TERM,
        name: str = "long_term",
    ) -> None:
        """Initialise the store.

        Args:
            config: Retention policy supplying the record ceiling.
            repository: Injected persistence backend.
            kind: Retention policy stamped onto stored records. Project memory
                reuses this class with a different kind.
            name: Logical store name.
        """
        super().__init__(kind, name=name)
        self._config = config
        self._repository = repository

    @property
    def repository(self) -> RecordRepository:
        """Return the backing repository."""
        return self._repository

    async def _persist(self, record: MemoryRecord) -> None:
        """Write the record, refusing once the namespace ceiling is reached.

        Raises:
            MemoryCapacityError: If the namespace is full. This is a hard stop
                rather than silent eviction — durable memory that quietly
                discards data is worse than durable memory that says no.
        """
        existing = await self._repository.read(record.namespace)
        if len(existing) >= self._config.long_term_max_records:
            raise MemoryCapacityError(
                "Long-term namespace has reached its configured record ceiling.",
                context={
                    "namespace": record.namespace,
                    "limit": self._config.long_term_max_records,
                },
            )
        await self._repository.append(record)

    async def _load(self, namespace: str) -> list[MemoryRecord]:
        """Return every record in ``namespace``."""
        return await self._repository.read(namespace)

    async def _remove(self, record_id: str, namespace: str) -> bool:
        """Delete one record. Returns whether it existed."""
        return await self._repository.remove(record_id, namespace)

    async def _purge(self, namespace: str) -> int:
        """Delete every record in ``namespace``."""
        return await self._repository.purge(namespace)
