"""Record persistence.

The stores do not know how records reach disk; they delegate to a
:class:`RecordRepository`. That indirection is what lets the same
:class:`~eden.memory.long_term.LongTermMemory` run against an append-only file
today and against SQLite or Postgres later without the store changing.

The shipped file repository is append-only with deferred compaction: writes are
a single ``open`` in append mode, and the file is only rewritten when a delete
actually removes something. Append-only also means an interrupted write costs
at most the final line, which the loader skips.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from eden.errors import MemoryStorageError, ValidationError
from eden.logging import get_logger
from eden.memory.types import MemoryRecord

_LOGGER = get_logger("memory.repository")
_SAFE_NAMESPACE = re.compile(r"[^A-Za-z0-9._-]+")


@runtime_checkable
class RecordRepository(Protocol):
    """Durable storage for memory records, partitioned by namespace."""

    async def append(self, record: MemoryRecord) -> None:
        """Persist one record."""
        ...

    async def read(self, namespace: str) -> list[MemoryRecord]:
        """Return every record in ``namespace``, oldest first."""
        ...

    async def remove(self, record_id: str, namespace: str) -> bool:
        """Delete one record. Returns whether it existed."""
        ...

    async def purge(self, namespace: str) -> int:
        """Delete every record in ``namespace``. Returns the count removed."""
        ...

    async def replace_all(self, namespace: str, records: Sequence[MemoryRecord]) -> None:
        """Overwrite ``namespace`` with exactly ``records``."""
        ...


class InMemoryRecordRepository:
    """A volatile repository. The default when persistence is disabled."""

    def __init__(self) -> None:
        """Create an empty repository."""
        self._data: dict[str, list[MemoryRecord]] = {}
        self._lock = asyncio.Lock()

    async def append(self, record: MemoryRecord) -> None:
        """Persist one record."""
        async with self._lock:
            self._data.setdefault(record.namespace, []).append(record)

    async def read(self, namespace: str) -> list[MemoryRecord]:
        """Return every record in ``namespace``, oldest first."""
        async with self._lock:
            return list(self._data.get(namespace, ()))

    async def remove(self, record_id: str, namespace: str) -> bool:
        """Delete one record. Returns whether it existed."""
        async with self._lock:
            records = self._data.get(namespace)
            if not records:
                return False
            remaining = [record for record in records if record.id != record_id]
            if len(remaining) == len(records):
                return False
            self._data[namespace] = remaining
            return True

    async def purge(self, namespace: str) -> int:
        """Delete every record in ``namespace``. Returns the count removed."""
        async with self._lock:
            return len(self._data.pop(namespace, []))

    async def replace_all(self, namespace: str, records: Sequence[MemoryRecord]) -> None:
        """Overwrite ``namespace`` with exactly ``records``."""
        async with self._lock:
            self._data[namespace] = list(records)


class JsonlRecordRepository:
    """An append-only JSON Lines repository, one file per namespace.

    Example:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> from eden.memory.types import MemoryRecord
        >>> async def demo() -> int:
        ...     with tempfile.TemporaryDirectory() as tmp:
        ...         repo = JsonlRecordRepository(Path(tmp))
        ...         await repo.append(MemoryRecord(content="hello"))
        ...         return len(await repo.read("default"))
        >>> asyncio.run(demo())
        1
    """

    def __init__(self, directory: Path, *, suffix: str = ".jsonl") -> None:
        """Initialise the repository.

        Args:
            directory: Root under which namespace files are written.
            suffix: File extension used for namespace files.
        """
        self._directory = directory
        self._suffix = suffix
        self._lock = asyncio.Lock()

    def path_for(self, namespace: str) -> Path:
        """Return the file backing ``namespace``.

        The namespace is sanitised so that a hostile value cannot escape the
        configured directory.
        """
        safe = _SAFE_NAMESPACE.sub("_", namespace).strip("._-") or "default"
        return self._directory / f"{safe}{self._suffix}"

    async def append(self, record: MemoryRecord) -> None:
        """Persist one record.

        Raises:
            MemoryStorageError: If the write fails.
        """
        async with self._lock:
            await asyncio.to_thread(self._append_sync, record)

    def _append_sync(self, record: MemoryRecord) -> None:
        """Append one JSON line, creating the directory when needed."""
        path = self.path_for(record.namespace)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), separators=(",", ":")))
                handle.write("\n")
        except OSError as exc:
            raise MemoryStorageError(
                "Could not append to the memory file.",
                context={"path": str(path)},
                cause=exc,
            ) from exc

    async def read(self, namespace: str) -> list[MemoryRecord]:
        """Return every record in ``namespace``, oldest first."""
        async with self._lock:
            return await asyncio.to_thread(self._read_sync, namespace)

    def _read_sync(self, namespace: str) -> list[MemoryRecord]:
        """Parse the namespace file, skipping unreadable lines.

        A corrupt or truncated line is logged and skipped rather than failing
        the whole recall: partial memory is far better than none.

        Raises:
            MemoryStorageError: If the file exists but cannot be opened.
        """
        path = self.path_for(namespace)
        if not path.is_file():
            return []
        records: list[MemoryRecord] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(MemoryRecord.from_dict(json.loads(stripped)))
                    except (json.JSONDecodeError, ValidationError):
                        _LOGGER.warning(
                            "Skipping unreadable memory line.",
                            extra={"path": str(path), "line": number},
                        )
        except OSError as exc:
            raise MemoryStorageError(
                "Could not read the memory file.",
                context={"path": str(path)},
                cause=exc,
            ) from exc
        return records

    async def remove(self, record_id: str, namespace: str) -> bool:
        """Delete one record, compacting the file only when it existed."""
        existing = await self.read(namespace)
        remaining = [record for record in existing if record.id != record_id]
        if len(remaining) == len(existing):
            return False
        await self.replace_all(namespace, remaining)
        return True

    async def purge(self, namespace: str) -> int:
        """Delete the namespace file. Returns the count removed."""
        existing = await self.read(namespace)
        async with self._lock:
            await asyncio.to_thread(self._unlink_sync, namespace)
        return len(existing)

    def _unlink_sync(self, namespace: str) -> None:
        """Remove the namespace file if present.

        Raises:
            MemoryStorageError: If deletion fails.
        """
        path = self.path_for(namespace)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise MemoryStorageError(
                "Could not delete the memory file.",
                context={"path": str(path)},
                cause=exc,
            ) from exc

    async def replace_all(self, namespace: str, records: Sequence[MemoryRecord]) -> None:
        """Rewrite ``namespace`` atomically via a temporary file."""
        async with self._lock:
            await asyncio.to_thread(self._replace_sync, namespace, list(records))

    def _replace_sync(self, namespace: str, records: Sequence[MemoryRecord]) -> None:
        """Write to a sibling temp file, then rename over the original.

        Rename is atomic on POSIX, so a crash mid-compaction leaves the old
        file intact rather than a half-written one.

        Raises:
            MemoryStorageError: If the rewrite fails.
        """
        path = self.path_for(namespace)
        temporary = path.with_suffix(f"{self._suffix}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_dict(), separators=(",", ":")))
                    handle.write("\n")
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise MemoryStorageError(
                "Could not rewrite the memory file.",
                context={"path": str(path)},
                cause=exc,
            ) from exc
