"""Execution audit journal.

Every action produces a journal entry, including the ones that were refused.
A record of what EDEN *declined* to do is as much a part of the audit trail as
a record of what it did, and an empty journal after a denied request would look
identical to one after a request that never arrived.

Journalling never fails the pipeline. A broken audit sink is logged loudly, but
it does not become a reason to abandon an action that policy already approved.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from eden.execution.types import ExecutionRecord
from eden.logging import get_logger

_LOGGER = get_logger("execution.journal")

JOURNAL_FILENAME = "execution.jsonl"


@runtime_checkable
class ExecutionJournal(Protocol):
    """Durable record of every pipeline decision."""

    async def append(self, record: ExecutionRecord) -> None:
        """Write one record."""
        ...

    async def read(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent entries, newest last."""
        ...


class NullJournal:
    """Discards everything. Used when journalling is switched off."""

    async def append(self, record: ExecutionRecord) -> None:
        """Discard the record."""
        del record

    async def read(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return nothing."""
        del limit
        return []


class InMemoryJournal:
    """Keeps entries in process. The default when persistence is disabled."""

    def __init__(self, *, capacity: int = 10_000) -> None:
        """Initialise the journal.

        Args:
            capacity: Maximum entries retained before the oldest are dropped.
        """
        self._entries: list[dict[str, Any]] = []
        self._capacity = capacity
        self._lock = asyncio.Lock()

    async def append(self, record: ExecutionRecord) -> None:
        """Store the record, trimming to capacity."""
        async with self._lock:
            self._entries.append(record.to_dict())
            overflow = len(self._entries) - self._capacity
            if overflow > 0:
                del self._entries[:overflow]

    async def read(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent entries, newest last."""
        async with self._lock:
            return list(self._entries[-limit:])

    async def entries(self) -> Sequence[dict[str, Any]]:
        """Return every retained entry."""
        async with self._lock:
            return list(self._entries)


class JsonlJournal:
    """Appends one JSON object per line to a file.

    Append-only, so an interrupted write costs at most the final line, which
    the reader skips.
    """

    def __init__(self, path: Path) -> None:
        """Initialise the journal.

        Args:
            path: File the journal is written to. Parents are created on demand.
        """
        self._path = path
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        """Return the journal file path."""
        return self._path

    async def append(self, record: ExecutionRecord) -> None:
        """Write one record, logging rather than raising on failure."""
        async with self._lock:
            try:
                await asyncio.to_thread(self._append_sync, record)
            except OSError as exc:
                _LOGGER.error(
                    "Could not write to the execution journal.",
                    extra={"path": str(self._path), "error_type": type(exc).__name__},
                )

    def _append_sync(self, record: ExecutionRecord) -> None:
        """Append one JSON line."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=str, separators=(",", ":")))
            handle.write("\n")

    async def read(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent entries, newest last."""
        async with self._lock:
            return await asyncio.to_thread(self._read_sync, limit)

    def _read_sync(self, limit: int) -> list[dict[str, Any]]:
        """Parse the journal, skipping unreadable lines."""
        if not self._path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        entries.append(parsed)
        except OSError as exc:
            _LOGGER.error(
                "Could not read the execution journal.",
                extra={"path": str(self._path), "error_type": type(exc).__name__},
            )
            return []
        return entries[-limit:]


def build_journal(*, enabled: bool, directory: Path | None) -> ExecutionJournal:
    """Return the journal implied by configuration.

    Args:
        enabled: Whether journalling is switched on at all.
        directory: Where to write. ``None`` keeps entries in process.

    Returns:
        A ready-to-use journal.
    """
    if not enabled:
        return NullJournal()
    if directory is None:
        return InMemoryJournal()
    return JsonlJournal(directory / JOURNAL_FILENAME)
