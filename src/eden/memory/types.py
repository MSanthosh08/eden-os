"""Memory domain types.

The central claim of this subsystem is that "short-term", "long-term",
"vector", "conversation" and "project" memory are five *retention policies*
over one data model — not five data models. So there is exactly one record
type, one query type and one result type, and the stores differ only in where
they persist and how they forget.

That is what makes cross-store recall possible: :class:`MemoryManager` can fan
a single query across every store and merge the results, because every store
speaks in :class:`SearchHit`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from eden.config.enums import MemoryKind, Role
from eden.errors import ValidationError
from eden.utils.ids import new_id

DEFAULT_NAMESPACE = "default"
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A single unit of remembered information.

    Attributes:
        id: Stable identifier, generated when omitted.
        content: The remembered text.
        kind: Which retention policy owns this record.
        namespace: Isolation boundary — a project id, conversation id or agent
            id. Records never leak across namespaces.
        created_at: UTC creation time, used for recency and expiry.
        importance: Operator or agent salience in ``[0, 1]``. Drives eviction:
            a full short-term buffer discards the least important record, not
            merely the oldest.
        tags: Free-form labels usable as a hard filter.
        metadata: Structured annotations, e.g. the role of a conversation turn.
        embedding: Vector representation, present only for vector memory.
    """

    content: str
    kind: MemoryKind = MemoryKind.LONG_TERM
    namespace: str = DEFAULT_NAMESPACE
    id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    importance: float = 0.5
    tags: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and fill in defaults.

        Raises:
            ValidationError: If the record is structurally unusable.
        """
        if not self.content.strip():
            raise ValidationError("Memory content must not be empty.")
        if not 0.0 <= self.importance <= 1.0:
            raise ValidationError(
                "Importance must be between 0 and 1.",
                context={"importance": self.importance},
            )
        if not self.namespace.strip():
            raise ValidationError("Memory namespace must not be empty.")
        if not self.id:
            object.__setattr__(self, "id", new_id("mem"))
        if self.created_at.tzinfo is None:
            object.__setattr__(self, "created_at", self.created_at.replace(tzinfo=UTC))

    @property
    def estimated_tokens(self) -> int:
        """Return a cheap token estimate for budget arithmetic."""
        return max(1, len(self.content) // _CHARS_PER_TOKEN)

    def age_seconds(self, now: datetime) -> float:
        """Return this record's age in seconds relative to ``now``."""
        return max(0.0, (now - self.created_at).total_seconds())

    def with_embedding(self, vector: Sequence[float]) -> MemoryRecord:
        """Return a copy carrying ``vector``."""
        return replace(self, embedding=tuple(float(value) for value in vector))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for durable storage."""
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind.value,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
            "tags": sorted(self.tags),
            "metadata": dict(self.metadata),
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryRecord:
        """Reconstruct a record from its stored representation.

        Args:
            payload: A mapping previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed record.

        Raises:
            ValidationError: If the payload is missing or malformed.
        """
        try:
            raw_embedding = payload.get("embedding")
            return cls(
                id=str(payload["id"]),
                content=str(payload["content"]),
                kind=MemoryKind(str(payload["kind"])),
                namespace=str(payload.get("namespace") or DEFAULT_NAMESPACE),
                created_at=datetime.fromisoformat(str(payload["created_at"])),
                importance=float(payload.get("importance", 0.5)),
                tags=frozenset(str(tag) for tag in payload.get("tags") or ()),
                metadata=dict(payload.get("metadata") or {}),
                embedding=(
                    tuple(float(value) for value in raw_embedding)
                    if isinstance(raw_embedding, list)
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                "Stored memory record is malformed.",
                context={"id": str(payload.get("id", "<unknown>"))},
                cause=exc,
            ) from exc

    @classmethod
    def from_turn(
        cls,
        role: Role,
        content: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        importance: float = 0.5,
    ) -> MemoryRecord:
        """Build a conversation record from one turn."""
        return cls(
            content=content,
            kind=MemoryKind.CONVERSATION,
            namespace=namespace,
            importance=importance,
            metadata={"role": role.value},
        )


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """A recall request.

    A query with empty ``text`` is a valid browse: stores return the most
    recent matching records rather than nothing.

    Attributes:
        text: Free text to match. Empty means "no relevance constraint".
        namespace: Isolation boundary to search within.
        kinds: Restrict to particular retention policies. Empty means all.
        tags: Every listed tag must be present on a record to match.
        limit: Maximum hits returned per store.
        min_score: Relevance floor below which hits are discarded.
        since: Ignore records created before this instant.
    """

    text: str = ""
    namespace: str = DEFAULT_NAMESPACE
    kinds: frozenset[MemoryKind] = frozenset()
    tags: frozenset[str] = frozenset()
    limit: int = 10
    min_score: float = 0.0
    since: datetime | None = None

    def __post_init__(self) -> None:
        """Validate the query.

        Raises:
            ValidationError: If a bound is invalid.
        """
        if self.limit <= 0:
            raise ValidationError("Query limit must be positive.", context={"limit": self.limit})
        if not self.namespace.strip():
            raise ValidationError("Query namespace must not be empty.")

    @property
    def tokens(self) -> frozenset[str]:
        """Return the lower-cased word set of :attr:`text`."""
        return frozenset(self.text.lower().split())

    def matches_kind(self, kind: MemoryKind) -> bool:
        """Return whether ``kind`` is in scope for this query."""
        return not self.kinds or kind in self.kinds


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One record returned by a search, with its relevance score.

    Attributes:
        record: The matched record.
        score: Relevance in ``[0, 1]``. Comparable only within one result set.
        store: Name of the store that produced the hit, for explainability.
    """

    record: MemoryRecord
    score: float
    store: str = ""

    def __post_init__(self) -> None:
        """Clamp the score into range rather than rejecting it.

        Scorers are pluggable, and a slightly out-of-range score is not worth
        failing a recall over.
        """
        object.__setattr__(self, "score", max(0.0, min(1.0, self.score)))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors.

    Args:
        left: First vector.
        right: Second vector. Must be the same length as ``left``.

    Returns:
        Similarity in ``[-1, 1]``. Zero when either vector has no magnitude.

    Raises:
        ValidationError: If the vectors have different lengths.

    Example:
        >>> round(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 6)
        1.0
    """
    if len(left) != len(right):
        raise ValidationError(
            "Cannot compare vectors of different widths.",
            context={"left": len(left), "right": len(right)},
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def keyword_score(query_tokens: frozenset[str], content: str) -> float:
    """Return a lexical overlap score in ``[0, 1]``.

    This is deliberately simple. Vector memory exists for semantic recall; the
    non-vector stores only need a cheap, dependency-free relevance signal that
    behaves predictably.

    Args:
        query_tokens: Lower-cased query words.
        content: Record content to score.

    Returns:
        The fraction of query tokens present in ``content``.

    Example:
        >>> keyword_score(frozenset({"deploy", "rollback"}), "deploy the service")
        0.5
    """
    if not query_tokens:
        return 1.0
    content_tokens = frozenset(content.lower().split())
    if not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)
