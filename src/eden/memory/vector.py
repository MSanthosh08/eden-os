"""Vector memory.

Semantic recall needs an embedder, and an embedder is exactly the kind of thing
that should not be hardwired. :class:`Embedder` is the contract;
:class:`GatewayEmbedder` routes through the AI Gateway (so embeddings inherit
failover, health tracking and privacy filtering), and :class:`HashEmbedder`
provides a deterministic offline fallback so vector memory works with no
credentials and no network.

Similarity search is brute-force cosine over the namespace. That is the correct
choice at this scale: it is exact, has no index to corrupt, and adds no
dependency. :class:`VectorMemory` takes its search through a
:class:`VectorIndex`, so swapping in FAISS or a hosted vector database later is
an injection rather than a rewrite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from eden.config.enums import MemoryKind
from eden.config.schema import MemoryConfig
from eden.core.types import EmbeddingRequest
from eden.errors import EdenError
from eden.gateway.client import GatewayClient
from eden.gateway.providers.mock import hash_vector
from eden.logging import get_logger
from eden.memory.base import BaseMemoryStore
from eden.memory.repository import RecordRepository
from eden.memory.types import MemoryQuery, MemoryRecord, cosine_similarity

_LOGGER = get_logger("memory.vector")


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def dimensions(self) -> int:
        """Return the width of vectors this embedder produces."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return one vector per input text, in order."""
        ...


class HashEmbedder:
    """A deterministic, dependency-free embedder.

    Not a placeholder: it is the default when no provider advertises
    embeddings, and it makes retrieval testable and reproducible. Documents
    sharing vocabulary land close together under cosine similarity, which is
    sufficient for lexical-semantic recall even though it captures no meaning
    beyond token identity.
    """

    def __init__(self, dimensions: int = 256) -> None:
        """Initialise the embedder.

        Args:
            dimensions: Vector width. Must be positive.

        Raises:
            ValueError: If ``dimensions`` is not positive.
        """
        if dimensions <= 0:
            message = "dimensions must be positive."
            raise ValueError(message)
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Return the width of vectors this embedder produces."""
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return one deterministic unit vector per input text."""
        return [hash_vector(text, self._dimensions) for text in texts]


class GatewayEmbedder:
    """Embeds through the AI Gateway, with a fallback embedder on failure.

    Falling back rather than raising is deliberate: losing the ability to
    *store* a memory because a vendor is down is a much worse outcome than
    storing it with a weaker vector. The fallback is logged, and the record is
    still retrievable.
    """

    def __init__(
        self,
        gateway: GatewayClient,
        config: MemoryConfig,
        *,
        fallback: Embedder | None = None,
    ) -> None:
        """Initialise the embedder.

        Args:
            gateway: Gateway façade used to reach an embedding provider.
            config: Memory policy supplying the model and fallback width.
            fallback: Embedder used when the gateway cannot serve the batch.
        """
        self._gateway = gateway
        self._config = config
        self._fallback: Embedder = fallback or HashEmbedder(config.vector_dimensions)
        self._dimensions = self._fallback.dimensions

    @property
    def dimensions(self) -> int:
        """Return the width of the most recently produced vectors."""
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return one vector per input text, degrading to the fallback."""
        try:
            response = await self._gateway.embed(
                EmbeddingRequest(texts=list(texts), model=self._config.vector_model)
            )
        except EdenError as exc:
            _LOGGER.warning(
                "Embedding provider unavailable; using the offline embedder.",
                extra={"error_code": exc.code},
            )
            return await self._fallback.embed(texts)
        if len(response.vectors) != len(texts):
            _LOGGER.warning(
                "Embedding provider returned the wrong number of vectors.",
                extra={"expected": len(texts), "received": len(response.vectors)},
            )
            return await self._fallback.embed(texts)
        self._dimensions = response.dimensions or self._dimensions
        return list(response.vectors)


@runtime_checkable
class VectorIndex(Protocol):
    """Ranks stored vectors against a query vector."""

    def rank(
        self,
        query_vector: Sequence[float],
        records: Sequence[MemoryRecord],
    ) -> dict[str, float]:
        """Return a similarity score per record id."""
        ...


class BruteForceIndex:
    """Exact cosine similarity over every candidate.

    Records whose embedding is missing or of a different width score zero
    rather than raising: a fallback embedder changing dimensions must not make
    old memories unretrievable, only less competitive.
    """

    def rank(
        self,
        query_vector: Sequence[float],
        records: Sequence[MemoryRecord],
    ) -> dict[str, float]:
        """Return a similarity score in ``[0, 1]`` per record id."""
        scores: dict[str, float] = {}
        for record in records:
            vector = record.embedding
            if vector is None or len(vector) != len(query_vector):
                scores[record.id] = 0.0
                continue
            # Map cosine from [-1, 1] onto [0, 1] so it composes with the
            # recency and importance blending applied by the base store.
            scores[record.id] = (cosine_similarity(query_vector, vector) + 1.0) / 2.0
        return scores


class VectorMemory(BaseMemoryStore):
    """Semantic memory backed by embeddings and cosine similarity.

    Example:
        >>> import asyncio
        >>> from eden.config.schema import MemoryConfig
        >>> from eden.memory.repository import InMemoryRecordRepository
        >>> from eden.memory.types import MemoryQuery, MemoryRecord
        >>> async def demo() -> str:
        ...     store = VectorMemory(
        ...         MemoryConfig(), InMemoryRecordRepository(), HashEmbedder(32)
        ...     )
        ...     await store.add(MemoryRecord(content="deploy the payment service"))
        ...     await store.add(MemoryRecord(content="feed the office cat"))
        ...     hits = await store.search(MemoryQuery(text="deploy payment", limit=1))
        ...     return hits[0].record.content
        >>> asyncio.run(demo())
        'deploy the payment service'
    """

    def __init__(
        self,
        config: MemoryConfig,
        repository: RecordRepository,
        embedder: Embedder,
        *,
        index: VectorIndex | None = None,
        name: str = "vector",
    ) -> None:
        """Initialise the store.

        Args:
            config: Retention policy supplying the similarity floor.
            repository: Injected persistence backend.
            embedder: Injected text-to-vector strategy.
            index: Injected similarity search. Defaults to exact brute force.
            name: Logical store name.
        """
        super().__init__(MemoryKind.VECTOR, name=name)
        self._config = config
        self._repository = repository
        self._embedder = embedder
        self._index = index or BruteForceIndex()

    @property
    def embedder(self) -> Embedder:
        """Return the embedder in use."""
        return self._embedder

    async def _prepare_many(self, records: Sequence[MemoryRecord]) -> list[MemoryRecord]:
        """Attach embeddings, batching the call across the whole input."""
        stamped = await super()._prepare_many(records)
        missing = [record for record in stamped if record.embedding is None]
        if not missing:
            return stamped
        vectors = await self._embedder.embed([record.content for record in missing])
        embedded = {
            record.id: record.with_embedding(vector)
            for record, vector in zip(missing, vectors, strict=True)
        }
        return [embedded.get(record.id, record) for record in stamped]

    async def _relevance(
        self,
        query: MemoryQuery,
        candidates: Sequence[MemoryRecord],
    ) -> dict[str, float]:
        """Return cosine similarity per record id.

        An empty query text has no direction to compare against, so relevance
        is uniform and the base store's recency blending decides the ordering.
        """
        if not query.text.strip() or not candidates:
            return {record.id: 1.0 for record in candidates}
        query_vector = (await self._embedder.embed([query.text]))[0]
        scores = self._index.rank(query_vector, candidates)
        floor = (self._config.vector_min_similarity + 1.0) / 2.0
        return {record_id: score if score >= floor else 0.0 for record_id, score in scores.items()}

    async def _persist(self, record: MemoryRecord) -> None:
        """Write the record to the repository."""
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
