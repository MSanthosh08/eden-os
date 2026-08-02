"""Memory subsystem.

Five retention policies over one record shape:

===============  ==========================  ==============================
Kind             Persistence                 Forgetting
===============  ==========================  ==============================
short_term       volatile, in process        capacity + time-to-live
long_term        durable, repository         explicit deletion only
vector           durable + embeddings        explicit deletion only
conversation     volatile or durable         turn limit + token windowing
project          durable, repository         explicit deletion only
===============  ==========================  ==============================

Callers normally use :class:`~eden.memory.manager.MemoryManager`, which fans a
single query across every store and merges the results.
"""

from __future__ import annotations

from eden.memory.base import BaseMemoryStore, MemoryStore
from eden.memory.consolidation import (
    CONSOLIDATED_TAG,
    Consolidator,
    ExtractiveSummariser,
    GatewaySummariser,
    Summariser,
)
from eden.memory.conversation import ConversationMemory, ProjectMemory
from eden.memory.facts import (
    ExtractedFact,
    FactExtractor,
    HeuristicFactExtractor,
    NullFactExtractor,
)
from eden.memory.manager import MemoryManager, build_memory_manager
from eden.memory.repository import (
    InMemoryRecordRepository,
    JsonlRecordRepository,
    RecordRepository,
)
from eden.memory.stores import LongTermMemory, ShortTermMemory
from eden.memory.types import (
    DEFAULT_NAMESPACE,
    MemoryQuery,
    MemoryRecord,
    SearchHit,
    cosine_similarity,
    keyword_score,
)
from eden.memory.vector import (
    BruteForceIndex,
    Embedder,
    GatewayEmbedder,
    HashEmbedder,
    VectorIndex,
    VectorMemory,
)

__all__ = [
    "CONSOLIDATED_TAG",
    "DEFAULT_NAMESPACE",
    "BaseMemoryStore",
    "BruteForceIndex",
    "Consolidator",
    "ConversationMemory",
    "Embedder",
    "ExtractedFact",
    "ExtractiveSummariser",
    "FactExtractor",
    "GatewayEmbedder",
    "GatewaySummariser",
    "HashEmbedder",
    "HeuristicFactExtractor",
    "InMemoryRecordRepository",
    "JsonlRecordRepository",
    "LongTermMemory",
    "MemoryManager",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryStore",
    "NullFactExtractor",
    "ProjectMemory",
    "RecordRepository",
    "SearchHit",
    "ShortTermMemory",
    "Summariser",
    "VectorIndex",
    "VectorMemory",
    "build_memory_manager",
    "cosine_similarity",
    "keyword_score",
]
