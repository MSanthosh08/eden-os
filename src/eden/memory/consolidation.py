"""Memory consolidation.

A long-running agent accumulates conversation turns without bound. Consolidation
is the answer: once a conversation grows past a threshold, its oldest turns are
summarised into a single long-term record and the originals are dropped.

The summariser is injected, and there are two implementations for the same
reason there are two embedders — an EDEN with no reachable provider must still
be able to forget gracefully. :class:`GatewaySummariser` produces a real
abstractive summary and falls back to :class:`ExtractiveSummariser`, which needs
nothing but the text itself.

Consolidation is lossy by design. That is the point: the alternative to losing
detail is losing the ability to hold a conversation at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from eden.config.enums import MemoryKind, Role
from eden.config.schema import MemoryConfig
from eden.core.types import ChatRequest, Message
from eden.errors import EdenError
from eden.gateway.client import GatewayClient
from eden.logging import get_logger, timed_block
from eden.memory.base import MemoryStore
from eden.memory.conversation import ConversationMemory
from eden.memory.types import MemoryRecord

_LOGGER = get_logger("memory.consolidation")

CONSOLIDATED_TAG = "consolidated"
_SUMMARY_SYSTEM_PROMPT = (
    "You compress conversation history for an AI system's long-term memory. "
    "Preserve decisions, commitments, named entities, constraints and unresolved "
    "questions. Discard pleasantries and repetition. Write plain prose, no preamble."
)
_WORDS_PER_TOKEN = 0.75
_EXTRACTIVE_SENTENCE_CHARS = 160


@runtime_checkable
class Summariser(Protocol):
    """Compresses several texts into one."""

    async def summarise(self, texts: Sequence[str], *, target_words: int) -> str:
        """Return a summary of ``texts`` no longer than roughly ``target_words``."""
        ...


class ExtractiveSummariser:
    """Selects text rather than generating it.

    Takes the opening of each turn and truncates to the word budget. Crude, but
    it never hallucinates, never fails, and needs no model — which is exactly
    what a fallback should be.
    """

    async def summarise(self, texts: Sequence[str], *, target_words: int) -> str:
        """Return the leading fragment of each text, trimmed to the budget."""
        fragments: list[str] = []
        for text in texts:
            cleaned = " ".join(text.split())
            if not cleaned:
                continue
            fragments.append(cleaned[:_EXTRACTIVE_SENTENCE_CHARS])
        joined = " | ".join(fragments)
        words = joined.split()
        if len(words) <= target_words:
            return joined
        return " ".join(words[:target_words]) + " …"


class GatewaySummariser:
    """Generates a summary through the AI Gateway, degrading on failure."""

    def __init__(
        self,
        gateway: GatewayClient,
        *,
        model: str = "",
        fallback: Summariser | None = None,
    ) -> None:
        """Initialise the summariser.

        Args:
            gateway: Gateway façade used to reach a model.
            model: Explicit model name. Empty lets the router choose.
            fallback: Used when the gateway cannot serve the request.
        """
        self._gateway = gateway
        self._model = model
        self._fallback: Summariser = fallback or ExtractiveSummariser()

    async def summarise(self, texts: Sequence[str], *, target_words: int) -> str:
        """Return a generated summary, falling back to extraction on failure."""
        if not texts:
            return ""
        body = "\n".join(f"- {' '.join(text.split())}" for text in texts)
        request = ChatRequest(
            messages=[
                Message.system(_SUMMARY_SYSTEM_PROMPT),
                Message.user(f"Summarise in at most {target_words} words:\n{body}"),
            ],
            model=self._model,
            temperature=0.2,
            max_tokens=max(64, int(target_words / _WORDS_PER_TOKEN)),
        )
        try:
            response = await self._gateway.chat(request)
        except EdenError as exc:
            _LOGGER.warning(
                "Summarisation provider unavailable; using extractive fallback.",
                extra={"error_code": exc.code},
            )
            return await self._fallback.summarise(texts, target_words=target_words)
        if not response.content.strip():
            return await self._fallback.summarise(texts, target_words=target_words)
        return response.content.strip()


class Consolidator:
    """Compresses old conversation turns into durable long-term memory."""

    def __init__(
        self,
        config: MemoryConfig,
        conversation: ConversationMemory,
        long_term: MemoryStore,
        summariser: Summariser,
    ) -> None:
        """Initialise the consolidator.

        Args:
            config: Retention policy supplying the thresholds.
            conversation: Source of turns to compress.
            long_term: Destination for the resulting summary.
            summariser: Injected compression strategy.
        """
        self._config = config
        self._conversation = conversation
        self._long_term = long_term
        self._summariser = summariser

    async def due(self, namespace: str) -> bool:
        """Return whether ``namespace`` has grown past the threshold."""
        return await self._conversation.count(namespace) > self._config.consolidate_after_turns

    async def consolidate(self, namespace: str, *, force: bool = False) -> MemoryRecord | None:
        """Summarise the oldest turns and drop them.

        System turns are never consolidated. They define how the agent behaves,
        and a summary of an instruction is not an instruction.

        Args:
            namespace: Conversation to compress.
            force: Compress even if the threshold has not been reached.

        Returns:
            The stored summary record, or ``None`` when there was nothing to do.
        """
        if not force and not await self.due(namespace):
            return None

        turns = await self._conversation.history(namespace)
        keep = self._config.consolidation_keep_turns
        candidates = [
            (index, message)
            for index, message in enumerate(turns)
            if message.role is not Role.SYSTEM
        ]
        compressible = candidates[: max(0, len(candidates) - keep)]
        if not compressible:
            return None

        with timed_block(
            _LOGGER,
            "memory.consolidate",
            namespace=namespace,
            turns=len(compressible),
        ):
            summary = await self._summariser.summarise(
                [f"{message.role.value}: {message.content}" for _, message in compressible],
                target_words=self._config.consolidation_summary_words,
            )
            if not summary.strip():
                return None

            record = await self._long_term.add(
                MemoryRecord(
                    content=summary,
                    kind=MemoryKind.LONG_TERM,
                    namespace=namespace,
                    importance=0.7,
                    tags=frozenset({CONSOLIDATED_TAG}),
                    metadata={
                        "source": "conversation",
                        "turns_compressed": len(compressible),
                    },
                )
            )
            await self._prune(namespace, len(compressible))

        _LOGGER.info(
            "Conversation consolidated.",
            extra={
                "namespace": namespace,
                "compressed": len(compressible),
                "record": record.id,
            },
        )
        return record

    async def _prune(self, namespace: str, count: int) -> None:
        """Delete the oldest ``count`` non-system turns."""
        removed = 0
        for record in await self._conversation.search_records(namespace):
            if removed >= count:
                break
            if record.metadata.get("role") == Role.SYSTEM.value:
                continue
            if await self._conversation.delete(record.id, namespace=namespace):
                removed += 1
