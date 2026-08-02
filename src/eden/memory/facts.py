"""Fact extraction from conversation.

A raw chat turn lives in :class:`~eden.memory.conversation.ConversationMemory`
and is subject to two kinds of forgetting: a turn limit, and a token budget
applied when building a prompt window. That is correct behaviour for a
transcript — but "the user's name" is not a transcript entry, it is a fact, and
a fact that disappears once a conversation gets long enough is not durable
memory at all.

:class:`FactExtractor` is the seam that promotes a handful of high-confidence
statements out of the transcript and into
:class:`~eden.memory.conversation.ProjectMemory`, where retention is
"forever, until told otherwise" rather than "until the token budget runs out".

The shipped extractor is deliberately a small set of regular expressions, not a
model call. Every other default in EDEN that could depend on network access
degrades to something dependency-free — the hash embedder is the other
example — and fact extraction runs on every single user turn, so it needs to be
cheap, offline, and its behaviour needs to be exactly reproducible in a test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_MAX_FACT_LENGTH = 200


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """One fact recognised in a piece of text.

    Attributes:
        key: Stable identifier, e.g. ``"name"``. Reusing a key overwrites the
            previous value rather than accumulating duplicates.
        value: The extracted content, already trimmed for storage.
        confidence: How sure the extractor is, in ``[0, 1]``. Reserved for
            future extractors that are not simple pattern matches; the
            heuristic extractor only ever emits ``1.0``.
    """

    key: str
    value: str
    confidence: float = 1.0


@runtime_checkable
class FactExtractor(Protocol):
    """Recognises durable facts inside free text."""

    def extract(self, text: str) -> list[ExtractedFact]:
        """Return every fact recognised in ``text``."""
        ...


# Ordered so a more specific phrase is tried before a more general one that
# could otherwise swallow it, e.g. "I live in London" before a hypothetical
# bare "in London" pattern. Each pattern captures exactly the value.
_NAME_PATTERNS = (
    re.compile(r"\bmy name is\s+([A-Za-z][\w' -]{0,60})", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Za-z][\w' -]{0,60})", re.IGNORECASE),
    re.compile(r"\bi'?m called\s+([A-Za-z][\w' -]{0,60})", re.IGNORECASE),
    re.compile(r"\bthis is\s+([A-Za-z][\w' -]{0,60})\s+speaking\b", re.IGNORECASE),
)
_LOCATION_PATTERNS = (
    re.compile(r"\bi live in\s+([A-Za-z][\w' ,-]{0,60})", re.IGNORECASE),
    re.compile(r"\bi'?m based in\s+([A-Za-z][\w' ,-]{0,60})", re.IGNORECASE),
    re.compile(r"\bi'?m from\s+([A-Za-z][\w' ,-]{0,60})", re.IGNORECASE),
)
_OCCUPATION_PATTERNS = (
    re.compile(r"\bi work as\s+(?:an?\s+)?([A-Za-z][\w' -]{0,60})", re.IGNORECASE),
    re.compile(r"\bi'?m an?\s+([A-Za-z][\w' -]{0,60})\s+by trade\b", re.IGNORECASE),
    re.compile(r"\bi work at\s+([A-Za-z][\w' &.,-]{0,60})", re.IGNORECASE),
)

# A trailing clause a sentence might continue with, which must not be pulled
# into the captured value: "my name is Sam and I live in Perth" should yield
# the name "Sam", not "Sam and I live in Perth". Punctuation may sit directly
# against the captured word ("Perth, by the way"), so it needs no preceding
# space; a word-level connector does need one, or it would cut inside a word.
_TRAILING_CLAUSE = re.compile(
    r"(?:\s*[,.!?].*|\s+(?:and|but|because|since|though|although)\b.*)$",
    re.IGNORECASE,
)


def _trim(raw: str) -> str:
    """Cut a captured span at its first trailing clause and tidy whitespace."""
    cleaned = _TRAILING_CLAUSE.sub("", raw).strip().strip(".,!? ")
    return cleaned[:_MAX_FACT_LENGTH]


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    """Return the trimmed capture of the first pattern that matches."""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            trimmed = _trim(match.group(1))
            if trimmed:
                return trimmed
    return None


class HeuristicFactExtractor:
    """Recognises a small set of common self-descriptions.

    Deliberately narrow: false positives write a wrong fact into durable
    memory, where it will keep being asserted back to the person until
    corrected. A missed fact costs nothing; a wrong one costs trust.

    Example:
        >>> HeuristicFactExtractor().extract("Hi, my name is Sudharshan.")
        [ExtractedFact(key='name', value='Sudharshan', confidence=1.0)]
    """

    def extract(self, text: str) -> list[ExtractedFact]:
        """Return every fact recognised in ``text``."""
        facts: list[ExtractedFact] = []
        name = _first_match(_NAME_PATTERNS, text)
        if name:
            facts.append(ExtractedFact(key="name", value=name))
        location = _first_match(_LOCATION_PATTERNS, text)
        if location:
            facts.append(ExtractedFact(key="location", value=location))
        occupation = _first_match(_OCCUPATION_PATTERNS, text)
        if occupation:
            facts.append(ExtractedFact(key="occupation", value=occupation))
        return facts


class NullFactExtractor:
    """Recognises nothing. Used when automatic extraction is switched off."""

    def extract(self, text: str) -> list[ExtractedFact]:
        """Return no facts."""
        del text
        return []
