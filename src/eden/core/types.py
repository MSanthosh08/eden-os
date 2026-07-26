"""Provider-neutral domain types.

These value objects are the lingua franca of EDEN. Business logic — agents,
memory, automation — speaks only this vocabulary; each provider adapter
translates between it and a vendor's wire format. That translation boundary is
the reason a vendor can be swapped without touching anything above the gateway.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from eden.config.enums import Capability, FinishReason, PrivacyTier, Role
from eden.errors import ValidationError

_MIN_TEMPERATURE = 0.0
_MAX_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a conversation.

    Attributes:
        role: Who authored the turn.
        content: Text content of the turn.
        name: Optional author label, used for tool results and named personas.
        metadata: Non-transmitted annotations, e.g. memory provenance.
    """

    role: Role
    content: str
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the turn.

        Raises:
            ValidationError: If a non-tool message carries empty content.
        """
        if self.role is not Role.TOOL and not self.content.strip():
            raise ValidationError(
                "Message content must not be empty.",
                context={"role": str(self.role)},
            )

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one generation.

    Attributes:
        prompt_tokens: Tokens consumed by the request.
        completion_tokens: Tokens produced in the response.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return the sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A provider-neutral generation request.

    Attributes:
        messages: Conversation history, oldest first.
        model: Explicit model name. Empty means "let the provider decide".
        temperature: Sampling temperature.
        max_tokens: Output ceiling. ``None`` defers to the provider default.
        stream: Whether the caller intends to consume a stream.
        required_capabilities: Features a candidate provider must advertise.
        minimum_privacy_tier: Data-residency floor for this specific request.
        preferred_provider: Logical provider name to try first, if healthy.
        stop: Stop sequences.
        metadata: Caller annotations echoed into logs and responses.
    """

    messages: Sequence[Message]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    required_capabilities: frozenset[Capability] = frozenset({Capability.CHAT})
    minimum_privacy_tier: PrivacyTier | None = None
    preferred_provider: str = ""
    stop: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the request.

        Raises:
            ValidationError: If the request is structurally unusable.
        """
        if not self.messages:
            raise ValidationError("A chat request requires at least one message.")
        if not _MIN_TEMPERATURE <= self.temperature <= _MAX_TEMPERATURE:
            raise ValidationError(
                "Temperature is out of range.",
                context={"temperature": self.temperature},
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValidationError(
                "max_tokens must be positive when supplied.",
                context={"max_tokens": self.max_tokens},
            )
        if self.stream and Capability.STREAMING not in self.required_capabilities:
            object.__setattr__(
                self,
                "required_capabilities",
                self.required_capabilities | {Capability.STREAMING},
            )

    @property
    def estimated_prompt_tokens(self) -> int:
        """Return a cheap prompt-size estimate used for cost scoring.

        Uses the conventional four-characters-per-token heuristic. It is
        deliberately provider-agnostic; exact accounting comes back in
        :class:`Usage` once the call completes.
        """
        characters = sum(len(message.content) for message in self.messages)
        return max(1, characters // 4)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A provider-neutral generation result.

    Attributes:
        content: Generated text.
        model: Model that actually served the request.
        provider: Logical provider name that served the request.
        usage: Token accounting.
        finish_reason: Why generation stopped.
        latency_ms: Measured wall-clock duration of the provider call.
        cost: Estimated spend in the provider's configured currency units.
        attempts: Number of providers tried, including the successful one.
        metadata: Non-sensitive provider annotations.
    """

    content: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    latency_ms: float = 0.0
    cost: float = 0.0
    attempts: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """A provider-neutral request to embed one or more texts.

    Attributes:
        texts: Inputs to embed, in order.
        model: Explicit embedding model. Empty defers to the provider default.
        dimensions: Requested vector width, where the provider supports it.
        metadata: Caller annotations echoed into logs.
    """

    texts: Sequence[str]
    model: str = ""
    dimensions: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the request.

        Raises:
            ValidationError: If the batch is empty, contains blank input, or
                requests a non-positive width.
        """
        if not self.texts:
            raise ValidationError("An embedding request requires at least one text.")
        if any(not text.strip() for text in self.texts):
            raise ValidationError("Embedding inputs must not be blank.")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValidationError(
                "dimensions must be positive when supplied.",
                context={"dimensions": self.dimensions},
            )

    @property
    def estimated_tokens(self) -> int:
        """Return a cheap token estimate for the whole batch."""
        return max(1, sum(len(text) for text in self.texts) // 4)


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """A provider-neutral embedding result.

    Attributes:
        vectors: One vector per input text, in the same order.
        model: Model that served the request.
        provider: Logical provider name that served the request.
        usage: Token accounting.
        latency_ms: Measured wall-clock duration.
        cost: Estimated spend.
    """

    vectors: tuple[tuple[float, ...], ...]
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cost: float = 0.0

    @property
    def dimensions(self) -> int:
        """Return the width of the returned vectors, or zero when empty."""
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One incremental fragment of a streamed generation.

    Attributes:
        delta: Text produced since the previous chunk.
        provider: Logical provider name serving the stream.
        model: Model serving the stream.
        finish_reason: Set on the final chunk only.
        index: Zero-based position in the stream.
    """

    delta: str
    provider: str
    model: str
    finish_reason: FinishReason | None = None
    index: int = 0

    @property
    def is_final(self) -> bool:
        """Return whether this chunk terminates the stream."""
        return self.finish_reason is not None
