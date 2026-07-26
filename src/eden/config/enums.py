"""Enumerations shared across every EDEN subsystem.

These types are the single source of truth for values that would otherwise
become magic strings scattered through the codebase. They are declared here —
in the lowest-level package — so that any layer may depend on them without
creating an import cycle.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum, unique


@unique
class Environment(StrEnum):
    """Deployment environment EDEN is running in."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@unique
class LogLevel(StrEnum):
    """Severity thresholds accepted by the logging subsystem."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@unique
class LogFormat(StrEnum):
    """Rendering style for log records."""

    CONSOLE = "console"
    JSON = "json"


@unique
class ProviderKind(StrEnum):
    """Wire protocol family a provider speaks.

    Note that many vendors share the OpenAI chat-completions contract; they all
    map to :attr:`OPENAI_COMPATIBLE` and are differentiated purely by
    configuration (base URL, model catalogue, pricing).
    """

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MOCK = "mock"
    CUSTOM = "custom"


@unique
class Capability(StrEnum):
    """Discrete features a model may support.

    Requests declare the capabilities they require; the router only considers
    providers whose advertised capability set is a superset of that demand.
    """

    CHAT = "chat"
    STREAMING = "streaming"
    TOOL_USE = "tool_use"
    VISION = "vision"
    JSON_MODE = "json_mode"
    LONG_CONTEXT = "long_context"
    CODE = "code"
    REASONING = "reasoning"
    EMBEDDING = "embedding"


@unique
class PrivacyTier(IntEnum):
    """Data-residency guarantee offered by a provider.

    Ordered from least to most private so that a request can express a floor
    (``minimum_privacy_tier``) with a simple comparison.
    """

    PUBLIC_CLOUD = 0
    PRIVATE_CLOUD = 1
    ON_PREMISE = 2
    LOCAL_ONLY = 3


@unique
class RoutingStrategyName(StrEnum):
    """Named routing strategies selectable from configuration."""

    BALANCED = "balanced"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    PRIVACY_FIRST = "privacy_first"
    ROUND_ROBIN = "round_robin"


@unique
class HealthState(StrEnum):
    """Observed health of a provider."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@unique
class CircuitState(StrEnum):
    """State of a provider circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@unique
class MemoryKind(StrEnum):
    """Retention policy of a stored memory.

    The five kinds share one record shape and one store contract; they differ
    in *where* they persist and *how* they forget, not in what they hold.
    """

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    VECTOR = "vector"
    CONVERSATION = "conversation"
    PROJECT = "project"


@unique
class ActionKind(StrEnum):
    """The class of effect an action will have on the world."""

    NOOP = "noop"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SHELL_COMMAND = "shell_command"
    DEVICE_COMMAND = "device_command"
    CUSTOM = "custom"


@unique
class RiskLevel(IntEnum):
    """Severity of an action, ordered so that policy can compare thresholds."""

    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@unique
class PermissionMode(StrEnum):
    """How an action of a given risk is authorised."""

    AUTOMATIC = "automatic"
    CONFIRM = "confirm"
    DENY = "deny"


@unique
class ExecutionStatus(StrEnum):
    """Where an action currently sits in the pipeline.

    The values form the state machine: nothing reaches ``SUCCEEDED`` without
    passing through ``VERIFIED`` and ``PERMITTED`` first.
    """

    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    PERMITTED = "permitted"
    DENIED = "denied"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"
    SKIPPED = "skipped"


@unique
class TaskStatus(StrEnum):
    """Where a task sits in an agent's lifecycle."""

    PENDING = "pending"
    REJECTED = "rejected"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@unique
class StepStatus(StrEnum):
    """Outcome of one step within a plan."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@unique
class DeviceKind(StrEnum):
    """Transport family a device driver speaks."""

    SIMULATED = "simulated"
    HTTP = "http"
    CUSTOM = "custom"


@unique
class DeviceState(StrEnum):
    """Connection state of one device."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    BUSY = "busy"
    FAULTED = "faulted"


@unique
class TriggerKind(StrEnum):
    """What causes an automation rule to fire."""

    INTERVAL = "interval"
    DAILY = "daily"
    EVENT = "event"
    MANUAL = "manual"


@unique
class AutomationStatus(StrEnum):
    """Outcome of one automation run."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@unique
class Role(StrEnum):
    """Author of a conversation message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@unique
class FinishReason(StrEnum):
    """Why a generation stopped."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
