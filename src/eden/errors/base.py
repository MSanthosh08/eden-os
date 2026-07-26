"""Root of the EDEN exception hierarchy.

Every failure inside EDEN is expressed as an :class:`EdenError` subclass so that
callers can react to *categories* of failure (configuration, transport, policy)
without importing the module that raised them. Each error carries a stable
machine-readable ``code``, a redaction-safe ``context`` mapping and a
``retryable`` hint used by the retry and failover machinery.
"""

from __future__ import annotations

from typing import Any


class EdenError(Exception):
    """Base class for every error raised by EDEN.

    Attributes:
        code: Stable, machine-readable identifier used by logs and APIs.
        retryable: Whether retrying the same operation may succeed.
    """

    code: str = "eden.error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description. Must never contain secrets.
            context: Structured, non-sensitive detail attached to logs.
            cause: Underlying exception, preserved for tracebacks.
        """
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the error."""
        payload: dict[str, Any] = {
            "code": self.code,
            "type": type(self).__name__,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.context:
            payload["context"] = self.context
        if self.cause is not None:
            payload["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return payload

    def __repr__(self) -> str:
        """Return an unambiguous representation for debugging."""
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigurationError(EdenError):
    """Raised when configuration cannot be loaded, parsed or validated."""

    code = "eden.config.error"


class MissingConfigError(ConfigurationError):
    """Raised when a required configuration key is absent."""

    code = "eden.config.missing"


class InvalidConfigError(ConfigurationError):
    """Raised when a configuration value fails validation."""

    code = "eden.config.invalid"


class SecretResolutionError(ConfigurationError):
    """Raised when a secret reference cannot be resolved from the environment."""

    code = "eden.config.secret_unresolved"


# ---------------------------------------------------------------------------
# Core / kernel
# ---------------------------------------------------------------------------
class CoreError(EdenError):
    """Base class for kernel, container and registry failures."""

    code = "eden.core.error"


class RegistryError(CoreError):
    """Raised for duplicate or unknown registry entries."""

    code = "eden.core.registry"


class DependencyResolutionError(CoreError):
    """Raised when the DI container cannot satisfy a dependency."""

    code = "eden.core.dependency"


class LifecycleError(CoreError):
    """Raised when a component fails to start or stop cleanly."""

    code = "eden.core.lifecycle"


class PluginLoadError(CoreError):
    """Raised when a dynamically imported component cannot be loaded."""

    code = "eden.core.plugin"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationError(EdenError):
    """Raised when externally supplied input fails validation."""

    code = "eden.validation.error"


# ---------------------------------------------------------------------------
# Gateway / providers
# ---------------------------------------------------------------------------
class GatewayError(EdenError):
    """Base class for AI gateway failures."""

    code = "eden.gateway.error"


class ProviderError(GatewayError):
    """Base class for errors originating from a specific provider."""

    code = "eden.gateway.provider"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Initialise the error and record the offending provider name.

        Args:
            message: Human-readable description.
            provider: Logical provider name, e.g. ``"openai-main"``.
            context: Structured, non-sensitive detail.
            cause: Underlying exception.
        """
        merged: dict[str, Any] = {"provider": provider}
        merged.update(context or {})
        super().__init__(message, context=merged, cause=cause)
        self.provider = provider


class ProviderAuthenticationError(ProviderError):
    """Raised when credentials are rejected. Never retried."""

    code = "eden.gateway.auth"
    retryable = False


class ProviderTimeoutError(ProviderError):
    """Raised when a provider exceeds its configured timeout."""

    code = "eden.gateway.timeout"
    retryable = True


class ProviderRateLimitError(ProviderError):
    """Raised when a provider signals rate limiting."""

    code = "eden.gateway.rate_limit"
    retryable = True


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is unreachable, unhealthy or circuit-broken."""

    code = "eden.gateway.unavailable"
    retryable = True


class ProviderResponseError(ProviderError):
    """Raised when a provider returns a malformed or unusable payload."""

    code = "eden.gateway.bad_response"
    retryable = False


class ModelNotSupportedError(ProviderError):
    """Raised when a provider cannot serve the requested model."""

    code = "eden.gateway.model_unsupported"


class NoProviderAvailableError(GatewayError):
    """Raised when the router finds no candidate satisfying the request."""

    code = "eden.gateway.no_candidate"


class EmbeddingNotSupportedError(GatewayError):
    """Raised when no configured provider can produce embeddings."""

    code = "eden.gateway.embedding_unsupported"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class MemorySubsystemError(EdenError):
    """Base class for memory failures.

    Named to avoid shadowing the builtin :class:`MemoryError`, which means
    something entirely different.
    """

    code = "eden.memory.error"


class MemoryNotFoundError(MemorySubsystemError):
    """Raised when a record is requested by an identifier that does not exist."""

    code = "eden.memory.not_found"


class MemoryStorageError(MemorySubsystemError):
    """Raised when the backing store cannot be read or written."""

    code = "eden.memory.storage"
    retryable = True


class MemoryCapacityError(MemorySubsystemError):
    """Raised when a write would exceed a hard configured limit."""

    code = "eden.memory.capacity"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
class ExecutionError(EdenError):
    """Base class for failures in the execution pipeline."""

    code = "eden.execution.error"


class VerificationError(ExecutionError):
    """Raised when an action fails verification and must not proceed."""

    code = "eden.execution.verification"


class PermissionDeniedError(ExecutionError):
    """Raised when policy or an approver refuses an action."""

    code = "eden.execution.permission_denied"


class HandlerNotFoundError(ExecutionError):
    """Raised when no handler is registered for an action kind."""

    code = "eden.execution.no_handler"


class ActionExecutionError(ExecutionError):
    """Raised when a handler fails while performing an action."""

    code = "eden.execution.failed"


class RollbackError(ExecutionError):
    """Raised when compensating an action fails.

    This is the most serious execution failure: the world is now in a state
    that neither the caller nor EDEN intended.
    """

    code = "eden.execution.rollback_failed"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class AgentError(EdenError):
    """Base class for agent failures."""

    code = "eden.agent.error"


class NoSuitableAgentError(AgentError):
    """Raised when no registered agent is willing to take a task."""

    code = "eden.agent.no_candidate"


class PlanningError(AgentError):
    """Raised when an agent cannot produce a usable plan."""

    code = "eden.agent.planning"


class InvalidPlanError(AgentError):
    """Raised when a produced plan violates a configured constraint."""

    code = "eden.agent.invalid_plan"


class AgentCapabilityError(AgentError):
    """Raised when an agent needs a subsystem that is not available to it."""

    code = "eden.agent.capability_unavailable"


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
class HardwareError(EdenError):
    """Base class for device failures."""

    code = "eden.hardware.error"


class DeviceNotFoundError(HardwareError):
    """Raised when a command names a device that is not registered."""

    code = "eden.hardware.not_found"


class DeviceUnavailableError(HardwareError):
    """Raised when a device is not in a state that can serve a request."""

    code = "eden.hardware.unavailable"
    retryable = True


class DeviceCommandError(HardwareError):
    """Raised when a device rejects or fails a command."""

    code = "eden.hardware.command_failed"


class DeviceSafetyError(HardwareError):
    """Raised when a command would exceed a declared safety limit.

    Never retryable: a command outside the safe envelope does not become safe
    on a second attempt.
    """

    code = "eden.hardware.safety_violation"
    retryable = False


# ---------------------------------------------------------------------------
# Automation
# ---------------------------------------------------------------------------
class AutomationError(EdenError):
    """Base class for automation failures."""

    code = "eden.automation.error"


class RuleNotFoundError(AutomationError):
    """Raised when a rule is referenced by a name that is not registered."""

    code = "eden.automation.rule_not_found"


class InvalidRuleError(AutomationError):
    """Raised when a rule definition is internally inconsistent."""

    code = "eden.automation.invalid_rule"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class InterfaceError(EdenError):
    """Base class for interface-layer failures."""

    code = "eden.interface.error"


class TransportError(EdenError):
    """Raised for network-level failures beneath a provider."""

    code = "eden.transport.error"
    retryable = True
