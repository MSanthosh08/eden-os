"""Immutable configuration schema.

Every tunable in EDEN lives here as a frozen dataclass. Nothing in the codebase
reads ``os.environ`` or a literal port, path, model name, timeout or provider
name directly; components receive a slice of this tree by dependency injection.

Frozen + ``slots`` gives three properties that matter for a long-lived product:
configuration cannot drift at runtime, it is cheap to pass around, and it is
trivially comparable in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from eden.config.enums import (
    Capability,
    DeviceKind,
    Environment,
    LogFormat,
    LogLevel,
    PermissionMode,
    PrivacyTier,
    ProviderKind,
    RiskLevel,
    RoutingStrategyName,
)
from eden.errors import InvalidConfigError

_DEFAULT_APP_NAME = "eden"
_MAX_PORT = 65535
_MAX_TEMPERATURE = 2.0


def _require_positive(value: float, name: str) -> None:
    """Raise if ``value`` is not strictly positive.

    Args:
        value: Value under validation.
        name: Dotted configuration key, used in the error context.

    Raises:
        InvalidConfigError: If ``value`` is zero or negative.
    """
    if value <= 0:
        raise InvalidConfigError(
            "Value must be strictly positive.",
            context={"key": name, "value": value},
        )


def _require_non_negative(value: float, name: str) -> None:
    """Raise if ``value`` is negative.

    Args:
        value: Value under validation.
        name: Dotted configuration key, used in the error context.

    Raises:
        InvalidConfigError: If ``value`` is negative.
    """
    if value < 0:
        raise InvalidConfigError(
            "Value must not be negative.",
            context={"key": name, "value": value},
        )


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Filesystem locations owned by EDEN.

    Attributes:
        root: Base directory for all mutable state.
        data_dir: Durable state (memory stores, project indexes).
        cache_dir: Regenerable artefacts.
        log_dir: Rotating log files.
        plugin_dir: Externally supplied plugin packages.
    """

    root: Path = Path(".eden")
    data_dir: Path = Path(".eden/data")
    cache_dir: Path = Path(".eden/cache")
    log_dir: Path = Path(".eden/logs")
    plugin_dir: Path = Path(".eden/plugins")

    def all_directories(self) -> tuple[Path, ...]:
        """Return every directory that must exist before startup."""
        return (self.root, self.data_dir, self.cache_dir, self.log_dir, self.plugin_dir)

    def ensure(self) -> None:
        """Create every configured directory, including parents."""
        for directory in self.all_directories():
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Central logging behaviour.

    Attributes:
        level: Minimum severity emitted.
        format: Console rendering for humans, JSON for aggregators.
        to_file: Whether to additionally write a rotating file.
        file_name: Log file name inside :attr:`PathsConfig.log_dir`.
        max_bytes: Rotation threshold per file.
        backup_count: Number of rotated files retained.
        redact_keys: Structured-field names scrubbed before emission.
        log_timings: Whether ``@timed`` decorators emit duration records.
    """

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.CONSOLE
    to_file: bool = False
    file_name: str = "eden.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    redact_keys: tuple[str, ...] = (
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "x-api-key",
    )
    log_timings: bool = True

    def __post_init__(self) -> None:
        """Validate numeric bounds.

        Raises:
            InvalidConfigError: If rotation settings are not sensible.
        """
        _require_positive(self.max_bytes, "logging.max_bytes")
        _require_non_negative(self.backup_count, "logging.backup_count")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Exponential-backoff policy applied to retryable failures.

    Attributes:
        max_attempts: Total attempts including the first.
        initial_backoff_seconds: Delay before the second attempt.
        max_backoff_seconds: Ceiling for any single delay.
        multiplier: Growth factor between attempts.
        jitter_ratio: Fraction of the delay randomised to avoid thundering herds.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 8.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        """Validate the policy.

        Raises:
            InvalidConfigError: If any bound is invalid.
        """
        _require_positive(self.max_attempts, "retry.max_attempts")
        _require_non_negative(self.initial_backoff_seconds, "retry.initial_backoff_seconds")
        _require_positive(self.max_backoff_seconds, "retry.max_backoff_seconds")
        _require_positive(self.multiplier, "retry.multiplier")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise InvalidConfigError(
                "Jitter ratio must be between 0 and 1.",
                context={"key": "retry.jitter_ratio", "value": self.jitter_ratio},
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A single model offered by a provider.

    Cost and latency figures are declarative inputs to routing, not measurements;
    the health tracker supplies observed latency at runtime.

    Attributes:
        name: Provider-specific model identifier.
        capabilities: Features this model supports.
        context_window: Maximum tokens the model accepts.
        input_cost_per_1k: Currency units per 1000 prompt tokens.
        output_cost_per_1k: Currency units per 1000 completion tokens.
        expected_latency_ms: Prior belief used before health data accumulates.
    """

    name: str
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})
    context_window: int = 8192
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    expected_latency_ms: float = 1000.0

    def __post_init__(self) -> None:
        """Validate the model entry.

        Raises:
            InvalidConfigError: If the name is empty or a bound is invalid.
        """
        if not self.name.strip():
            raise InvalidConfigError("Model name must not be empty.", context={"key": "model.name"})
        _require_positive(self.context_window, "model.context_window")
        _require_non_negative(self.input_cost_per_1k, "model.input_cost_per_1k")
        _require_non_negative(self.output_cost_per_1k, "model.output_cost_per_1k")
        _require_positive(self.expected_latency_ms, "model.expected_latency_ms")


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Declarative description of one AI provider.

    A provider is identified by a logical ``name`` chosen by the operator, which
    means the same vendor may appear several times (different regions, keys or
    tiers) without any code change.

    Attributes:
        name: Unique logical name, e.g. ``"groq-fast"``.
        kind: Wire protocol family.
        enabled: Whether the provider participates in routing.
        base_url: API root. Required for every non-mock kind.
        api_key_env: Environment variable holding the credential.
        default_model: Model used when a request does not name one.
        models: Catalogue used for capability and cost decisions.
        timeout_seconds: Per-request wall-clock budget.
        max_concurrency: Maximum simultaneous in-flight requests.
        requests_per_minute: Client-side rate limit; ``0`` disables it.
        privacy_tier: Data-residency guarantee.
        weight: Operator preference multiplier applied to the final score.
        implementation: Import path for :attr:`ProviderKind.CUSTOM` providers.
        retry: Provider-specific retry policy.
        headers: Extra static headers, e.g. vendor API versions.
    """

    name: str
    kind: ProviderKind
    enabled: bool = True
    base_url: str = ""
    api_key_env: str = ""
    default_model: str = ""
    models: tuple[ModelConfig, ...] = ()
    timeout_seconds: float = 60.0
    max_concurrency: int = 8
    requests_per_minute: int = 0
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC_CLOUD
    weight: float = 1.0
    implementation: str = ""
    retry: RetryConfig = field(default_factory=RetryConfig)
    headers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate the provider entry.

        Raises:
            InvalidConfigError: If the declaration is internally inconsistent.
        """
        if not self.name.strip():
            raise InvalidConfigError(
                "Provider name must not be empty.", context={"key": "provider.name"}
            )
        _require_positive(self.timeout_seconds, f"provider.{self.name}.timeout_seconds")
        _require_positive(self.max_concurrency, f"provider.{self.name}.max_concurrency")
        _require_non_negative(self.requests_per_minute, f"provider.{self.name}.requests_per_minute")
        _require_positive(self.weight, f"provider.{self.name}.weight")
        if self.kind not in (ProviderKind.MOCK, ProviderKind.CUSTOM) and not self.base_url:
            raise InvalidConfigError(
                "A base_url is required for network providers.",
                context={"key": f"provider.{self.name}.base_url", "kind": str(self.kind)},
            )
        if self.kind is ProviderKind.CUSTOM and not self.implementation:
            raise InvalidConfigError(
                "Custom providers must declare an implementation import path.",
                context={"key": f"provider.{self.name}.implementation"},
            )
        if self.models and self.default_model:
            known = {model.name for model in self.models}
            if self.default_model not in known:
                raise InvalidConfigError(
                    "default_model is not present in the model catalogue.",
                    context={
                        "key": f"provider.{self.name}.default_model",
                        "value": self.default_model,
                        "known": sorted(known),
                    },
                )

    def model(self, name: str) -> ModelConfig | None:
        """Return the catalogue entry for ``name`` if present."""
        for candidate in self.models:
            if candidate.name == name:
                return candidate
        return None

    def capabilities(self) -> frozenset[Capability]:
        """Return the union of capabilities across the model catalogue."""
        union: set[Capability] = set()
        for candidate in self.models:
            union |= candidate.capabilities
        return frozenset(union)


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    """Failure thresholds that take a provider out of rotation.

    Attributes:
        failure_threshold: Consecutive failures that open the circuit.
        reset_seconds: Cool-down before a probe request is permitted.
        half_open_successes: Successes required to fully close the circuit.
    """

    failure_threshold: int = 5
    reset_seconds: float = 30.0
    half_open_successes: int = 2

    def __post_init__(self) -> None:
        """Validate the breaker.

        Raises:
            InvalidConfigError: If any threshold is not positive.
        """
        _require_positive(self.failure_threshold, "circuit_breaker.failure_threshold")
        _require_positive(self.reset_seconds, "circuit_breaker.reset_seconds")
        _require_positive(self.half_open_successes, "circuit_breaker.half_open_successes")


@dataclass(frozen=True, slots=True)
class RouterWeights:
    """Relative importance of each routing signal.

    Weights are normalised at scoring time, so only their ratios matter.

    Attributes:
        cost: Preference for cheaper providers.
        latency: Preference for faster providers.
        health: Preference for providers with a clean recent record.
        privacy: Preference for stronger data-residency guarantees.
        preference: Weight given to the operator's per-provider multiplier.
    """

    cost: float = 1.0
    latency: float = 1.0
    health: float = 1.5
    privacy: float = 0.5
    preference: float = 1.0

    def __post_init__(self) -> None:
        """Validate the weights.

        Raises:
            InvalidConfigError: If a weight is negative or all are zero.
        """
        for name, value in (
            ("cost", self.cost),
            ("latency", self.latency),
            ("health", self.health),
            ("privacy", self.privacy),
            ("preference", self.preference),
        ):
            _require_non_negative(value, f"router.weights.{name}")
        if self.total() <= 0:
            raise InvalidConfigError(
                "At least one router weight must be greater than zero.",
                context={"key": "router.weights"},
            )

    def total(self) -> float:
        """Return the sum of all weights."""
        return self.cost + self.latency + self.health + self.privacy + self.preference


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Behaviour of the Omni Router.

    Attributes:
        strategy: Named scoring profile.
        weights: Signal weights used by the balanced strategy.
        minimum_privacy_tier: Hard floor applied to every request.
        failover_enabled: Whether to try the next candidate on failure.
        max_failovers: Maximum additional providers attempted per request.
        circuit_breaker: Thresholds for removing providers from rotation.
    """

    strategy: RoutingStrategyName = RoutingStrategyName.BALANCED
    weights: RouterWeights = field(default_factory=RouterWeights)
    minimum_privacy_tier: PrivacyTier = PrivacyTier.PUBLIC_CLOUD
    failover_enabled: bool = True
    max_failovers: int = 2
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    def __post_init__(self) -> None:
        """Validate failover bounds.

        Raises:
            InvalidConfigError: If ``max_failovers`` is negative.
        """
        _require_non_negative(self.max_failovers, "router.max_failovers")


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """AI Gateway configuration.

    Attributes:
        providers: Every provider EDEN may use.
        router: Provider-selection behaviour.
        request_timeout_seconds: Ceiling across all failover attempts.
        stream_chunk_timeout_seconds: Idle timeout between streamed chunks.
    """

    providers: tuple[ProviderConfig, ...] = ()
    router: RouterConfig = field(default_factory=RouterConfig)
    request_timeout_seconds: float = 120.0
    stream_chunk_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate provider uniqueness and timeouts.

        Raises:
            InvalidConfigError: If names collide or timeouts are invalid.
        """
        _require_positive(self.request_timeout_seconds, "gateway.request_timeout_seconds")
        _require_positive(self.stream_chunk_timeout_seconds, "gateway.stream_chunk_timeout_seconds")
        seen: set[str] = set()
        for provider in self.providers:
            if provider.name in seen:
                raise InvalidConfigError(
                    "Duplicate provider name.",
                    context={"key": "gateway.providers", "name": provider.name},
                )
            seen.add(provider.name)

    def enabled_providers(self) -> tuple[ProviderConfig, ...]:
        """Return only the providers that are switched on."""
        return tuple(provider for provider in self.providers if provider.enabled)


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Retention and capacity policy for the memory subsystem.

    Attributes:
        enabled: Whether the memory subsystem is started at all.
        short_term_capacity: Records retained per namespace before eviction.
        short_term_ttl_seconds: Age at which a short-term record expires.
        conversation_turn_limit: Turns retained per conversation.
        conversation_token_budget: Ceiling used when building a prompt window.
        long_term_max_records: Hard ceiling per namespace for durable stores.
        vector_provider: Logical provider name used for embeddings. Empty lets
            the router choose any provider advertising the capability.
        vector_model: Embedding model name. Empty defers to the provider.
        vector_dimensions: Width used by the offline hash embedder.
        vector_min_similarity: Cosine floor below which hits are discarded.
        default_search_limit: Records returned per store when unspecified.
        persist: Whether durable stores write to disk at all.
        consolidate_after_turns: Turn count above which a conversation is
            summarised into long-term memory.
        consolidation_keep_turns: Recent turns left untouched by consolidation.
        consolidation_summary_words: Target length of a generated summary.
    """

    enabled: bool = True
    short_term_capacity: int = 200
    short_term_ttl_seconds: float = 3600.0
    conversation_turn_limit: int = 100
    conversation_token_budget: int = 4000
    long_term_max_records: int = 100_000
    vector_provider: str = ""
    vector_model: str = ""
    vector_dimensions: int = 256
    vector_min_similarity: float = 0.0
    default_search_limit: int = 10
    persist: bool = True
    consolidate_after_turns: int = 50
    consolidation_keep_turns: int = 10
    consolidation_summary_words: int = 200

    def __post_init__(self) -> None:
        """Validate the policy.

        Raises:
            InvalidConfigError: If any bound is invalid.
        """
        _require_positive(self.short_term_capacity, "memory.short_term_capacity")
        _require_positive(self.short_term_ttl_seconds, "memory.short_term_ttl_seconds")
        _require_positive(self.conversation_turn_limit, "memory.conversation_turn_limit")
        _require_positive(self.conversation_token_budget, "memory.conversation_token_budget")
        _require_positive(self.long_term_max_records, "memory.long_term_max_records")
        _require_positive(self.vector_dimensions, "memory.vector_dimensions")
        _require_positive(self.default_search_limit, "memory.default_search_limit")
        _require_positive(self.consolidate_after_turns, "memory.consolidate_after_turns")
        _require_positive(self.consolidation_keep_turns, "memory.consolidation_keep_turns")
        _require_positive(self.consolidation_summary_words, "memory.consolidation_summary_words")
        if self.consolidation_keep_turns >= self.consolidate_after_turns:
            raise InvalidConfigError(
                "consolidation_keep_turns must be below consolidate_after_turns, "
                "otherwise consolidation would never have anything to summarise.",
                context={
                    "keep": self.consolidation_keep_turns,
                    "threshold": self.consolidate_after_turns,
                },
            )
        if not -1.0 <= self.vector_min_similarity <= 1.0:
            raise InvalidConfigError(
                "Cosine similarity floor must be between -1 and 1.",
                context={
                    "key": "memory.vector_min_similarity",
                    "value": self.vector_min_similarity,
                },
            )


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Constraints applied to every agent.

    Attributes:
        enabled: Whether the agent subsystem is started at all.
        max_plan_steps: Ceiling on steps in one plan, refused at planning time.
        min_suitability: Score below which an agent is not considered for a task.
        rollback_on_verification_failure: Undo an agent's completed work when
            its own post-execution check says the goal was not met.
        planning_temperature: Sampling temperature used for planning prompts.
        planning_model: Model used for planning. Empty lets the router choose.
        recall_limit: Memories retrieved when building an agent's context.
        default_namespace: Namespace used when a task does not name one.
        task_timeout_seconds: Wall-clock budget for one whole task.
    """

    enabled: bool = True
    max_plan_steps: int = 20
    min_suitability: float = 0.1
    rollback_on_verification_failure: bool = True
    planning_temperature: float = 0.2
    planning_model: str = ""
    recall_limit: int = 5
    default_namespace: str = "default"
    task_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate the constraints.

        Raises:
            InvalidConfigError: If a bound is invalid.
        """
        _require_positive(self.max_plan_steps, "agents.max_plan_steps")
        _require_positive(self.recall_limit, "agents.recall_limit")
        _require_positive(self.task_timeout_seconds, "agents.task_timeout_seconds")
        if not 0.0 <= self.min_suitability <= 1.0:
            raise InvalidConfigError(
                "Minimum suitability must be between 0 and 1.",
                context={"key": "agents.min_suitability", "value": self.min_suitability},
            )
        if not 0.0 <= self.planning_temperature <= _MAX_TEMPERATURE:
            raise InvalidConfigError(
                "Planning temperature is out of range.",
                context={"key": "agents.planning_temperature"},
            )


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Policy governing what EDEN is permitted to do to the world.

    The defaults are deliberately restrictive. An operator opts *in* to
    capability — ``allowed_commands`` starts empty, meaning no shell command
    can run until one is named — rather than opting out of danger.

    Attributes:
        enabled: Whether the execution subsystem is started at all.
        dry_run: Verify and permit as normal, but never perform side effects.
        default_mode: Authorisation applied when no rule matches.
        auto_approve_max_risk: Risk at or below which reversible actions run
            without asking. Irreversible actions never qualify.
        deny_above_risk: Risk above which an action is refused outright, with
            no approval path.
        require_reversible: When set, an action with no rollback plan is denied.
        workspace_root: Directory outside which filesystem actions are refused.
        allowed_commands: Executables permitted by name. Empty forbids all.
        denied_path_globs: Patterns refused regardless of workspace scope.
        timeout_seconds: Wall-clock budget for one action.
        max_transaction_actions: Ceiling on actions in one transaction.
        max_payload_bytes: Ceiling on a single action's content payload.
        max_output_bytes: Captured output beyond this is truncated.
        journal_enabled: Whether every decision is written to the audit journal.
    """

    enabled: bool = True
    dry_run: bool = False
    default_mode: PermissionMode = PermissionMode.CONFIRM
    auto_approve_max_risk: RiskLevel = RiskLevel.LOW
    deny_above_risk: RiskLevel = RiskLevel.HIGH
    require_reversible: bool = False
    workspace_root: Path = Path(".eden/workspace")
    allowed_commands: tuple[str, ...] = ()
    denied_path_globs: tuple[str, ...] = (
        "**/.env",
        "**/.env.*",
        "**/.git/**",
        "**/.ssh/**",
        "**/id_rsa*",
        "**/*.pem",
        "**/*.key",
        "**/credentials*",
    )
    timeout_seconds: float = 60.0
    max_transaction_actions: int = 50
    max_payload_bytes: int = 5_242_880
    max_output_bytes: int = 65_536
    journal_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the policy.

        Raises:
            InvalidConfigError: If a bound is invalid or the thresholds are
                mutually contradictory.
        """
        _require_positive(self.timeout_seconds, "execution.timeout_seconds")
        _require_positive(self.max_transaction_actions, "execution.max_transaction_actions")
        _require_positive(self.max_payload_bytes, "execution.max_payload_bytes")
        _require_positive(self.max_output_bytes, "execution.max_output_bytes")
        if self.auto_approve_max_risk > self.deny_above_risk:
            raise InvalidConfigError(
                "auto_approve_max_risk cannot exceed deny_above_risk; that would "
                "auto-approve actions the policy also forbids.",
                context={
                    "auto_approve_max_risk": self.auto_approve_max_risk.name,
                    "deny_above_risk": self.deny_above_risk.name,
                },
            )


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Declarative description of one physical or simulated device.

    Safety limits live here rather than inside driver code, so an operator can
    tighten them without a deploy and so verification can see them *before* a
    command is ever sent to equipment.

    Attributes:
        name: Unique logical device name.
        kind: Driver family.
        enabled: Whether the device participates at all.
        endpoint: Connection target — a URL for HTTP devices, otherwise unused.
        implementation: Import path for :attr:`DeviceKind.CUSTOM` drivers.
        channels: Named channels this device exposes.
        limits: Per-channel ``(channel, minimum, maximum)`` bounds.
        timeout_seconds: Per-command wall-clock budget.
        max_commands_per_minute: Client-side command rate ceiling.
        connect_on_start: Whether the manager connects during startup.
        metadata: Free-form annotations, e.g. physical location.
    """

    name: str
    kind: DeviceKind = DeviceKind.SIMULATED
    enabled: bool = True
    endpoint: str = ""
    implementation: str = ""
    channels: tuple[str, ...] = ()
    limits: tuple[tuple[str, float, float], ...] = ()
    timeout_seconds: float = 10.0
    max_commands_per_minute: int = 600
    connect_on_start: bool = True
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate the device declaration.

        Raises:
            InvalidConfigError: If the declaration is unusable.
        """
        if not self.name.strip():
            raise InvalidConfigError(
                "Device name must not be empty.", context={"key": "device.name"}
            )
        _require_positive(self.timeout_seconds, f"device.{self.name}.timeout_seconds")
        _require_non_negative(
            self.max_commands_per_minute, f"device.{self.name}.max_commands_per_minute"
        )
        if self.kind is DeviceKind.HTTP and not self.endpoint:
            raise InvalidConfigError(
                "HTTP devices require an endpoint.",
                context={"key": f"device.{self.name}.endpoint"},
            )
        if self.kind is DeviceKind.CUSTOM and not self.implementation:
            raise InvalidConfigError(
                "Custom devices must declare an implementation import path.",
                context={"key": f"device.{self.name}.implementation"},
            )
        for channel, lowest, highest in self.limits:
            if lowest > highest:
                raise InvalidConfigError(
                    "Channel limit has its minimum above its maximum.",
                    context={"device": self.name, "channel": channel},
                )
            if self.channels and channel not in self.channels:
                raise InvalidConfigError(
                    "Limit names a channel the device does not declare.",
                    context={"device": self.name, "channel": channel},
                )

    def limit_for(self, channel: str) -> tuple[float, float] | None:
        """Return the ``(minimum, maximum)`` bounds for ``channel``, if declared."""
        for name, lowest, highest in self.limits:
            if name == channel:
                return (lowest, highest)
        return None


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    """Device fleet and global hardware policy.

    ``enabled`` defaults to ``False``. Software that can move things should not
    be able to move things the moment it is installed.

    Attributes:
        enabled: Whether the hardware subsystem is started at all.
        devices: Every device EDEN may talk to.
        connect_timeout_seconds: Budget for one connection attempt.
        read_only: Refuse every actuating command, permitting reads only.
    """

    enabled: bool = False
    devices: tuple[DeviceConfig, ...] = ()
    connect_timeout_seconds: float = 15.0
    read_only: bool = False

    def __post_init__(self) -> None:
        """Validate the fleet.

        Raises:
            InvalidConfigError: If names collide or a bound is invalid.
        """
        _require_positive(self.connect_timeout_seconds, "hardware.connect_timeout_seconds")
        seen: set[str] = set()
        for device in self.devices:
            if device.name in seen:
                raise InvalidConfigError(
                    "Duplicate device name.",
                    context={"key": "hardware.devices", "name": device.name},
                )
            seen.add(device.name)

    def enabled_devices(self) -> tuple[DeviceConfig, ...]:
        """Return only the devices that are switched on."""
        return tuple(device for device in self.devices if device.enabled)

    def device(self, name: str) -> DeviceConfig | None:
        """Return the declaration for ``name``, if present."""
        for device in self.devices:
            if device.name == name:
                return device
        return None


@dataclass(frozen=True, slots=True)
class AutomationConfig:
    """Scheduling policy for automation rules.

    Attributes:
        enabled: Whether the scheduler is started at all.
        tick_seconds: How often triggers are evaluated.
        max_concurrent_runs: Rules permitted to run simultaneously.
        run_timeout_seconds: Budget for one rule run.
        catch_up: Whether a rule whose window was missed fires immediately on
            the next tick. ``False`` skips it, which is usually what an
            operator wants after a restart — a day of missed hourly jobs
            should not all fire at once.
        history_limit: Completed runs retained for inspection.
    """

    enabled: bool = False
    tick_seconds: float = 1.0
    max_concurrent_runs: int = 4
    run_timeout_seconds: float = 300.0
    catch_up: bool = False
    history_limit: int = 200

    def __post_init__(self) -> None:
        """Validate the policy.

        Raises:
            InvalidConfigError: If any bound is invalid.
        """
        _require_positive(self.tick_seconds, "automation.tick_seconds")
        _require_positive(self.max_concurrent_runs, "automation.max_concurrent_runs")
        _require_positive(self.run_timeout_seconds, "automation.run_timeout_seconds")
        _require_positive(self.history_limit, "automation.history_limit")


@dataclass(frozen=True, slots=True)
class InterfaceConfig:
    """Local interface surface.

    Attributes:
        enabled: Whether the HTTP interface may be started.
        host: Interface to bind. Loopback by default and deliberately — this
            server has no authentication and must not face a network.
        port: TCP port to bind. ``0`` asks the operating system for an
            ephemeral port, which is how tests and multi-instance setups avoid
            collisions; read the real port back from the running server.
        allow_actions: Whether the interface may submit tasks and actions
            rather than only observing.
        approval_timeout_seconds: How long a pending approval waits for a human
            before it is treated as refused.
        max_request_bytes: Ceiling on one request body.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8420
    allow_actions: bool = True
    approval_timeout_seconds: float = 120.0
    max_request_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        """Validate the binding.

        Raises:
            InvalidConfigError: If the port or limits are out of range.
        """
        if not 0 <= self.port <= _MAX_PORT:
            raise InvalidConfigError(
                "Port must be between 0 and 65535, where 0 requests an ephemeral port.",
                context={"key": "interface.port", "value": self.port},
            )
        _require_positive(self.approval_timeout_seconds, "interface.approval_timeout_seconds")
        _require_positive(self.max_request_bytes, "interface.max_request_bytes")


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Network binding for the (future) interface layer.

    Attributes:
        host: Interface to bind.
        port: TCP port to bind.
    """

    host: str = "127.0.0.1"
    port: int = 8420

    def __post_init__(self) -> None:
        """Validate the port range.

        Raises:
            InvalidConfigError: If the port is outside 1-65535.
        """
        if not 1 <= self.port <= _MAX_PORT:
            raise InvalidConfigError(
                "Port must be between 1 and 65535.",
                context={"key": "server.port", "value": self.port},
            )


@dataclass(frozen=True, slots=True)
class EdenConfig:
    """Root configuration object for a running EDEN instance.

    Attributes:
        app_name: Logical instance name used in logs and telemetry.
        environment: Deployment environment.
        version: Semantic version of the running build.
        paths: Filesystem layout.
        logging: Logging behaviour.
        server: Network binding.
        gateway: AI Gateway and routing behaviour.
        memory: Retention and capacity policy.
        execution: What EDEN is permitted to do to the world.
        agents: Constraints applied to every agent.
        hardware: Device fleet and hardware policy.
        automation: Scheduling policy for automation rules.
        interface: Local interface surface.
    """

    app_name: str = _DEFAULT_APP_NAME
    environment: Environment = Environment.DEVELOPMENT
    version: str = "0.1.0"
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)

    def __post_init__(self) -> None:
        """Validate the instance name.

        Raises:
            InvalidConfigError: If the application name is empty.
        """
        if not self.app_name.strip():
            raise InvalidConfigError("app_name must not be empty.", context={"key": "app_name"})

    @property
    def is_production(self) -> bool:
        """Return whether this instance runs in production."""
        return self.environment is Environment.PRODUCTION
