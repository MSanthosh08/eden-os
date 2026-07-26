"""Configuration loader.

:class:`ConfigLoader` is a Builder: sources are appended in ascending precedence
and :meth:`ConfigLoader.build` merges them, coerces the raw values and
constructs the frozen :class:`~eden.config.schema.EdenConfig` tree.

Precedence, lowest to highest::

    schema defaults  <  TOML file  <  environment  <  runtime overrides
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
from eden.config.schema import (
    AgentConfig,
    AutomationConfig,
    CircuitBreakerConfig,
    DeviceConfig,
    EdenConfig,
    ExecutionConfig,
    GatewayConfig,
    HardwareConfig,
    InterfaceConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    PathsConfig,
    ProviderConfig,
    RetryConfig,
    RouterConfig,
    RouterWeights,
    ServerConfig,
)
from eden.config.sources import (
    ConfigSource,
    EnvironmentSource,
    MappingSource,
    TomlFileSource,
    deep_merge,
)
from eden.errors import ConfigurationError, InvalidConfigError

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on", "y"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off", "n"})

DEFAULT_CONFIG_FILENAME = "eden.toml"


def _as_mapping(value: Any, key: str) -> Mapping[str, Any]:  # noqa: ANN401 - raw payload
    """Return ``value`` as a mapping.

    Raises:
        InvalidConfigError: If ``value`` is not a mapping.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise InvalidConfigError(
        "Expected a table of settings.",
        context={"key": key, "actual_type": type(value).__name__},
    )


def _as_sequence(value: Any, key: str) -> Sequence[Any]:  # noqa: ANN401 - raw payload
    """Return ``value`` as a sequence.

    Raises:
        InvalidConfigError: If ``value`` is not a list.
    """
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return value
    raise InvalidConfigError(
        "Expected a list of entries.",
        context={"key": key, "actual_type": type(value).__name__},
    )


def _as_bool(value: Any, key: str) -> bool:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to ``bool``.

    Raises:
        InvalidConfigError: If the token is not recognisably boolean.
    """
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise InvalidConfigError(
        "Expected a boolean value.",
        context={"key": key, "value": str(value)},
    )


def _as_int(value: Any, key: str) -> int:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to ``int``.

    Raises:
        InvalidConfigError: If the value is not an integer.
    """
    if isinstance(value, bool):
        raise InvalidConfigError("Expected an integer, not a boolean.", context={"key": key})
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise InvalidConfigError(
            "Expected an integer value.",
            context={"key": key, "value": str(value)},
            cause=exc,
        ) from exc


def _as_float(value: Any, key: str) -> float:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to ``float``.

    Raises:
        InvalidConfigError: If the value is not numeric.
    """
    if isinstance(value, bool):
        raise InvalidConfigError("Expected a number, not a boolean.", context={"key": key})
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise InvalidConfigError(
            "Expected a numeric value.",
            context={"key": key, "value": str(value)},
            cause=exc,
        ) from exc


def _as_str(value: Any, key: str) -> str:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to a stripped ``str``.

    Raises:
        InvalidConfigError: If the value is a container.
    """
    if isinstance(value, Mapping | list | tuple):
        raise InvalidConfigError(
            "Expected a scalar string value.",
            context={"key": key, "actual_type": type(value).__name__},
        )
    return str(value).strip()


def _as_str_tuple(value: Any, key: str) -> tuple[str, ...]:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to a tuple of strings, accepting a comma-separated string."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(_as_str(item, key) for item in _as_sequence(value, key))


def _as_enum[
    EnumT: DeviceKind | Environment | LogFormat | LogLevel | PermissionMode | ProviderKind
](
    value: Any,  # noqa: ANN401 - raw payload
    key: str,
    enum_type: type[EnumT],
) -> EnumT:
    """Coerce ``value`` to a member of ``enum_type``.

    Raises:
        InvalidConfigError: If no member matches.
    """
    token = _as_str(value, key)
    # Every enum reaching this helper defines ``value == name`` or
    # ``value == name.lower()``, so trying the case variants of the value is
    # equivalent to also matching on the member name.
    for candidate in (token, token.lower(), token.upper()):
        try:
            return enum_type(candidate)
        except ValueError:
            continue
    raise InvalidConfigError(
        "Value is not a recognised option.",
        context={
            "key": key,
            "value": token,
            "allowed": [member.value for member in enum_type],
        },
    )


def _as_privacy_tier(value: Any, key: str) -> PrivacyTier:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to a :class:`PrivacyTier`, accepting names or ordinals.

    Raises:
        InvalidConfigError: If no tier matches.
    """
    token = _as_str(value, key)
    for member in PrivacyTier:
        if member.name.lower() == token.lower() or str(member.value) == token:
            return member
    raise InvalidConfigError(
        "Value is not a recognised privacy tier.",
        context={"key": key, "value": token, "allowed": [m.name.lower() for m in PrivacyTier]},
    )


def _as_strategy(value: Any, key: str) -> RoutingStrategyName:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to a :class:`RoutingStrategyName`.

    Raises:
        InvalidConfigError: If no strategy matches.
    """
    token = _as_str(value, key)
    for member in RoutingStrategyName:
        if member.value.lower() == token.lower():
            return member
    raise InvalidConfigError(
        "Value is not a recognised routing strategy.",
        context={
            "key": key,
            "value": token,
            "allowed": [member.value for member in RoutingStrategyName],
        },
    )


def _as_capabilities(value: Any, key: str) -> frozenset[Capability]:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to a set of :class:`Capability` members.

    Raises:
        InvalidConfigError: If any token is unknown.
    """
    tokens = _as_str_tuple(value, key)
    resolved: set[Capability] = set()
    allowed = {member.value: member for member in Capability}
    for token in tokens:
        member = allowed.get(token.lower())
        if member is None:
            raise InvalidConfigError(
                "Unknown capability.",
                context={"key": key, "value": token, "allowed": sorted(allowed)},
            )
        resolved.add(member)
    return frozenset(resolved)


def _as_headers(value: Any, key: str) -> tuple[tuple[str, str], ...]:  # noqa: ANN401 - raw payload
    """Coerce a header table into an ordered tuple of pairs."""
    mapping = _as_mapping(value, key)
    return tuple((str(name), _as_str(header_value, key)) for name, header_value in mapping.items())


def _pick(
    source: Mapping[str, Any],
    key: str,
    default: Any,  # noqa: ANN401 - schema-supplied default
    coerce: Callable[[Any, str], Any],
    prefix: str,
) -> Any:  # noqa: ANN401 - value type follows the coercer
    """Return the coerced value for ``key`` or ``default`` when absent."""
    if key not in source:
        return default
    return coerce(source[key], f"{prefix}{key}")


class ConfigLoader:
    """Builds an :class:`EdenConfig` from an ordered list of sources.

    Example:
        >>> from pathlib import Path
        >>> config = (
        ...     ConfigLoader()
        ...     .with_toml(Path("eden.toml"))
        ...     .with_environ({"EDEN__LOGGING__LEVEL": "DEBUG"})
        ...     .build()
        ... )
        >>> config.logging.level.value
        'DEBUG'
    """

    def __init__(self) -> None:
        """Create an empty loader."""
        self._sources: list[ConfigSource] = []

    def with_source(self, source: ConfigSource) -> ConfigLoader:
        """Append an arbitrary source at the current highest precedence."""
        self._sources.append(source)
        return self

    def with_mapping(self, data: Mapping[str, Any], *, name: str = "mapping") -> ConfigLoader:
        """Append an in-memory mapping."""
        return self.with_source(MappingSource(data, name=name))

    def with_toml(self, path: Path, *, required: bool = False) -> ConfigLoader:
        """Append a TOML file."""
        return self.with_source(TomlFileSource(path, required=required))

    def with_environ(self, environ: Mapping[str, str] | None = None) -> ConfigLoader:
        """Append the ``EDEN__``-prefixed environment variables."""
        return self.with_source(EnvironmentSource(os.environ if environ is None else environ))

    def with_overrides(self, data: Mapping[str, Any]) -> ConfigLoader:
        """Append the highest-precedence runtime overrides."""
        return self.with_source(MappingSource(data, name="overrides"))

    @property
    def source_names(self) -> tuple[str, ...]:
        """Return the identifiers of every registered source, in order."""
        return tuple(source.name for source in self._sources)

    def merged(self) -> dict[str, Any]:
        """Return the merged raw mapping without constructing the schema.

        Raises:
            ConfigurationError: If a source fails to load.
        """
        merged: dict[str, Any] = {}
        for source in self._sources:
            try:
                fragment = source.load()
            except ConfigurationError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise ConfigurationError(
                    "Configuration source failed to load.",
                    context={"source": source.name},
                    cause=exc,
                ) from exc
            merged = deep_merge(merged, fragment)
        return merged

    def build(self) -> EdenConfig:
        """Merge every source and construct the validated configuration tree.

        Returns:
            A fully validated, immutable configuration object.

        Raises:
            ConfigurationError: If loading, coercion or validation fails.
        """
        raw = self.merged()
        return build_config(raw)


def build_config(raw: Mapping[str, Any]) -> EdenConfig:
    """Construct an :class:`EdenConfig` from a raw nested mapping.

    Args:
        raw: Merged configuration mapping.

    Returns:
        The validated configuration tree.

    Raises:
        ConfigurationError: If any value is missing, mistyped or out of range.
    """
    defaults = EdenConfig()
    return EdenConfig(
        app_name=_pick(raw, "app_name", defaults.app_name, _as_str, ""),
        environment=_pick(
            raw,
            "environment",
            defaults.environment,
            lambda value, key: _as_enum(value, key, Environment),
            "",
        ),
        version=_pick(raw, "version", defaults.version, _as_str, ""),
        paths=_build_paths(_as_mapping(raw.get("paths"), "paths")),
        logging=_build_logging(_as_mapping(raw.get("logging"), "logging")),
        server=_build_server(_as_mapping(raw.get("server"), "server")),
        gateway=_build_gateway(_as_mapping(raw.get("gateway"), "gateway")),
        memory=_build_memory(_as_mapping(raw.get("memory"), "memory")),
        execution=_build_execution(_as_mapping(raw.get("execution"), "execution")),
        agents=_build_agents(_as_mapping(raw.get("agents"), "agents")),
        hardware=_build_hardware(_as_mapping(raw.get("hardware"), "hardware")),
        automation=_build_automation(_as_mapping(raw.get("automation"), "automation")),
        interface=_build_interface(_as_mapping(raw.get("interface"), "interface")),
    )


def _as_limits(value: Any, key: str) -> tuple[tuple[str, float, float], ...]:  # noqa: ANN401
    """Coerce a channel-limit table into ordered triples.

    Accepts ``{channel = [min, max]}`` or ``{channel = {min = .., max = ..}}``.

    Raises:
        InvalidConfigError: If an entry is not a two-value range.
    """
    mapping = _as_mapping(value, key)
    limits: list[tuple[str, float, float]] = []
    for channel, bounds in mapping.items():
        if isinstance(bounds, Mapping):
            lowest = _as_float(bounds.get("min", 0.0), f"{key}.{channel}.min")
            highest = _as_float(bounds.get("max", 0.0), f"{key}.{channel}.max")
        else:
            pair = _as_sequence(bounds, f"{key}.{channel}")
            if len(pair) != 2:  # noqa: PLR2004 - a range has exactly two ends
                raise InvalidConfigError(
                    "A channel limit must be a minimum and a maximum.",
                    context={"key": f"{key}.{channel}"},
                )
            lowest = _as_float(pair[0], f"{key}.{channel}[0]")
            highest = _as_float(pair[1], f"{key}.{channel}[1]")
        limits.append((str(channel), lowest, highest))
    return tuple(limits)


def _build_device(raw: Mapping[str, Any], index: int) -> DeviceConfig:
    """Construct a single :class:`DeviceConfig`.

    Raises:
        InvalidConfigError: If the mandatory ``name`` key is absent.
    """
    prefix = f"hardware.devices[{index}]."
    if "name" not in raw:
        raise InvalidConfigError("Device entries require a name.", context={"key": f"{prefix}name"})
    name = _as_str(raw["name"], f"{prefix}name")
    d = DeviceConfig(name=name)
    return DeviceConfig(
        name=name,
        kind=_pick(raw, "kind", d.kind, lambda v, k: _as_enum(v, k, DeviceKind), prefix),
        enabled=_pick(raw, "enabled", d.enabled, _as_bool, prefix),
        endpoint=_pick(raw, "endpoint", d.endpoint, _as_str, prefix).rstrip("/"),
        implementation=_pick(raw, "implementation", d.implementation, _as_str, prefix),
        channels=_pick(raw, "channels", d.channels, _as_str_tuple, prefix),
        limits=_pick(raw, "limits", d.limits, _as_limits, prefix),
        timeout_seconds=_pick(raw, "timeout_seconds", d.timeout_seconds, _as_float, prefix),
        max_commands_per_minute=_pick(
            raw, "max_commands_per_minute", d.max_commands_per_minute, _as_int, prefix
        ),
        connect_on_start=_pick(raw, "connect_on_start", d.connect_on_start, _as_bool, prefix),
        metadata=_pick(raw, "metadata", d.metadata, _as_headers, prefix),
    )


def _build_hardware(raw: Mapping[str, Any]) -> HardwareConfig:
    """Construct :class:`HardwareConfig`."""
    d = HardwareConfig()
    devices_raw = _as_sequence(raw.get("devices"), "hardware.devices")
    devices = tuple(
        _build_device(_as_mapping(entry, f"hardware.devices[{i}]"), i)
        for i, entry in enumerate(devices_raw)
    )
    return HardwareConfig(
        enabled=_pick(raw, "enabled", d.enabled, _as_bool, "hardware."),
        devices=devices,
        connect_timeout_seconds=_pick(
            raw, "connect_timeout_seconds", d.connect_timeout_seconds, _as_float, "hardware."
        ),
        read_only=_pick(raw, "read_only", d.read_only, _as_bool, "hardware."),
    )


def _build_automation(raw: Mapping[str, Any]) -> AutomationConfig:
    """Construct :class:`AutomationConfig`."""
    d = AutomationConfig()
    prefix = "automation."
    return AutomationConfig(
        enabled=_pick(raw, "enabled", d.enabled, _as_bool, prefix),
        tick_seconds=_pick(raw, "tick_seconds", d.tick_seconds, _as_float, prefix),
        max_concurrent_runs=_pick(
            raw, "max_concurrent_runs", d.max_concurrent_runs, _as_int, prefix
        ),
        run_timeout_seconds=_pick(
            raw, "run_timeout_seconds", d.run_timeout_seconds, _as_float, prefix
        ),
        catch_up=_pick(raw, "catch_up", d.catch_up, _as_bool, prefix),
        history_limit=_pick(raw, "history_limit", d.history_limit, _as_int, prefix),
    )


def _build_interface(raw: Mapping[str, Any]) -> InterfaceConfig:
    """Construct :class:`InterfaceConfig`."""
    d = InterfaceConfig()
    prefix = "interface."
    return InterfaceConfig(
        enabled=_pick(raw, "enabled", d.enabled, _as_bool, prefix),
        host=_pick(raw, "host", d.host, _as_str, prefix),
        port=_pick(raw, "port", d.port, _as_int, prefix),
        allow_actions=_pick(raw, "allow_actions", d.allow_actions, _as_bool, prefix),
        approval_timeout_seconds=_pick(
            raw, "approval_timeout_seconds", d.approval_timeout_seconds, _as_float, prefix
        ),
        max_request_bytes=_pick(raw, "max_request_bytes", d.max_request_bytes, _as_int, prefix),
    )


def _build_agents(raw: Mapping[str, Any]) -> AgentConfig:
    """Construct :class:`AgentConfig`."""
    d = AgentConfig()
    prefix = "agents."
    return AgentConfig(
        enabled=_pick(raw, "enabled", d.enabled, _as_bool, prefix),
        max_plan_steps=_pick(raw, "max_plan_steps", d.max_plan_steps, _as_int, prefix),
        min_suitability=_pick(raw, "min_suitability", d.min_suitability, _as_float, prefix),
        rollback_on_verification_failure=_pick(
            raw,
            "rollback_on_verification_failure",
            d.rollback_on_verification_failure,
            _as_bool,
            prefix,
        ),
        planning_temperature=_pick(
            raw, "planning_temperature", d.planning_temperature, _as_float, prefix
        ),
        planning_model=_pick(raw, "planning_model", d.planning_model, _as_str, prefix),
        recall_limit=_pick(raw, "recall_limit", d.recall_limit, _as_int, prefix),
        default_namespace=_pick(raw, "default_namespace", d.default_namespace, _as_str, prefix),
        task_timeout_seconds=_pick(
            raw, "task_timeout_seconds", d.task_timeout_seconds, _as_float, prefix
        ),
    )


def _as_risk(value: Any, key: str) -> RiskLevel:  # noqa: ANN401 - raw payload
    """Coerce ``value`` to a :class:`RiskLevel`, accepting names or ordinals.

    Raises:
        InvalidConfigError: If no level matches.
    """
    token = _as_str(value, key)
    for member in RiskLevel:
        if member.name.lower() == token.lower() or str(member.value) == token:
            return member
    raise InvalidConfigError(
        "Value is not a recognised risk level.",
        context={"key": key, "value": token, "allowed": [m.name.lower() for m in RiskLevel]},
    )


def _build_execution(raw: Mapping[str, Any]) -> ExecutionConfig:
    """Construct :class:`ExecutionConfig`."""
    d = ExecutionConfig()
    prefix = "execution."
    return ExecutionConfig(
        enabled=_pick(raw, "enabled", d.enabled, _as_bool, prefix),
        dry_run=_pick(raw, "dry_run", d.dry_run, _as_bool, prefix),
        default_mode=_pick(
            raw,
            "default_mode",
            d.default_mode,
            lambda v, k: _as_enum(v, k, PermissionMode),
            prefix,
        ),
        auto_approve_max_risk=_pick(
            raw, "auto_approve_max_risk", d.auto_approve_max_risk, _as_risk, prefix
        ),
        deny_above_risk=_pick(raw, "deny_above_risk", d.deny_above_risk, _as_risk, prefix),
        require_reversible=_pick(raw, "require_reversible", d.require_reversible, _as_bool, prefix),
        workspace_root=Path(_pick(raw, "workspace_root", str(d.workspace_root), _as_str, prefix)),
        allowed_commands=_pick(raw, "allowed_commands", d.allowed_commands, _as_str_tuple, prefix),
        denied_path_globs=_pick(
            raw, "denied_path_globs", d.denied_path_globs, _as_str_tuple, prefix
        ),
        timeout_seconds=_pick(raw, "timeout_seconds", d.timeout_seconds, _as_float, prefix),
        max_transaction_actions=_pick(
            raw, "max_transaction_actions", d.max_transaction_actions, _as_int, prefix
        ),
        max_payload_bytes=_pick(raw, "max_payload_bytes", d.max_payload_bytes, _as_int, prefix),
        max_output_bytes=_pick(raw, "max_output_bytes", d.max_output_bytes, _as_int, prefix),
        journal_enabled=_pick(raw, "journal_enabled", d.journal_enabled, _as_bool, prefix),
    )


def _build_memory(raw: Mapping[str, Any]) -> MemoryConfig:
    """Construct :class:`MemoryConfig`."""
    d = MemoryConfig()
    prefix = "memory."
    return MemoryConfig(
        enabled=_pick(raw, "enabled", d.enabled, _as_bool, prefix),
        short_term_capacity=_pick(
            raw, "short_term_capacity", d.short_term_capacity, _as_int, prefix
        ),
        short_term_ttl_seconds=_pick(
            raw, "short_term_ttl_seconds", d.short_term_ttl_seconds, _as_float, prefix
        ),
        conversation_turn_limit=_pick(
            raw, "conversation_turn_limit", d.conversation_turn_limit, _as_int, prefix
        ),
        conversation_token_budget=_pick(
            raw, "conversation_token_budget", d.conversation_token_budget, _as_int, prefix
        ),
        long_term_max_records=_pick(
            raw, "long_term_max_records", d.long_term_max_records, _as_int, prefix
        ),
        vector_provider=_pick(raw, "vector_provider", d.vector_provider, _as_str, prefix),
        vector_model=_pick(raw, "vector_model", d.vector_model, _as_str, prefix),
        vector_dimensions=_pick(raw, "vector_dimensions", d.vector_dimensions, _as_int, prefix),
        vector_min_similarity=_pick(
            raw, "vector_min_similarity", d.vector_min_similarity, _as_float, prefix
        ),
        default_search_limit=_pick(
            raw, "default_search_limit", d.default_search_limit, _as_int, prefix
        ),
        persist=_pick(raw, "persist", d.persist, _as_bool, prefix),
        consolidate_after_turns=_pick(
            raw, "consolidate_after_turns", d.consolidate_after_turns, _as_int, prefix
        ),
        consolidation_keep_turns=_pick(
            raw, "consolidation_keep_turns", d.consolidation_keep_turns, _as_int, prefix
        ),
        consolidation_summary_words=_pick(
            raw,
            "consolidation_summary_words",
            d.consolidation_summary_words,
            _as_int,
            prefix,
        ),
    )


def _build_paths(raw: Mapping[str, Any]) -> PathsConfig:
    """Construct :class:`PathsConfig`, deriving sub-directories from ``root``."""
    defaults = PathsConfig()
    root = Path(_pick(raw, "root", str(defaults.root), _as_str, "paths."))
    return PathsConfig(
        root=root,
        data_dir=Path(_pick(raw, "data_dir", str(root / "data"), _as_str, "paths.")),
        cache_dir=Path(_pick(raw, "cache_dir", str(root / "cache"), _as_str, "paths.")),
        log_dir=Path(_pick(raw, "log_dir", str(root / "logs"), _as_str, "paths.")),
        plugin_dir=Path(_pick(raw, "plugin_dir", str(root / "plugins"), _as_str, "paths.")),
    )


def _build_logging(raw: Mapping[str, Any]) -> LoggingConfig:
    """Construct :class:`LoggingConfig`."""
    defaults = LoggingConfig()
    return LoggingConfig(
        level=_pick(
            raw, "level", defaults.level, lambda v, k: _as_enum(v, k, LogLevel), "logging."
        ),
        format=_pick(
            raw, "format", defaults.format, lambda v, k: _as_enum(v, k, LogFormat), "logging."
        ),
        to_file=_pick(raw, "to_file", defaults.to_file, _as_bool, "logging."),
        file_name=_pick(raw, "file_name", defaults.file_name, _as_str, "logging."),
        max_bytes=_pick(raw, "max_bytes", defaults.max_bytes, _as_int, "logging."),
        backup_count=_pick(raw, "backup_count", defaults.backup_count, _as_int, "logging."),
        redact_keys=_pick(raw, "redact_keys", defaults.redact_keys, _as_str_tuple, "logging."),
        log_timings=_pick(raw, "log_timings", defaults.log_timings, _as_bool, "logging."),
    )


def _build_server(raw: Mapping[str, Any]) -> ServerConfig:
    """Construct :class:`ServerConfig`."""
    defaults = ServerConfig()
    return ServerConfig(
        host=_pick(raw, "host", defaults.host, _as_str, "server."),
        port=_pick(raw, "port", defaults.port, _as_int, "server."),
    )


def _build_retry(raw: Mapping[str, Any], prefix: str) -> RetryConfig:
    """Construct :class:`RetryConfig`."""
    defaults = RetryConfig()
    return RetryConfig(
        max_attempts=_pick(raw, "max_attempts", defaults.max_attempts, _as_int, prefix),
        initial_backoff_seconds=_pick(
            raw,
            "initial_backoff_seconds",
            defaults.initial_backoff_seconds,
            _as_float,
            prefix,
        ),
        max_backoff_seconds=_pick(
            raw, "max_backoff_seconds", defaults.max_backoff_seconds, _as_float, prefix
        ),
        multiplier=_pick(raw, "multiplier", defaults.multiplier, _as_float, prefix),
        jitter_ratio=_pick(raw, "jitter_ratio", defaults.jitter_ratio, _as_float, prefix),
    )


def _build_model(raw: Mapping[str, Any], prefix: str) -> ModelConfig:
    """Construct a single :class:`ModelConfig`.

    Raises:
        InvalidConfigError: If the mandatory ``name`` key is absent.
    """
    if "name" not in raw:
        raise InvalidConfigError("Model entries require a name.", context={"key": f"{prefix}name"})
    defaults = ModelConfig(name=_as_str(raw["name"], f"{prefix}name"))
    return ModelConfig(
        name=defaults.name,
        capabilities=_pick(raw, "capabilities", defaults.capabilities, _as_capabilities, prefix),
        context_window=_pick(raw, "context_window", defaults.context_window, _as_int, prefix),
        input_cost_per_1k=_pick(
            raw, "input_cost_per_1k", defaults.input_cost_per_1k, _as_float, prefix
        ),
        output_cost_per_1k=_pick(
            raw, "output_cost_per_1k", defaults.output_cost_per_1k, _as_float, prefix
        ),
        expected_latency_ms=_pick(
            raw, "expected_latency_ms", defaults.expected_latency_ms, _as_float, prefix
        ),
    )


def _build_provider(raw: Mapping[str, Any], index: int) -> ProviderConfig:
    """Construct a single :class:`ProviderConfig`.

    Raises:
        InvalidConfigError: If mandatory keys are absent.
    """
    prefix = f"gateway.providers[{index}]."
    for required_key in ("name", "kind"):
        if required_key not in raw:
            raise InvalidConfigError(
                "Provider entries require this key.",
                context={"key": f"{prefix}{required_key}"},
            )
    name = _as_str(raw["name"], f"{prefix}name")
    kind = _as_enum(raw["kind"], f"{prefix}kind", ProviderKind)
    models_raw = _as_sequence(raw.get("models"), f"{prefix}models")
    models = tuple(
        _build_model(_as_mapping(entry, f"{prefix}models[{i}]"), f"{prefix}models[{i}].")
        for i, entry in enumerate(models_raw)
    )
    defaults = ProviderConfig(name=name, kind=ProviderKind.MOCK)
    return ProviderConfig(
        name=name,
        kind=kind,
        enabled=_pick(raw, "enabled", defaults.enabled, _as_bool, prefix),
        base_url=_pick(raw, "base_url", defaults.base_url, _as_str, prefix).rstrip("/"),
        api_key_env=_pick(raw, "api_key_env", defaults.api_key_env, _as_str, prefix),
        default_model=_pick(raw, "default_model", defaults.default_model, _as_str, prefix),
        models=models,
        timeout_seconds=_pick(raw, "timeout_seconds", defaults.timeout_seconds, _as_float, prefix),
        max_concurrency=_pick(raw, "max_concurrency", defaults.max_concurrency, _as_int, prefix),
        requests_per_minute=_pick(
            raw, "requests_per_minute", defaults.requests_per_minute, _as_int, prefix
        ),
        privacy_tier=_pick(raw, "privacy_tier", defaults.privacy_tier, _as_privacy_tier, prefix),
        weight=_pick(raw, "weight", defaults.weight, _as_float, prefix),
        implementation=_pick(raw, "implementation", defaults.implementation, _as_str, prefix),
        retry=_build_retry(_as_mapping(raw.get("retry"), f"{prefix}retry"), f"{prefix}retry."),
        headers=_pick(raw, "headers", defaults.headers, _as_headers, prefix),
    )


def _build_router(raw: Mapping[str, Any]) -> RouterConfig:
    """Construct :class:`RouterConfig`."""
    defaults = RouterConfig()
    weights_raw = _as_mapping(raw.get("weights"), "gateway.router.weights")
    weight_defaults = RouterWeights()
    weights = RouterWeights(
        cost=_pick(weights_raw, "cost", weight_defaults.cost, _as_float, "gateway.router.weights."),
        latency=_pick(
            weights_raw, "latency", weight_defaults.latency, _as_float, "gateway.router.weights."
        ),
        health=_pick(
            weights_raw, "health", weight_defaults.health, _as_float, "gateway.router.weights."
        ),
        privacy=_pick(
            weights_raw, "privacy", weight_defaults.privacy, _as_float, "gateway.router.weights."
        ),
        preference=_pick(
            weights_raw,
            "preference",
            weight_defaults.preference,
            _as_float,
            "gateway.router.weights.",
        ),
    )
    breaker_raw = _as_mapping(raw.get("circuit_breaker"), "gateway.router.circuit_breaker")
    breaker_defaults = CircuitBreakerConfig()
    breaker = CircuitBreakerConfig(
        failure_threshold=_pick(
            breaker_raw,
            "failure_threshold",
            breaker_defaults.failure_threshold,
            _as_int,
            "gateway.router.circuit_breaker.",
        ),
        reset_seconds=_pick(
            breaker_raw,
            "reset_seconds",
            breaker_defaults.reset_seconds,
            _as_float,
            "gateway.router.circuit_breaker.",
        ),
        half_open_successes=_pick(
            breaker_raw,
            "half_open_successes",
            breaker_defaults.half_open_successes,
            _as_int,
            "gateway.router.circuit_breaker.",
        ),
    )
    return RouterConfig(
        strategy=_pick(raw, "strategy", defaults.strategy, _as_strategy, "gateway.router."),
        weights=weights,
        minimum_privacy_tier=_pick(
            raw,
            "minimum_privacy_tier",
            defaults.minimum_privacy_tier,
            _as_privacy_tier,
            "gateway.router.",
        ),
        failover_enabled=_pick(
            raw, "failover_enabled", defaults.failover_enabled, _as_bool, "gateway.router."
        ),
        max_failovers=_pick(
            raw, "max_failovers", defaults.max_failovers, _as_int, "gateway.router."
        ),
        circuit_breaker=breaker,
    )


def _build_gateway(raw: Mapping[str, Any]) -> GatewayConfig:
    """Construct :class:`GatewayConfig`."""
    defaults = GatewayConfig()
    providers_raw: Iterable[Any] = _as_sequence(raw.get("providers"), "gateway.providers")
    providers = tuple(
        _build_provider(_as_mapping(entry, f"gateway.providers[{index}]"), index)
        for index, entry in enumerate(providers_raw)
    )
    return GatewayConfig(
        providers=providers,
        router=_build_router(_as_mapping(raw.get("router"), "gateway.router")),
        request_timeout_seconds=_pick(
            raw,
            "request_timeout_seconds",
            defaults.request_timeout_seconds,
            _as_float,
            "gateway.",
        ),
        stream_chunk_timeout_seconds=_pick(
            raw,
            "stream_chunk_timeout_seconds",
            defaults.stream_chunk_timeout_seconds,
            _as_float,
            "gateway.",
        ),
    )


def load_config(
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> EdenConfig:
    """Load configuration using EDEN's standard precedence chain.

    Args:
        config_path: TOML file to read. Defaults to ``./eden.toml`` when present.
        environ: Environment mapping. Defaults to the process environment.
        overrides: Highest-precedence runtime values, e.g. CLI flags.

    Returns:
        The validated configuration tree.

    Raises:
        ConfigurationError: If any source fails to load or validate.
    """
    loader = ConfigLoader()
    path = config_path if config_path is not None else Path(DEFAULT_CONFIG_FILENAME)
    loader.with_toml(path, required=config_path is not None)
    loader.with_environ(environ)
    if overrides:
        loader.with_overrides(overrides)
    return loader.build()
