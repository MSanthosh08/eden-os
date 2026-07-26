"""Provider construction.

The factory maps a :class:`~eden.config.enums.ProviderKind` onto a builder. It
knows about *protocol families*, not vendors — which is why OpenAI, Groq,
DeepSeek, Together, OpenRouter and Ollama all arrive through one branch, and why
a vendor EDEN has never heard of can be added with
``kind = "custom"`` plus an ``implementation`` import path, with no change to
this file.

Builders are registered rather than hard-coded, so a plugin package may add a
new protocol family at runtime.
"""

from __future__ import annotations

from collections.abc import Callable

from eden.config.enums import ProviderKind
from eden.config.schema import ProviderConfig
from eden.config.secrets import SecretResolver
from eden.errors import InvalidConfigError, PluginLoadError
from eden.gateway.provider import BaseProvider
from eden.gateway.providers.anthropic import AnthropicProvider
from eden.gateway.providers.gemini import GeminiProvider
from eden.gateway.providers.mock import MockProvider
from eden.gateway.providers.openai_compatible import OpenAICompatibleProvider
from eden.logging import get_logger
from eden.transport.base import HttpTransport
from eden.utils.clock import Clock, SystemClock
from eden.utils.imports import import_subclass

_LOGGER = get_logger("gateway.factory")

ProviderBuilder = Callable[["ProviderBuildContext"], BaseProvider]


class ProviderBuildContext:
    """Everything a builder needs to construct one provider.

    Bundling the dependencies keeps builder signatures stable as the system
    grows, so adding a new shared dependency does not break third-party plugins.
    """

    __slots__ = ("clock", "config", "secrets", "transport")

    def __init__(
        self,
        config: ProviderConfig,
        transport: HttpTransport | None,
        secrets: SecretResolver,
        clock: Clock,
    ) -> None:
        """Store the construction dependencies."""
        self.config = config
        self.transport = transport
        self.secrets = secrets
        self.clock = clock

    def require_transport(self) -> HttpTransport:
        """Return the transport, asserting that one was supplied.

        Raises:
            InvalidConfigError: If a network provider was configured without a
                transport, which is a wiring error rather than a user error.
        """
        if self.transport is None:
            raise InvalidConfigError(
                "This provider requires an HTTP transport but none was supplied.",
                context={"provider": self.config.name, "kind": str(self.config.kind)},
            )
        return self.transport


def _build_openai_compatible(context: ProviderBuildContext) -> BaseProvider:
    """Build an adapter for any OpenAI-compatible endpoint."""
    return OpenAICompatibleProvider(
        context.config,
        context.require_transport(),
        secrets=context.secrets,
        clock=context.clock,
    )


def _build_anthropic(context: ProviderBuildContext) -> BaseProvider:
    """Build an Anthropic Messages adapter."""
    return AnthropicProvider(
        context.config,
        context.require_transport(),
        secrets=context.secrets,
        clock=context.clock,
    )


def _build_gemini(context: ProviderBuildContext) -> BaseProvider:
    """Build a Google Gemini adapter."""
    return GeminiProvider(
        context.config,
        context.require_transport(),
        secrets=context.secrets,
        clock=context.clock,
    )


def _build_mock(context: ProviderBuildContext) -> BaseProvider:
    """Build the deterministic in-process provider."""
    return MockProvider(context.config, secrets=context.secrets, clock=context.clock)


def _build_custom(context: ProviderBuildContext) -> BaseProvider:
    """Build a provider named by import path in configuration.

    Raises:
        PluginLoadError: If the class cannot be imported or has an incompatible
            constructor.
    """
    provider_class = import_subclass(
        context.config.implementation,
        BaseProvider,  # type: ignore[type-abstract]  # a plugin base is abstract by design
    )
    try:
        return provider_class(
            context.config,
            secrets=context.secrets,
            clock=context.clock,
        )
    except TypeError as exc:
        raise PluginLoadError(
            "Custom provider constructor is incompatible. Expected " "(config, *, secrets, clock).",
            context={
                "provider": context.config.name,
                "implementation": context.config.implementation,
            },
            cause=exc,
        ) from exc


class ProviderFactory:
    """Builds provider adapters from configuration.

    Example:
        >>> from eden.config.schema import ProviderConfig
        >>> from eden.config.enums import ProviderKind
        >>> factory = ProviderFactory()
        >>> provider = factory.create(
        ...     ProviderConfig(name="local", kind=ProviderKind.MOCK, default_model="m")
        ... )
        >>> provider.name
        'local'
    """

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        secrets: SecretResolver | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the factory with the dependencies shared by all providers.

        Args:
            transport: HTTP transport injected into network providers.
            secrets: Credential resolver.
            clock: Time source.
        """
        self._transport = transport
        self._secrets = secrets or SecretResolver()
        self._clock = clock or SystemClock()
        self._builders: dict[ProviderKind, ProviderBuilder] = {
            ProviderKind.OPENAI_COMPATIBLE: _build_openai_compatible,
            ProviderKind.ANTHROPIC: _build_anthropic,
            ProviderKind.GEMINI: _build_gemini,
            ProviderKind.MOCK: _build_mock,
            ProviderKind.CUSTOM: _build_custom,
        }

    def register_builder(
        self,
        kind: ProviderKind,
        builder: ProviderBuilder,
        *,
        replace: bool = False,
    ) -> None:
        """Register a builder for a protocol family.

        Args:
            kind: Protocol family handled by the builder.
            builder: Callable receiving a build context.
            replace: Permit overwriting an existing builder.

        Raises:
            InvalidConfigError: If the kind is taken and ``replace`` is ``False``.
        """
        if kind in self._builders and not replace:
            raise InvalidConfigError(
                "A builder is already registered for this provider kind.",
                context={"kind": str(kind)},
            )
        self._builders[kind] = builder

    def create(self, config: ProviderConfig) -> BaseProvider:
        """Construct the provider described by ``config``.

        Args:
            config: Declarative provider description.

        Returns:
            A ready-to-use adapter.

        Raises:
            InvalidConfigError: If no builder handles the configured kind.
        """
        builder = self._builders.get(config.kind)
        if builder is None:
            raise InvalidConfigError(
                "No builder is registered for this provider kind.",
                context={
                    "provider": config.name,
                    "kind": str(config.kind),
                    "known": sorted(str(key) for key in self._builders),
                },
            )
        context = ProviderBuildContext(config, self._transport, self._secrets, self._clock)
        provider = builder(context)
        _LOGGER.debug(
            "Provider constructed.",
            extra={"provider": config.name, "kind": str(config.kind)},
        )
        return provider

    def create_all(self, configs: tuple[ProviderConfig, ...]) -> list[BaseProvider]:
        """Construct every provider in ``configs``.

        A provider that fails to construct is logged and skipped rather than
        aborting startup, so one bad credential cannot take the system down.

        Args:
            configs: Provider declarations, typically the enabled subset.

        Returns:
            Successfully constructed providers, in configuration order.
        """
        providers: list[BaseProvider] = []
        for config in configs:
            try:
                providers.append(self.create(config))
            except (InvalidConfigError, PluginLoadError) as exc:
                _LOGGER.error(
                    "Skipping provider that failed to initialise.",
                    extra={"provider": config.name, "error": exc.to_dict()},
                )
        return providers
