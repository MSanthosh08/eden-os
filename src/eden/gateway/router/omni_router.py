"""The Omni Router.

Selection happens in two clearly separated phases.

**Filter** applies hard constraints — capability coverage, model availability,
the privacy floor and circuit-breaker admission. A provider that fails any of
these is not a worse choice, it is *not a choice at all*, and no weighting can
resurrect it.

**Rank** applies soft preferences through a :class:`~eden.gateway.router.strategy.Strategy`.

Keeping these apart is what stops a cheap-but-non-compliant provider from ever
winning on price, and it makes routing decisions explainable: every selection
records why each candidate was excluded or how it scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from eden.config.enums import PrivacyTier
from eden.config.schema import RouterConfig
from eden.core.types import ChatRequest
from eden.gateway.health import HealthTracker
from eden.gateway.provider import BaseProvider
from eden.gateway.router.strategy import ScoredProvider, Strategy, build_strategy
from eden.logging import get_logger

_LOGGER = get_logger("gateway.router")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The full, explainable outcome of one selection.

    Attributes:
        ranked: Eligible candidates ordered best-first.
        excluded: Provider name mapped to the reason it was filtered out.
    """

    ranked: tuple[ScoredProvider, ...] = ()
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def has_candidates(self) -> bool:
        """Return whether any provider survived filtering."""
        return bool(self.ranked)

    def provider_names(self) -> tuple[str, ...]:
        """Return the ordered names of eligible providers."""
        return tuple(scored.provider.name for scored in self.ranked)


class OmniRouter:
    """Chooses providers from availability, capability, privacy, cost and health."""

    def __init__(
        self,
        config: RouterConfig,
        health: HealthTracker,
        *,
        strategy: Strategy | None = None,
    ) -> None:
        """Initialise the router.

        Args:
            config: Routing behaviour from configuration.
            health: Shared health tracker consulted for admission and scoring.
            strategy: Override for the configured strategy, used by tests and
                by callers that need per-tenant routing profiles.
        """
        self._config = config
        self._health = health
        self._strategy = strategy or build_strategy(config.strategy, config.weights)

    @property
    def strategy_name(self) -> str:
        """Return the active strategy name."""
        return str(self._strategy.name)

    def decide(
        self,
        request: ChatRequest,
        providers: Sequence[BaseProvider],
    ) -> RoutingDecision:
        """Return the full routing decision for ``request``.

        Args:
            request: The pending request.
            providers: Every configured and enabled provider.

        Returns:
            The ranked shortlist together with exclusion reasons.
        """
        floor = self._effective_floor(request)
        eligible: list[BaseProvider] = []
        excluded: dict[str, str] = {}

        for provider in providers:
            reason = self._exclusion_reason(request, provider, floor)
            if reason is None:
                eligible.append(provider)
            else:
                excluded[provider.name] = reason

        ranked = tuple(self._strategy.rank(request, eligible, self._health))
        decision = RoutingDecision(ranked=ranked, excluded=excluded)

        _LOGGER.debug(
            "Routing decision computed.",
            extra={
                "strategy": self.strategy_name,
                "eligible": list(decision.provider_names()),
                "excluded": excluded,
                "privacy_floor": floor.name.lower(),
            },
        )
        return decision

    def select(
        self,
        request: ChatRequest,
        providers: Sequence[BaseProvider],
    ) -> list[BaseProvider]:
        """Return the ordered shortlist of providers to attempt.

        The list is truncated to one entry plus ``max_failovers`` so the caller
        cannot accidentally walk the entire fleet on a pathological request.

        Args:
            request: The pending request.
            providers: Every configured and enabled provider.

        Returns:
            Providers ordered best-first, possibly empty.
        """
        decision = self.decide(request, providers)
        ordered = [scored.provider for scored in decision.ranked]
        if not self._config.failover_enabled:
            return ordered[:1]
        return ordered[: 1 + self._config.max_failovers]

    def _effective_floor(self, request: ChatRequest) -> PrivacyTier:
        """Return the stricter of the global and per-request privacy floors."""
        if request.minimum_privacy_tier is None:
            return self._config.minimum_privacy_tier
        return max(self._config.minimum_privacy_tier, request.minimum_privacy_tier)

    def _exclusion_reason(
        self,
        request: ChatRequest,
        provider: BaseProvider,
        floor: PrivacyTier,
    ) -> str | None:
        """Return why ``provider`` cannot serve ``request``, or ``None``."""
        if not provider.config.enabled:
            return "disabled"
        if provider.privacy_tier < floor:
            return "privacy_tier_below_floor"
        missing = request.required_capabilities - provider.capabilities
        if missing:
            return f"missing_capabilities:{','.join(sorted(c.value for c in missing))}"
        if not provider.supports(request):
            return "model_unavailable"
        if not self._health.is_available(provider.name):
            return "circuit_open"
        return None
