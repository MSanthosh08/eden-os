"""Routing strategies.

Each signal the router cares about — cost, latency, health, privacy, operator
preference — is an independent :class:`Scorer` returning a normalised value in
``[0, 1]`` where higher is better. A :class:`Strategy` is nothing more than a
set of weights over those scorers.

Normalisation happens *within* the candidate set for each request, which means
the scores stay meaningful whether a deployment has two providers or twenty, and
whether costs are measured in cents or thousandths of a cent.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from eden.config.enums import RoutingStrategyName
from eden.config.schema import RouterWeights
from eden.core.types import ChatRequest
from eden.errors import InvalidConfigError
from eden.gateway.health import HealthTracker
from eden.gateway.provider import BaseProvider

_NEUTRAL_SCORE = 0.5
_MAX_PRIVACY_TIER = 3.0
_PREFERENCE_DECAY = 0.5


@dataclass(frozen=True, slots=True)
class ScoredProvider:
    """A candidate with its computed score and per-signal breakdown.

    Attributes:
        provider: The candidate adapter.
        score: Final weighted score. Higher is better.
        breakdown: Per-signal contributions, retained for explainability.
    """

    provider: BaseProvider
    score: float
    breakdown: Mapping[str, float]


class Scorer(abc.ABC):
    """Scores every candidate on one signal."""

    @property
    @abc.abstractmethod
    def signal(self) -> str:
        """Return the signal name used in the score breakdown."""

    @abc.abstractmethod
    def score(
        self,
        request: ChatRequest,
        candidates: Sequence[BaseProvider],
        health: HealthTracker,
    ) -> dict[str, float]:
        """Return a normalised score per provider name.

        Args:
            request: The pending request.
            candidates: Providers already filtered for eligibility.
            health: Observed runtime health.

        Returns:
            Mapping of provider name to a score in ``[0, 1]``.
        """


def _normalise_lower_is_better(values: Mapping[str, float]) -> dict[str, float]:
    """Invert and rescale a cost-like signal so that higher is better."""
    if not values:
        return {}
    finite = [value for value in values.values() if value != float("inf")]
    if not finite:
        return dict.fromkeys(values, _NEUTRAL_SCORE)
    lowest = min(finite)
    highest = max(finite)
    if highest == lowest:
        return dict.fromkeys(values, 1.0)
    span = highest - lowest
    return {
        name: 0.0 if value == float("inf") else 1.0 - ((value - lowest) / span)
        for name, value in values.items()
    }


class CostScorer(Scorer):
    """Prefers providers with the lowest estimated spend for this request."""

    @property
    def signal(self) -> str:
        """Return the signal name."""
        return "cost"

    def score(
        self,
        request: ChatRequest,
        candidates: Sequence[BaseProvider],
        health: HealthTracker,  # noqa: ARG002 - signature fixed by the Scorer contract
    ) -> dict[str, float]:
        """Score candidates by estimated cost, cheapest first."""
        estimates = {provider.name: provider.estimate_cost(request) for provider in candidates}
        return _normalise_lower_is_better(estimates)


class LatencyScorer(Scorer):
    """Prefers providers that have been fastest recently.

    Observed latency wins over the configured prior as soon as any data exists,
    so a vendor that degrades in production loses traffic without a config edit.
    """

    @property
    def signal(self) -> str:
        """Return the signal name."""
        return "latency"

    def score(
        self,
        request: ChatRequest,
        candidates: Sequence[BaseProvider],
        health: HealthTracker,
    ) -> dict[str, float]:
        """Score candidates by expected latency, fastest first."""
        estimates = {
            provider.name: health.latency_ms(
                provider.name,
                provider.expected_latency_ms(request),
            )
            for provider in candidates
        }
        return _normalise_lower_is_better(estimates)


class HealthScorer(Scorer):
    """Prefers providers with a clean recent record."""

    @property
    def signal(self) -> str:
        """Return the signal name."""
        return "health"

    def score(
        self,
        request: ChatRequest,  # noqa: ARG002 - signature fixed by the Scorer contract
        candidates: Sequence[BaseProvider],
        health: HealthTracker,
    ) -> dict[str, float]:
        """Score candidates by observed reliability."""
        return {provider.name: health.record(provider.name).score() for provider in candidates}


class PrivacyScorer(Scorer):
    """Prefers stronger data-residency guarantees."""

    @property
    def signal(self) -> str:
        """Return the signal name."""
        return "privacy"

    def score(
        self,
        request: ChatRequest,  # noqa: ARG002 - signature fixed by the Scorer contract
        candidates: Sequence[BaseProvider],
        health: HealthTracker,  # noqa: ARG002 - signature fixed by the Scorer contract
    ) -> dict[str, float]:
        """Score candidates by privacy tier, most private first."""
        return {
            provider.name: float(provider.privacy_tier.value) / _MAX_PRIVACY_TIER
            for provider in candidates
        }


class PreferenceScorer(Scorer):
    """Honours the operator's per-provider weight and the request's hint."""

    @property
    def signal(self) -> str:
        """Return the signal name."""
        return "preference"

    def score(
        self,
        request: ChatRequest,
        candidates: Sequence[BaseProvider],
        health: HealthTracker,  # noqa: ARG002 - signature fixed by the Scorer contract
    ) -> dict[str, float]:
        """Score candidates by configured weight, boosting an explicit request."""
        weights = {provider.name: provider.config.weight for provider in candidates}
        highest = max(weights.values(), default=1.0)
        scores = {
            name: (weight / highest if highest > 0 else _NEUTRAL_SCORE)
            for name, weight in weights.items()
        }
        if request.preferred_provider in scores:
            # Award the hinted provider a full score and halve the rest.
            # Assigning 1.0 alone would be a no-op whenever every provider
            # carries the same configured weight, which is the common case.
            return {
                name: 1.0 if name == request.preferred_provider else value * _PREFERENCE_DECAY
                for name, value in scores.items()
            }
        return scores


class Strategy(abc.ABC):
    """Combines scorers into a single ordering."""

    def __init__(self, scorers: Sequence[Scorer]) -> None:
        """Store the scorers this strategy consults."""
        self._scorers = tuple(scorers)

    @property
    @abc.abstractmethod
    def name(self) -> RoutingStrategyName:
        """Return the configuration name of this strategy."""

    @abc.abstractmethod
    def weight_for(self, signal: str) -> float:
        """Return the relative weight applied to ``signal``."""

    def rank(
        self,
        request: ChatRequest,
        candidates: Sequence[BaseProvider],
        health: HealthTracker,
    ) -> list[ScoredProvider]:
        """Return candidates ordered best-first.

        Args:
            request: The pending request.
            candidates: Providers already filtered for eligibility.
            health: Observed runtime health.

        Returns:
            Scored candidates sorted by descending score. Ties break on the
            provider name so that ordering is deterministic and testable.
        """
        if not candidates:
            return []
        signal_scores = {
            scorer.signal: scorer.score(request, candidates, health) for scorer in self._scorers
        }
        total_weight = sum(self.weight_for(signal) for signal in signal_scores) or 1.0

        results: list[ScoredProvider] = []
        for provider in candidates:
            breakdown = {
                signal: scores.get(provider.name, _NEUTRAL_SCORE)
                for signal, scores in signal_scores.items()
            }
            weighted = sum(self.weight_for(signal) * value for signal, value in breakdown.items())
            results.append(
                ScoredProvider(
                    provider=provider,
                    score=weighted / total_weight,
                    breakdown=breakdown,
                )
            )
        results.sort(key=lambda scored: (-scored.score, scored.provider.name))
        return results


class WeightedStrategy(Strategy):
    """Balances every signal using operator-supplied weights."""

    def __init__(
        self,
        weights: RouterWeights,
        *,
        name: RoutingStrategyName = RoutingStrategyName.BALANCED,
        scorers: Sequence[Scorer] | None = None,
    ) -> None:
        """Initialise the strategy.

        Args:
            weights: Relative importance of each signal.
            name: Configuration name reported by :attr:`name`.
            scorers: Override for the default scorer set.
        """
        super().__init__(
            scorers
            if scorers is not None
            else (
                CostScorer(),
                LatencyScorer(),
                HealthScorer(),
                PrivacyScorer(),
                PreferenceScorer(),
            )
        )
        self._weights = weights
        self._name = name

    @property
    def name(self) -> RoutingStrategyName:
        """Return the configuration name of this strategy."""
        return self._name

    def weight_for(self, signal: str) -> float:
        """Return the weight configured for ``signal``."""
        mapping = {
            "cost": self._weights.cost,
            "latency": self._weights.latency,
            "health": self._weights.health,
            "privacy": self._weights.privacy,
            "preference": self._weights.preference,
        }
        return mapping.get(signal, 0.0)


class RoundRobinStrategy(Strategy):
    """Distributes traffic evenly, subject only to health.

    Useful for load testing and for deployments where every provider is
    equivalent and the goal is even wear rather than optimisation.
    """

    def __init__(self) -> None:
        """Initialise the strategy with a health scorer only."""
        super().__init__((HealthScorer(),))
        self._cursor = 0

    @property
    def name(self) -> RoutingStrategyName:
        """Return the configuration name of this strategy."""
        return RoutingStrategyName.ROUND_ROBIN

    def weight_for(self, signal: str) -> float:
        """Return the weight configured for ``signal``."""
        return 1.0 if signal == "health" else 0.0

    def rank(
        self,
        request: ChatRequest,
        candidates: Sequence[BaseProvider],
        health: HealthTracker,
    ) -> list[ScoredProvider]:
        """Return candidates rotated by an advancing cursor."""
        ranked = super().rank(request, candidates, health)
        if not ranked:
            return ranked
        offset = self._cursor % len(ranked)
        self._cursor += 1
        return ranked[offset:] + ranked[:offset]


def build_strategy(
    name: RoutingStrategyName,
    weights: RouterWeights,
) -> Strategy:
    """Construct the strategy named in configuration.

    Named profiles are expressed as preset weight vectors over the same
    scorers, so a new profile is a data change rather than a new class.

    Args:
        name: Strategy selected in configuration.
        weights: Operator weights, used by the balanced profile.

    Returns:
        A ready-to-use strategy.

    Raises:
        InvalidConfigError: If the strategy name is unhandled.
    """
    if name is RoutingStrategyName.BALANCED:
        return WeightedStrategy(weights, name=name)
    if name is RoutingStrategyName.CHEAPEST:
        return WeightedStrategy(
            RouterWeights(cost=8.0, latency=0.5, health=2.0, privacy=0.0, preference=0.5),
            name=name,
        )
    if name is RoutingStrategyName.FASTEST:
        return WeightedStrategy(
            RouterWeights(cost=0.5, latency=8.0, health=2.0, privacy=0.0, preference=0.5),
            name=name,
        )
    if name is RoutingStrategyName.PRIVACY_FIRST:
        return WeightedStrategy(
            RouterWeights(cost=0.5, latency=0.5, health=2.0, privacy=8.0, preference=0.5),
            name=name,
        )
    if name is RoutingStrategyName.ROUND_ROBIN:
        return RoundRobinStrategy()
    raise InvalidConfigError(
        "Unhandled routing strategy.",
        context={"key": "gateway.router.strategy", "value": str(name)},
    )
