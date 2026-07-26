"""Provider selection.

The router never talks to a vendor. It answers one question — *in what order
should these providers be attempted?* — and the gateway client does the calling.
"""

from __future__ import annotations

from eden.gateway.router.omni_router import OmniRouter, RoutingDecision
from eden.gateway.router.strategy import (
    CostScorer,
    HealthScorer,
    LatencyScorer,
    PreferenceScorer,
    PrivacyScorer,
    RoundRobinStrategy,
    ScoredProvider,
    Scorer,
    Strategy,
    WeightedStrategy,
    build_strategy,
)

__all__ = [
    "CostScorer",
    "HealthScorer",
    "LatencyScorer",
    "OmniRouter",
    "PreferenceScorer",
    "PrivacyScorer",
    "RoundRobinStrategy",
    "RoutingDecision",
    "ScoredProvider",
    "Scorer",
    "Strategy",
    "WeightedStrategy",
    "build_strategy",
]
