"""AI Gateway.

The gateway is the boundary between EDEN's business logic and the outside world
of model vendors. Above it, code speaks only :mod:`eden.core.types`. Below it,
each adapter speaks one vendor's dialect.
"""

from __future__ import annotations

from eden.gateway.client import GatewayClient, ProviderHealthSummary
from eden.gateway.factory import ProviderBuildContext, ProviderFactory
from eden.gateway.health import HealthTracker, ProviderHealth
from eden.gateway.provider import BaseProvider
from eden.gateway.router import OmniRouter, RoutingDecision

__all__ = [
    "BaseProvider",
    "GatewayClient",
    "HealthTracker",
    "OmniRouter",
    "ProviderBuildContext",
    "ProviderFactory",
    "ProviderHealth",
    "ProviderHealthSummary",
    "RoutingDecision",
]
