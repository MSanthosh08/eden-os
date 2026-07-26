"""Provider health tracking and circuit breaking.

Routing on *declared* cost and latency alone is naive: the cheapest provider is
worthless while it is returning 503s. This module turns observed outcomes into
two signals the router consumes — a continuous health score and a binary
admission decision from the circuit breaker.

Latency is smoothed with an exponentially weighted moving average so that one
slow call does not evict an otherwise good provider, while a sustained
regression is reflected within a handful of requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eden.config.enums import CircuitState, HealthState
from eden.config.schema import CircuitBreakerConfig
from eden.logging import get_logger
from eden.utils.clock import Clock, SystemClock

_LOGGER = get_logger("gateway.health")

EWMA_ALPHA = 0.3
_DEGRADED_SUCCESS_RATE = 0.85
_UNHEALTHY_SUCCESS_RATE = 0.5
_MIN_SAMPLES_FOR_RATE = 4


@dataclass(slots=True)
class ProviderHealth:
    """Mutable health record for one provider.

    Attributes:
        provider: Logical provider name.
        successes: Total successful calls observed.
        failures: Total failed calls observed.
        consecutive_failures: Failures since the last success.
        consecutive_successes: Successes since the last failure.
        ewma_latency_ms: Smoothed observed latency.
        circuit: Current breaker state.
        opened_at: Monotonic timestamp when the breaker last opened.
        last_error_code: Code of the most recent failure.
    """

    provider: str
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    ewma_latency_ms: float = 0.0
    circuit: CircuitState = CircuitState.CLOSED
    opened_at: float = 0.0
    last_error_code: str = ""

    @property
    def total_calls(self) -> int:
        """Return the number of observed calls."""
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        """Return the observed success rate, optimistic before any data."""
        if self.total_calls == 0:
            return 1.0
        return self.successes / self.total_calls

    @property
    def state(self) -> HealthState:
        """Return the coarse health state derived from the counters."""
        if self.circuit is CircuitState.OPEN:
            return HealthState.UNHEALTHY
        if self.total_calls == 0:
            return HealthState.UNKNOWN
        if self.total_calls < _MIN_SAMPLES_FOR_RATE:
            return HealthState.HEALTHY if self.consecutive_failures == 0 else HealthState.DEGRADED
        if self.success_rate < _UNHEALTHY_SUCCESS_RATE:
            return HealthState.UNHEALTHY
        if self.success_rate < _DEGRADED_SUCCESS_RATE:
            return HealthState.DEGRADED
        return HealthState.HEALTHY

    def score(self) -> float:
        """Return a health score in ``[0, 1]``, higher being better."""
        if self.circuit is CircuitState.OPEN:
            return 0.0
        base = self.success_rate
        penalty = min(0.5, 0.1 * self.consecutive_failures)
        if self.circuit is CircuitState.HALF_OPEN:
            penalty += 0.2
        return max(0.0, min(1.0, base - penalty))


@dataclass(slots=True)
class HealthTracker:
    """Records outcomes and answers admission questions for every provider."""

    config: CircuitBreakerConfig
    clock: Clock = field(default_factory=SystemClock)
    _records: dict[str, ProviderHealth] = field(default_factory=dict, init=False)

    def record(self, provider: str) -> ProviderHealth:
        """Return the health record for ``provider``, creating it if absent."""
        record = self._records.get(provider)
        if record is None:
            record = ProviderHealth(provider=provider)
            self._records[provider] = record
        return record

    def snapshot(self) -> dict[str, ProviderHealth]:
        """Return a shallow copy of every health record."""
        return dict(self._records)

    def observe_success(self, provider: str, latency_ms: float) -> None:
        """Record a successful call and update the smoothed latency."""
        record = self.record(provider)
        record.successes += 1
        record.consecutive_failures = 0
        record.consecutive_successes += 1
        record.ewma_latency_ms = (
            latency_ms
            if record.ewma_latency_ms == 0.0
            else EWMA_ALPHA * latency_ms + (1 - EWMA_ALPHA) * record.ewma_latency_ms
        )
        if (
            record.circuit is CircuitState.HALF_OPEN
            and record.consecutive_successes >= self.config.half_open_successes
        ):
            record.circuit = CircuitState.CLOSED
            _LOGGER.info("Circuit closed.", extra={"provider": provider})

    def observe_failure(self, provider: str, *, error_code: str = "") -> None:
        """Record a failed call and open the circuit once the threshold is hit."""
        record = self.record(provider)
        record.failures += 1
        record.consecutive_successes = 0
        record.consecutive_failures += 1
        record.last_error_code = error_code
        if (
            record.circuit is not CircuitState.OPEN
            and record.consecutive_failures >= self.config.failure_threshold
        ):
            record.circuit = CircuitState.OPEN
            record.opened_at = self.clock.monotonic()
            _LOGGER.warning(
                "Circuit opened.",
                extra={
                    "provider": provider,
                    "consecutive_failures": record.consecutive_failures,
                    "error_code": error_code,
                },
            )

    def is_available(self, provider: str) -> bool:
        """Return whether ``provider`` may currently receive traffic.

        An open circuit transitions to half-open once the cool-down elapses,
        which admits a single probe rather than the full load.
        """
        record = self.record(provider)
        if record.circuit is not CircuitState.OPEN:
            return True
        elapsed = self.clock.monotonic() - record.opened_at
        if elapsed >= self.config.reset_seconds:
            record.circuit = CircuitState.HALF_OPEN
            record.consecutive_successes = 0
            _LOGGER.info("Circuit half-opened for probing.", extra={"provider": provider})
            return True
        return False

    def latency_ms(self, provider: str, fallback: float) -> float:
        """Return observed latency for ``provider``, or ``fallback`` if unknown."""
        record = self.record(provider)
        return record.ewma_latency_ms if record.ewma_latency_ms > 0 else fallback

    def reset(self, provider: str | None = None) -> None:
        """Clear health for one provider, or all when ``provider`` is ``None``."""
        if provider is None:
            self._records.clear()
        else:
            self._records.pop(provider, None)
