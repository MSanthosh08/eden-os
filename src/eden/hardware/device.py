"""Device contract and safety envelope.

Hardware is where a mistake stops being recoverable. A wrong file can be
restored from the rollback plan; a wrong servo angle can break a machine or a
person. Two decisions follow from that.

**Reads are free, writes are gated.** :meth:`BaseDevice.read` is a direct call.
Actuation is *not* reachable from outside this package except as an
:class:`~eden.execution.types.Action` submitted to the Phase 3 pipeline, which
means every command is verified, authorised and journalled by machinery that
already exists and is already tested.

**Limits are configuration, not code.** A driver cannot decide its own safe
range. Bounds come from :class:`~eden.config.schema.DeviceConfig`, so an
operator can tighten them without a deploy and verification can inspect them
before anything is sent.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from eden.config.enums import DeviceState
from eden.config.schema import DeviceConfig
from eden.errors import (
    DeviceCommandError,
    DeviceSafetyError,
    DeviceUnavailableError,
    ValidationError,
)
from eden.logging import get_logger, timed_block
from eden.utils.async_tools import with_timeout
from eden.utils.clock import Clock, SystemClock
from eden.utils.ratelimit import TokenBucket


@dataclass(frozen=True, slots=True)
class Reading:
    """One observation taken from a device.

    Attributes:
        device: Logical device name.
        channel: Channel the value came from.
        value: Observed value.
        unit: Unit of measurement, for display only.
        taken_at: When the observation was made.
    """

    device: str
    channel: str
    value: float
    unit: str = ""
    taken_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "device": self.device,
            "channel": self.channel,
            "value": self.value,
            "unit": self.unit,
            "taken_at": self.taken_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class DeviceCommand:
    """An instruction to change a device's state.

    Attributes:
        device: Logical device name.
        channel: Channel to drive.
        value: Target value.
        metadata: Non-transmitted annotations.
    """

    device: str
    channel: str
    value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the command.

        Raises:
            ValidationError: If the device or channel is unnamed.
        """
        if not self.device.strip():
            raise ValidationError("Device command must name a device.")
        if not self.channel.strip():
            raise ValidationError("Device command must name a channel.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {"device": self.device, "channel": self.channel, "value": self.value}


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    """A snapshot of one device.

    Attributes:
        name: Logical device name.
        state: Current connection state.
        detail: Short human-readable explanation.
        channels: Channels this device exposes.
        last_error: Code of the most recent failure.
        commands_sent: Successful commands since startup.
    """

    name: str
    state: DeviceState
    detail: str = ""
    channels: tuple[str, ...] = ()
    last_error: str = ""
    commands_sent: int = 0

    @property
    def ready(self) -> bool:
        """Return whether the device can currently serve requests."""
        return self.state is DeviceState.READY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "state": self.state.value,
            "detail": self.detail,
            "channels": list(self.channels),
            "last_error": self.last_error,
            "commands_sent": self.commands_sent,
        }


@runtime_checkable
class Device(Protocol):
    """A physical or simulated device."""

    @property
    def name(self) -> str:
        """Return the logical device name."""
        ...

    @property
    def state(self) -> DeviceState:
        """Return the current connection state."""
        ...

    async def connect(self) -> None:
        """Establish a connection. Must be idempotent."""
        ...

    async def disconnect(self) -> None:
        """Release the connection. Must be idempotent and must not raise."""
        ...

    async def read(self, channel: str) -> Reading:
        """Return the current value of ``channel``."""
        ...

    async def send(self, command: DeviceCommand) -> Reading:
        """Apply ``command`` and return the resulting reading."""
        ...

    def status(self) -> DeviceStatus:
        """Return a snapshot of this device."""
        ...


class BaseDevice(abc.ABC):
    """Common behaviour for every device driver.

    Owns state tracking, rate limiting, timeout enforcement, bounds checking,
    logging and error translation. A driver implements only the three
    transport-specific operations: connect, read one channel, write one channel.
    """

    def __init__(self, config: DeviceConfig, *, clock: Clock | None = None) -> None:
        """Initialise the driver from its configuration slice.

        Args:
            config: Declarative device description including safety limits.
            clock: Time source used by the command rate limiter.
        """
        self._config = config
        self._clock = clock or SystemClock()
        self._logger = get_logger(f"hardware.{config.name}")
        self._state = DeviceState.DISCONNECTED
        self._last_error = ""
        self._commands_sent = 0
        self._bucket = TokenBucket(
            requests_per_minute=config.max_commands_per_minute,
            clock=self._clock,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Return the logical device name."""
        return self._config.name

    @property
    def config(self) -> DeviceConfig:
        """Return this device's configuration slice."""
        return self._config

    @property
    def state(self) -> DeviceState:
        """Return the current connection state."""
        return self._state

    @property
    def channels(self) -> tuple[str, ...]:
        """Return the channels this device exposes."""
        return self._config.channels

    def status(self) -> DeviceStatus:
        """Return a snapshot of this device."""
        return DeviceStatus(
            name=self.name,
            state=self._state,
            detail=self._detail(),
            channels=self.channels,
            last_error=self._last_error,
            commands_sent=self._commands_sent,
        )

    def _detail(self) -> str:
        """Return a short human-readable state description."""
        if self._state is DeviceState.FAULTED and self._last_error:
            return f"Faulted: {self._last_error}"
        return self._state.value

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Establish a connection. Idempotent.

        Raises:
            DeviceUnavailableError: If the connection cannot be established.
        """
        if self._state in (DeviceState.READY, DeviceState.BUSY):
            return
        self._state = DeviceState.CONNECTING
        try:
            await with_timeout(
                self._open(),
                self._config.timeout_seconds,
                on_timeout=lambda: DeviceUnavailableError(
                    "Device did not respond within its connection budget.",
                    context={"device": self.name},
                ),
            )
        except Exception as exc:
            self._state = DeviceState.FAULTED
            self._last_error = type(exc).__name__
            self._logger.error(
                "Device failed to connect.",
                extra={"device": self.name, "error_type": type(exc).__name__},
            )
            raise
        self._state = DeviceState.READY
        self._last_error = ""
        self._logger.info("Device connected.", extra={"device": self.name})

    async def disconnect(self) -> None:
        """Release the connection. Idempotent and never raises."""
        if self._state is DeviceState.DISCONNECTED:
            return
        try:
            await self._close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            self._logger.warning(
                "Device did not disconnect cleanly.",
                extra={"device": self.name, "error_type": type(exc).__name__},
            )
        self._state = DeviceState.DISCONNECTED
        self._logger.info("Device disconnected.", extra={"device": self.name})

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    async def read(self, channel: str) -> Reading:
        """Return the current value of ``channel``.

        Reads are not gated by the execution pipeline: observing a device
        changes nothing, and requiring approval to look at a sensor would make
        the system unusable without making it safer.

        Raises:
            DeviceUnavailableError: If the device is not ready.
            DeviceCommandError: If the channel is unknown or the read fails.
        """
        self._require_ready()
        self._require_channel(channel)
        with timed_block(self._logger, "device.read", device=self.name, channel=channel):
            try:
                value = await with_timeout(
                    self._read_channel(channel),
                    self._config.timeout_seconds,
                    on_timeout=lambda: DeviceCommandError(
                        "Device read exceeded its budget.",
                        context={"device": self.name, "channel": channel},
                    ),
                )
            except Exception as exc:
                self._fault(exc)
                raise
        return Reading(device=self.name, channel=channel, value=value)

    async def send(self, command: DeviceCommand) -> Reading:
        """Apply ``command`` after checking it against the safety envelope.

        This method is intentionally *not* the public way to actuate. The
        hardware action handler calls it, and that handler is only reachable
        through the execution pipeline.

        Raises:
            DeviceSafetyError: If the value is outside the configured bounds.
            DeviceUnavailableError: If the device is not ready.
            DeviceCommandError: If the channel is unknown or the write fails.
        """
        self._require_ready()
        self._require_channel(command.channel)
        self._require_within_limits(command)
        await self._bucket.acquire()

        previous = self._state
        self._state = DeviceState.BUSY
        try:
            with timed_block(
                self._logger,
                "device.send",
                device=self.name,
                channel=command.channel,
            ):
                value = await with_timeout(
                    self._write_channel(command.channel, command.value),
                    self._config.timeout_seconds,
                    on_timeout=lambda: DeviceCommandError(
                        "Device command exceeded its budget.",
                        context={"device": self.name, "channel": command.channel},
                    ),
                )
        except Exception as exc:
            self._fault(exc)
            raise
        else:
            self._state = previous
            self._commands_sent += 1
        return Reading(device=self.name, channel=command.channel, value=value)

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    def _require_ready(self) -> None:
        """Raise unless the device can serve a request.

        Raises:
            DeviceUnavailableError: If the device is not ready.
        """
        if self._state is not DeviceState.READY:
            raise DeviceUnavailableError(
                "Device is not ready.",
                context={"device": self.name, "state": self._state.value},
            )

    def _require_channel(self, channel: str) -> None:
        """Raise if ``channel`` is not declared by this device.

        A device that declares no channels accepts any, which supports drivers
        whose channel set is discovered at runtime.

        Raises:
            DeviceCommandError: If the channel is not declared.
        """
        if self._config.channels and channel not in self._config.channels:
            raise DeviceCommandError(
                "Device does not expose this channel.",
                context={
                    "device": self.name,
                    "channel": channel,
                    "known": list(self._config.channels),
                },
            )

    def _require_within_limits(self, command: DeviceCommand) -> None:
        """Raise if the command falls outside its declared bounds.

        Raises:
            DeviceSafetyError: If the value is out of range.
        """
        bounds = self._config.limit_for(command.channel)
        if bounds is None:
            return
        lowest, highest = bounds
        if not lowest <= command.value <= highest:
            raise DeviceSafetyError(
                "Command value is outside the configured safe range.",
                context={
                    "device": self.name,
                    "channel": command.channel,
                    "value": command.value,
                    "minimum": lowest,
                    "maximum": highest,
                },
            )

    def _fault(self, error: BaseException) -> None:
        """Mark the device faulted and record why."""
        self._state = DeviceState.FAULTED
        self._last_error = type(error).__name__

    # ------------------------------------------------------------------
    # Driver hooks
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def _open(self) -> None:
        """Establish the underlying connection."""

    @abc.abstractmethod
    async def _close(self) -> None:
        """Release the underlying connection."""

    @abc.abstractmethod
    async def _read_channel(self, channel: str) -> float:
        """Return the raw current value of ``channel``."""

    @abc.abstractmethod
    async def _write_channel(self, channel: str, value: float) -> float:
        """Apply ``value`` to ``channel`` and return the resulting value."""
