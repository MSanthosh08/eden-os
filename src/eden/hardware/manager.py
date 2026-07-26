"""Device drivers, the fleet manager, and the actuation gate.

Two drivers ship. :class:`SimulatedDevice` is a real, deterministic device that
needs no equipment — it is how the hardware layer is developed and tested, and
how an operator dry-runs a rig before wiring it up. :class:`HttpDevice` talks to
anything exposing a small JSON endpoint, which covers most hobby and industrial
bridges without EDEN taking on a serial dependency.

:class:`DeviceCommandHandler` is the important piece. It is the *only* route
from an intent to a moving part, and it is an
:class:`~eden.execution.handlers.ActionHandler`, which means actuation inherits
the entire Phase 3 pipeline: verification, permission, journalling, rollback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from eden.config.enums import ActionKind, DeviceKind
from eden.config.schema import DeviceConfig, ExecutionConfig, HardwareConfig
from eden.core.registry import Registry
from eden.errors import (
    DeviceCommandError,
    DeviceNotFoundError,
    DeviceSafetyError,
    HardwareError,
    InvalidConfigError,
    PluginLoadError,
    ValidationError,
)
from eden.execution.handlers import ActionHandler
from eden.execution.types import Action, ExecutionResult, Preparation, RollbackPlan
from eden.hardware.device import (
    BaseDevice,
    Device,
    DeviceCommand,
    DeviceStatus,
    Reading,
)
from eden.logging import get_logger
from eden.transport.base import HttpTransport
from eden.utils.clock import Clock, SystemClock
from eden.utils.imports import import_subclass

_LOGGER = get_logger("hardware.manager")

COMPONENT_NAME = "hardware"

DEVICE_PARAMETER = "device"
CHANNEL_PARAMETER = "channel"
VALUE_PARAMETER = "value"


class SimulatedDevice(BaseDevice):
    """An in-process device with deterministic, inspectable behaviour.

    Not a stub. Channels hold real state, writes are reflected in subsequent
    reads, and unwritten channels return a stable value derived from the channel
    name — so a test or a dry run sees plausible, repeatable data.

    Example:
        >>> import asyncio
        >>> from eden.config.schema import DeviceConfig
        >>> async def demo() -> float:
        ...     device = SimulatedDevice(DeviceConfig(name="rig", channels=("servo",)))
        ...     await device.connect()
        ...     reading = await device.send(
        ...         DeviceCommand(device="rig", channel="servo", value=42.0)
        ...     )
        ...     return reading.value
        >>> asyncio.run(demo())
        42.0
    """

    def __init__(self, config: DeviceConfig, *, clock: Clock | None = None) -> None:
        """Initialise the device with an empty channel state."""
        super().__init__(config, clock=clock)
        self._values: dict[str, float] = {}

    def preset(self, channel: str, value: float) -> None:
        """Seed a channel value, for tests and for replaying recorded data."""
        self._values[channel] = value

    async def _open(self) -> None:
        """Connect instantly."""
        return

    async def _close(self) -> None:
        """Disconnect instantly."""
        return

    async def _read_channel(self, channel: str) -> float:
        """Return the stored value, or a stable synthetic one."""
        if channel in self._values:
            return self._values[channel]
        # Deterministic, bounded, and different per channel, so a test can
        # depend on the value without it being a magic constant.
        return round(50.0 + 50.0 * math.sin(float(sum(channel.encode()))), 4)

    async def _write_channel(self, channel: str, value: float) -> float:
        """Store and echo the value."""
        self._values[channel] = value
        return value


class HttpDevice(BaseDevice):
    """Talks to a device exposing a minimal JSON control endpoint.

    The contract is deliberately tiny — ``POST {endpoint}/read`` and
    ``POST {endpoint}/write``, each returning ``{"value": <number>}`` — so a
    bridge can be written in a few lines on almost any microcontroller stack.
    """

    def __init__(
        self,
        config: DeviceConfig,
        transport: HttpTransport,
        *,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the driver with an injected transport."""
        super().__init__(config, clock=clock)
        self._transport = transport

    async def _open(self) -> None:
        """Probe the endpoint to confirm the device answers.

        Raises:
            DeviceCommandError: If the probe is rejected.
        """
        await self._post("status", {})

    async def _close(self) -> None:
        """Nothing to release; the transport is shared and closed elsewhere."""
        return

    async def _read_channel(self, channel: str) -> float:
        """Read one channel over HTTP."""
        return _as_value(await self._post("read", {"channel": channel}), self.name)

    async def _write_channel(self, channel: str, value: float) -> float:
        """Write one channel over HTTP."""
        payload = await self._post("write", {"channel": channel, "value": value})
        return _as_value(payload, self.name)

    async def _post(self, path: str, body: Mapping[str, Any]) -> Any:  # noqa: ANN401
        """Send one request to the device bridge.

        Raises:
            DeviceCommandError: If the device returns a non-2xx status.
        """
        response = await self._transport.post_json(
            f"{self._config.endpoint}/{path}",
            headers={"content-type": "application/json"},
            payload=body,
            timeout=self._config.timeout_seconds,
        )
        if not response.is_success:
            raise DeviceCommandError(
                "Device bridge returned an error status.",
                context={
                    "device": self.name,
                    "status_code": response.status_code,
                    "detail": response.text()[:256],
                },
            )
        return response.json()


def _as_value(payload: object, device: str) -> float:
    """Extract a numeric value from a device response.

    Raises:
        DeviceCommandError: If the payload has no usable number.
    """
    if isinstance(payload, Mapping):
        raw = payload.get("value")
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        if isinstance(raw, int | float):
            return float(raw)
    raise DeviceCommandError(
        "Device response did not contain a numeric value.",
        context={"device": device},
    )


class DeviceManager:
    """Owns the device fleet and its lifecycle."""

    def __init__(
        self,
        config: HardwareConfig,
        devices: Sequence[Device],
    ) -> None:
        """Initialise the manager.

        Args:
            config: Hardware policy.
            devices: Constructed drivers.
        """
        self._config = config
        self._devices = {device.name: device for device in devices}
        self._started = False

    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return COMPONENT_NAME

    @property
    def read_only(self) -> bool:
        """Return whether actuation is globally disabled."""
        return self._config.read_only

    @property
    def devices(self) -> tuple[Device, ...]:
        """Return every registered device."""
        return tuple(self._devices.values())

    def device(self, name: str) -> Device:
        """Return the device registered under ``name``.

        Raises:
            DeviceNotFoundError: If no such device exists.
        """
        found = self._devices.get(name)
        if found is None:
            raise DeviceNotFoundError(
                "No device is registered under this name.",
                context={"device": name, "known": sorted(self._devices)},
            )
        return found

    def statuses(self) -> list[DeviceStatus]:
        """Return a snapshot of every device."""
        return [device.status() for device in self._devices.values()]

    async def start(self) -> None:
        """Connect every device configured to connect on start. Idempotent.

        A device that fails to connect is logged and left faulted rather than
        aborting startup: one broken sensor must not take down the system.
        """
        if self._started:
            return
        self._started = True
        for device in self._devices.values():
            declared = self._config.device(device.name)
            if declared is not None and not declared.connect_on_start:
                continue
            try:
                await device.connect()
            except HardwareError as exc:
                _LOGGER.error(
                    "Device failed to connect; continuing without it.",
                    extra={"device": device.name, "error_code": exc.code},
                )
        _LOGGER.info(
            "Hardware subsystem started.",
            extra={
                "devices": sorted(self._devices),
                "read_only": self._config.read_only,
            },
        )

    async def stop(self) -> None:
        """Disconnect every device. Idempotent and never raises."""
        if not self._started:
            return
        self._started = False
        for device in self._devices.values():
            await device.disconnect()
        _LOGGER.info("Hardware subsystem stopped.")

    async def read(self, device: str, channel: str) -> Reading:
        """Return the current value of ``channel`` on ``device``."""
        return await self.device(device).read(channel)

    async def read_all(self, device: str) -> list[Reading]:
        """Return a reading for every channel on ``device``.

        A channel that fails is skipped rather than failing the sweep.
        """
        target = self.device(device)
        readings: list[Reading] = []
        declared = self._config.device(device)
        for channel in declared.channels if declared else ():
            try:
                readings.append(await target.read(channel))
            except HardwareError as exc:
                _LOGGER.warning(
                    "Channel read failed; omitting it from the sweep.",
                    extra={"device": device, "channel": channel, "error_code": exc.code},
                )
        return readings


class DeviceCommandHandler(ActionHandler):
    """Routes device actuation through the execution pipeline.

    ``prepare`` reads the channel's current value and returns a rollback plan
    that restores it, which makes most device commands reversible and therefore
    eligible for the ordinary policy ladder. A channel that cannot be read back
    is honestly reported as irreversible, and policy will insist on approval.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        manager: DeviceManager,
    ) -> None:
        """Initialise the handler.

        Args:
            config: Execution policy, used for timeouts and output limits.
            manager: Fleet the handler is permitted to drive.
        """
        super().__init__(config)
        self._manager = manager

    @property
    def kind(self) -> ActionKind:
        """Return the action kind this handler performs."""
        return ActionKind.DEVICE_COMMAND

    async def prepare(self, action: Action) -> Preparation:
        """Read the current value so the command can be undone."""
        command = _command_from(action)
        try:
            current = await self._manager.read(command.device, command.channel)
        except (HardwareError, ValidationError) as exc:
            return Preparation(
                rollback=None,
                notes={
                    "device": command.device,
                    "channel": command.channel,
                    "reason": f"the channel could not be read back ({exc.code})",
                },
            )
        return Preparation(
            rollback=RollbackPlan(
                steps=(
                    Action(
                        kind=ActionKind.DEVICE_COMMAND,
                        summary=(
                            f"Restore {command.device}.{command.channel} " f"to {current.value}."
                        ),
                        parameters={
                            DEVICE_PARAMETER: command.device,
                            CHANNEL_PARAMETER: command.channel,
                            VALUE_PARAMETER: current.value,
                        },
                        actor=action.actor,
                        namespace=action.namespace,
                        metadata={"rollback_for": action.id},
                    ),
                ),
                description=(f"Return {command.device}.{command.channel} to {current.value}."),
            ),
            notes={
                "device": command.device,
                "channel": command.channel,
                "previous_value": current.value,
                "existed": True,
            },
        )

    async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
        """Send the command to the device.

        Raises:
            DeviceSafetyError: If actuation is globally disabled, or the value
                is outside the device's configured bounds.
        """
        del preparation
        command = _command_from(action)
        if self._manager.read_only:
            raise DeviceSafetyError(
                "Hardware is in read-only mode; no command may be sent.",
                context={"device": command.device, "channel": command.channel},
            )
        device = self._manager.device(command.device)
        reading = await device.send(command)
        return ExecutionResult(
            succeeded=True,
            output=f"{command.device}.{command.channel} = {reading.value}",
            detail=reading.to_dict(),
        )


def _command_from(action: Action) -> DeviceCommand:
    """Build a :class:`DeviceCommand` from an action's parameters.

    Raises:
        ValidationError: If a parameter is missing or the value is not numeric.
    """
    raw = action.parameter(VALUE_PARAMETER)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValidationError(
            "Device command value must be a number.",
            context={"action": action.id, "value": str(raw)},
        )
    return DeviceCommand(
        device=action.text_parameter(DEVICE_PARAMETER),
        channel=action.text_parameter(CHANNEL_PARAMETER),
        value=float(raw),
    )


def build_device(
    config: DeviceConfig,
    *,
    transport: HttpTransport | None = None,
    clock: Clock | None = None,
) -> Device:
    """Construct the driver described by ``config``.

    Args:
        config: Device declaration.
        transport: HTTP transport, required for :attr:`DeviceKind.HTTP`.
        clock: Time source.

    Returns:
        A ready-to-connect driver.

    Raises:
        InvalidConfigError: If the kind is unhandled or its dependencies are
            missing.
    """
    resolved_clock = clock or SystemClock()
    if config.kind is DeviceKind.SIMULATED:
        return SimulatedDevice(config, clock=resolved_clock)
    if config.kind is DeviceKind.HTTP:
        if transport is None:
            raise InvalidConfigError(
                "HTTP devices require an HTTP transport but none was supplied.",
                context={"device": config.name},
            )
        return HttpDevice(config, transport, clock=resolved_clock)
    if config.kind is DeviceKind.CUSTOM:
        driver = import_subclass(config.implementation, BaseDevice)  # type: ignore[type-abstract]
        return driver(config, clock=resolved_clock)
    raise InvalidConfigError(
        "No driver is registered for this device kind.",
        context={"device": config.name, "kind": str(config.kind)},
    )


def build_device_manager(
    config: HardwareConfig,
    *,
    transport: HttpTransport | None = None,
    clock: Clock | None = None,
) -> DeviceManager:
    """Construct the whole fleet.

    A device that fails to construct is logged and skipped rather than aborting
    startup, matching how the gateway treats a bad provider.

    Args:
        config: Hardware policy and device declarations.
        transport: HTTP transport for network devices.
        clock: Time source.

    Returns:
        A ready-to-start manager.
    """
    registry: Registry[Device] = Registry("device")
    devices: list[Device] = []
    for declaration in config.enabled_devices():
        try:
            device = build_device(declaration, transport=transport, clock=clock)
        except (InvalidConfigError, HardwareError, PluginLoadError) as exc:
            _LOGGER.error(
                "Skipping device that failed to initialise.",
                extra={"device": declaration.name, "error": exc.to_dict()},
            )
            continue
        registry.register(declaration.name, lambda d=device: d)  # type: ignore[misc]
        devices.append(device)
    return DeviceManager(config, devices)
