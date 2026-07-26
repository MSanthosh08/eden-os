"""Hardware subsystem.

Reads are direct calls: observing a device changes nothing, and requiring
approval to look at a sensor would make the system unusable without making it
safer.

Actuation is not directly reachable. The only route to a moving part is a
:class:`~eden.execution.types.Action` of kind ``device_command`` submitted to
the Phase 3 execution pipeline, so every command is verified against the
device's configured safety envelope, authorised, journalled, and — because
``prepare`` reads the channel's current value first — usually reversible.

``hardware.enabled`` defaults to ``False``. Software that can move things should
not be able to move things the moment it is installed.
"""

from __future__ import annotations

from eden.hardware.device import (
    BaseDevice,
    Device,
    DeviceCommand,
    DeviceStatus,
    Reading,
)
from eden.hardware.manager import (
    CHANNEL_PARAMETER,
    COMPONENT_NAME,
    DEVICE_PARAMETER,
    VALUE_PARAMETER,
    DeviceCommandHandler,
    DeviceManager,
    HttpDevice,
    SimulatedDevice,
    build_device,
    build_device_manager,
)

__all__ = [
    "CHANNEL_PARAMETER",
    "COMPONENT_NAME",
    "DEVICE_PARAMETER",
    "VALUE_PARAMETER",
    "BaseDevice",
    "Device",
    "DeviceCommand",
    "DeviceCommandHandler",
    "DeviceManager",
    "DeviceStatus",
    "HttpDevice",
    "Reading",
    "SimulatedDevice",
    "build_device",
    "build_device_manager",
]
