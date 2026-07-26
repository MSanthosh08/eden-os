"""Automation subsystem.

A rule pairs a trigger — *when* — with a payload: *what*. The payload is always
either an agent task or an execution action, never a callable, so a rule is
data that can be listed, inspected, disabled and audited, and cannot smuggle
arbitrary code past the execution pipeline.

``automation.enabled`` defaults to ``False``: nothing fires on a schedule until
an operator says so.
"""

from __future__ import annotations

from eden.automation.scheduler import (
    COMPONENT_NAME,
    AutomationRun,
    AutomationScheduler,
    DailyTrigger,
    EventTrigger,
    IntervalTrigger,
    ManualTrigger,
    Rule,
    Trigger,
    build_scheduler,
    daily_at,
    every,
    every_minutes,
    on_event,
)

__all__ = [
    "COMPONENT_NAME",
    "AutomationRun",
    "AutomationScheduler",
    "DailyTrigger",
    "EventTrigger",
    "IntervalTrigger",
    "ManualTrigger",
    "Rule",
    "Trigger",
    "build_scheduler",
    "daily_at",
    "every",
    "every_minutes",
    "on_event",
]
