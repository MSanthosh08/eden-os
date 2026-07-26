"""Automation.

A rule pairs a :class:`Trigger` — *when* — with a payload: *what*. The payload
is always either an agent :class:`~eden.agents.types.Task` or an
:class:`~eden.execution.types.Action`, never a callable. That matters: a rule is
data, so it can be listed, inspected, disabled and audited, and it cannot
smuggle arbitrary code past the execution pipeline.

Time comes from an injected :class:`~eden.utils.clock.Clock`, so the whole
scheduler is testable without waiting. A test advances a
:class:`~eden.utils.clock.ManualClock` and ticks; nothing sleeps.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from eden.agents.types import Task
from eden.config.enums import AutomationStatus, TriggerKind
from eden.config.schema import AutomationConfig
from eden.errors import (
    AutomationError,
    EdenError,
    InvalidRuleError,
    RuleNotFoundError,
)
from eden.execution.types import Action
from eden.logging import correlation_scope, get_logger, timed_block
from eden.utils.clock import Clock, SystemClock
from eden.utils.ids import new_id

_LOGGER = get_logger("automation.scheduler")

COMPONENT_NAME = "automation"

_SECONDS_PER_DAY = 86_400.0
_MINUTES_PER_HOUR = 60


class Trigger(abc.ABC):
    """Decides whether a rule should fire."""

    @property
    @abc.abstractmethod
    def kind(self) -> TriggerKind:
        """Return the trigger family."""

    @abc.abstractmethod
    def should_fire(self, now: float, last_fired: float | None) -> bool:
        """Return whether the rule should fire.

        Args:
            now: Current monotonic time in seconds.
            last_fired: Monotonic time of the previous firing, or ``None``.
        """

    def describe(self) -> str:
        """Return a short human-readable description."""
        return self.kind.value


class IntervalTrigger(Trigger):
    """Fires every ``seconds``, measured from the previous firing.

    Example:
        >>> trigger = IntervalTrigger(60.0)
        >>> trigger.should_fire(now=0.0, last_fired=None)
        True
        >>> trigger.should_fire(now=30.0, last_fired=0.0)
        False
    """

    def __init__(self, seconds: float, *, fire_immediately: bool = True) -> None:
        """Initialise the trigger.

        Args:
            seconds: Interval between firings. Must be positive.
            fire_immediately: Whether the first tick fires before one full
                interval has elapsed.

        Raises:
            InvalidRuleError: If ``seconds`` is not positive.
        """
        if seconds <= 0:
            raise InvalidRuleError("Interval must be positive.", context={"seconds": seconds})
        self._seconds = seconds
        self._fire_immediately = fire_immediately

    @property
    def kind(self) -> TriggerKind:
        """Return the trigger family."""
        return TriggerKind.INTERVAL

    @property
    def seconds(self) -> float:
        """Return the configured interval."""
        return self._seconds

    def should_fire(self, now: float, last_fired: float | None) -> bool:
        """Return whether one interval has elapsed."""
        if last_fired is None:
            return self._fire_immediately
        return (now - last_fired) >= self._seconds

    def describe(self) -> str:
        """Return a short human-readable description."""
        return f"every {self._seconds:g}s"


class DailyTrigger(Trigger):
    """Fires once per day at a wall-clock time.

    Wall-clock scheduling needs a wall clock, so this trigger takes the same
    injected :class:`~eden.utils.clock.Clock` the scheduler uses rather than
    reading the system time itself.
    """

    def __init__(self, hour: int, minute: int = 0, *, clock: Clock | None = None) -> None:
        """Initialise the trigger.

        Args:
            hour: Hour of day in UTC, from 0 to 23.
            minute: Minute of the hour, from 0 to 59.
            clock: Time source.

        Raises:
            InvalidRuleError: If the time of day is out of range.
        """
        if not 0 <= hour <= 23:  # noqa: PLR2004 - hours in a day
            raise InvalidRuleError("Hour must be 0-23.", context={"hour": hour})
        if not 0 <= minute < _MINUTES_PER_HOUR:
            raise InvalidRuleError("Minute must be 0-59.", context={"minute": minute})
        self._hour = hour
        self._minute = minute
        self._clock = clock or SystemClock()

    @property
    def kind(self) -> TriggerKind:
        """Return the trigger family."""
        return TriggerKind.DAILY

    def should_fire(self, now: float, last_fired: float | None) -> bool:
        """Return whether the scheduled time has arrived and not yet fired today."""
        current = self._clock.now()
        target = current.replace(hour=self._hour, minute=self._minute, second=0, microsecond=0)
        if current < target:
            return False
        if last_fired is None:
            return True
        return (now - last_fired) >= _SECONDS_PER_DAY

    def describe(self) -> str:
        """Return a short human-readable description."""
        return f"daily at {self._hour:02d}:{self._minute:02d} UTC"


class EventTrigger(Trigger):
    """Fires when a named event has been signalled since the last firing."""

    def __init__(self, event: str) -> None:
        """Initialise the trigger.

        Args:
            event: Event name callers signal through the scheduler.

        Raises:
            InvalidRuleError: If the event name is empty.
        """
        if not event.strip():
            raise InvalidRuleError("Event name must not be empty.")
        self._event = event
        self._pending = False

    @property
    def kind(self) -> TriggerKind:
        """Return the trigger family."""
        return TriggerKind.EVENT

    @property
    def event(self) -> str:
        """Return the event name."""
        return self._event

    def signal(self) -> None:
        """Mark the event as having occurred."""
        self._pending = True

    def should_fire(self, now: float, last_fired: float | None) -> bool:
        """Return whether the event fired, consuming the signal."""
        del now, last_fired
        if self._pending:
            self._pending = False
            return True
        return False

    def describe(self) -> str:
        """Return a short human-readable description."""
        return f"on event '{self._event}'"


class ManualTrigger(Trigger):
    """Never fires on its own; the rule runs only when explicitly invoked."""

    @property
    def kind(self) -> TriggerKind:
        """Return the trigger family."""
        return TriggerKind.MANUAL

    def should_fire(self, now: float, last_fired: float | None) -> bool:
        """Always return ``False``."""
        del now, last_fired
        return False

    def describe(self) -> str:
        """Return a short human-readable description."""
        return "manual only"


@dataclass(frozen=True, slots=True)
class Rule:
    """A trigger paired with the work it causes.

    Exactly one of ``task`` or ``action`` must be set. Both routes end up
    supervised — a task goes through the agent orchestrator, an action through
    the execution pipeline — so automation cannot do anything a human operator
    could not have done through the same machinery.

    Attributes:
        name: Unique rule name.
        trigger: When the rule fires.
        task: Agent task to dispatch, if this rule delegates to an agent.
        action: Action to submit, if this rule acts directly.
        enabled: Whether the scheduler considers this rule at all.
        description: Human-readable purpose.
        metadata: Free-form annotations.
    """

    name: str
    trigger: Trigger
    task: Task | None = None
    action: Action | None = None
    enabled: bool = True
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the rule.

        Raises:
            InvalidRuleError: If the payload is missing or ambiguous.
        """
        if not self.name.strip():
            raise InvalidRuleError("Rule name must not be empty.")
        if (self.task is None) == (self.action is None):
            raise InvalidRuleError(
                "A rule must carry exactly one of a task or an action.",
                context={"rule": self.name},
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "trigger": self.trigger.describe(),
            "kind": self.trigger.kind.value,
            "enabled": self.enabled,
            "description": self.description,
            "payload": "task" if self.task is not None else "action",
            "summary": (
                self.task.goal
                if self.task is not None
                else (self.action.summary if self.action is not None else "")
            ),
        }


@dataclass(frozen=True, slots=True)
class AutomationRun:
    """The record of one rule firing.

    Attributes:
        id: Unique run identifier.
        rule: Name of the rule that fired.
        status: Outcome.
        detail: Short human-readable result.
        started_at: When the run began.
        duration_ms: Measured wall-clock duration.
        error: Serialised failure, when one occurred.
    """

    id: str
    rule: str
    status: AutomationStatus
    detail: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    duration_ms: float = 0.0
    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "rule": self.rule,
            "status": self.status.value,
            "detail": self.detail,
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 3),
            "error": dict(self.error) if self.error else None,
        }


TaskRunner = Callable[[Task], Awaitable[Any]]
ActionRunner = Callable[[Action], Awaitable[Any]]


class AutomationScheduler:
    """Evaluates triggers and dispatches the work they cause.

    The scheduler owns no execution machinery of its own. It holds two injected
    coroutines — one that dispatches a task, one that submits an action — which
    keeps it decoupled from the agent and execution subsystems and makes it
    trivial to test with recording doubles.
    """

    def __init__(
        self,
        config: AutomationConfig,
        *,
        task_runner: TaskRunner | None = None,
        action_runner: ActionRunner | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the scheduler.

        Args:
            config: Scheduling policy.
            task_runner: Dispatches an agent task. Rules carrying a task are
                skipped when absent.
            action_runner: Submits an action to the execution pipeline. Rules
                carrying an action are skipped when absent.
            clock: Time source.
        """
        self._config = config
        self._task_runner = task_runner
        self._action_runner = action_runner
        self._clock = clock or SystemClock()
        self._rules: dict[str, Rule] = {}
        self._last_fired: dict[str, float] = {}
        self._history: list[AutomationRun] = []
        self._semaphore = asyncio.Semaphore(config.max_concurrent_runs)
        self._loop_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._started = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return COMPONENT_NAME

    @property
    def rules(self) -> tuple[Rule, ...]:
        """Return every registered rule."""
        return tuple(self._rules.values())

    def register(self, rule: Rule, *, replace: bool = False) -> None:
        """Register ``rule``.

        Raises:
            InvalidRuleError: If the name is taken and ``replace`` is ``False``.
        """
        if rule.name in self._rules and not replace:
            raise InvalidRuleError(
                "A rule with this name is already registered.",
                context={"rule": rule.name},
            )
        self._rules[rule.name] = rule
        _LOGGER.info(
            "Automation rule registered.",
            extra={"rule": rule.name, "trigger": rule.trigger.describe()},
        )

    def unregister(self, name: str) -> None:
        """Remove ``name``.

        Raises:
            RuleNotFoundError: If the rule is not registered.
        """
        if name not in self._rules:
            raise RuleNotFoundError(
                "No rule is registered under this name.",
                context={"rule": name, "known": sorted(self._rules)},
            )
        del self._rules[name]
        self._last_fired.pop(name, None)

    def rule(self, name: str) -> Rule:
        """Return the rule registered under ``name``.

        Raises:
            RuleNotFoundError: If the rule is not registered.
        """
        found = self._rules.get(name)
        if found is None:
            raise RuleNotFoundError(
                "No rule is registered under this name.",
                context={"rule": name, "known": sorted(self._rules)},
            )
        return found

    def signal(self, event: str) -> int:
        """Signal ``event`` to every rule listening for it.

        Returns:
            The number of rules that were armed.
        """
        armed = 0
        for rule in self._rules.values():
            trigger = rule.trigger
            if isinstance(trigger, EventTrigger) and trigger.event == event:
                trigger.signal()
                armed += 1
        return armed

    @property
    def history(self) -> tuple[AutomationRun, ...]:
        """Return recent runs, oldest first."""
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Begin the tick loop. Idempotent."""
        if self._started:
            return
        self._started = True
        self._stopping.clear()
        self._loop_task = asyncio.create_task(self._run_loop())
        _LOGGER.info(
            "Automation subsystem started.",
            extra={"rules": sorted(self._rules), "tick_seconds": self._config.tick_seconds},
        )

    async def stop(self) -> None:
        """Stop the tick loop. Idempotent and never raises."""
        if not self._started:
            return
        self._started = False
        self._stopping.set()
        task = self._loop_task
        self._loop_task = None
        if task is not None:
            task.cancel()
            # Shutdown must not fail. A cancelled loop raising is expected;
            # anything else is logged rather than allowed to escape stop().
            with suppress(asyncio.CancelledError):
                try:
                    await task
                except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                    _LOGGER.warning(
                        "Automation loop did not stop cleanly.",
                        extra={"error_type": type(exc).__name__},
                    )
        _LOGGER.info("Automation subsystem stopped.")

    async def _run_loop(self) -> None:
        """Tick until stopped."""
        while not self._stopping.is_set():
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                _LOGGER.error(
                    "Automation tick failed; continuing.",
                    extra={"error_type": type(exc).__name__},
                )
            await self._clock.sleep(self._config.tick_seconds)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    async def tick(self) -> list[AutomationRun]:
        """Evaluate every rule once and run those that fire.

        Returns:
            The runs produced by this tick.
        """
        now = self._clock.monotonic()
        due = [
            rule
            for rule in self._rules.values()
            if rule.enabled and rule.trigger.should_fire(now, self._last_fired.get(rule.name))
        ]
        if not due:
            return []
        runs = await asyncio.gather(*(self._guarded_run(rule, now) for rule in due))
        return list(runs)

    async def _guarded_run(self, rule: Rule, now: float) -> AutomationRun:
        """Run one rule under the concurrency ceiling."""
        async with self._semaphore:
            self._last_fired[rule.name] = now
            return await self.run(rule.name)

    async def run(self, name: str) -> AutomationRun:
        """Run one rule immediately, regardless of its trigger.

        Args:
            name: Rule to run.

        Returns:
            The run record. Failures are recorded, not raised, so one broken
            rule cannot stop the scheduler.

        Raises:
            RuleNotFoundError: If the rule is not registered.
        """
        rule = self.rule(name)
        run_id = new_id("run")
        started = datetime.now(tz=UTC)

        with correlation_scope(run_id), timed_block(_LOGGER, "automation.run", rule=name) as watch:
            try:
                detail = await self._dispatch(rule)
            except EdenError as exc:
                return self._record(
                    AutomationRun(
                        id=run_id,
                        rule=name,
                        status=AutomationStatus.FAILED,
                        detail=exc.message,
                        started_at=started,
                        duration_ms=watch.elapsed_ms,
                        error=exc.to_dict(),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a rule must not kill the loop
                return self._record(
                    AutomationRun(
                        id=run_id,
                        rule=name,
                        status=AutomationStatus.FAILED,
                        detail=f"{type(exc).__name__}: {exc}",
                        started_at=started,
                        duration_ms=watch.elapsed_ms,
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
                )

        if detail is None:
            return self._record(
                AutomationRun(
                    id=run_id,
                    rule=name,
                    status=AutomationStatus.SKIPPED,
                    detail="No runner is configured for this rule's payload.",
                    started_at=started,
                    duration_ms=watch.elapsed_ms,
                )
            )
        return self._record(
            AutomationRun(
                id=run_id,
                rule=name,
                status=AutomationStatus.SUCCEEDED,
                detail=detail,
                started_at=started,
                duration_ms=watch.elapsed_ms,
            )
        )

    async def _dispatch(self, rule: Rule) -> str | None:
        """Send the rule's payload to the appropriate runner.

        Returns:
            A short description of what happened, or ``None`` when no runner is
            available for this payload.

        Raises:
            AutomationError: If the run exceeds its budget.
        """
        if rule.task is not None:
            if self._task_runner is None:
                return None
            await self._with_budget(self._task_runner(rule.task), rule.name)
            return f"Dispatched task: {rule.task.goal}"
        if rule.action is not None:
            if self._action_runner is None:
                return None
            await self._with_budget(self._action_runner(rule.action), rule.name)
            return f"Submitted action: {rule.action.summary}"
        return None

    async def _with_budget(self, work: Awaitable[Any], rule: str) -> Any:  # noqa: ANN401
        """Await ``work`` under the configured run timeout.

        Raises:
            AutomationError: If the budget is exceeded.
        """
        try:
            return await asyncio.wait_for(work, timeout=self._config.run_timeout_seconds)
        except TimeoutError as exc:
            raise AutomationError(
                "Automation run exceeded its time budget.",
                context={"rule": rule, "timeout": self._config.run_timeout_seconds},
                cause=exc,
            ) from exc

    def _record(self, run: AutomationRun) -> AutomationRun:
        """Append the run to history, trimming to the configured limit."""
        self._history.append(run)
        overflow = len(self._history) - self._config.history_limit
        if overflow > 0:
            del self._history[:overflow]
        _LOGGER.info(
            "Automation run finished.",
            extra={"rule": run.rule, "status": run.status.value},
        )
        return run


def every(seconds: float) -> IntervalTrigger:
    """Return an interval trigger. Sugar for readable rule definitions."""
    return IntervalTrigger(seconds)


def every_minutes(minutes: float) -> IntervalTrigger:
    """Return an interval trigger measured in minutes."""
    return IntervalTrigger(minutes * 60.0)


def daily_at(hour: int, minute: int = 0, *, clock: Clock | None = None) -> DailyTrigger:
    """Return a daily trigger at a UTC time."""
    return DailyTrigger(hour, minute, clock=clock)


def on_event(event: str) -> EventTrigger:
    """Return an event trigger."""
    return EventTrigger(event)


def build_scheduler(
    config: AutomationConfig,
    *,
    task_runner: TaskRunner | None = None,
    action_runner: ActionRunner | None = None,
    clock: Clock | None = None,
    rules: Sequence[Rule] = (),
) -> AutomationScheduler:
    """Construct a scheduler with an initial rule set.

    Args:
        config: Scheduling policy.
        task_runner: Dispatches agent tasks.
        action_runner: Submits actions.
        clock: Time source.
        rules: Rules registered at construction time.

    Returns:
        A ready-to-start scheduler.
    """
    scheduler = AutomationScheduler(
        config,
        task_runner=task_runner,
        action_runner=action_runner,
        clock=clock,
    )
    for rule in rules:
        scheduler.register(rule)
    return scheduler


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
