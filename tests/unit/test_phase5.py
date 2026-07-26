"""Unit tests for the Phase 5 subsystems: hardware, automation and interface."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from eden.agents.types import Task
from eden.automation.scheduler import (
    AutomationScheduler,
    DailyTrigger,
    EventTrigger,
    IntervalTrigger,
    ManualTrigger,
    Rule,
    build_scheduler,
    every,
    every_minutes,
    on_event,
)
from eden.config.enums import (
    ActionKind,
    AutomationStatus,
    DeviceKind,
    DeviceState,
    ExecutionStatus,
    RiskLevel,
)
from eden.config.schema import (
    AutomationConfig,
    DeviceConfig,
    ExecutionConfig,
    HardwareConfig,
    InterfaceConfig,
)
from eden.errors import (
    DeviceCommandError,
    DeviceNotFoundError,
    DeviceSafetyError,
    DeviceUnavailableError,
    InvalidConfigError,
    InvalidRuleError,
    RuleNotFoundError,
    ValidationError,
)
from eden.execution.engine import ExecutionEngine
from eden.execution.permissions import AlwaysApproveGate, PolicyEngine
from eden.execution.types import Action, Verdict
from eden.hardware.device import DeviceCommand
from eden.hardware.manager import (
    DeviceCommandHandler,
    HttpDevice,
    SimulatedDevice,
    build_device,
    build_device_manager,
)
from eden.interface.server import Request, Response, Router, WebApprovalGate
from eden.utils.clock import ManualClock
from tests.conftest import FakeTransport


def device_config(name: str = "rig", **overrides: Any) -> DeviceConfig:
    """Return a simulated device declaration."""
    base: dict[str, Any] = {"channels": ("servo", "temp"), "limits": (("servo", 0.0, 90.0),)}
    base.update(overrides)
    return DeviceConfig(name=name, **base)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
class TestDeviceConfigValidation:
    def test_http_device_requires_an_endpoint(self) -> None:
        with pytest.raises(InvalidConfigError):
            DeviceConfig(name="x", kind=DeviceKind.HTTP)

    def test_custom_device_requires_an_implementation(self) -> None:
        with pytest.raises(InvalidConfigError):
            DeviceConfig(name="x", kind=DeviceKind.CUSTOM)

    def test_inverted_limit_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            DeviceConfig(name="x", channels=("a",), limits=(("a", 10.0, 1.0),))

    def test_limit_for_an_undeclared_channel_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            DeviceConfig(name="x", channels=("a",), limits=(("b", 0.0, 1.0),))

    def test_duplicate_device_names_are_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            HardwareConfig(devices=(device_config("a"), device_config("a")))

    def test_hardware_is_off_by_default(self) -> None:
        """Software that can move things must not do so on install."""
        assert HardwareConfig().enabled is False


class TestSimulatedDevice:
    async def test_connect_read_write_cycle(self, clock: ManualClock) -> None:
        device = SimulatedDevice(device_config(), clock=clock)
        assert device.status().state is DeviceState.DISCONNECTED
        await device.connect()
        assert device.status().state is DeviceState.READY
        reading = await device.send(DeviceCommand(device="rig", channel="servo", value=45.0))
        assert reading.value == 45.0
        assert (await device.read("servo")).value == 45.0
        await device.disconnect()
        assert device.status().state is DeviceState.DISCONNECTED

    async def test_unconnected_device_refuses_work(self, clock: ManualClock) -> None:
        device = SimulatedDevice(device_config(), clock=clock)
        with pytest.raises(DeviceUnavailableError):
            await device.read("servo")

    async def test_unknown_channel_is_refused(self, clock: ManualClock) -> None:
        device = SimulatedDevice(device_config(), clock=clock)
        await device.connect()
        with pytest.raises(DeviceCommandError):
            await device.read("nonexistent")

    @pytest.mark.parametrize("value", [-1.0, 90.1, 1000.0])
    async def test_out_of_range_command_is_refused(self, value: float, clock: ManualClock) -> None:
        device = SimulatedDevice(device_config(), clock=clock)
        await device.connect()
        with pytest.raises(DeviceSafetyError):
            await device.send(DeviceCommand(device="rig", channel="servo", value=value))

    async def test_safety_errors_are_never_retryable(self, clock: ManualClock) -> None:
        """A command outside the envelope does not become safe on a retry."""
        device = SimulatedDevice(device_config(), clock=clock)
        await device.connect()
        with pytest.raises(DeviceSafetyError) as caught:
            await device.send(DeviceCommand(device="rig", channel="servo", value=999.0))
        assert caught.value.retryable is False

    async def test_channel_without_a_limit_is_unbounded(self, clock: ManualClock) -> None:
        device = SimulatedDevice(device_config(), clock=clock)
        await device.connect()
        assert (await device.send(DeviceCommand(device="rig", channel="temp", value=1e6))).value

    async def test_unwritten_channel_reads_deterministically(self, clock: ManualClock) -> None:
        first = SimulatedDevice(device_config(), clock=clock)
        second = SimulatedDevice(device_config(), clock=clock)
        await first.connect()
        await second.connect()
        assert (await first.read("temp")).value == (await second.read("temp")).value

    async def test_status_reports_state_and_counts(self, clock: ManualClock) -> None:
        device = SimulatedDevice(device_config(), clock=clock)
        await device.connect()
        await device.send(DeviceCommand(device="rig", channel="servo", value=10.0))
        status = device.status()
        assert status.ready is True
        assert status.commands_sent == 1
        assert "servo" in status.channels

    async def test_command_validation(self) -> None:
        with pytest.raises(ValidationError):
            DeviceCommand(device="", channel="a", value=1.0)
        with pytest.raises(ValidationError):
            DeviceCommand(device="a", channel=" ", value=1.0)


class TestHttpDevice:
    async def test_reads_and_writes_over_the_bridge(
        self, transport: FakeTransport, clock: ManualClock
    ) -> None:
        config = DeviceConfig(
            name="bridge",
            kind=DeviceKind.HTTP,
            endpoint="http://device.test",
            channels=("angle",),
        )
        device = HttpDevice(config, transport, clock=clock)
        transport.queue_json({"ok": True, "value": 0})
        await device.connect()
        transport.queue_json({"value": 12.5})
        assert (await device.read("angle")).value == 12.5
        transport.queue_json({"value": 30.0})
        result = await device.send(DeviceCommand(device="bridge", channel="angle", value=30.0))
        assert result.value == 30.0

    async def test_error_status_faults_the_device(
        self, transport: FakeTransport, clock: ManualClock
    ) -> None:
        config = DeviceConfig(name="bridge", kind=DeviceKind.HTTP, endpoint="http://device.test")
        device = HttpDevice(config, transport, clock=clock)
        transport.queue_error(500)
        with pytest.raises(DeviceCommandError):
            await device.connect()
        assert device.status().state is DeviceState.FAULTED

    async def test_non_numeric_response_is_rejected(
        self, transport: FakeTransport, clock: ManualClock
    ) -> None:
        config = DeviceConfig(name="bridge", kind=DeviceKind.HTTP, endpoint="http://device.test")
        device = HttpDevice(config, transport, clock=clock)
        transport.queue_json({"value": 0})
        await device.connect()
        transport.queue_json({"nope": True})
        with pytest.raises(DeviceCommandError):
            await device.read("angle")


class TestDeviceManager:
    def manager(self, clock: ManualClock, **overrides: Any) -> Any:
        config = HardwareConfig(enabled=True, devices=(device_config(),), **overrides)
        return build_device_manager(config, clock=clock)

    async def test_starts_and_stops_the_fleet(self, clock: ManualClock) -> None:
        manager = self.manager(clock)
        await manager.start()
        assert manager.statuses()[0].state is DeviceState.READY
        await manager.stop()
        assert manager.statuses()[0].state is DeviceState.DISCONNECTED

    async def test_unknown_device_reports_the_known_set(self, clock: ManualClock) -> None:
        manager = self.manager(clock)
        with pytest.raises(DeviceNotFoundError) as caught:
            manager.device("absent")
        assert "known" in caught.value.context

    async def test_read_all_skips_failing_channels(self, clock: ManualClock) -> None:
        manager = self.manager(clock)
        await manager.start()
        readings = await manager.read_all("rig")
        assert {r.channel for r in readings} == {"servo", "temp"}

    async def test_a_broken_device_does_not_abort_startup(self, clock: ManualClock) -> None:
        good = device_config("good")
        bad = DeviceConfig(name="bad", kind=DeviceKind.CUSTOM, implementation="nowhere:Absent")
        manager = build_device_manager(
            HardwareConfig(enabled=True, devices=(good, bad)), clock=clock
        )
        assert [d.name for d in manager.devices] == ["good"]

    def test_http_device_without_transport_is_rejected(self, clock: ManualClock) -> None:
        config = DeviceConfig(name="x", kind=DeviceKind.HTTP, endpoint="http://a.test")
        with pytest.raises(InvalidConfigError):
            build_device(config, clock=clock)


class TestActuationGoesThroughThePipeline:
    """The architectural claim: nothing reaches a moving part unsupervised."""

    def engine(self, clock: ManualClock, *, approve: bool = True, **hw: Any) -> Any:
        hardware = HardwareConfig(enabled=True, devices=(device_config(),), **hw)
        manager = build_device_manager(hardware, clock=clock)
        execution = ExecutionConfig()
        gate = AlwaysApproveGate() if approve else None
        return manager, ExecutionEngine(
            execution,
            policy=PolicyEngine(execution, gate),
            handlers=[DeviceCommandHandler(execution, manager)],
        )

    def command(self, value: float = 45.0) -> Action:
        return Action(
            kind=ActionKind.DEVICE_COMMAND,
            summary=f"Move servo to {value}",
            parameters={"device": "rig", "channel": "servo", "value": value},
        )

    async def test_a_command_is_verified_permitted_and_executed(self, clock: ManualClock) -> None:
        manager, engine = self.engine(clock)
        await manager.start()
        record = await engine.submit(self.command(45.0))
        assert record.status is ExecutionStatus.SUCCEEDED
        assert (await manager.read("rig", "servo")).value == 45.0

    async def test_prepare_captures_the_previous_value(self, clock: ManualClock) -> None:
        """Reversibility is proven before permission, exactly as ADR-0004 requires."""
        manager, engine = self.engine(clock)
        await manager.start()
        await engine.submit(self.command(10.0))
        verdict, preparation = await engine.review(self.command(80.0))
        assert verdict.reversible is True
        assert preparation.notes["previous_value"] == 10.0

    async def test_rollback_returns_the_channel(self, clock: ManualClock) -> None:
        manager, engine = self.engine(clock)
        await manager.start()
        await engine.submit(self.command(10.0))
        action = self.command(80.0)
        _, preparation = await engine.review(action)
        record = await engine.submit(action)
        assert (await manager.read("rig", "servo")).value == 80.0
        await engine.rollback(record, preparation)
        assert (await manager.read("rig", "servo")).value == 10.0

    async def test_an_unsafe_command_fails_the_action(self, clock: ManualClock) -> None:
        manager, engine = self.engine(clock)
        await manager.start()
        record = await engine.submit(self.command(500.0))
        assert record.status in (ExecutionStatus.FAILED, ExecutionStatus.ROLLED_BACK)
        assert record.error is not None

    async def test_read_only_mode_refuses_every_command(self, clock: ManualClock) -> None:
        manager, engine = self.engine(clock, read_only=True)
        await manager.start()
        before = (await manager.read("rig", "servo")).value
        record = await engine.submit(self.command(45.0))
        assert not record.succeeded
        assert (await manager.read("rig", "servo")).value == before

    async def test_a_non_numeric_value_is_refused_at_verification(self, clock: ManualClock) -> None:
        manager, engine = self.engine(clock)
        await manager.start()
        record = await engine.submit(
            Action(
                kind=ActionKind.DEVICE_COMMAND,
                summary="bad",
                parameters={"device": "rig", "channel": "servo", "value": "up a bit"},
            )
        )
        assert not record.succeeded


# ---------------------------------------------------------------------------
# Automation
# ---------------------------------------------------------------------------
class TestTriggers:
    def test_interval_fires_immediately_then_waits(self) -> None:
        trigger = IntervalTrigger(60.0)
        assert trigger.should_fire(0.0, None) is True
        assert trigger.should_fire(30.0, 0.0) is False
        assert trigger.should_fire(60.0, 0.0) is True

    def test_interval_can_defer_the_first_firing(self) -> None:
        assert IntervalTrigger(60.0, fire_immediately=False).should_fire(0.0, None) is False

    def test_non_positive_interval_is_rejected(self) -> None:
        with pytest.raises(InvalidRuleError):
            IntervalTrigger(0.0)

    def test_daily_respects_the_clock(self, clock: ManualClock) -> None:
        trigger = DailyTrigger(0, 0, clock=clock)
        assert trigger.should_fire(0.0, None) is True
        assert trigger.should_fire(100.0, 0.0) is False
        assert trigger.should_fire(90_000.0, 0.0) is True

    def test_daily_rejects_impossible_times(self) -> None:
        with pytest.raises(InvalidRuleError):
            DailyTrigger(24)
        with pytest.raises(InvalidRuleError):
            DailyTrigger(1, 60)

    def test_event_fires_once_per_signal(self) -> None:
        trigger = EventTrigger("deploy")
        assert trigger.should_fire(0.0, None) is False
        trigger.signal()
        assert trigger.should_fire(0.0, None) is True
        assert trigger.should_fire(0.0, None) is False

    def test_manual_never_fires_on_its_own(self) -> None:
        assert ManualTrigger().should_fire(1e9, None) is False

    def test_helpers_describe_themselves(self) -> None:
        assert "60" in every(60).describe()
        assert every_minutes(2).seconds == 120.0
        assert "deploy" in on_event("deploy").describe()


class TestRule:
    def test_a_rule_needs_exactly_one_payload(self) -> None:
        with pytest.raises(InvalidRuleError):
            Rule(name="a", trigger=every(1))
        with pytest.raises(InvalidRuleError):
            Rule(
                name="a",
                trigger=every(1),
                task=Task(goal="g"),
                action=Action(kind=ActionKind.NOOP, summary="s"),
            )

    def test_a_rule_carries_data_not_a_callable(self) -> None:
        """Rules are inspectable; they cannot smuggle code past the pipeline."""
        rule = Rule(name="a", trigger=every(1), task=Task(goal="tidy up"))
        payload = rule.to_dict()
        assert payload["payload"] == "task"
        assert payload["summary"] == "tidy up"
        assert not callable(rule.task)


class TestScheduler:
    def scheduler(self, clock: ManualClock, **overrides: Any) -> AutomationScheduler:
        return AutomationScheduler(AutomationConfig(enabled=True, **overrides), clock=clock)

    async def test_tick_fires_due_rules_only(self, clock: ManualClock) -> None:
        seen: list[str] = []

        async def runner(task: Task) -> None:
            seen.append(task.goal)

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True), task_runner=runner, clock=clock
        )
        scheduler.register(Rule(name="fast", trigger=every(1), task=Task(goal="fast")))
        scheduler.register(
            Rule(
                name="slow",
                trigger=IntervalTrigger(1000.0, fire_immediately=False),
                task=Task(goal="slow"),
            )
        )
        await scheduler.tick()
        assert seen == ["fast"]

    async def test_a_rule_does_not_refire_before_its_interval(self, clock: ManualClock) -> None:
        calls = {"n": 0}

        async def runner(task: Task) -> None:
            del task
            calls["n"] += 1

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True), task_runner=runner, clock=clock
        )
        scheduler.register(Rule(name="r", trigger=every(60), task=Task(goal="g")))
        await scheduler.tick()
        await scheduler.tick()
        assert calls["n"] == 1
        clock.advance(61)
        await scheduler.tick()
        assert calls["n"] == 2

    async def test_disabled_rules_are_skipped(self, clock: ManualClock) -> None:
        calls: list[str] = []

        async def runner(task: Task) -> None:
            calls.append(task.goal)

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True), task_runner=runner, clock=clock
        )
        scheduler.register(Rule(name="off", trigger=every(1), task=Task(goal="g"), enabled=False))
        await scheduler.tick()
        assert calls == []

    async def test_a_failing_rule_is_recorded_not_raised(self, clock: ManualClock) -> None:
        async def runner(task: Task) -> None:
            del task
            message = "rule blew up"
            raise RuntimeError(message)

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True), task_runner=runner, clock=clock
        )
        scheduler.register(Rule(name="bad", trigger=every(1), task=Task(goal="g")))
        runs = await scheduler.tick()
        assert runs[0].status is AutomationStatus.FAILED
        assert scheduler.history[-1].error is not None

    async def test_a_rule_with_no_runner_is_skipped_not_failed(self, clock: ManualClock) -> None:
        scheduler = self.scheduler(clock)
        scheduler.register(Rule(name="r", trigger=every(1), task=Task(goal="g")))
        runs = await scheduler.tick()
        assert runs[0].status is AutomationStatus.SKIPPED

    async def test_actions_route_to_the_action_runner(self, clock: ManualClock) -> None:
        submitted: list[Action] = []

        async def runner(action: Action) -> None:
            submitted.append(action)

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True), action_runner=runner, clock=clock
        )
        scheduler.register(
            Rule(
                name="act",
                trigger=every(1),
                action=Action(kind=ActionKind.NOOP, summary="tick"),
            )
        )
        await scheduler.tick()
        assert submitted[0].summary == "tick"

    async def test_events_arm_matching_rules(self, clock: ManualClock) -> None:
        calls: list[str] = []

        async def runner(task: Task) -> None:
            calls.append(task.goal)

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True), task_runner=runner, clock=clock
        )
        scheduler.register(Rule(name="e", trigger=on_event("deploy"), task=Task(goal="notify")))
        await scheduler.tick()
        assert calls == []
        assert scheduler.signal("deploy") == 1
        await scheduler.tick()
        assert calls == ["notify"]

    async def test_manual_rules_run_only_on_request(self, clock: ManualClock) -> None:
        calls: list[str] = []

        async def runner(task: Task) -> None:
            calls.append(task.goal)

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True), task_runner=runner, clock=clock
        )
        scheduler.register(Rule(name="m", trigger=ManualTrigger(), task=Task(goal="manual")))
        await scheduler.tick()
        assert calls == []
        assert (await scheduler.run("m")).status is AutomationStatus.SUCCEEDED
        assert calls == ["manual"]

    def test_duplicate_and_unknown_rules_raise(self, clock: ManualClock) -> None:
        scheduler = self.scheduler(clock)
        scheduler.register(Rule(name="r", trigger=every(1), task=Task(goal="g")))
        with pytest.raises(InvalidRuleError):
            scheduler.register(Rule(name="r", trigger=every(1), task=Task(goal="g")))
        with pytest.raises(RuleNotFoundError):
            scheduler.unregister("absent")

    async def test_history_is_bounded(self, clock: ManualClock) -> None:
        async def runner(task: Task) -> None:
            del task

        scheduler = AutomationScheduler(
            AutomationConfig(enabled=True, history_limit=3),
            task_runner=runner,
            clock=clock,
        )
        scheduler.register(Rule(name="r", trigger=ManualTrigger(), task=Task(goal="g")))
        for _ in range(6):
            await scheduler.run("r")
        assert len(scheduler.history) == 3

    async def test_lifecycle_is_idempotent(self, clock: ManualClock) -> None:
        scheduler = build_scheduler(AutomationConfig(enabled=True), clock=clock)
        await scheduler.start()
        await scheduler.start()
        await scheduler.stop()
        await scheduler.stop()

    def test_automation_is_off_by_default(self) -> None:
        assert AutomationConfig().enabled is False


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class TestRouter:
    def test_exact_and_wildcard_matching(self) -> None:
        router = Router()

        async def handler(request: Request) -> Response:
            del request
            return Response.json({})

        router.add("GET", "/api/status", handler)
        router.add("GET", "/api/thing/*", handler)
        assert router.get("GET", "/api/status") is handler
        assert router.get("GET", "/api/thing/42") is handler
        assert router.get("POST", "/api/status") is None
        assert router.get("GET", "/nope") is None


class TestRequestResponse:
    def test_json_body_parsing(self) -> None:
        request = Request(method="POST", path="/", body=b'{"a": 1}')
        assert request.json() == {"a": 1}

    def test_empty_body_is_an_empty_object(self) -> None:
        assert Request(method="POST", path="/").json() == {}

    @pytest.mark.parametrize("body", [b"not json", b"[1,2,3]"])
    def test_bad_bodies_are_rejected(self, body: bytes) -> None:
        from eden.errors import InterfaceError

        with pytest.raises(InterfaceError):
            Request(method="POST", path="/", body=body).json()

    def test_response_helpers(self) -> None:
        assert json.loads(Response.json({"a": 1}).body) == {"a": 1}
        assert Response.html("<p>x</p>").content_type.startswith("text/html")
        assert Response.error(404, "gone").status == 404


class TestWebApprovalGate:
    def config(self, **overrides: Any) -> InterfaceConfig:
        return InterfaceConfig(enabled=True, **overrides)

    def verdict(self) -> Verdict:
        return Verdict(action_id="a", findings=(), reversible=True)

    async def test_approval_unblocks_the_waiting_action(self) -> None:
        gate = WebApprovalGate(self.config())
        action = Action(kind=ActionKind.NOOP, summary="waiting")

        async def approve_shortly() -> None:
            for _ in range(50):
                await asyncio.sleep(0)
                if gate.pending:
                    gate.resolve(gate.pending[0].id, approved=True)
                    return

        task = asyncio.create_task(approve_shortly())
        assert await gate.request(action, self.verdict()) is True
        await task

    async def test_refusal_is_recorded(self) -> None:
        gate = WebApprovalGate(self.config())

        async def refuse_shortly() -> None:
            for _ in range(50):
                await asyncio.sleep(0)
                if gate.pending:
                    gate.resolve(gate.pending[0].id, approved=False)
                    return

        task = asyncio.create_task(refuse_shortly())
        assert (
            await gate.request(Action(kind=ActionKind.NOOP, summary="x"), self.verdict()) is False
        )
        await task

    async def test_silence_becomes_refusal_not_consent(self) -> None:
        """An interface that cannot reach a human must behave like no human."""
        gate = WebApprovalGate(self.config(approval_timeout_seconds=0.01))
        approved = await gate.request(
            Action(kind=ActionKind.NOOP, summary="ignored"), self.verdict()
        )
        assert approved is False
        assert gate.pending == []

    def test_resolving_an_unknown_approval_is_harmless(self) -> None:
        assert WebApprovalGate(self.config()).resolve("nope", approved=True) is False

    async def test_pending_entries_describe_the_risk(self) -> None:
        gate = WebApprovalGate(self.config(approval_timeout_seconds=0.05))
        action = Action(kind=ActionKind.SHELL_COMMAND, summary="run something")
        verdict = Verdict(
            action_id=action.id,
            findings=(),
            reversible=False,
        )
        task = asyncio.create_task(gate.request(action, verdict))
        for _ in range(50):
            await asyncio.sleep(0)
            if gate.pending:
                break
        entry = gate.pending[0].to_dict()
        assert "cannot be undone" in entry["rollback"]
        assert entry["action"]["summary"] == "run something"
        await task

    def test_interface_defaults_to_loopback_and_off(self) -> None:
        config = InterfaceConfig()
        assert config.enabled is False
        assert config.host == "127.0.0.1"

    def test_invalid_port_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            InterfaceConfig(port=70000)


class TestRiskOrdering:
    def test_risk_levels_compare(self) -> None:
        assert RiskLevel.LOW < RiskLevel.HIGH
        assert max(RiskLevel.NONE, RiskLevel.CRITICAL) is RiskLevel.CRITICAL
