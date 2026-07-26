"""Unit tests for the core primitives, logging and shared utilities."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pytest

from eden.config.enums import Capability, LogFormat, Role
from eden.config.schema import LoggingConfig, PathsConfig, RetryConfig
from eden.config.secrets import REDACTED
from eden.core.container import Container, Scope
from eden.core.registry import Registry
from eden.core.types import ChatRequest, Message, Usage
from eden.errors import (
    DependencyResolutionError,
    EdenError,
    PluginLoadError,
    RegistryError,
    ValidationError,
)
from eden.logging import configure_logging, correlation_scope, get_correlation_id, get_logger
from eden.logging.formatters import RedactionFilter
from eden.logging.timing import timed, timed_block
from eden.utils.async_tools import compute_backoff, gather_limited, retry_async, with_timeout
from eden.utils.clock import ManualClock
from eden.utils.imports import import_object, import_subclass
from eden.utils.ratelimit import TokenBucket


class _Engine:
    def __init__(self, label: str = "engine") -> None:
        self.label = label


class _Car:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine


class TestContainer:
    def test_singletons_are_cached(self) -> None:
        container = Container()
        container.register(_Engine, lambda _: _Engine())
        assert container.resolve(_Engine) is container.resolve(_Engine)

    def test_transients_are_fresh_each_time(self) -> None:
        container = Container()
        container.register(_Engine, lambda _: _Engine(), scope=Scope.TRANSIENT)
        assert container.resolve(_Engine) is not container.resolve(_Engine)

    def test_dependencies_are_injected_not_fetched(self) -> None:
        container = Container()
        container.register(_Engine, lambda _: _Engine("v8"))
        container.register(_Car, lambda c: _Car(c.resolve(_Engine)))
        assert container.resolve(_Car).engine.label == "v8"

    def test_named_bindings_coexist(self) -> None:
        container = Container()
        container.register(_Engine, lambda _: _Engine("a"), name="a")
        container.register(_Engine, lambda _: _Engine("b"), name="b")
        assert container.resolve(_Engine, name="b").label == "b"

    def test_unknown_binding_raises_with_known_keys(self) -> None:
        with pytest.raises(DependencyResolutionError) as caught:
            Container().resolve(_Engine)
        assert "known" in caught.value.context

    def test_duplicate_registration_raises_unless_replacing(self) -> None:
        container = Container()
        container.register(_Engine, lambda _: _Engine())
        with pytest.raises(DependencyResolutionError):
            container.register(_Engine, lambda _: _Engine())
        container.register(_Engine, lambda _: _Engine("new"), replace=True)
        assert container.resolve(_Engine).label == "new"

    def test_circular_dependency_is_detected(self) -> None:
        container = Container()
        container.register(_Engine, lambda c: _Engine(c.resolve(_Car).engine.label))
        container.register(_Car, lambda c: _Car(c.resolve(_Engine)))
        with pytest.raises(DependencyResolutionError) as caught:
            container.resolve(_Engine)
        assert "chain" in caught.value.context or caught.value.cause is not None

    def test_factory_failure_is_wrapped_not_swallowed(self) -> None:
        container = Container()

        def explode(_: Container) -> _Engine:
            message = "kaboom"
            raise RuntimeError(message)

        container.register(_Engine, explode)
        with pytest.raises(DependencyResolutionError) as caught:
            container.resolve(_Engine)
        assert isinstance(caught.value.cause, RuntimeError)


class TestRegistry:
    def test_lookup_is_case_insensitive(self) -> None:
        registry: Registry[int] = Registry("demo")
        registry.register("Answer", lambda: 42)
        assert registry.create("answer") == 42
        assert "ANSWER" in registry

    def test_duplicate_and_unknown_names_raise(self) -> None:
        registry: Registry[int] = Registry("demo")
        registry.register("a", lambda: 1)
        with pytest.raises(RegistryError):
            registry.register("a", lambda: 2)
        with pytest.raises(RegistryError):
            registry.create("missing")

    def test_empty_name_is_rejected(self) -> None:
        registry: Registry[int] = Registry("demo")
        with pytest.raises(RegistryError):
            registry.register("  ", lambda: 1)


class TestDomainTypes:
    def test_empty_message_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message.user("   ")

    def test_request_requires_messages(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_temperature_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(messages=[Message.user("hi")], temperature=5.0)

    def test_streaming_implies_the_streaming_capability(self) -> None:
        request = ChatRequest(messages=[Message.user("hi")], stream=True)
        assert Capability.STREAMING in request.required_capabilities

    def test_usage_totals(self) -> None:
        assert Usage(prompt_tokens=10, completion_tokens=5).total_tokens == 15

    def test_role_helpers(self) -> None:
        assert Message.system("s").role is Role.SYSTEM
        assert Message.assistant("a").role is Role.ASSISTANT


class TestLogging:
    def test_configure_is_idempotent(self) -> None:
        first = configure_logging(LoggingConfig())
        configure_logging(LoggingConfig())
        assert len(first.handlers) == 1

    def test_credentials_never_reach_a_handler(self) -> None:
        record = logging.LogRecord(
            name="eden.test.redaction",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="calling with sk-abcdefghijklmnopqrstuvwx",
            args=(),
            exc_info=None,
        )
        record.__dict__["api_key"] = "leak"
        assert RedactionFilter().filter(record) is True
        assert "sk-abcdefghijklmnopqrstuvwx" not in record.getMessage()
        assert record.__dict__["api_key"] == REDACTED

    def test_redaction_survives_a_hostile_payload(self) -> None:
        class Explodes:
            def __str__(self) -> str:
                message = "cannot render"
                raise RuntimeError(message)

        record = logging.LogRecord(
            name="eden.test.redaction",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=Explodes(),
            args=(),
            exc_info=None,
        )
        assert RedactionFilter().filter(record) is True
        assert "suppressed" in str(record.msg)

    def test_json_format_is_selectable(self) -> None:
        root = configure_logging(LoggingConfig(format=LogFormat.JSON), force=True)
        assert type(root.handlers[0].formatter).__name__ == "JsonFormatter"

    def test_file_logging_requires_paths(self) -> None:
        with pytest.raises(ValueError, match="PathsConfig"):
            configure_logging(LoggingConfig(to_file=True), force=True)

    def test_file_handler_is_attached(self, tmp_path: Path) -> None:
        paths = PathsConfig(root=tmp_path / "r", log_dir=tmp_path / "r" / "logs")
        root = configure_logging(LoggingConfig(to_file=True), paths, force=True)
        assert len(root.handlers) == 2

    def test_correlation_id_is_scoped(self) -> None:
        assert get_correlation_id() == "-"
        with correlation_scope("req-42") as value:
            assert value == "req-42"
            assert get_correlation_id() == "req-42"
        assert get_correlation_id() == "-"


class TestTiming:
    def test_block_records_duration_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("eden.test.timing")
        with caplog.at_level(logging.DEBUG, logger="eden.test.timing"), timed_block(logger, "unit"):
            pass
        assert caplog.records[-1].__dict__["outcome"] == "success"
        assert caplog.records[-1].__dict__["duration_ms"] >= 0

    def test_block_records_duration_on_failure_and_reraises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = get_logger("eden.test.timing")
        with (
            caplog.at_level(logging.DEBUG, logger="eden.test.timing"),
            pytest.raises(EdenError),
            timed_block(logger, "unit"),
        ):
            raise EdenError("bad")
        assert caplog.records[-1].__dict__["outcome"] == "error"

    async def test_decorator_supports_async(self) -> None:
        logger = get_logger("eden.test.timing")

        @timed(logger, "async-op")
        async def work() -> int:
            return 7

        assert await work() == 7

    def test_decorator_supports_sync(self) -> None:
        logger = get_logger("eden.test.timing")

        @timed(logger, "sync-op")
        def work() -> int:
            return 7

        assert work() == 7


class TestBackoff:
    def test_first_attempt_never_waits(self) -> None:
        assert compute_backoff(1, RetryConfig()) == 0.0

    def test_growth_is_capped(self) -> None:
        policy = RetryConfig(initial_backoff_seconds=1.0, multiplier=10.0, max_backoff_seconds=5.0)
        assert compute_backoff(6, policy, rng=random.Random(0)) <= 5.0

    def test_jitter_stays_within_bounds(self) -> None:
        policy = RetryConfig(initial_backoff_seconds=1.0, multiplier=1.0, jitter_ratio=0.5)
        values = [compute_backoff(2, policy, rng=random.Random(seed)) for seed in range(20)]
        assert all(0.5 <= value <= 1.5 for value in values)


class TestRetry:
    async def test_succeeds_without_sleeping_on_first_try(self, clock: ManualClock) -> None:
        async def work() -> str:
            return "ok"

        result = await retry_async(
            work,
            RetryConfig(),
            logger=get_logger("eden.test.retry"),
            operation_name="work",
            clock=clock,
        )
        assert result == "ok"
        assert clock.slept == ()

    async def test_retries_only_retryable_errors(self, clock: ManualClock) -> None:
        calls = {"n": 0}

        class Retryable(EdenError):
            retryable = True

        async def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise Retryable("later")
            return "ok"

        result = await retry_async(
            flaky,
            RetryConfig(max_attempts=3, initial_backoff_seconds=0.1, jitter_ratio=0.0),
            logger=get_logger("eden.test.retry"),
            operation_name="flaky",
            clock=clock,
        )
        assert result == "ok"
        assert calls["n"] == 3
        assert len(clock.slept) == 2

    async def test_non_retryable_error_fails_immediately(self, clock: ManualClock) -> None:
        calls = {"n": 0}

        async def hard_fail() -> str:
            calls["n"] += 1
            raise EdenError("permanent")

        with pytest.raises(EdenError):
            await retry_async(
                hard_fail,
                RetryConfig(max_attempts=5),
                logger=get_logger("eden.test.retry"),
                operation_name="hard",
                clock=clock,
            )
        assert calls["n"] == 1


class TestAsyncHelpers:
    async def test_timeout_uses_the_supplied_error(self) -> None:
        import asyncio

        async def slow() -> None:
            await asyncio.sleep(1)

        with pytest.raises(EdenError):
            await with_timeout(slow(), 0.01, on_timeout=lambda: EdenError("too slow"))

    async def test_gather_limited_returns_failures_inline(self) -> None:
        async def good() -> int:
            return 1

        async def bad() -> int:
            message = "nope"
            raise RuntimeError(message)

        results = await gather_limited([good, bad, good], limit=2)
        assert results[0] == 1
        assert isinstance(results[1], RuntimeError)

    async def test_gather_limited_rejects_zero_limit(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            await gather_limited([], limit=0)


class TestTokenBucket:
    def test_zero_quota_means_unlimited(self) -> None:
        bucket = TokenBucket(requests_per_minute=0)
        assert bucket.unlimited is True
        assert bucket.try_acquire(1000) is True

    def test_bucket_drains_and_refills(self, clock: ManualClock) -> None:
        bucket = TokenBucket(requests_per_minute=60, clock=clock, burst=2)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False
        clock.advance(1.0)
        assert bucket.try_acquire() is True

    async def test_acquire_waits_for_refill(self, clock: ManualClock) -> None:
        bucket = TokenBucket(requests_per_minute=60, clock=clock, burst=1)
        await bucket.acquire()
        waited = await bucket.acquire()
        assert waited > 0

    def test_negative_quota_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            TokenBucket(requests_per_minute=-1)


class TestDynamicImports:
    def test_colon_and_dotted_paths_both_work(self) -> None:
        assert import_object("math:sqrt")(9.0) == 3.0
        assert import_object("math.sqrt")(16.0) == 4.0

    @pytest.mark.parametrize("path", ["", "nomodule", "eden.utils:absent", "no_such_pkg:x"])
    def test_bad_paths_raise_plugin_errors(self, path: str) -> None:
        with pytest.raises(PluginLoadError):
            import_object(path)

    def test_subclass_check_is_enforced(self) -> None:
        with pytest.raises(PluginLoadError):
            import_subclass("math:sqrt", _Engine)
        with pytest.raises(PluginLoadError):
            import_subclass("decimal:Decimal", _Engine)
