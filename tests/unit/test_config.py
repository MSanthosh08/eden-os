"""Unit tests for the configuration, error and secret layers."""

from __future__ import annotations

from pathlib import Path

import pytest

from eden.config import ConfigLoader, EdenConfig, build_config, load_config
from eden.config.enums import Environment, LogLevel, PrivacyTier, ProviderKind
from eden.config.schema import GatewayConfig, ProviderConfig, RetryConfig, RouterWeights
from eden.config.secrets import REDACTED, SecretResolver, SecretStr
from eden.config.sources import EnvironmentSource, deep_merge
from eden.errors import (
    ConfigurationError,
    EdenError,
    InvalidConfigError,
    ProviderTimeoutError,
    SecretResolutionError,
)
from eden.utils.redaction import redact_text, redact_value


class TestErrorHierarchy:
    def test_carries_code_context_and_retry_hint(self) -> None:
        error = ProviderTimeoutError("slow", provider="openai", context={"model": "x"})
        assert error.code == "eden.gateway.timeout"
        assert error.retryable is True
        assert error.context == {"provider": "openai", "model": "x"}
        assert isinstance(error, EdenError)

    def test_serialises_without_leaking_the_cause_object(self) -> None:
        cause = ValueError("underlying")
        error = ProviderTimeoutError("slow", provider="p", cause=cause)
        payload = error.to_dict()
        assert payload["code"] == "eden.gateway.timeout"
        assert payload["cause"] == "ValueError: underlying"
        assert error.__cause__ is cause


class TestSecrets:
    def test_secret_is_redacted_in_every_rendering(self) -> None:
        secret = SecretStr("sk-live-1234567890")
        assert str(secret) == REDACTED
        assert REDACTED in repr(secret)
        assert f"{secret}" == REDACTED
        assert secret.reveal() == "sk-live-1234567890"

    def test_missing_required_secret_raises(self) -> None:
        resolver = SecretResolver({})
        with pytest.raises(SecretResolutionError):
            resolver.resolve("ABSENT_KEY")
        assert resolver.resolve("ABSENT_KEY", required=False) is None

    def test_blank_value_counts_as_missing(self) -> None:
        resolver = SecretResolver({"K": "   "})
        assert resolver.has("K") is False


class TestRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "key sk-abcdefghijklmnopqrstuvwx here",
            "token gsk_abcdefghijklmnopqrstuv here",
            "google AIzaSyABCDEFGHIJKLMNOPQRSTUV here",
            "Authorization: Bearer abcdefghijklmnop",
        ],
    )
    def test_credential_shapes_are_masked(self, text: str) -> None:
        assert REDACTED in redact_text(text)

    def test_sensitive_keys_are_masked_recursively(self) -> None:
        payload = {"outer": {"api_key": "anything", "safe": "visible"}}
        cleaned = redact_value(payload)
        assert cleaned == {"outer": {"api_key": REDACTED, "safe": "visible"}}

    def test_secret_objects_are_masked(self) -> None:
        assert redact_value({"x": SecretStr("hunter2")}) == {"x": REDACTED}


class TestDeepMerge:
    def test_mappings_merge_and_scalars_replace(self) -> None:
        base = {"a": {"b": 1, "c": 2}, "list": [1, 2]}
        overlay = {"a": {"c": 3}, "list": [9]}
        assert deep_merge(base, overlay) == {"a": {"b": 1, "c": 3}, "list": [9]}

    def test_inputs_are_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestEnvironmentSource:
    def test_double_underscore_becomes_nesting(self) -> None:
        source = EnvironmentSource({"EDEN__LOGGING__LEVEL": "DEBUG", "OTHER": "ignored"})
        assert source.load() == {"logging": {"level": "DEBUG"}}

    def test_conflicting_scalar_and_branch_raises(self) -> None:
        source = EnvironmentSource({"EDEN__A": "1", "EDEN__A__B": "2"})
        with pytest.raises(ConfigurationError):
            source.load()


class TestConfigLoader:
    def test_defaults_are_usable_without_any_source(self) -> None:
        config = ConfigLoader().build()
        assert config.app_name == "eden"
        assert config.environment is Environment.DEVELOPMENT

    def test_precedence_is_defaults_then_file_then_env_then_overrides(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "eden.toml"
        toml_file.write_text(
            "\n".join(
                [
                    'app_name = "from-file"',
                    'version = "9.9.9"',
                    "[logging]",
                    'level = "WARNING"',
                ]
            ),
            encoding="utf-8",
        )
        config = (
            ConfigLoader()
            .with_toml(toml_file)
            .with_environ({"EDEN__LOGGING__LEVEL": "ERROR", "EDEN__APP_NAME": "from-env"})
            .with_overrides({"app_name": "from-override"})
            .build()
        )
        assert config.app_name == "from-override"
        assert config.logging.level is LogLevel.ERROR
        assert config.version == "9.9.9"

    def test_missing_optional_file_is_not_an_error(self, tmp_path: Path) -> None:
        config = ConfigLoader().with_toml(tmp_path / "absent.toml").build()
        assert isinstance(config, EdenConfig)

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            ConfigLoader().with_toml(tmp_path / "absent.toml", required=True).build()

    def test_malformed_toml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "eden.toml"
        bad.write_text("this is [not valid", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            ConfigLoader().with_toml(bad).build()

    def test_env_strings_are_coerced_to_schema_types(self) -> None:
        config = (
            ConfigLoader()
            .with_environ(
                {
                    "EDEN__SERVER__PORT": "9000",
                    "EDEN__LOGGING__TO_FILE": "yes",
                    "EDEN__GATEWAY__REQUEST_TIMEOUT_SECONDS": "12.5",
                }
            )
            .build()
        )
        assert config.server.port == 9000
        assert config.logging.to_file is True
        assert config.gateway.request_timeout_seconds == 12.5

    def test_paths_derive_from_root(self) -> None:
        config = ConfigLoader().with_environ({"EDEN__PATHS__ROOT": "/tmp/eden-x"}).build()
        assert config.paths.data_dir == Path("/tmp/eden-x/data")
        assert config.paths.log_dir == Path("/tmp/eden-x/logs")

    def test_provider_table_is_parsed(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "eden.toml"
        toml_file.write_text(
            "\n".join(
                [
                    "[[gateway.providers]]",
                    'name = "groq"',
                    'kind = "openai_compatible"',
                    'base_url = "https://api.groq.com/openai/v1"',
                    'api_key_env = "GROQ_API_KEY"',
                    'default_model = "llama-3.3-70b"',
                    'privacy_tier = "public_cloud"',
                    "[[gateway.providers.models]]",
                    'name = "llama-3.3-70b"',
                    'capabilities = ["chat", "streaming"]',
                    "input_cost_per_1k = 0.59",
                ]
            ),
            encoding="utf-8",
        )
        config = ConfigLoader().with_toml(toml_file).build()
        provider = config.gateway.providers[0]
        assert provider.kind is ProviderKind.OPENAI_COMPATIBLE
        assert provider.privacy_tier is PrivacyTier.PUBLIC_CLOUD
        assert provider.models[0].input_cost_per_1k == 0.59

    def test_load_config_helper_uses_standard_chain(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "eden.toml"
        toml_file.write_text('app_name = "helper"', encoding="utf-8")
        config = load_config(config_path=toml_file, environ={}, overrides={"version": "2.0"})
        assert config.app_name == "helper"
        assert config.version == "2.0"


class TestConfigValidation:
    def test_unknown_enum_value_is_rejected_with_the_allowed_set(self) -> None:
        with pytest.raises(InvalidConfigError) as caught:
            build_config({"environment": "banana"})
        assert "allowed" in caught.value.context

    def test_non_numeric_where_number_expected_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            build_config({"server": {"port": "not-a-port"}})

    def test_out_of_range_port_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            build_config({"server": {"port": "70000"}})

    def test_network_provider_without_base_url_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            ProviderConfig(name="x", kind=ProviderKind.ANTHROPIC)

    def test_custom_provider_without_implementation_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            ProviderConfig(name="x", kind=ProviderKind.CUSTOM)

    def test_default_model_outside_catalogue_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            build_config(
                {
                    "gateway": {
                        "providers": [
                            {
                                "name": "p",
                                "kind": "mock",
                                "default_model": "absent",
                                "models": [{"name": "present"}],
                            }
                        ]
                    }
                }
            )

    def test_duplicate_provider_names_are_rejected(self) -> None:
        duplicate = ProviderConfig(name="same", kind=ProviderKind.MOCK)
        with pytest.raises(InvalidConfigError):
            GatewayConfig(providers=(duplicate, duplicate))

    def test_all_zero_router_weights_are_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            RouterWeights(cost=0, latency=0, health=0, privacy=0, preference=0)

    def test_invalid_jitter_ratio_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigError):
            RetryConfig(jitter_ratio=1.5)

    def test_config_is_immutable(self) -> None:
        config = EdenConfig()
        with pytest.raises(AttributeError):
            config.app_name = "mutated"  # type: ignore[misc]
