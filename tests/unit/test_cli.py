"""Tests for CLI configuration loading, in particular its logging defaults.

The CLI is interactive, so it defaults to a quiet log level: the subsystem
startup/shutdown trace is useful for a long-running service, but is noise for
someone running one command and reading its answer. These tests exist to pin
that default and, just as importantly, to prove it never overrides a level the
person actually asked for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from eden.config.enums import LogLevel
from eden.interface.cli import _load, build_parser


def parse(*argv: str) -> argparse.Namespace:
    """Parse CLI arguments the way ``main`` would."""
    return build_parser().parse_args(argv)


class TestQuietDefault:
    def test_no_flag_and_no_file_setting_defaults_to_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        args = parse("status")
        config = _load(args)
        assert config.logging.level is LogLevel.WARNING

    def test_explicit_flag_overrides_the_quiet_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        args = parse("--log-level", "DEBUG", "status")
        config = _load(args)
        assert config.logging.level is LogLevel.DEBUG

    def test_a_level_set_in_eden_toml_is_respected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "eden.toml").write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        args = parse("status")
        config = _load(args)
        assert config.logging.level is LogLevel.INFO

    def test_the_flag_still_wins_over_a_file_setting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "eden.toml").write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        args = parse("--log-level", "ERROR", "status")
        config = _load(args)
        assert config.logging.level is LogLevel.ERROR

    def test_quiet_default_does_not_disable_other_subsystems(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fix must only touch verbosity, nothing else in configuration."""
        monkeypatch.chdir(tmp_path)
        args = parse("status")
        config = _load(args)
        assert config.gateway is not None
        assert config.memory.enabled is True
