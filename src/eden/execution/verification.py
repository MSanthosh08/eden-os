"""Verification.

Verification answers one question: *what could this action do, and how bad
would that be?* It never decides whether the action is allowed — that is
:mod:`eden.execution.permissions` — and it never performs anything.

Each rule is an independent :class:`Verifier` producing :class:`Finding`
objects. :class:`CompositeVerifier` runs them all and aggregates. Splitting the
rules apart means each is a handful of lines with an obvious test, and a
deployment can add a domain-specific rule without touching the others.

A finding marked ``blocking`` is a hard stop: no risk threshold, approval or
override can clear it. That is the same filter-versus-preference discipline the
router uses in ADR-0002.
"""

from __future__ import annotations

import abc
import os
import re
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from eden.config.enums import ActionKind, RiskLevel
from eden.config.schema import ExecutionConfig
from eden.errors import ValidationError
from eden.execution.handlers import (
    ARGUMENTS_PARAMETER,
    COMMAND_PARAMETER,
    CONTENT_PARAMETER,
    PATH_PARAMETER,
)
from eden.execution.types import Action, Finding, Preparation, Verdict
from eden.logging import get_logger

_LOGGER = get_logger("execution.verification")

# Characters that only matter if something later hands the string to a shell.
# Nothing in EDEN does, but a command containing them signals either confusion
# or an attempt, and both are worth refusing loudly.
_SHELL_METACHARACTERS = re.compile(r"[;&|<>$`\n\r\\]")

_REQUIRED_PARAMETERS: dict[ActionKind, tuple[str, ...]] = {
    ActionKind.FILE_WRITE: (PATH_PARAMETER, CONTENT_PARAMETER),
    ActionKind.FILE_DELETE: (PATH_PARAMETER,),
    ActionKind.SHELL_COMMAND: (COMMAND_PARAMETER,),
}


class Verifier(abc.ABC):
    """One verification rule."""

    def __init__(self, config: ExecutionConfig) -> None:
        """Initialise the rule with the execution policy."""
        self._config = config

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the rule name, used in diagnostics."""

    @abc.abstractmethod
    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Return every finding this rule observes about ``action``."""


class SchemaVerifier(Verifier):
    """Checks that an action carries the parameters its kind requires."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "schema"

    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Report any missing or mistyped parameter."""
        del preparation
        findings: list[Finding] = []
        for parameter in _REQUIRED_PARAMETERS.get(action.kind, ()):
            if parameter not in action.parameters:
                findings.append(
                    Finding(
                        code="schema.missing_parameter",
                        message=f"Action of kind '{action.kind.value}' requires '{parameter}'.",
                        risk=RiskLevel.HIGH,
                        blocking=True,
                    )
                )
            elif not isinstance(action.parameters[parameter], str) and parameter != (
                ARGUMENTS_PARAMETER
            ):
                findings.append(
                    Finding(
                        code="schema.invalid_parameter",
                        message=f"Parameter '{parameter}' must be a string.",
                        risk=RiskLevel.HIGH,
                        blocking=True,
                    )
                )
        return findings


class WorkspaceScopeVerifier(Verifier):
    """Confines filesystem actions to the configured workspace.

    Paths are compared after full symlink resolution, so neither ``..`` nor a
    symlink pointing outside the workspace can escape the boundary.
    """

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "workspace_scope"

    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Report a blocking finding when the target lies outside the workspace."""
        del preparation
        if action.kind not in (ActionKind.FILE_WRITE, ActionKind.FILE_DELETE):
            return []
        target = _target_path(action, self._config)
        if target is None:
            return []
        root = _resolve(self._config.workspace_root)
        if not _is_within(target, root):
            return [
                Finding(
                    code="path.outside_workspace",
                    message=(
                        f"Target resolves to {target}, which is outside the " f"workspace {root}."
                    ),
                    risk=RiskLevel.CRITICAL,
                    blocking=True,
                )
            ]
        return []


class SensitivePathVerifier(Verifier):
    """Refuses paths matching the configured sensitive-file patterns."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "sensitive_path"

    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Report a blocking finding for credential-bearing paths."""
        del preparation
        if action.kind not in (ActionKind.FILE_WRITE, ActionKind.FILE_DELETE):
            return []
        target = _target_path(action, self._config)
        if target is None:
            return []
        for pattern in self._config.denied_path_globs:
            if _matches_glob(target, pattern):
                return [
                    Finding(
                        code="path.sensitive",
                        message=(
                            f"Target matches the protected pattern '{pattern}'. "
                            "Files matching it may hold credentials."
                        ),
                        risk=RiskLevel.CRITICAL,
                        blocking=True,
                    )
                ]
        return []


class CommandAllowlistVerifier(Verifier):
    """Permits only explicitly allowlisted executables."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "command_allowlist"

    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Report findings for disallowed or suspicious commands."""
        del preparation
        if action.kind is not ActionKind.SHELL_COMMAND:
            return []
        raw = action.parameters.get(COMMAND_PARAMETER)
        if not isinstance(raw, str):
            return []
        command = raw.strip()
        findings: list[Finding] = []

        if _SHELL_METACHARACTERS.search(command):
            findings.append(
                Finding(
                    code="command.metacharacters",
                    message=(
                        "Command contains shell metacharacters. EDEN never invokes a "
                        "shell, so these cannot do what they appear to; the command "
                        "is refused because it is not what it claims to be."
                    ),
                    risk=RiskLevel.CRITICAL,
                    blocking=True,
                )
            )

        if not self._config.allowed_commands:
            findings.append(
                Finding(
                    code="command.no_allowlist",
                    message=(
                        "No commands are allowlisted. Add the executable to "
                        "execution.allowed_commands to permit it."
                    ),
                    risk=RiskLevel.HIGH,
                    blocking=True,
                )
            )
        elif command not in self._config.allowed_commands:
            findings.append(
                Finding(
                    code="command.not_allowlisted",
                    message=f"'{command}' is not in execution.allowed_commands.",
                    risk=RiskLevel.HIGH,
                    blocking=True,
                )
            )
        else:
            findings.append(
                Finding(
                    code="command.execution",
                    message=f"Runs the external program '{command}'.",
                    risk=RiskLevel.MODERATE,
                )
            )
        return findings


class PayloadSizeVerifier(Verifier):
    """Bounds the size of content an action may write."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "payload_size"

    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Report a blocking finding for oversized payloads."""
        del preparation
        content = action.parameters.get(CONTENT_PARAMETER)
        if not isinstance(content, str):
            return []
        size = len(content.encode("utf-8"))
        if size > self._config.max_payload_bytes:
            return [
                Finding(
                    code="payload.too_large",
                    message=(
                        f"Payload of {size} bytes exceeds the configured limit of "
                        f"{self._config.max_payload_bytes}."
                    ),
                    risk=RiskLevel.HIGH,
                    blocking=True,
                )
            ]
        return []


class ReversibilityVerifier(Verifier):
    """Raises the assessed risk of anything that cannot be undone.

    This is the rule that gives :class:`~eden.execution.types.Preparation` its
    teeth. A handler that cannot describe an undo makes its action strictly
    more dangerous, and policy sees that as a risk level rather than as a
    footnote.
    """

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "reversibility"

    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Report a finding when no rollback plan exists."""
        if preparation.reversible:
            return []
        reason = preparation.notes.get("reason", "no rollback plan was produced")
        finding = Finding(
            code="rollback.unavailable",
            message=f"This action cannot be undone: {reason}.",
            risk=RiskLevel.HIGH,
            blocking=self._config.require_reversible,
        )
        _LOGGER.debug(
            "Irreversible action verified.",
            extra={"action": action.id, "kind": action.kind.value},
        )
        return [finding]


class DestructiveActionVerifier(Verifier):
    """Flags actions that destroy or overwrite existing state."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "destructive"

    def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
        """Report a finding proportional to what is being destroyed."""
        existed = bool(preparation.notes.get("existed", False))
        if action.kind is ActionKind.FILE_DELETE and existed:
            return [
                Finding(
                    code="effect.delete",
                    message="Deletes an existing file.",
                    risk=RiskLevel.MODERATE,
                )
            ]
        if action.kind is ActionKind.FILE_WRITE and existed:
            return [
                Finding(
                    code="effect.overwrite",
                    message="Overwrites an existing file.",
                    risk=RiskLevel.MODERATE,
                )
            ]
        if action.kind is ActionKind.FILE_WRITE:
            return [
                Finding(
                    code="effect.create",
                    message="Creates a new file.",
                    risk=RiskLevel.LOW,
                )
            ]
        return []


class CompositeVerifier:
    """Runs every rule and aggregates the findings into one verdict."""

    def __init__(self, verifiers: Sequence[Verifier]) -> None:
        """Initialise the composite with its rules, in order."""
        self._verifiers = tuple(verifiers)

    @property
    def rules(self) -> tuple[str, ...]:
        """Return the names of every registered rule."""
        return tuple(verifier.name for verifier in self._verifiers)

    def verify(self, action: Action, preparation: Preparation) -> Verdict:
        """Return the aggregated verdict for ``action``.

        A rule that raises is itself treated as a blocking finding: a
        verification step that cannot complete must never be read as approval.
        """
        findings: list[Finding] = []
        for verifier in self._verifiers:
            try:
                findings.extend(verifier.verify(action, preparation))
            except ValidationError as exc:
                findings.append(
                    Finding(
                        code="verification.invalid_action",
                        message=exc.message,
                        risk=RiskLevel.HIGH,
                        blocking=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a failed check must not pass
                findings.append(
                    Finding(
                        code="verification.rule_failed",
                        message=(
                            f"Verification rule '{verifier.name}' failed to complete "
                            f"({type(exc).__name__}); the action is refused."
                        ),
                        risk=RiskLevel.CRITICAL,
                        blocking=True,
                    )
                )
        return Verdict(
            action_id=action.id,
            findings=tuple(findings),
            reversible=preparation.reversible,
        )


def default_verifier(config: ExecutionConfig) -> CompositeVerifier:
    """Return the verification suite EDEN ships with.

    Args:
        config: Execution policy shared by every rule.

    Returns:
        A composite running schema, scope, sensitivity, allowlist, size,
        reversibility and destructiveness checks.
    """
    return CompositeVerifier(
        [
            SchemaVerifier(config),
            WorkspaceScopeVerifier(config),
            SensitivePathVerifier(config),
            CommandAllowlistVerifier(config),
            PayloadSizeVerifier(config),
            ReversibilityVerifier(config),
            DestructiveActionVerifier(config),
        ]
    )


def _target_path(action: Action, config: ExecutionConfig) -> Path | None:
    """Return the resolved target path of a filesystem action, if well-formed."""
    raw = action.parameters.get(PATH_PARAMETER)
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = config.workspace_root / candidate
    return _resolve(candidate)


def _matches_glob(path: Path, pattern: str) -> bool:
    """Return whether ``path`` matches ``pattern``.

    ``Path.full_match`` is Python 3.13+, so matching is done with
    :func:`fnmatch.fnmatch`, whose ``*`` spans separators. That is *more*
    permissive than a strict glob, which is the correct direction of error for
    a denylist: over-matching refuses something harmless, under-matching lets a
    credential file through.

    Both the whole path and the bare filename are tested, so ``**/id_rsa*``
    catches the file wherever it sits. The filename fallback is skipped when the
    pattern's final segment is pure wildcard — the tail of ``**/.git/**`` is
    ``**``, which would otherwise match every file in existence.
    """
    text = path.as_posix()
    if fnmatch(text, pattern):
        return True
    if pattern.startswith("**/") and fnmatch(text, pattern[3:]):
        return True
    tail = PurePosixPath(pattern).name
    if not tail or set(tail) <= {"*"}:
        return False
    return fnmatch(path.name, tail)


def _resolve(path: Path) -> Path:
    """Return an absolute, symlink-resolved path even if it does not exist."""
    return Path(os.path.realpath(path.expanduser()))


def _is_within(candidate: Path, root: Path) -> bool:
    """Return whether ``candidate`` lies inside ``root``."""
    return candidate == root or root in candidate.parents
