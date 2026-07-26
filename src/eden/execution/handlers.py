"""Action handlers.

A handler is the only thing in EDEN that touches the world, and it does so
through two clearly separated methods:

``prepare`` reads. It captures whatever is needed to undo the action *before*
the action happens, and returns ``None`` for the rollback plan when the effect
genuinely cannot be undone. It must have no side effects.

``execute`` writes. It is reached only after verification and permission have
both passed, and never directly by a caller.

Shipping both a reversible handler and an irreversible one is deliberate: it
makes the reversibility contract concrete rather than theoretical, and lets
policy be tested against both.
"""

from __future__ import annotations

import abc
import asyncio
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eden.config.enums import ActionKind
from eden.config.schema import ExecutionConfig
from eden.errors import ActionExecutionError, ValidationError
from eden.execution.types import (
    Action,
    ExecutionResult,
    Preparation,
    RollbackPlan,
)
from eden.logging import get_logger

PATH_PARAMETER = "path"
CONTENT_PARAMETER = "content"
COMMAND_PARAMETER = "command"
ARGUMENTS_PARAMETER = "arguments"

_SAFE_ENVIRONMENT_KEYS = ("PATH", "LANG", "LC_ALL", "TZ", "HOME")


class ActionHandler(abc.ABC):
    """Performs one kind of action."""

    def __init__(self, config: ExecutionConfig) -> None:
        """Initialise the handler with the execution policy."""
        self._config = config
        self._logger = get_logger(f"execution.handler.{self.kind.value}")

    @property
    @abc.abstractmethod
    def kind(self) -> ActionKind:
        """Return the action kind this handler performs."""

    @abc.abstractmethod
    async def prepare(self, action: Action) -> Preparation:
        """Capture undo state without causing any side effect.

        Returns:
            A preparation whose rollback plan is ``None`` when the action
            cannot be undone.
        """

    @abc.abstractmethod
    async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
        """Perform the action.

        Called only after verification and permission have passed.

        Raises:
            ActionExecutionError: If the effect could not be achieved.
        """

    def resolve_path(self, action: Action, parameter: str = PATH_PARAMETER) -> Path:
        """Return the absolute, symlink-resolved path named by ``parameter``.

        Resolution happens here rather than in verification so that both use
        the *same* path. Verifying one path and writing to another is the
        classic time-of-check-to-time-of-use hole.

        Raises:
            ValidationError: If the parameter is missing or not a string.
        """
        raw = action.text_parameter(parameter)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._config.workspace_root / candidate
        return _resolve_without_requiring_existence(candidate)


class NoOpHandler(ActionHandler):
    """Records an intent without touching anything.

    Used for dry-run planning and as the trivially reversible case: it has no
    effect, so its rollback plan is empty rather than absent.
    """

    @property
    def kind(self) -> ActionKind:
        """Return the action kind this handler performs."""
        return ActionKind.NOOP

    async def prepare(self, action: Action) -> Preparation:
        """Return an empty, reversible plan."""
        return Preparation(
            rollback=RollbackPlan(description="No effect to undo."),
            notes={"summary": action.summary},
        )

    async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
        """Return success without performing anything."""
        del preparation
        return ExecutionResult(succeeded=True, output=action.summary)


class FileWriteHandler(ActionHandler):
    """Writes text to a file, capturing the previous contents for rollback.

    Reversibility depends on what is already there. Overwriting a UTF-8 file is
    reversible by restoring the old text; creating a new file is reversible by
    deleting it; overwriting a *binary* file is not reversible through this
    handler, so it honestly reports that rather than silently corrupting the
    original on undo.
    """

    @property
    def kind(self) -> ActionKind:
        """Return the action kind this handler performs."""
        return ActionKind.FILE_WRITE

    async def prepare(self, action: Action) -> Preparation:
        """Capture the existing file contents, if any."""
        path = self.resolve_path(action)
        if not path.exists():
            return Preparation(
                rollback=RollbackPlan(
                    steps=(
                        Action(
                            kind=ActionKind.FILE_DELETE,
                            summary=f"Delete {path} created by rollback of {action.id}.",
                            parameters={PATH_PARAMETER: str(path)},
                            actor=action.actor,
                            namespace=action.namespace,
                            metadata={"rollback_for": action.id},
                        ),
                    ),
                    description=f"Delete the newly created file {path}.",
                ),
                notes={"existed": False, "path": str(path)},
            )

        previous = await asyncio.to_thread(_read_text_or_none, path)
        if previous is None:
            return Preparation(
                rollback=None,
                notes={
                    "existed": True,
                    "path": str(path),
                    "reason": "existing file is not valid UTF-8 text",
                },
            )
        return Preparation(
            rollback=RollbackPlan(
                steps=(
                    Action(
                        kind=ActionKind.FILE_WRITE,
                        summary=f"Restore previous contents of {path}.",
                        parameters={PATH_PARAMETER: str(path), CONTENT_PARAMETER: previous},
                        actor=action.actor,
                        namespace=action.namespace,
                        metadata={"rollback_for": action.id},
                    ),
                ),
                description=f"Restore the previous {len(previous)} characters of {path}.",
            ),
            notes={"existed": True, "path": str(path), "previous_length": len(previous)},
        )

    async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
        """Write the content to disk.

        Raises:
            ActionExecutionError: If the write fails.
        """
        del preparation
        path = self.resolve_path(action)
        content = action.text_parameter(CONTENT_PARAMETER)
        try:
            await asyncio.to_thread(_write_text, path, content)
        except OSError as exc:
            raise ActionExecutionError(
                "Could not write the file.",
                context={"action": action.id, "path": str(path)},
                cause=exc,
            ) from exc
        return ExecutionResult(
            succeeded=True,
            output=f"Wrote {len(content)} characters to {path}.",
            detail={"path": str(path), "bytes": len(content.encode("utf-8"))},
        )


class FileDeleteHandler(ActionHandler):
    """Deletes a file, capturing its contents so the delete can be undone."""

    @property
    def kind(self) -> ActionKind:
        """Return the action kind this handler performs."""
        return ActionKind.FILE_DELETE

    async def prepare(self, action: Action) -> Preparation:
        """Capture the file contents so it can be restored."""
        path = self.resolve_path(action)
        if not path.exists():
            return Preparation(
                rollback=RollbackPlan(description="Nothing to delete; nothing to undo."),
                notes={"existed": False, "path": str(path)},
            )
        if path.is_dir():
            return Preparation(
                rollback=None,
                notes={"path": str(path), "reason": "target is a directory"},
            )
        previous = await asyncio.to_thread(_read_text_or_none, path)
        if previous is None:
            return Preparation(
                rollback=None,
                notes={"path": str(path), "reason": "file is not valid UTF-8 text"},
            )
        return Preparation(
            rollback=RollbackPlan(
                steps=(
                    Action(
                        kind=ActionKind.FILE_WRITE,
                        summary=f"Restore deleted file {path}.",
                        parameters={PATH_PARAMETER: str(path), CONTENT_PARAMETER: previous},
                        actor=action.actor,
                        namespace=action.namespace,
                        metadata={"rollback_for": action.id},
                    ),
                ),
                description=f"Recreate {path} with its previous contents.",
            ),
            notes={"existed": True, "path": str(path), "previous_length": len(previous)},
        )

    async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
        """Delete the file.

        Raises:
            ActionExecutionError: If deletion fails.
        """
        del preparation
        path = self.resolve_path(action)
        try:
            existed = await asyncio.to_thread(_unlink, path)
        except OSError as exc:
            raise ActionExecutionError(
                "Could not delete the file.",
                context={"action": action.id, "path": str(path)},
                cause=exc,
            ) from exc
        return ExecutionResult(
            succeeded=True,
            output=f"Deleted {path}." if existed else f"{path} did not exist.",
            detail={"path": str(path), "existed": existed},
        )


class ShellCommandHandler(ActionHandler):
    """Runs an allowlisted executable with explicit arguments.

    Four properties make this safe enough to ship:

    * **No shell.** The command is executed directly, so quoting, globbing,
      pipes, redirection and ``;`` chaining have no meaning. There is no string
      for an injection to hide in.
    * **Allowlist, not denylist.** Only executables named in
      ``execution.allowed_commands`` may run, and that list starts empty.
    * **Scrubbed environment.** The child inherits a small fixed set of
      variables, so credentials in the parent process are never passed down.
    * **Irreversible by contract.** ``prepare`` returns no rollback plan,
      because a handler cannot know how to undo an arbitrary program. Policy
      therefore treats every command as irreversible and refuses to
      auto-approve it.
    """

    @property
    def kind(self) -> ActionKind:
        """Return the action kind this handler performs."""
        return ActionKind.SHELL_COMMAND

    async def prepare(self, action: Action) -> Preparation:
        """Report that command execution cannot be undone."""
        return Preparation(
            rollback=None,
            notes={
                "command": action.text_parameter(COMMAND_PARAMETER),
                "reason": "the effects of an arbitrary program cannot be described in advance",
            },
        )

    async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
        """Run the command and capture its truncated output.

        Raises:
            ActionExecutionError: If the executable is missing, the command
                times out, or it exits non-zero.
        """
        del preparation
        command = action.text_parameter(COMMAND_PARAMETER)
        arguments = _string_sequence(action.parameters.get(ARGUMENTS_PARAMETER, ()))
        executable = shutil.which(command)
        if executable is None:
            raise ActionExecutionError(
                "Command was not found on the executable path.",
                context={"action": action.id, "command": command},
            )

        workspace = _resolve_without_requiring_existence(self._config.workspace_root)
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *arguments,
                cwd=workspace,
                env=_scrubbed_environment(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ActionExecutionError(
                "Command could not be started.",
                context={"action": action.id, "command": command},
                cause=exc,
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._config.timeout_seconds
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ActionExecutionError(
                "Command exceeded its time budget and was terminated.",
                context={
                    "action": action.id,
                    "command": command,
                    "timeout_seconds": self._config.timeout_seconds,
                },
                cause=exc,
            ) from exc

        output = _truncate(stdout, self._config.max_output_bytes)
        errors = _truncate(stderr, self._config.max_output_bytes)
        if process.returncode != 0:
            raise ActionExecutionError(
                "Command exited with a non-zero status.",
                context={
                    "action": action.id,
                    "command": command,
                    "exit_code": process.returncode,
                    "stderr": errors,
                },
            )
        return ExecutionResult(
            succeeded=True,
            output=output,
            detail={"command": command, "exit_code": 0, "stderr": errors},
        )


def default_handlers(config: ExecutionConfig) -> list[ActionHandler]:
    """Return the handlers EDEN ships with.

    Args:
        config: Execution policy passed to each handler.

    Returns:
        One handler per built-in action kind.
    """
    return [
        NoOpHandler(config),
        FileWriteHandler(config),
        FileDeleteHandler(config),
        ShellCommandHandler(config),
    ]


# ---------------------------------------------------------------------------
# Blocking helpers, run off the event loop
# ---------------------------------------------------------------------------
def _read_text_or_none(path: Path) -> str | None:
    """Return the file's text, or ``None`` when it is not UTF-8 decodable."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _unlink(path: Path) -> bool:
    """Delete ``path``. Returns whether it existed."""
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def _resolve_without_requiring_existence(path: Path) -> Path:
    """Return an absolute, symlink-resolved path even if it does not yet exist."""
    return Path(os.path.realpath(path.expanduser()))


def _scrubbed_environment() -> dict[str, str]:
    """Return a minimal environment, so credentials are never inherited."""
    return {key: os.environ[key] for key in _SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _truncate(raw: bytes, limit: int) -> str:
    """Decode and truncate captured output."""
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n<truncated at {limit} characters>"


def _string_sequence(value: Any) -> list[str]:  # noqa: ANN401 - raw parameter payload
    """Coerce an argument list to strings.

    Raises:
        ValidationError: If the value is not a list of scalars.
    """
    if isinstance(value, str):
        raise ValidationError(
            "Command arguments must be a list, not a single string. A string "
            "would have to be split, and splitting is where injection lives."
        )
    if not isinstance(value, Sequence):
        raise ValidationError("Command arguments must be a list.")
    for item in value:
        if isinstance(item, Mapping | Sequence) and not isinstance(item, str):
            raise ValidationError("Command arguments must be scalars.")
    return [str(item) for item in value]
