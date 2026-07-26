"""Unit tests for the execution pipeline.

A large share of these are adversarial: path traversal, symlink escape,
credential filenames, shell metacharacters. They exist because the pipeline's
entire purpose is to refuse those, and a guard nobody tries to defeat is a
guard nobody knows works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eden.config.enums import (
    ActionKind,
    ExecutionStatus,
    PermissionMode,
    RiskLevel,
)
from eden.config.schema import ExecutionConfig
from eden.errors import (
    ActionExecutionError,
    HandlerNotFoundError,
    InvalidConfigError,
    RollbackError,
    ValidationError,
)
from eden.execution.engine import ExecutionEngine
from eden.execution.handlers import (
    FileDeleteHandler,
    FileWriteHandler,
    NoOpHandler,
    ShellCommandHandler,
)
from eden.execution.journal import InMemoryJournal, JsonlJournal, NullJournal
from eden.execution.permissions import (
    AlwaysApproveGate,
    CallbackGate,
    DenyingGate,
    PolicyEngine,
)
from eden.execution.types import (
    Action,
    ExecutionResult,
    Finding,
    Preparation,
    RollbackPlan,
    Verdict,
)
from eden.execution.verification import default_verifier


def config_for(tmp_path: Path, **overrides: object) -> ExecutionConfig:
    """Return an execution policy rooted in a temporary workspace."""
    base: dict[str, object] = {"workspace_root": tmp_path / "workspace"}
    base.update(overrides)
    return ExecutionConfig(**base)  # type: ignore[arg-type]


def write_action(path: str, content: str = "hello", **kwargs: object) -> Action:
    """Return a file-write action."""
    return Action(
        kind=ActionKind.FILE_WRITE,
        summary=f"Write {path}",
        parameters={"path": path, "content": content},
        **kwargs,  # type: ignore[arg-type]
    )


def engine_for(tmp_path: Path, *, approve: bool = True, **overrides: object) -> ExecutionEngine:
    """Return an engine with an explicit approval posture."""
    config = config_for(tmp_path, **overrides)
    gate = AlwaysApproveGate() if approve else DenyingGate()
    return ExecutionEngine(config, policy=PolicyEngine(config, gate))


class TestActionType:
    def test_identifier_is_generated(self) -> None:
        assert write_action("a.txt").id.startswith("act-")

    def test_empty_summary_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Action(kind=ActionKind.NOOP, summary="  ")

    def test_missing_parameter_reports_the_name(self) -> None:
        action = Action(kind=ActionKind.NOOP, summary="x")
        with pytest.raises(ValidationError) as caught:
            action.parameter("absent")
        assert caught.value.context["parameter"] == "absent"

    def test_non_string_parameter_is_rejected(self) -> None:
        action = Action(kind=ActionKind.NOOP, summary="x", parameters={"n": 5})
        with pytest.raises(ValidationError):
            action.text_parameter("n")

    def test_journal_form_truncates_large_payloads(self) -> None:
        action = write_action("a.txt", content="x" * 5000)
        assert "characters" in str(action.to_dict()["parameters"]["content"])

    def test_action_has_no_way_to_execute_itself(self) -> None:
        """The central invariant: an action is data, not code."""
        for attribute in ("run", "execute", "apply", "perform", "__call__"):
            assert not hasattr(Action(kind=ActionKind.NOOP, summary="x"), attribute)


class TestVerdict:
    def test_risk_is_the_maximum_finding(self) -> None:
        verdict = Verdict(
            action_id="a",
            findings=(
                Finding(code="x", message="m", risk=RiskLevel.LOW),
                Finding(code="y", message="m", risk=RiskLevel.HIGH),
            ),
        )
        assert verdict.risk is RiskLevel.HIGH

    def test_no_findings_means_no_risk(self) -> None:
        assert Verdict(action_id="a").risk is RiskLevel.NONE

    def test_blocking_is_surfaced(self) -> None:
        verdict = Verdict(
            action_id="a",
            findings=(Finding(code="x", message="m", blocking=True),),
        )
        assert verdict.blocked is True
        assert len(verdict.blocking_findings) == 1


class TestPreparationContract:
    def test_absent_plan_means_irreversible(self) -> None:
        assert Preparation(rollback=None).reversible is False

    def test_empty_plan_means_reversible_with_nothing_to_do(self) -> None:
        preparation = Preparation(rollback=RollbackPlan())
        assert preparation.reversible is True
        assert preparation.rollback is not None
        assert preparation.rollback.is_empty is True


class TestVerification:
    async def test_missing_parameters_block(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        action = Action(kind=ActionKind.FILE_WRITE, summary="incomplete")
        verdict = default_verifier(config).verify(action, Preparation())
        assert verdict.blocked is True
        assert any(f.code == "schema.missing_parameter" for f in verdict.findings)

    @pytest.mark.parametrize(
        "target",
        ["../escape.txt", "../../etc/passwd", "/etc/passwd", "sub/../../out.txt"],
    )
    async def test_paths_outside_the_workspace_are_blocked(
        self, tmp_path: Path, target: str
    ) -> None:
        config = config_for(tmp_path)
        verdict = default_verifier(config).verify(write_action(target), Preparation())
        assert verdict.blocked is True
        assert any(f.code == "path.outside_workspace" for f in verdict.findings)

    async def test_symlink_escape_is_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (workspace / "link").symlink_to(outside)
        config = config_for(tmp_path)
        verdict = default_verifier(config).verify(write_action("link/captured.txt"), Preparation())
        assert verdict.blocked is True
        assert any(f.code == "path.outside_workspace" for f in verdict.findings)

    @pytest.mark.parametrize(
        "target", [".env", "nested/.env", "id_rsa", "deploy.pem", "credentials.json"]
    )
    async def test_credential_filenames_are_blocked(self, tmp_path: Path, target: str) -> None:
        config = config_for(tmp_path)
        verdict = default_verifier(config).verify(write_action(target), Preparation())
        assert verdict.blocked is True
        assert any(f.code == "path.sensitive" for f in verdict.findings)

    async def test_ordinary_workspace_path_is_permitted(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        verdict = default_verifier(config).verify(
            write_action("notes/report.md"),
            Preparation(rollback=RollbackPlan()),
        )
        assert verdict.blocked is False

    @pytest.mark.parametrize(
        "command", ["ls; rm -rf /", "echo $(whoami)", "cat a | tee b", "sh\nrm"]
    )
    async def test_shell_metacharacters_are_blocked(self, tmp_path: Path, command: str) -> None:
        config = config_for(tmp_path, allowed_commands=("ls", "echo", "cat", "sh"))
        action = Action(
            kind=ActionKind.SHELL_COMMAND,
            summary="run",
            parameters={"command": command},
        )
        verdict = default_verifier(config).verify(action, Preparation())
        assert verdict.blocked is True
        assert any(f.code == "command.metacharacters" for f in verdict.findings)

    async def test_empty_allowlist_forbids_every_command(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        action = Action(
            kind=ActionKind.SHELL_COMMAND, summary="run", parameters={"command": "echo"}
        )
        verdict = default_verifier(config).verify(action, Preparation())
        assert verdict.blocked is True
        assert any(f.code == "command.no_allowlist" for f in verdict.findings)

    async def test_unlisted_command_is_blocked(self, tmp_path: Path) -> None:
        config = config_for(tmp_path, allowed_commands=("echo",))
        action = Action(
            kind=ActionKind.SHELL_COMMAND, summary="run", parameters={"command": "curl"}
        )
        verdict = default_verifier(config).verify(action, Preparation())
        assert any(f.code == "command.not_allowlisted" for f in verdict.findings)

    async def test_oversized_payload_is_blocked(self, tmp_path: Path) -> None:
        config = config_for(tmp_path, max_payload_bytes=10)
        verdict = default_verifier(config).verify(
            write_action("a.txt", content="x" * 100), Preparation()
        )
        assert any(f.code == "payload.too_large" for f in verdict.findings)

    async def test_irreversibility_raises_risk(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        verdict = default_verifier(config).verify(write_action("a.txt"), Preparation(rollback=None))
        assert any(f.code == "rollback.unavailable" for f in verdict.findings)
        assert verdict.risk >= RiskLevel.HIGH

    async def test_require_reversible_turns_it_into_a_block(self, tmp_path: Path) -> None:
        config = config_for(tmp_path, require_reversible=True)
        verdict = default_verifier(config).verify(write_action("a.txt"), Preparation(rollback=None))
        assert verdict.blocked is True

    async def test_a_rule_that_raises_becomes_a_block(self, tmp_path: Path) -> None:
        """A verification step that cannot complete must never read as approval."""
        from eden.execution.verification import CompositeVerifier, Verifier

        class Exploding(Verifier):
            @property
            def name(self) -> str:
                return "exploding"

            def verify(self, action: Action, preparation: Preparation) -> list[Finding]:
                message = "rule is broken"
                raise RuntimeError(message)

        verdict = CompositeVerifier([Exploding(config_for(tmp_path))]).verify(
            write_action("a.txt"), Preparation()
        )
        assert verdict.blocked is True
        assert verdict.risk is RiskLevel.CRITICAL


class TestPolicy:
    def verdict(self, *, risk: RiskLevel, reversible: bool, blocked: bool = False) -> Verdict:
        return Verdict(
            action_id="a",
            findings=(Finding(code="c", message="m", risk=risk, blocking=blocked),),
            reversible=reversible,
        )

    async def test_blocking_finding_denies_regardless_of_the_gate(self, tmp_path: Path) -> None:
        policy = PolicyEngine(config_for(tmp_path), AlwaysApproveGate())
        permit = await policy.decide(
            write_action("a.txt"),
            self.verdict(risk=RiskLevel.LOW, reversible=True, blocked=True),
        )
        assert permit.granted is False
        assert permit.mode is PermissionMode.DENY

    async def test_risk_above_the_ceiling_denies(self, tmp_path: Path) -> None:
        policy = PolicyEngine(
            config_for(tmp_path, deny_above_risk=RiskLevel.MODERATE), AlwaysApproveGate()
        )
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.HIGH, reversible=True)
        )
        assert permit.granted is False

    async def test_low_risk_reversible_runs_automatically(self, tmp_path: Path) -> None:
        policy = PolicyEngine(config_for(tmp_path), DenyingGate())
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.LOW, reversible=True)
        )
        assert permit.granted is True
        assert permit.mode is PermissionMode.AUTOMATIC

    async def test_irreversible_never_auto_approves_however_mild(self, tmp_path: Path) -> None:
        """The reversibility rule outranks the risk threshold."""
        policy = PolicyEngine(config_for(tmp_path), DenyingGate())
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.NONE, reversible=False)
        )
        assert permit.mode is PermissionMode.CONFIRM
        assert permit.granted is False

    async def test_missing_approver_refuses(self, tmp_path: Path) -> None:
        policy = PolicyEngine(config_for(tmp_path))
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.MODERATE, reversible=True)
        )
        assert permit.granted is False

    async def test_approval_is_recorded_with_the_approver(self, tmp_path: Path) -> None:
        policy = PolicyEngine(config_for(tmp_path), AlwaysApproveGate("alice"))
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.MODERATE, reversible=True)
        )
        assert permit.granted is True
        assert permit.approver == "alice"

    async def test_callback_gate_is_consulted(self, tmp_path: Path) -> None:
        seen: list[str] = []

        async def approve(action: Action, verdict: Verdict) -> bool:
            seen.append(action.id)
            return verdict.risk <= RiskLevel.MODERATE

        policy = PolicyEngine(config_for(tmp_path), CallbackGate(approve))
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.MODERATE, reversible=True)
        )
        assert permit.granted is True
        assert len(seen) == 1

    async def test_a_broken_callback_refuses_rather_than_raising(self, tmp_path: Path) -> None:
        async def explode(action: Action, verdict: Verdict) -> bool:
            message = "approval service is down"
            raise RuntimeError(message)

        policy = PolicyEngine(config_for(tmp_path), CallbackGate(explode))
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.MODERATE, reversible=True)
        )
        assert permit.granted is False

    async def test_default_deny_mode_refuses_everything_unmatched(self, tmp_path: Path) -> None:
        policy = PolicyEngine(
            config_for(tmp_path, default_mode=PermissionMode.DENY), AlwaysApproveGate()
        )
        permit = await policy.decide(
            write_action("a.txt"), self.verdict(risk=RiskLevel.LOW, reversible=True)
        )
        assert permit.granted is False

    def test_contradictory_thresholds_are_rejected_at_config_time(self) -> None:
        with pytest.raises(InvalidConfigError):
            ExecutionConfig(auto_approve_max_risk=RiskLevel.CRITICAL, deny_above_risk=RiskLevel.LOW)


class TestHandlers:
    async def test_write_to_a_new_file_rolls_back_by_deleting(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        handler = FileWriteHandler(config)
        action = write_action("new.txt")
        preparation = await handler.prepare(action)
        assert preparation.reversible is True
        assert preparation.rollback is not None
        assert preparation.rollback.steps[0].kind is ActionKind.FILE_DELETE

    async def test_overwrite_rolls_back_by_restoring(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        target = config.workspace_root / "existing.txt"
        target.parent.mkdir(parents=True)
        target.write_text("original", encoding="utf-8")
        preparation = await FileWriteHandler(config).prepare(write_action("existing.txt"))
        assert preparation.rollback is not None
        assert preparation.rollback.steps[0].parameters["content"] == "original"

    async def test_overwriting_a_binary_file_is_reported_irreversible(self, tmp_path: Path) -> None:
        """Honest irreversibility beats a rollback that would corrupt the file."""
        config = config_for(tmp_path)
        target = config.workspace_root / "image.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"\xff\xfe\x00\x01binary")
        preparation = await FileWriteHandler(config).prepare(write_action("image.bin"))
        assert preparation.reversible is False
        assert "UTF-8" in str(preparation.notes["reason"])

    async def test_delete_captures_content_for_restoration(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        target = config.workspace_root / "doomed.txt"
        target.parent.mkdir(parents=True)
        target.write_text("save me", encoding="utf-8")
        action = Action(
            kind=ActionKind.FILE_DELETE,
            summary="delete",
            parameters={"path": "doomed.txt"},
        )
        preparation = await FileDeleteHandler(config).prepare(action)
        assert preparation.rollback is not None
        assert preparation.rollback.steps[0].parameters["content"] == "save me"

    async def test_deleting_a_directory_is_irreversible(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        (config.workspace_root / "folder").mkdir(parents=True)
        action = Action(kind=ActionKind.FILE_DELETE, summary="d", parameters={"path": "folder"})
        assert (await FileDeleteHandler(config).prepare(action)).reversible is False

    async def test_shell_commands_are_always_irreversible(self, tmp_path: Path) -> None:
        action = Action(
            kind=ActionKind.SHELL_COMMAND, summary="run", parameters={"command": "echo"}
        )
        preparation = await ShellCommandHandler(config_for(tmp_path)).prepare(action)
        assert preparation.reversible is False

    async def test_noop_is_reversible_with_nothing_to_undo(self, tmp_path: Path) -> None:
        preparation = await NoOpHandler(config_for(tmp_path)).prepare(
            Action(kind=ActionKind.NOOP, summary="x")
        )
        assert preparation.reversible is True

    async def test_shell_arguments_must_be_a_list_not_a_string(self, tmp_path: Path) -> None:
        """A string would have to be split, and splitting is where injection lives."""
        config = config_for(tmp_path, allowed_commands=("echo",))
        action = Action(
            kind=ActionKind.SHELL_COMMAND,
            summary="run",
            parameters={"command": "echo", "arguments": "a b; rm -rf /"},
        )
        with pytest.raises(ValidationError):
            await ShellCommandHandler(config).execute(action, Preparation())

    async def test_unknown_executable_fails_cleanly(self, tmp_path: Path) -> None:
        config = config_for(tmp_path, allowed_commands=("definitely-not-a-real-binary",))
        action = Action(
            kind=ActionKind.SHELL_COMMAND,
            summary="run",
            parameters={"command": "definitely-not-a-real-binary"},
        )
        with pytest.raises(ActionExecutionError):
            await ShellCommandHandler(config).execute(action, Preparation())


class TestEnginePipeline:
    async def test_noop_succeeds(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        record = await engine.submit(Action(kind=ActionKind.NOOP, summary="nothing"))
        assert record.status is ExecutionStatus.SUCCEEDED

    async def test_write_reaches_the_filesystem(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        record = await engine.submit(write_action("out.txt", "content"))
        assert record.succeeded is True
        assert (engine._config.workspace_root / "out.txt").read_text(encoding="utf-8") == "content"

    async def test_escape_attempt_is_rejected_before_touching_disk(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        record = await engine.submit(write_action("../escaped.txt"))
        assert record.status is ExecutionStatus.REJECTED
        assert record.executed is False
        assert not (tmp_path / "escaped.txt").exists()

    async def test_denied_action_never_executes(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path, approve=False)
        target = tmp_path / "workspace" / "existing.txt"
        target.parent.mkdir(parents=True)
        target.write_text("untouched", encoding="utf-8")
        record = await engine.submit(write_action("existing.txt", "overwritten"))
        assert record.status is ExecutionStatus.DENIED
        assert target.read_text(encoding="utf-8") == "untouched"

    async def test_dry_run_verifies_and_permits_but_does_nothing(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path, dry_run=True)
        record = await engine.submit(write_action("planned.txt"))
        assert record.status is ExecutionStatus.SKIPPED
        assert record.permit is not None
        assert record.permit.granted is True
        assert not (tmp_path / "workspace" / "planned.txt").exists()

    async def test_review_inspects_without_executing(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        verdict, preparation = await engine.review(write_action("preview.txt"))
        assert verdict.reversible is True
        assert preparation.notes["existed"] is False
        assert not (tmp_path / "workspace" / "preview.txt").exists()

    async def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        engine = ExecutionEngine(config_for(tmp_path), handlers=[])
        with pytest.raises(HandlerNotFoundError):
            await engine.submit(Action(kind=ActionKind.NOOP, summary="x"))

    async def test_handler_failure_triggers_automatic_rollback(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        target = config.workspace_root / "fragile.txt"
        target.parent.mkdir(parents=True)
        target.write_text("original", encoding="utf-8")

        class FailingWrite(FileWriteHandler):
            """Writes, then fails — the worst case, because the effect landed.

            Compensating steps carry ``rollback_for`` metadata and are allowed
            through, which mirrors reality: the write itself works, something
            after it (a checksum, a permission fix) is what failed.
            """

            async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
                result = await super().execute(action, preparation)
                if "rollback_for" in action.metadata:
                    return result
                raise ActionExecutionError("write failed after the effect landed")

        engine = ExecutionEngine(
            config,
            policy=PolicyEngine(config, AlwaysApproveGate()),
            handlers=[FailingWrite(config), FileDeleteHandler(config)],
        )
        record = await engine.submit(write_action("fragile.txt", "corrupt"))
        assert record.status is ExecutionStatus.ROLLED_BACK
        assert record.rollback_applied is True
        assert target.read_text(encoding="utf-8") == "original"

    async def test_a_crashing_handler_does_not_escape(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)

        class Crashing(NoOpHandler):
            async def execute(self, action: Action, preparation: Preparation) -> ExecutionResult:
                message = "unexpected"
                raise RuntimeError(message)

        engine = ExecutionEngine(
            config,
            policy=PolicyEngine(config, AlwaysApproveGate()),
            handlers=[Crashing(config)],
        )
        record = await engine.submit(Action(kind=ActionKind.NOOP, summary="boom"))
        assert record.status in (ExecutionStatus.FAILED, ExecutionStatus.ROLLED_BACK)
        assert record.error is not None

    async def test_explicit_rollback_restores_state(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        config = config_for(tmp_path)
        target = config.workspace_root / "doc.txt"
        target.parent.mkdir(parents=True)
        target.write_text("v1", encoding="utf-8")

        action = write_action("doc.txt", "v2")
        _, preparation = await engine.review(action)
        record = await engine.submit(action)
        assert target.read_text(encoding="utf-8") == "v2"

        rolled = await engine.rollback(record, preparation)
        assert rolled.status is ExecutionStatus.ROLLED_BACK
        assert target.read_text(encoding="utf-8") == "v1"

    async def test_rolling_back_an_irreversible_action_raises(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        record = await engine.submit(Action(kind=ActionKind.NOOP, summary="x"))
        with pytest.raises(RollbackError):
            await engine.rollback(record, Preparation(rollback=None))


class TestTransactions:
    async def test_all_actions_run_when_none_fail(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        result = await engine.submit_transaction([write_action("one.txt"), write_action("two.txt")])
        assert result.succeeded is True
        assert len(result.records) == 2

    async def test_failure_compensates_earlier_actions_in_reverse(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        workspace = tmp_path / "workspace"
        result = await engine.submit_transaction(
            [
                write_action("kept.txt", "first"),
                write_action("second.txt", "second"),
                write_action("../escape.txt", "bad"),
            ]
        )
        assert result.succeeded is False
        assert result.compensated is True
        assert not (workspace / "kept.txt").exists()
        assert not (workspace / "second.txt").exists()

    async def test_failed_record_is_identifiable(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path)
        result = await engine.submit_transaction(
            [write_action("ok.txt"), write_action("/etc/passwd")]
        )
        failed = result.failed_record
        assert failed is not None
        assert failed.status is ExecutionStatus.REJECTED

    async def test_oversized_transaction_is_refused(self, tmp_path: Path) -> None:
        engine = engine_for(tmp_path, max_transaction_actions=2)
        with pytest.raises(ActionExecutionError):
            await engine.submit_transaction([write_action(f"{i}.txt") for i in range(5)])


class TestJournal:
    async def test_every_outcome_is_recorded_including_refusals(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        journal = InMemoryJournal()
        engine = ExecutionEngine(
            config, policy=PolicyEngine(config, AlwaysApproveGate()), journal=journal
        )
        await engine.submit(write_action("fine.txt"))
        await engine.submit(write_action("../escape.txt"))

        entries = await journal.read()
        statuses = [entry["status"] for entry in entries]
        assert statuses == ["succeeded", "rejected"]

    async def test_journal_records_the_reason_for_refusal(self, tmp_path: Path) -> None:
        config = config_for(tmp_path)
        journal = InMemoryJournal()
        engine = ExecutionEngine(config, journal=journal)
        await engine.submit(write_action(".env", "leak"))
        entry = (await journal.read())[-1]
        codes = [finding["code"] for finding in entry["verdict"]["findings"]]
        assert "path.sensitive" in codes

    async def test_jsonl_journal_persists(self, tmp_path: Path) -> None:
        journal = JsonlJournal(tmp_path / "audit.jsonl")
        config = config_for(tmp_path)
        engine = ExecutionEngine(
            config, policy=PolicyEngine(config, AlwaysApproveGate()), journal=journal
        )
        await engine.submit(write_action("audited.txt"))
        reopened = JsonlJournal(tmp_path / "audit.jsonl")
        assert len(await reopened.read()) == 1

    async def test_a_broken_journal_does_not_fail_the_action(self, tmp_path: Path) -> None:
        class Broken(NullJournal):
            async def append(self, record: object) -> None:
                message = "audit sink is down"
                raise RuntimeError(message)

        config = config_for(tmp_path)
        engine = ExecutionEngine(
            config,
            policy=PolicyEngine(config, AlwaysApproveGate()),
            journal=Broken(),
        )
        record = await engine.submit(write_action("still-written.txt"))
        assert record.succeeded is True
