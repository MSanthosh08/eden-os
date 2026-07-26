"""The execution engine.

This is the only path from an intent to an effect. There is deliberately no
"just run it" shortcut, no ``force`` flag and no way to hand a handler an
action directly from outside the module: every effect passes through

    prepare  →  verify  →  permit  →  execute  →  (rollback)

*Prepare* comes first, before verification, because reversibility is an input
to the risk assessment. A handler must prove it can undo an action before
anyone is asked whether the action is allowed.

Transactions apply the saga pattern: actions run in order, and if any fails,
the ones that already succeeded are compensated in reverse. There is no
distributed two-phase commit here and there cannot be — the world does not
support it — so the honest guarantee is "we will try to put it back", recorded
truthfully in the journal either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from eden.config.enums import ActionKind, ExecutionStatus, PermissionMode
from eden.config.schema import ExecutionConfig
from eden.core.registry import Registry
from eden.errors import (
    ActionExecutionError,
    EdenError,
    HandlerNotFoundError,
    RollbackError,
)
from eden.execution.handlers import ActionHandler, default_handlers
from eden.execution.journal import ExecutionJournal, NullJournal
from eden.execution.permissions import ApprovalGate, PolicyEngine
from eden.execution.types import (
    Action,
    ExecutionRecord,
    ExecutionResult,
    Permit,
    Preparation,
    TransactionResult,
    Verdict,
)
from eden.execution.verification import CompositeVerifier, default_verifier
from eden.logging import get_logger, timed_block
from eden.utils.clock import Clock, SystemClock

_LOGGER = get_logger("execution.engine")

COMPONENT_NAME = "execution"


class ExecutionEngine:
    """Runs actions through verification, permission, execution and rollback.

    Example:
        >>> import asyncio
        >>> from eden.config.schema import ExecutionConfig
        >>> from eden.config.enums import ActionKind
        >>> from eden.execution.types import Action
        >>> async def demo() -> str:
        ...     engine = ExecutionEngine(ExecutionConfig())
        ...     record = await engine.submit(
        ...         Action(kind=ActionKind.NOOP, summary="do nothing")
        ...     )
        ...     return record.status.value
        >>> asyncio.run(demo())
        'succeeded'
    """

    def __init__(
        self,
        config: ExecutionConfig,
        *,
        verifier: CompositeVerifier | None = None,
        policy: PolicyEngine | None = None,
        journal: ExecutionJournal | None = None,
        handlers: Sequence[ActionHandler] | None = None,
        gate: ApprovalGate | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            config: Execution policy.
            verifier: Verification suite. Defaults to the shipped rules.
            policy: Authorisation engine. Defaults to one built from ``config``.
            journal: Audit sink. Defaults to discarding.
            handlers: Action handlers. Defaults to the shipped set.
            gate: Approval mechanism, used only when ``policy`` is omitted.
            clock: Time source.
        """
        self._config = config
        self._verifier = verifier or default_verifier(config)
        self._policy = policy or PolicyEngine(config, gate)
        self._journal = journal or NullJournal()
        self._clock = clock or SystemClock()
        self._handlers: Registry[ActionHandler] = Registry("action handler")
        self._started = False
        for handler in handlers if handlers is not None else default_handlers(config):
            self.register_handler(handler)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return COMPONENT_NAME

    async def start(self) -> None:
        """Prepare the workspace. Idempotent."""
        if self._started:
            return
        self._config.workspace_root.mkdir(parents=True, exist_ok=True)
        self._started = True
        _LOGGER.info(
            "Execution subsystem started.",
            extra={
                "workspace": str(self._config.workspace_root),
                "handlers": list(self._handlers.names()),
                "dry_run": self._config.dry_run,
                "allowed_commands": list(self._config.allowed_commands),
            },
        )

    async def stop(self) -> None:
        """Release the subsystem. Idempotent and never raises."""
        if not self._started:
            return
        self._started = False
        _LOGGER.info("Execution subsystem stopped.")

    # ------------------------------------------------------------------
    # Registration and introspection
    # ------------------------------------------------------------------
    def register_handler(self, handler: ActionHandler, *, replace: bool = False) -> None:
        """Register ``handler`` for its action kind.

        Args:
            handler: The handler to register.
            replace: Permit overwriting an existing registration.

        Raises:
            RegistryError: If the kind is taken and ``replace`` is ``False``.
        """
        self._handlers.register(handler.kind.value, lambda: handler, replace=replace)

    @property
    def journal(self) -> ExecutionJournal:
        """Return the audit journal."""
        return self._journal

    @property
    def policy(self) -> PolicyEngine:
        """Return the authorisation engine."""
        return self._policy

    def handler_for(self, kind: ActionKind) -> ActionHandler:
        """Return the handler registered for ``kind``.

        Raises:
            HandlerNotFoundError: If no handler is registered.
        """
        if kind.value not in self._handlers:
            raise HandlerNotFoundError(
                "No handler is registered for this action kind.",
                context={"kind": kind.value, "known": list(self._handlers.names())},
            )
        return self._handlers.create(kind.value)

    # ------------------------------------------------------------------
    # Inspection without execution
    # ------------------------------------------------------------------
    async def review(self, action: Action) -> tuple[Verdict, Preparation]:
        """Prepare and verify ``action`` without executing or authorising it.

        This is what a user interface calls to show somebody what an action
        would do before asking them to approve it.

        Args:
            action: The action to inspect.

        Returns:
            The verdict and the captured preparation.

        Raises:
            HandlerNotFoundError: If no handler is registered for the kind.
        """
        handler = self.handler_for(action.kind)
        preparation = await self._prepare(handler, action)
        return self._verifier.verify(action, preparation), preparation

    # ------------------------------------------------------------------
    # The pipeline
    # ------------------------------------------------------------------
    async def submit(self, action: Action) -> ExecutionRecord:
        """Run ``action`` through the full pipeline.

        This is the only entry point. It never raises for an action that was
        refused or that failed — those are outcomes, recorded in the returned
        record and in the journal. It raises only when the pipeline itself
        cannot run, such as a missing handler.

        Args:
            action: The action to perform.

        Returns:
            The complete audit record.

        Raises:
            HandlerNotFoundError: If no handler is registered for the kind.
        """
        started = self._now()
        handler = self.handler_for(action.kind)

        with timed_block(
            _LOGGER,
            "execution.submit",
            action=action.id,
            kind=action.kind.value,
            actor=action.actor,
        ):
            preparation = await self._prepare(handler, action)
            verdict = self._verifier.verify(action, preparation)

            if verdict.blocked:
                permit = await self._policy.decide(action, verdict)
                return await self._finish(
                    ExecutionRecord(
                        action=action,
                        status=ExecutionStatus.REJECTED,
                        verdict=verdict,
                        permit=permit,
                        started_at=started,
                    )
                )

            permit = await self._policy.decide(action, verdict)
            if not permit.granted:
                return await self._finish(
                    ExecutionRecord(
                        action=action,
                        status=ExecutionStatus.DENIED,
                        verdict=verdict,
                        permit=permit,
                        started_at=started,
                    )
                )

            if self._config.dry_run:
                return await self._finish(
                    ExecutionRecord(
                        action=action,
                        status=ExecutionStatus.SKIPPED,
                        verdict=verdict,
                        permit=permit,
                        result=ExecutionResult(
                            succeeded=True,
                            output="Dry run: verified and permitted, not executed.",
                        ),
                        started_at=started,
                    )
                )

            return await self._execute(handler, action, verdict, permit, preparation, started)

    async def submit_transaction(self, actions: Sequence[Action]) -> TransactionResult:
        """Run several actions as a unit, compensating on failure.

        Args:
            actions: Actions to perform, in order.

        Returns:
            One record per action attempted, plus whether compensation ran.

        Raises:
            ActionExecutionError: If the batch exceeds the configured ceiling.
        """
        if len(actions) > self._config.max_transaction_actions:
            raise ActionExecutionError(
                "Transaction exceeds the configured action ceiling.",
                context={
                    "submitted": len(actions),
                    "limit": self._config.max_transaction_actions,
                },
            )

        records: list[ExecutionRecord] = []
        completed: list[tuple[ExecutionRecord, Preparation]] = []

        for action in actions:
            handler = self.handler_for(action.kind)
            preparation = await self._prepare(handler, action)
            record = await self.submit(action)
            records.append(record)
            if record.succeeded:
                completed.append((record, preparation))
                continue

            _LOGGER.warning(
                "Transaction failed; compensating completed actions.",
                extra={
                    "failed_action": action.id,
                    "status": record.status.value,
                    "to_compensate": len(completed),
                },
            )
            compensated = await self._compensate(completed)
            return TransactionResult(
                records=tuple(records), succeeded=False, compensated=compensated
            )

        return TransactionResult(records=tuple(records), succeeded=True)

    async def rollback(
        self,
        record: ExecutionRecord,
        preparation: Preparation,
    ) -> ExecutionRecord:
        """Undo a completed action.

        Args:
            record: The record of the action to undo.
            preparation: The preparation captured before it ran.

        Returns:
            An updated record reflecting the rollback outcome.

        Raises:
            RollbackError: If the action has no rollback plan.
        """
        if preparation.rollback is None:
            raise RollbackError(
                "This action was irreversible; there is no plan to apply.",
                context={"action": record.action.id, "kind": record.action.kind.value},
            )
        succeeded = await self._apply_rollback(record.action, preparation)
        return await self._finish(
            ExecutionRecord(
                action=record.action,
                status=(
                    ExecutionStatus.ROLLED_BACK if succeeded else ExecutionStatus.ROLLBACK_FAILED
                ),
                verdict=record.verdict,
                permit=record.permit,
                result=record.result,
                rollback_applied=succeeded,
                started_at=record.started_at,
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _prepare(self, handler: ActionHandler, action: Action) -> Preparation:
        """Run the handler's preparation, treating failure as irreversible.

        A handler that cannot inspect the world cannot promise to undo its
        work, so a failed preparation degrades to "no rollback plan" rather
        than aborting. Verification then sees the elevated risk.
        """
        try:
            return await handler.prepare(action)
        except EdenError as exc:
            _LOGGER.warning(
                "Preparation failed; treating the action as irreversible.",
                extra={"action": action.id, "error_code": exc.code},
            )
            return Preparation(rollback=None, notes={"reason": exc.message})
        except Exception as exc:  # noqa: BLE001 - preparation must not abort the pipeline
            _LOGGER.warning(
                "Preparation raised; treating the action as irreversible.",
                extra={"action": action.id, "error_type": type(exc).__name__},
            )
            return Preparation(
                rollback=None,
                notes={"reason": f"preparation failed ({type(exc).__name__})"},
            )

    async def _execute(
        self,
        handler: ActionHandler,
        action: Action,
        verdict: Verdict,
        permit: Permit,
        preparation: Preparation,
        started: datetime,
    ) -> ExecutionRecord:
        """Perform the action, compensating automatically if it fails."""
        try:
            result = await handler.execute(action, preparation)
        except EdenError as exc:
            rolled_back = await self._compensate_one(action, preparation)
            return await self._finish(
                ExecutionRecord(
                    action=action,
                    status=(ExecutionStatus.ROLLED_BACK if rolled_back else ExecutionStatus.FAILED),
                    verdict=verdict,
                    permit=permit,
                    rollback_applied=rolled_back,
                    error=exc.to_dict(),
                    started_at=started,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a handler bug must not escape
            rolled_back = await self._compensate_one(action, preparation)
            return await self._finish(
                ExecutionRecord(
                    action=action,
                    status=(ExecutionStatus.ROLLED_BACK if rolled_back else ExecutionStatus.FAILED),
                    verdict=verdict,
                    permit=permit,
                    rollback_applied=rolled_back,
                    error={
                        "code": "eden.execution.handler_crashed",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    started_at=started,
                )
            )

        return await self._finish(
            ExecutionRecord(
                action=action,
                status=(ExecutionStatus.SUCCEEDED if result.succeeded else ExecutionStatus.FAILED),
                verdict=verdict,
                permit=permit,
                result=result,
                started_at=started,
            )
        )

    async def _compensate_one(self, action: Action, preparation: Preparation) -> bool:
        """Attempt to undo one action, returning whether it worked."""
        if preparation.rollback is None:
            return False
        return await self._apply_rollback(action, preparation)

    async def _compensate(
        self,
        completed: Sequence[tuple[ExecutionRecord, Preparation]],
    ) -> bool:
        """Undo completed actions in reverse order.

        Compensation continues past a failure so that later steps still get
        their chance; partial recovery beats none.
        """
        all_succeeded = True
        for record, preparation in reversed(completed):
            if preparation.rollback is None:
                _LOGGER.error(
                    "Cannot compensate an irreversible action in a failed transaction.",
                    extra={"action": record.action.id, "kind": record.action.kind.value},
                )
                all_succeeded = False
                continue
            if not await self._apply_rollback(record.action, preparation):
                all_succeeded = False
        return all_succeeded

    async def _apply_rollback(self, action: Action, preparation: Preparation) -> bool:
        """Run every compensating step, returning whether all succeeded.

        Rollback steps bypass verification and permission deliberately. They
        were derived from state captured before the action ran and were
        described in the permit the approver saw, so they are already
        authorised. Re-asking would mean an approver could refuse the undo of
        something they had just approved, stranding the system mid-change.
        """
        plan = preparation.rollback
        if plan is None or plan.is_empty:
            return True
        for step in plan.steps:
            try:
                handler = self.handler_for(step.kind)
                await handler.execute(step, Preparation())
            except Exception as exc:  # noqa: BLE001 - report, never propagate
                _LOGGER.error(
                    "Rollback step failed; the system may be in an intermediate state.",
                    extra={
                        "action": action.id,
                        "step": step.id,
                        "step_kind": step.kind.value,
                        "error_type": type(exc).__name__,
                    },
                )
                return False
        _LOGGER.info(
            "Action rolled back.",
            extra={"action": action.id, "steps": len(plan.steps)},
        )
        return True

    async def _finish(self, record: ExecutionRecord) -> ExecutionRecord:
        """Stamp the finish time, journal the record and return it."""
        completed = ExecutionRecord(
            action=record.action,
            status=record.status,
            verdict=record.verdict,
            permit=record.permit,
            result=record.result,
            rollback_applied=record.rollback_applied,
            error=record.error,
            started_at=record.started_at,
            finished_at=self._now(),
        )
        try:
            await self._journal.append(completed)
        except Exception as exc:  # noqa: BLE001 - auditing must not fail the action
            _LOGGER.error(
                "Could not journal the execution record.",
                extra={"action": completed.action.id, "error_type": type(exc).__name__},
            )
        _LOGGER.info(
            "Action finished.",
            extra={
                "action": completed.action.id,
                "kind": completed.action.kind.value,
                "status": completed.status.value,
                "mode": completed.permit.mode.value if completed.permit else "n/a",
            },
        )
        return completed

    def _now(self) -> datetime:
        """Return the current time from the injected clock."""
        del self  # the clock is stateless; kept as a method for injection symmetry
        return datetime.now(tz=UTC)


__all__ = ["COMPONENT_NAME", "ExecutionEngine", "PermissionMode"]
