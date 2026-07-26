"""Execution domain types.

The central rule of this subsystem is that **an action is data, not code**. An
:class:`Action` describes an intended effect; it has no ``run`` method and
cannot perform anything. Only a registered handler, reached through
:class:`~eden.execution.engine.ExecutionEngine`, can turn a description into an
effect — and only after verification and permission have both passed.

That separation is what makes an action plan inspectable, loggable, reviewable
and *approvable before anything happens*. A plan that could execute itself
could never be safely shown to a human first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from eden.config.enums import ActionKind, ExecutionStatus, PermissionMode, RiskLevel
from eden.errors import ValidationError
from eden.utils.ids import new_id

SYSTEM_ACTOR = "system"
_MAX_JOURNALLED_SEQUENCE = 50
DEFAULT_NAMESPACE = "default"


@dataclass(frozen=True, slots=True)
class Action:
    """A declarative description of one intended effect.

    Attributes:
        kind: The class of effect, which selects a handler.
        summary: One-line human-readable description. This is what an approver
            reads, so it must be accurate rather than reassuring.
        parameters: Kind-specific inputs, validated by verification.
        id: Stable identifier, generated when omitted.
        actor: Who requested it — an agent name, a user, or ``"system"``.
        namespace: Isolation boundary for journalling and policy.
        created_at: UTC creation time.
        metadata: Non-executable annotations.
    """

    kind: ActionKind
    summary: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    id: str = ""
    actor: str = SYSTEM_ACTOR
    namespace: str = DEFAULT_NAMESPACE
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and fill in defaults.

        Raises:
            ValidationError: If the description is unusable.
        """
        if not self.summary.strip():
            raise ValidationError("Action summary must not be empty.")
        if not self.actor.strip():
            raise ValidationError("Action actor must not be empty.")
        if not self.id:
            object.__setattr__(self, "id", new_id("act"))

    def parameter(self, name: str) -> Any:  # noqa: ANN401 - kind-specific payload
        """Return a required parameter.

        Raises:
            ValidationError: If the parameter is absent.
        """
        if name not in self.parameters:
            raise ValidationError(
                "Action is missing a required parameter.",
                context={"action": self.id, "kind": self.kind.value, "parameter": name},
            )
        return self.parameters[name]

    def text_parameter(self, name: str) -> str:
        """Return a required parameter as a string.

        Raises:
            ValidationError: If the parameter is absent or not a string.
        """
        value = self.parameter(name)
        if not isinstance(value, str):
            raise ValidationError(
                "Action parameter must be a string.",
                context={"action": self.id, "parameter": name},
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the audit journal."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "summary": self.summary,
            "parameters": _summarise(self.parameters),
            "actor": self.actor,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Finding:
    """One observation made during verification.

    Attributes:
        code: Stable machine-readable identifier, e.g. ``"path.outside_workspace"``.
        message: Human-readable explanation shown to an approver.
        risk: Severity contributed by this finding.
        blocking: Whether this alone must prevent execution, regardless of any
            risk threshold or approval.
    """

    code: str
    message: str
    risk: RiskLevel = RiskLevel.LOW
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "code": self.code,
            "message": self.message,
            "risk": self.risk.name.lower(),
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """The compensating actions that undo one executed action.

    Rollback steps are themselves :class:`Action` objects, so undo reuses the
    very same handlers as do. There is no second, less-tested code path for the
    operation that runs when things have already gone wrong.

    An *empty* plan is meaningful and different from no plan at all: it means
    "this action had no effect to undo", which is reversible. ``None`` means
    "this effect cannot be undone".

    Attributes:
        steps: Compensating actions, applied in the order given.
        description: Human-readable summary shown to an approver.
    """

    steps: tuple[Action, ...] = ()
    description: str = ""

    @property
    def is_empty(self) -> bool:
        """Return whether this plan has no compensating steps."""
        return not self.steps


@dataclass(frozen=True, slots=True)
class Preparation:
    """The state a handler captured before an action ran.

    Preparation must be free of side effects apart from reads. Its purpose is
    to *prove* reversibility up front rather than hoping for it afterwards: if
    a handler cannot describe how to undo an action before performing it, the
    action is irreversible and policy treats it accordingly.

    Attributes:
        rollback: How to undo the action, or ``None`` if it cannot be undone.
        notes: Handler-specific context surfaced to verification and approvers.
    """

    rollback: RollbackPlan | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reversible(self) -> bool:
        """Return whether this action can be undone."""
        return self.rollback is not None


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of verifying one action.

    Attributes:
        action_id: Action this verdict describes.
        findings: Everything verification observed.
        reversible: Whether a rollback plan exists.
    """

    action_id: str
    findings: tuple[Finding, ...] = ()
    reversible: bool = False

    @property
    def risk(self) -> RiskLevel:
        """Return the highest risk observed, or :attr:`RiskLevel.NONE`."""
        if not self.findings:
            return RiskLevel.NONE
        return max(finding.risk for finding in self.findings)

    @property
    def blocked(self) -> bool:
        """Return whether any finding forbids execution outright."""
        return any(finding.blocking for finding in self.findings)

    @property
    def blocking_findings(self) -> tuple[Finding, ...]:
        """Return only the findings that forbid execution."""
        return tuple(finding for finding in self.findings if finding.blocking)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "action_id": self.action_id,
            "risk": self.risk.name.lower(),
            "blocked": self.blocked,
            "reversible": self.reversible,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class Permit:
    """The authorisation decision for one action.

    Attributes:
        action_id: Action this permit describes.
        granted: Whether execution may proceed.
        mode: How the decision was reached.
        reason: Human-readable justification, recorded in the journal.
        approver: Who approved, when approval was sought.
    """

    action_id: str
    granted: bool
    mode: PermissionMode
    reason: str
    approver: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "action_id": self.action_id,
            "granted": self.granted,
            "mode": self.mode.value,
            "reason": self.reason,
            "approver": self.approver,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What a handler produced.

    Attributes:
        succeeded: Whether the effect was achieved.
        output: Captured, truncated output. Never contains credentials.
        detail: Structured handler-specific result data.
        duration_ms: Measured wall-clock duration.
    """

    succeeded: bool
    output: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "succeeded": self.succeeded,
            "output": self.output,
            "detail": _summarise(self.detail),
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """The complete audit trail for one action through the pipeline.

    Every action produces exactly one record, whether it ran, was refused at
    verification, was denied by policy, failed, or was rolled back. A record
    with no result is not a gap in the audit trail — it is the evidence that
    something was stopped.

    Attributes:
        action: The action submitted.
        status: Where it ended up.
        verdict: Verification outcome, absent only if verification itself failed.
        permit: Authorisation decision, absent if verification rejected it.
        result: Handler outcome, absent if it never ran.
        rollback_applied: Whether compensation was performed.
        error: Serialised failure, when one occurred.
        started_at: When the pipeline began.
        finished_at: When the pipeline ended.
    """

    action: Action
    status: ExecutionStatus
    verdict: Verdict | None = None
    permit: Permit | None = None
    result: ExecutionResult | None = None
    rollback_applied: bool = False
    error: Mapping[str, Any] | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the action completed successfully."""
        return self.status is ExecutionStatus.SUCCEEDED

    @property
    def executed(self) -> bool:
        """Return whether a handler actually ran, successfully or not."""
        return self.result is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the journal."""
        return {
            "action": self.action.to_dict(),
            "status": self.status.value,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "permit": self.permit.to_dict() if self.permit else None,
            "result": self.result.to_dict() if self.result else None,
            "rollback_applied": self.rollback_applied,
            "error": dict(self.error) if self.error else None,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@dataclass(frozen=True, slots=True)
class TransactionResult:
    """The outcome of executing several actions as one unit.

    Attributes:
        records: One record per submitted action, in submission order.
        succeeded: Whether every action completed.
        compensated: Whether already-successful actions were rolled back.
    """

    records: tuple[ExecutionRecord, ...]
    succeeded: bool
    compensated: bool = False

    @property
    def failed_record(self) -> ExecutionRecord | None:
        """Return the first record that did not succeed, if any."""
        for record in self.records:
            if not record.succeeded:
                return record
        return None


def _summarise(payload: Mapping[str, Any], *, max_length: int = 512) -> dict[str, Any]:
    """Return a journal-safe copy of a payload with long values truncated.

    Action parameters can carry an entire file's contents. Writing those into
    every audit line would make the journal unreadable and enormous, so long
    strings are replaced with a length marker.
    """
    summarised: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > max_length:
            summarised[key] = f"<{len(value)} characters>"
        elif isinstance(value, bytes):
            summarised[key] = f"<{len(value)} bytes>"
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            and len(value) > _MAX_JOURNALLED_SEQUENCE
        ):
            summarised[key] = f"<{len(value)} items>"
        else:
            summarised[key] = value
    return summarised
