"""Permission.

Verification says how dangerous an action is. Permission decides whether it may
proceed anyway. Keeping those separate means the risk assessment is honest —
it is not tempted to under-report in order to get something approved.

The default :class:`ApprovalGate` is :class:`DenyingGate`. An EDEN with no
approver wired in cannot perform anything that needs confirmation. Silence is
refusal, never consent.
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable

from eden.config.enums import PermissionMode, RiskLevel
from eden.config.schema import ExecutionConfig
from eden.execution.types import Action, Permit, Verdict
from eden.logging import get_logger

_LOGGER = get_logger("execution.permissions")

AUTOMATIC_APPROVER = "policy"


class ApprovalGate(abc.ABC):
    """Asks somebody — or something — whether an action may proceed."""

    @property
    @abc.abstractmethod
    def approver(self) -> str:
        """Return the identity recorded in the journal when this gate approves."""

    @abc.abstractmethod
    async def request(self, action: Action, verdict: Verdict) -> bool:
        """Return whether the action is approved.

        Implementations must be safe to call concurrently and must never raise;
        an approval mechanism that throws would otherwise become an outage.
        """


class DenyingGate(ApprovalGate):
    """Refuses everything. The default, so that absence of an approver is safe."""

    @property
    def approver(self) -> str:
        """Return the identity recorded when this gate decides."""
        return "denying-gate"

    async def request(self, action: Action, verdict: Verdict) -> bool:
        """Always refuse."""
        _LOGGER.info(
            "Approval required but no approver is configured; refusing.",
            extra={"action": action.id, "risk": verdict.risk.name.lower()},
        )
        return False


class AlwaysApproveGate(ApprovalGate):
    """Approves everything policy is willing to ask about.

    Intended for tests and for explicitly trusted automation. It cannot approve
    a blocked action or one above ``deny_above_risk``, because those never
    reach a gate at all.
    """

    def __init__(self, approver: str = "auto-approver") -> None:
        """Initialise the gate with the identity it records."""
        self._approver = approver

    @property
    def approver(self) -> str:
        """Return the identity recorded when this gate approves."""
        return self._approver

    async def request(self, action: Action, verdict: Verdict) -> bool:
        """Always approve."""
        del action, verdict
        return True


class CallbackGate(ApprovalGate):
    """Delegates approval to an injected coroutine.

    This is the seam a user interface plugs into: a chat prompt, a Slack
    button, a CLI confirmation. A callback that raises is treated as a refusal
    rather than allowed to propagate.
    """

    def __init__(
        self,
        callback: Callable[[Action, Verdict], Awaitable[bool]],
        *,
        approver: str = "human",
    ) -> None:
        """Initialise the gate.

        Args:
            callback: Coroutine receiving the action and its verdict.
            approver: Identity recorded in the journal on approval.
        """
        self._callback = callback
        self._approver = approver

    @property
    def approver(self) -> str:
        """Return the identity recorded when this gate approves."""
        return self._approver

    async def request(self, action: Action, verdict: Verdict) -> bool:
        """Ask the callback, treating any failure as refusal."""
        try:
            return await self._callback(action, verdict)
        except Exception as exc:  # noqa: BLE001 - a broken approver must not approve
            _LOGGER.warning(
                "Approval callback failed; treating as refusal.",
                extra={"action": action.id, "error_type": type(exc).__name__},
            )
            return False


class PolicyEngine:
    """Turns a verdict into an authorisation decision.

    The decision ladder, in order:

    1. A blocking finding refuses outright. Nothing overrides it.
    2. Risk above ``deny_above_risk`` refuses outright.
    3. Irreversible actions never auto-approve, whatever their risk.
    4. Risk at or below ``auto_approve_max_risk`` proceeds automatically.
    5. Everything else asks the gate.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        gate: ApprovalGate | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            config: Execution policy supplying the thresholds.
            gate: Approval mechanism. Defaults to refusing.
        """
        self._config = config
        self._gate = gate or DenyingGate()

    @property
    def gate(self) -> ApprovalGate:
        """Return the approval gate in use."""
        return self._gate

    async def decide(self, action: Action, verdict: Verdict) -> Permit:
        """Return the authorisation decision for ``action``.

        Args:
            action: The action under consideration.
            verdict: Its verification outcome.

        Returns:
            A permit recording the decision and why it was reached.
        """
        if verdict.blocked:
            reasons = "; ".join(finding.message for finding in verdict.blocking_findings)
            return self._refuse(action, PermissionMode.DENY, reasons)

        if verdict.risk > self._config.deny_above_risk:
            return self._refuse(
                action,
                PermissionMode.DENY,
                (
                    f"Risk {verdict.risk.name.lower()} exceeds the configured ceiling "
                    f"{self._config.deny_above_risk.name.lower()}."
                ),
            )

        if self._qualifies_for_automatic(verdict):
            _LOGGER.debug(
                "Action auto-approved by policy.",
                extra={"action": action.id, "risk": verdict.risk.name.lower()},
            )
            return Permit(
                action_id=action.id,
                granted=True,
                mode=PermissionMode.AUTOMATIC,
                reason=(
                    f"Reversible and rated {verdict.risk.name.lower()}, at or below the "
                    f"automatic threshold {self._config.auto_approve_max_risk.name.lower()}."
                ),
                approver=AUTOMATIC_APPROVER,
            )

        if self._config.default_mode is PermissionMode.DENY:
            return self._refuse(
                action, PermissionMode.DENY, "Default policy denies unmatched actions."
            )

        approved = await self._gate.request(action, verdict)
        reason = (
            "Approved by the configured gate."
            if approved
            else "The configured gate refused approval."
        )
        _LOGGER.info(
            "Approval decision recorded.",
            extra={
                "action": action.id,
                "granted": approved,
                "risk": verdict.risk.name.lower(),
                "reversible": verdict.reversible,
            },
        )
        return Permit(
            action_id=action.id,
            granted=approved,
            mode=PermissionMode.CONFIRM,
            reason=reason,
            approver=self._gate.approver if approved else "",
        )

    def _qualifies_for_automatic(self, verdict: Verdict) -> bool:
        """Return whether the verdict permits running without asking.

        Irreversibility disqualifies unconditionally. An action nobody can undo
        is one a human should have seen, however mild it looks.
        """
        if not verdict.reversible:
            return False
        if self._config.default_mode is PermissionMode.DENY:
            return False
        return verdict.risk <= self._config.auto_approve_max_risk

    @staticmethod
    def _refuse(action: Action, mode: PermissionMode, reason: str) -> Permit:
        """Return a refusal permit and log it."""
        _LOGGER.warning(
            "Action denied.",
            extra={"action": action.id, "kind": action.kind.value, "reason": reason},
        )
        return Permit(action_id=action.id, granted=False, mode=mode, reason=reason)


def risk_at_most(level: RiskLevel, ceiling: RiskLevel) -> bool:
    """Return whether ``level`` is within ``ceiling``.

    Provided so callers outside this module compare risk through one named
    helper rather than open-coding an operator whose direction is easy to
    invert by accident.

    Example:
        >>> risk_at_most(RiskLevel.LOW, RiskLevel.MODERATE)
        True
    """
    return level <= ceiling
