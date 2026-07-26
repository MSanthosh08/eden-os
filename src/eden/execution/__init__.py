"""Execution subsystem.

Nothing in EDEN performs an effect except through
:class:`~eden.execution.engine.ExecutionEngine`, and everything it performs
passes through four phases::

    prepare  →  verify  →  permit  →  execute  →  (rollback)

``prepare`` runs first and is read-only: its job is to capture how the action
would be undone, *before* anyone decides whether to allow it. Reversibility is
therefore an input to the risk assessment rather than an afterthought, and an
action nobody can undo is never auto-approved.
"""

from __future__ import annotations

from eden.execution.engine import COMPONENT_NAME, ExecutionEngine
from eden.execution.handlers import (
    ActionHandler,
    FileDeleteHandler,
    FileWriteHandler,
    NoOpHandler,
    ShellCommandHandler,
    default_handlers,
)
from eden.execution.journal import (
    ExecutionJournal,
    InMemoryJournal,
    JsonlJournal,
    NullJournal,
    build_journal,
)
from eden.execution.permissions import (
    AlwaysApproveGate,
    ApprovalGate,
    CallbackGate,
    DenyingGate,
    PolicyEngine,
)
from eden.execution.types import (
    Action,
    ExecutionRecord,
    ExecutionResult,
    Finding,
    Permit,
    Preparation,
    RollbackPlan,
    TransactionResult,
    Verdict,
)
from eden.execution.verification import (
    CommandAllowlistVerifier,
    CompositeVerifier,
    DestructiveActionVerifier,
    PayloadSizeVerifier,
    ReversibilityVerifier,
    SchemaVerifier,
    SensitivePathVerifier,
    Verifier,
    WorkspaceScopeVerifier,
    default_verifier,
)

__all__ = [
    "COMPONENT_NAME",
    "Action",
    "ActionHandler",
    "AlwaysApproveGate",
    "ApprovalGate",
    "CallbackGate",
    "CommandAllowlistVerifier",
    "CompositeVerifier",
    "DenyingGate",
    "DestructiveActionVerifier",
    "ExecutionEngine",
    "ExecutionJournal",
    "ExecutionRecord",
    "ExecutionResult",
    "FileDeleteHandler",
    "FileWriteHandler",
    "Finding",
    "InMemoryJournal",
    "JsonlJournal",
    "NoOpHandler",
    "NullJournal",
    "PayloadSizeVerifier",
    "Permit",
    "PolicyEngine",
    "Preparation",
    "ReversibilityVerifier",
    "RollbackPlan",
    "SchemaVerifier",
    "SensitivePathVerifier",
    "ShellCommandHandler",
    "TransactionResult",
    "Verdict",
    "Verifier",
    "WorkspaceScopeVerifier",
    "build_journal",
    "default_handlers",
    "default_verifier",
]
