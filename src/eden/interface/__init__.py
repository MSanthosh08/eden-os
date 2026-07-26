"""Interface subsystem.

Two surfaces, one kernel. The CLI and the web console both boot EDEN through
the same composition root as any library caller, so there is no "CLI mode" that
can drift from the real system.

The web console is served by a small dependency-free asyncio HTTP server. It is
also where :class:`~eden.interface.server.WebApprovalGate` lives — the human
that Phase 3's :class:`~eden.execution.permissions.ApprovalGate` was designed to
wait for. A pending approval blocks its action, shows the risk findings and the
rollback plan, and times out into a *refusal*, never into consent.

``interface.enabled`` defaults to ``False`` and the host defaults to loopback:
this server has no authentication and must not face a network.
"""

from __future__ import annotations

from eden.interface.api import CONSOLE_HTML, build_router
from eden.interface.cli import build_parser, main
from eden.interface.server import (
    COMPONENT_NAME,
    HttpServer,
    PendingApproval,
    Request,
    Response,
    Router,
    WebApprovalGate,
)

__all__ = [
    "COMPONENT_NAME",
    "CONSOLE_HTML",
    "HttpServer",
    "PendingApproval",
    "Request",
    "Response",
    "Router",
    "WebApprovalGate",
    "build_parser",
    "build_router",
    "main",
]
