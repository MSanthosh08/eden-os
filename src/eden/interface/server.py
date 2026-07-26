"""Local HTTP interface.

A deliberately small asyncio HTTP/1.1 server with no framework dependency. EDEN
has kept its runtime dependency list at zero for five phases and a web
framework is not worth breaking that for: the surface here is a dozen JSON
routes and one HTML page.

The interesting piece is :class:`WebApprovalGate`. Phase 3 defined
:class:`~eden.execution.permissions.ApprovalGate` and shipped a refusing default
precisely so a human could be plugged in later. This is that human. A pending
approval blocks the action, surfaces in the UI with its risk findings and
rollback plan, and times out into a *refusal* — never into consent.

Security posture: this server binds loopback by default, has no authentication,
and must not face a network. That is stated in the config docstring, enforced by
the default, and repeated in the startup log.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from eden.config.schema import InterfaceConfig
from eden.errors import EdenError, InterfaceError
from eden.execution.permissions import ApprovalGate
from eden.execution.types import Action, Verdict
from eden.logging import get_logger
from eden.utils.ids import new_id

_LOGGER = get_logger("interface.server")

COMPONENT_NAME = "interface"

_STATUS_TEXT: Mapping[int, str] = {
    200: "OK",
    201: "Created",
    204: "No Content",
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    500: "Internal Server Error",
}

_MAX_HEADER_BYTES = 16_384


@dataclass(slots=True)
class PendingApproval:
    """One action waiting for a human decision.

    Attributes:
        id: Identifier the UI uses to resolve this request.
        action: The action awaiting approval.
        verdict: Its verification outcome, shown to the approver.
        rollback: Human-readable description of how it would be undone.
        requested_at: When the approval was requested.
        future: Resolved when a decision arrives.
    """

    id: str
    action: Action
    verdict: Verdict
    rollback: str
    requested_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    future: asyncio.Future[bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the UI."""
        return {
            "id": self.id,
            "action": self.action.to_dict(),
            "verdict": self.verdict.to_dict(),
            "rollback": self.rollback,
            "requested_at": self.requested_at.isoformat(),
        }


class WebApprovalGate(ApprovalGate):
    """Asks a human through the web interface, and refuses on silence.

    Example:
        >>> gate = WebApprovalGate(InterfaceConfig())
        >>> gate.approver
        'web'
    """

    def __init__(self, config: InterfaceConfig) -> None:
        """Initialise the gate.

        Args:
            config: Interface policy supplying the approval timeout.
        """
        self._config = config
        self._pending: dict[str, PendingApproval] = {}

    @property
    def approver(self) -> str:
        """Return the identity recorded when this gate approves."""
        return "web"

    @property
    def pending(self) -> list[PendingApproval]:
        """Return every approval currently waiting, oldest first."""
        return sorted(self._pending.values(), key=lambda item: item.requested_at)

    def resolve(self, approval_id: str, *, approved: bool) -> bool:
        """Record a human decision.

        Args:
            approval_id: Identifier from :meth:`pending`.
            approved: The decision.

        Returns:
            ``True`` if a waiting request was resolved.
        """
        entry = self._pending.pop(approval_id, None)
        if entry is None or entry.future is None or entry.future.done():
            return False
        entry.future.set_result(approved)
        return True

    async def request(self, action: Action, verdict: Verdict) -> bool:
        """Surface the action to the UI and wait for a decision.

        A timeout resolves to refusal. An interface that cannot reach a human
        must behave exactly like an interface with no human behind it.
        """
        approval_id = new_id("apr")
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[bool] = loop.create_future()
        entry = PendingApproval(
            id=approval_id,
            action=action,
            verdict=verdict,
            rollback=("Reversible." if verdict.reversible else "This action cannot be undone."),
            future=waiter,
        )
        self._pending[approval_id] = entry
        _LOGGER.info(
            "Approval requested from the web interface.",
            extra={
                "approval": approval_id,
                "action": action.id,
                "risk": verdict.risk.name.lower(),
            },
        )
        try:
            return await asyncio.wait_for(waiter, timeout=self._config.approval_timeout_seconds)
        except TimeoutError:
            _LOGGER.warning(
                "Approval timed out; refusing.",
                extra={"approval": approval_id, "action": action.id},
            )
            return False
        finally:
            self._pending.pop(approval_id, None)


@dataclass(frozen=True, slots=True)
class Response:
    """An HTTP response.

    Attributes:
        status: Numeric status code.
        body: Encoded response body.
        content_type: MIME type.
    """

    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"

    @classmethod
    def json(cls, payload: Any, *, status: int = 200) -> Response:  # noqa: ANN401
        """Return a JSON response."""
        return cls(
            status=status,
            body=json.dumps(payload, default=str).encode("utf-8"),
            content_type="application/json",
        )

    @classmethod
    def html(cls, markup: str, *, status: int = 200) -> Response:
        """Return an HTML response."""
        return cls(
            status=status, body=markup.encode("utf-8"), content_type="text/html; charset=utf-8"
        )

    @classmethod
    def error(cls, status: int, message: str) -> Response:
        """Return a JSON error response."""
        return cls.json({"error": message, "status": status}, status=status)


@dataclass(frozen=True, slots=True)
class Request:
    """A parsed HTTP request.

    Attributes:
        method: HTTP verb.
        path: Path without the query string.
        query: Parsed query parameters.
        body: Raw request body.
    """

    method: str
    path: str
    query: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> dict[str, Any]:
        """Parse the body as a JSON object.

        Returns:
            The decoded object, or an empty mapping when the body is empty.

        Raises:
            InterfaceError: If the body is not a JSON object.
        """
        if not self.body:
            return {}
        try:
            parsed = json.loads(self.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InterfaceError("Request body was not valid JSON.", cause=exc) from exc
        if not isinstance(parsed, dict):
            raise InterfaceError("Request body must be a JSON object.")
        return parsed


Handler = Callable[[Request], Awaitable[Response]]


class Router:
    """Maps ``(method, path)`` to a handler.

    Paths are matched exactly, with one trailing ``*`` permitted as a prefix
    wildcard. That is enough for this surface and avoids a pattern engine.
    """

    def __init__(self) -> None:
        """Create an empty router."""
        self._routes: dict[tuple[str, str], Handler] = {}

    def add(self, method: str, path: str, handler: Handler) -> None:
        """Register ``handler`` for ``method`` and ``path``."""
        self._routes[(method.upper(), path)] = handler

    def get(self, method: str, path: str) -> Handler | None:
        """Return the handler for a request, or ``None``."""
        exact = self._routes.get((method.upper(), path))
        if exact is not None:
            return exact
        for (route_method, route_path), handler in self._routes.items():
            if route_method != method.upper() or not route_path.endswith("*"):
                continue
            if path.startswith(route_path[:-1]):
                return handler
        return None

    @property
    def paths(self) -> tuple[str, ...]:
        """Return every registered path."""
        return tuple(sorted({path for _, path in self._routes}))


class HttpServer:
    """A minimal asyncio HTTP/1.1 server."""

    def __init__(self, config: InterfaceConfig, router: Router) -> None:
        """Initialise the server.

        Args:
            config: Binding and request limits.
            router: Route table.
        """
        self._config = config
        self._router = router
        self._server: asyncio.AbstractServer | None = None

    @property
    def component_name(self) -> str:
        """Return the name used in startup and shutdown logs."""
        return COMPONENT_NAME

    @property
    def url(self) -> str:
        """Return the URL the server is reachable at."""
        return f"http://{self._config.host}:{self._config.port}"

    @property
    def port(self) -> int:
        """Return the bound port, which may differ when port 0 was requested."""
        sockets = getattr(self._server, "sockets", None) if self._server else None
        if not sockets:
            return self._config.port
        return int(sockets[0].getsockname()[1])

    async def start(self) -> None:
        """Bind and begin serving. Idempotent."""
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._handle_connection, self._config.host, self._config.port
        )
        _LOGGER.info(
            "Interface listening. This server has no authentication; keep it on loopback.",
            extra={
                "url": f"http://{self._config.host}:{self.port}",
                "routes": len(self._router.paths),
                "allow_actions": self._config.allow_actions,
            },
        )

    async def stop(self) -> None:
        """Stop serving. Idempotent and never raises."""
        server = self._server
        self._server = None
        if server is None:
            return
        server.close()
        with contextlib.suppress(Exception):
            await server.wait_closed()
        _LOGGER.info("Interface stopped.")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve one connection, one request, then close."""
        try:
            request = await self._read_request(reader)
            response = (
                await self._dispatch(request)
                if request
                else Response.error(400, "Malformed request.")
            )
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the server
            _LOGGER.error("Request handling failed.", extra={"error_type": type(exc).__name__})
            response = Response.error(500, "Internal error.")
        try:
            writer.write(_encode(response))
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _read_request(self, reader: asyncio.StreamReader) -> Request | None:
        """Parse one request, or return ``None`` when it is unusable."""
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        except (TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            return None
        if len(head) > _MAX_HEADER_BYTES:
            return None

        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) < 2:  # noqa: PLR2004 - "METHOD path [version]"
            return None
        method, target = parts[0], parts[1]

        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()

        length = min(int(headers.get("content-length", "0") or 0), self._config.max_request_bytes)
        body = await reader.readexactly(length) if length > 0 else b""

        path, _, raw_query = target.partition("?")
        query = {}
        for pair in raw_query.split("&"):
            if "=" in pair:
                key, _, value = pair.partition("=")
                query[_unquote(key)] = _unquote(value)
        return Request(method=method, path=path, query=query, body=body)

    async def _dispatch(self, request: Request) -> Response:
        """Route the request and translate any EDEN error into a status."""
        handler = self._router.get(request.method, request.path)
        if handler is None:
            return Response.error(404, f"No route for {request.method} {request.path}.")
        try:
            return await handler(request)
        except InterfaceError as exc:
            return Response.error(400, exc.message)
        except EdenError as exc:
            return Response.json(exc.to_dict(), status=400)


def _encode(response: Response) -> bytes:
    """Serialise a response to bytes."""
    reason = _STATUS_TEXT.get(response.status, "Unknown")
    head = (
        f"HTTP/1.1 {response.status} {reason}\r\n"
        f"Content-Type: {response.content_type}\r\n"
        f"Content-Length: {len(response.body)}\r\n"
        "Cache-Control: no-store\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "Connection: close\r\n\r\n"
    )
    return head.encode("latin-1") + response.body


def _unquote(text: str) -> str:
    """Decode percent-encoding and ``+`` in a query component."""
    from urllib.parse import unquote_plus  # noqa: PLC0415 - keeps the import graph flat

    return unquote_plus(text)
