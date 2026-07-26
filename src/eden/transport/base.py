"""HTTP transport contract.

Providers never import an HTTP library. They depend on :class:`HttpTransport`,
which means the entire provider suite can be exercised in unit tests against an
in-memory fake, and the networking library can be replaced without editing a
single provider.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from eden.errors import TransportError

_CLIENT_ERROR_FLOOR = 400
_SERVER_ERROR_FLOOR = 500
_RATE_LIMIT_STATUS = 429
_AUTH_STATUSES = frozenset({401, 403})


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """An HTTP response decoupled from any client library.

    Attributes:
        status_code: Numeric HTTP status.
        headers: Response headers, lower-cased by the transport.
        body: Raw response bytes.
    """

    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def is_success(self) -> bool:
        """Return whether the status indicates success."""
        return self.status_code < _CLIENT_ERROR_FLOOR

    @property
    def is_server_error(self) -> bool:
        """Return whether the status indicates a server-side fault."""
        return self.status_code >= _SERVER_ERROR_FLOOR

    @property
    def is_rate_limited(self) -> bool:
        """Return whether the status indicates throttling."""
        return self.status_code == _RATE_LIMIT_STATUS

    @property
    def is_auth_failure(self) -> bool:
        """Return whether the status indicates rejected credentials."""
        return self.status_code in _AUTH_STATUSES

    def text(self) -> str:
        """Decode the body as UTF-8, replacing undecodable bytes."""
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:  # noqa: ANN401 - vendor payloads are heterogeneous
        """Parse the body as JSON.

        Returns:
            The decoded payload.

        Raises:
            TransportError: If the body is not valid JSON.
        """
        try:
            return json.loads(self.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TransportError(
                "Response body was not valid JSON.",
                context={"status_code": self.status_code},
                cause=exc,
            ) from exc


@runtime_checkable
class HttpTransport(Protocol):
    """Minimal asynchronous HTTP surface required by EDEN providers."""

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse:
        """Send a JSON POST and return the complete response."""
        ...

    def stream_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> AsyncIterator[str]:
        """Send a JSON POST and yield the response body line by line."""
        ...

    async def aclose(self) -> None:
        """Release pooled connections. Must be idempotent."""
        ...
