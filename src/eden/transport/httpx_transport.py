"""Production HTTP transport backed by ``httpx``.

``httpx`` is imported lazily inside the constructor so that importing
:mod:`eden` costs nothing when a deployment only uses local or mocked
providers. Connections are pooled across every provider sharing this instance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from eden.errors import TransportError
from eden.logging import get_logger
from eden.transport.base import HttpResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

_LOGGER = get_logger("transport.httpx")

DEFAULT_MAX_CONNECTIONS = 64
DEFAULT_MAX_KEEPALIVE = 16


class HttpxTransport:
    """An :class:`~eden.transport.base.HttpTransport` implementation using ``httpx``."""

    def __init__(
        self,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE,
        default_timeout: float = 60.0,
    ) -> None:
        """Create the transport and its connection pool.

        Args:
            max_connections: Pool ceiling across all hosts.
            max_keepalive_connections: Idle connections kept warm.
            default_timeout: Fallback timeout in seconds.

        Raises:
            TransportError: If ``httpx`` is not installed.
        """
        try:
            import httpx as httpx_module  # noqa: PLC0415 - deliberate lazy import
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise TransportError(
                "The 'http' extra is required for network providers. "
                "Install it with: pip install eden-os[http]",
                cause=exc,
            ) from exc

        self._httpx = httpx_module
        self._client: httpx.AsyncClient = httpx_module.AsyncClient(
            timeout=default_timeout,
            limits=httpx_module.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )
        self._closed = False
        _LOGGER.debug(
            "HTTP transport initialised.",
            extra={"max_connections": max_connections},
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> HttpResponse:
        """Send a JSON POST.

        Args:
            url: Absolute request URL.
            headers: Request headers, including authentication.
            payload: JSON-serialisable request body.
            timeout: Wall-clock budget in seconds.

        Returns:
            The complete response.

        Raises:
            TransportError: On timeout, connection failure or a closed transport.
        """
        self._ensure_open()
        try:
            response = await self._client.post(
                url,
                headers=dict(headers),
                json=dict(payload),
                timeout=timeout,
            )
        except self._httpx.TimeoutException as exc:
            raise TransportError(
                "HTTP request timed out.",
                context={"url": _safe_url(url), "timeout": timeout},
                cause=exc,
            ) from exc
        except self._httpx.HTTPError as exc:
            raise TransportError(
                "HTTP request failed.",
                context={"url": _safe_url(url)},
                cause=exc,
            ) from exc
        return HttpResponse(
            status_code=response.status_code,
            headers={key.lower(): value for key, value in response.headers.items()},
            body=response.content,
        )

    async def stream_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> AsyncIterator[str]:
        """Send a JSON POST and yield response lines as they arrive.

        Args:
            url: Absolute request URL.
            headers: Request headers, including authentication.
            payload: JSON-serialisable request body.
            timeout: Wall-clock budget in seconds.

        Yields:
            Decoded response lines with trailing newlines stripped.

        Raises:
            TransportError: On timeout, connection failure or a non-2xx status.
        """
        self._ensure_open()
        try:
            async with self._client.stream(
                "POST",
                url,
                headers=dict(headers),
                json=dict(payload),
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:  # noqa: PLR2004 - HTTP error floor
                    body = await response.aread()
                    raise TransportError(
                        "Streaming request returned an error status.",
                        context={
                            "url": _safe_url(url),
                            "status_code": response.status_code,
                            "body": body.decode("utf-8", errors="replace")[:512],
                        },
                    )
                async for line in response.aiter_lines():
                    if line:
                        yield line
        except self._httpx.TimeoutException as exc:
            raise TransportError(
                "Streaming request timed out.",
                context={"url": _safe_url(url), "timeout": timeout},
                cause=exc,
            ) from exc
        except self._httpx.HTTPError as exc:
            raise TransportError(
                "Streaming request failed.",
                context={"url": _safe_url(url)},
                cause=exc,
            ) from exc

    async def aclose(self) -> None:
        """Close the pool. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()
        _LOGGER.debug("HTTP transport closed.")

    def _ensure_open(self) -> None:
        """Raise if the transport has already been closed.

        Raises:
            TransportError: If the transport is closed.
        """
        if self._closed:
            raise TransportError("HTTP transport has already been closed.")


def _safe_url(url: str) -> str:
    """Return ``url`` without its query string, which may carry credentials."""
    return url.split("?", 1)[0]
