"""Adapter for Anthropic's Messages API.

Anthropic differs from the OpenAI contract in three ways that matter here: the
system prompt is a top-level field rather than a message, content arrives as a
list of typed blocks, and ``max_tokens`` is mandatory. Everything else —
retry, timing, rate limiting — is inherited unchanged.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from eden.config.enums import FinishReason, Role
from eden.config.schema import ProviderConfig
from eden.config.secrets import SecretResolver
from eden.core.types import ChatRequest, ChatResponse, StreamChunk, Usage
from eden.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from eden.gateway.provider import BaseProvider
from eden.transport.base import HttpResponse, HttpTransport
from eden.utils.clock import Clock

MESSAGES_PATH = "/v1/messages"
DEFAULT_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
STREAM_PREFIX = "data: "

_STOP_REASONS: Mapping[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALL,
}


class AnthropicProvider(BaseProvider):
    """Talks to the Anthropic Messages API."""

    def __init__(
        self,
        config: ProviderConfig,
        transport: HttpTransport,
        *,
        secrets: SecretResolver | None = None,
        clock: Clock | None = None,
        api_version: str = DEFAULT_API_VERSION,
    ) -> None:
        """Initialise the adapter.

        Args:
            config: Provider configuration slice.
            transport: Injected HTTP transport.
            secrets: Credential resolver.
            clock: Time source for rate limiting and backoff.
            api_version: Value sent in the ``anthropic-version`` header.
        """
        super().__init__(config, secrets=secrets, clock=clock)
        self._transport = transport
        self._api_version = api_version

    def _endpoint(self) -> str:
        """Return the absolute messages URL."""
        return f"{self._config.base_url}{MESSAGES_PATH}"

    def _headers(self) -> dict[str, str]:
        """Return request headers including the API key and version."""
        headers = {
            "content-type": "application/json",
            "anthropic-version": self._api_version,
            **self.base_headers(),
        }
        secret = self.credential(required=bool(self._config.api_key_env))
        if secret is not None:
            headers["x-api-key"] = secret.reveal()
        return headers

    def _payload(
        self,
        request: ChatRequest,
        model_name: str,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the vendor request body, hoisting system turns to the top level."""
        system_parts = [
            message.content for message in request.messages if message.role is Role.SYSTEM
        ]
        turns = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
            if message.role is not Role.SYSTEM
        ]
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": turns,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": request.temperature,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.stop:
            payload["stop_sequences"] = list(request.stop)
        return payload

    async def _perform_chat(self, request: ChatRequest, model_name: str) -> ChatResponse:
        """Send one request and parse the message.

        Raises:
            ProviderError: Subclass appropriate to the returned status.
        """
        response = await self._transport.post_json(
            self._endpoint(),
            headers=self._headers(),
            payload=self._payload(request, model_name, stream=False),
            timeout=self._config.timeout_seconds,
        )
        self._raise_for_status(response)
        return self._parse_message(response.json(), model_name)

    async def _perform_stream(
        self,
        request: ChatRequest,
        model_name: str,
    ) -> AsyncIterator[StreamChunk]:
        """Send one request and yield content-block deltas."""
        index = 0
        lines = self._transport.stream_json(
            self._endpoint(),
            headers=self._headers(),
            payload=self._payload(request, model_name, stream=True),
            timeout=self._config.timeout_seconds,
        )
        async for line in lines:
            fragment = line.strip()
            if not fragment.startswith(STREAM_PREFIX):
                continue
            try:
                event = json.loads(fragment[len(STREAM_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            if event_type == "content_block_delta":
                delta = event.get("delta")
                text = str(delta.get("text") or "") if isinstance(delta, dict) else ""
                if text:
                    yield StreamChunk(
                        delta=text,
                        provider=self.name,
                        model=model_name,
                        index=index,
                    )
                    index += 1
            elif event_type == "message_stop":
                yield StreamChunk(
                    delta="",
                    provider=self.name,
                    model=model_name,
                    finish_reason=FinishReason.STOP,
                    index=index,
                )
                break

    def _raise_for_status(self, response: HttpResponse) -> None:
        """Translate a non-2xx status into the right EDEN error.

        Raises:
            ProviderAuthenticationError: On 401 or 403.
            ProviderRateLimitError: On 429.
            ProviderUnavailableError: On 5xx.
            ProviderResponseError: On any other client error.
        """
        if response.is_success:
            return
        context = {"status_code": response.status_code, "detail": response.text()[:512]}
        if response.is_auth_failure:
            raise ProviderAuthenticationError(
                "Provider rejected the credential.", provider=self.name, context=context
            )
        if response.is_rate_limited:
            raise ProviderRateLimitError(
                "Provider is rate limiting this client.", provider=self.name, context=context
            )
        if response.is_server_error:
            raise ProviderUnavailableError(
                "Provider returned a server error.", provider=self.name, context=context
            )
        raise ProviderResponseError(
            "Provider rejected the request.", provider=self.name, context=context
        )

    def _parse_message(
        self,
        payload: Any,  # noqa: ANN401 - vendor payloads are heterogeneous
        model_name: str,
    ) -> ChatResponse:
        """Convert a vendor message into a neutral response.

        Raises:
            ProviderResponseError: If the payload is not a usable message.
        """
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                "Message payload was not a JSON object.", provider=self.name
            )
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ProviderResponseError(
                "Message payload contained no content blocks.", provider=self.name
            )
        text = "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage_raw = payload.get("usage")
        usage = Usage()
        if isinstance(usage_raw, dict):
            usage = Usage(
                prompt_tokens=_as_int(usage_raw.get("input_tokens")),
                completion_tokens=_as_int(usage_raw.get("output_tokens")),
            )
        stop_reason = payload.get("stop_reason")
        finish = (
            _STOP_REASONS.get(stop_reason, FinishReason.STOP)
            if isinstance(stop_reason, str)
            else FinishReason.STOP
        )
        return ChatResponse(
            content=text,
            model=str(payload.get("model") or model_name),
            provider=self.name,
            usage=usage,
            finish_reason=finish,
            metadata={"id": str(payload.get("id") or "")},
        )


def _as_int(value: object) -> int:
    """Coerce a vendor token count to ``int``, defaulting to zero."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
