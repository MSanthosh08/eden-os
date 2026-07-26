"""Adapter for every vendor speaking the OpenAI chat-completions contract.

OpenAI, Groq, DeepSeek, Together, OpenRouter, Fireworks, vLLM and Ollama all
expose the same ``/chat/completions`` shape. They are therefore *one* adapter
differentiated purely by configuration — base URL, credential variable, model
catalogue and pricing. Adding Fireworks is a config entry, not a code change.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from eden.config.enums import FinishReason
from eden.config.schema import ProviderConfig
from eden.config.secrets import SecretResolver
from eden.core.types import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Usage,
)
from eden.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from eden.gateway.provider import BaseProvider
from eden.transport.base import HttpResponse, HttpTransport
from eden.utils.clock import Clock

CHAT_COMPLETIONS_PATH = "/chat/completions"
EMBEDDINGS_PATH = "/embeddings"
STREAM_PREFIX = "data: "
STREAM_SENTINEL = "[DONE]"

_FINISH_REASONS: Mapping[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALL,
    "function_call": FinishReason.TOOL_CALL,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAICompatibleProvider(BaseProvider):
    """Talks to any endpoint implementing OpenAI's chat-completions API."""

    def __init__(
        self,
        config: ProviderConfig,
        transport: HttpTransport,
        *,
        secrets: SecretResolver | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            config: Provider configuration slice.
            transport: Injected HTTP transport.
            secrets: Credential resolver.
            clock: Time source for rate limiting and backoff.
        """
        super().__init__(config, secrets=secrets, clock=clock)
        self._transport = transport

    # ------------------------------------------------------------------
    # Wire format
    # ------------------------------------------------------------------
    def _endpoint(self) -> str:
        """Return the absolute chat-completions URL."""
        return f"{self._config.base_url}{CHAT_COMPLETIONS_PATH}"

    def _headers(self) -> dict[str, str]:
        """Return request headers including the bearer credential."""
        headers = {"content-type": "application/json", **self.base_headers()}
        secret = self.credential(required=bool(self._config.api_key_env))
        if secret is not None:
            headers["authorization"] = f"Bearer {secret.reveal()}"
        return headers

    def _payload(
        self,
        request: ChatRequest,
        model_name: str,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the vendor request body."""
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = list(request.stop)
        return payload

    # ------------------------------------------------------------------
    # Template method hooks
    # ------------------------------------------------------------------
    async def _perform_chat(self, request: ChatRequest, model_name: str) -> ChatResponse:
        """Send one request and parse the completion.

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
        return self._parse_completion(response.json(), model_name)

    async def _perform_stream(
        self,
        request: ChatRequest,
        model_name: str,
    ) -> AsyncIterator[StreamChunk]:
        """Send one request and yield server-sent-event deltas."""
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
            body = fragment[len(STREAM_PREFIX) :].strip()
            if body == STREAM_SENTINEL:
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                self._logger.debug(
                    "Skipping unparsable stream fragment.",
                    extra={"provider": self.name},
                )
                continue
            chunk = self._parse_stream_event(event, model_name, index)
            if chunk is not None:
                yield chunk
                index += 1

    async def _perform_embed(
        self,
        request: EmbeddingRequest,
        model_name: str,
    ) -> EmbeddingResponse:
        """Send one embedding request and parse the vectors."""
        payload: dict[str, Any] = {"model": model_name, "input": list(request.texts)}
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions
        response = await self._transport.post_json(
            f"{self._config.base_url}{EMBEDDINGS_PATH}",
            headers=self._headers(),
            payload=payload,
            timeout=self._config.timeout_seconds,
        )
        self._raise_for_status(response)
        return self._parse_embeddings(response.json(), model_name)

    def _parse_embeddings(
        self,
        payload: Any,  # noqa: ANN401 - vendor payloads are heterogeneous
        model_name: str,
    ) -> EmbeddingResponse:
        """Convert a vendor embedding payload into a neutral response.

        Raises:
            ProviderResponseError: If the payload holds no usable vectors.
        """
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                "Embedding payload was not a JSON object.", provider=self.name
            )
        entries = payload.get("data")
        if not isinstance(entries, list) or not entries:
            raise ProviderResponseError(
                "Embedding payload contained no vectors.", provider=self.name
            )
        # Vendors do not guarantee ordering, but they do return an index.
        ordered = sorted(
            (entry for entry in entries if isinstance(entry, dict)),
            key=lambda entry: _as_int(entry.get("index")),
        )
        vectors: list[tuple[float, ...]] = []
        for entry in ordered:
            raw = entry.get("embedding")
            if not isinstance(raw, list):
                raise ProviderResponseError("Embedding entry was malformed.", provider=self.name)
            vectors.append(tuple(float(value) for value in raw))
        usage_raw = payload.get("usage")
        usage = Usage()
        if isinstance(usage_raw, dict):
            usage = Usage(prompt_tokens=_as_int(usage_raw.get("prompt_tokens")))
        return EmbeddingResponse(
            vectors=tuple(vectors),
            model=str(payload.get("model") or model_name),
            provider=self.name,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
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
        detail = response.text()[:512]
        context = {"status_code": response.status_code, "detail": detail}
        if response.is_auth_failure:
            raise ProviderAuthenticationError(
                "Provider rejected the credential.",
                provider=self.name,
                context=context,
            )
        if response.is_rate_limited:
            raise ProviderRateLimitError(
                "Provider is rate limiting this client.",
                provider=self.name,
                context=context,
            )
        if response.is_server_error:
            raise ProviderUnavailableError(
                "Provider returned a server error.",
                provider=self.name,
                context=context,
            )
        raise ProviderResponseError(
            "Provider rejected the request.",
            provider=self.name,
            context=context,
        )

    def _parse_completion(
        self,
        payload: Any,  # noqa: ANN401 - vendor payloads are heterogeneous
        model_name: str,
    ) -> ChatResponse:
        """Convert a vendor completion into a neutral response.

        Raises:
            ProviderResponseError: If the payload lacks a usable choice.
        """
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                "Completion payload was not a JSON object.",
                provider=self.name,
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError(
                "Completion payload contained no choices.",
                provider=self.name,
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise ProviderResponseError(
                "Completion choice was malformed.",
                provider=self.name,
            )
        message = first.get("message")
        content = ""
        if isinstance(message, dict):
            content = str(message.get("content") or "")
        usage_raw = payload.get("usage")
        usage = Usage()
        if isinstance(usage_raw, dict):
            usage = Usage(
                prompt_tokens=_as_int(usage_raw.get("prompt_tokens")),
                completion_tokens=_as_int(usage_raw.get("completion_tokens")),
            )
        return ChatResponse(
            content=content,
            model=str(payload.get("model") or model_name),
            provider=self.name,
            usage=usage,
            finish_reason=_finish_reason(first.get("finish_reason")),
            metadata={"id": str(payload.get("id") or "")},
        )

    def _parse_stream_event(
        self,
        event: Any,  # noqa: ANN401 - vendor payloads are heterogeneous
        model_name: str,
        index: int,
    ) -> StreamChunk | None:
        """Convert one stream event into a neutral chunk, or ``None`` to skip."""
        if not isinstance(event, dict):
            return None
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        delta = first.get("delta")
        text = ""
        if isinstance(delta, dict):
            text = str(delta.get("content") or "")
        raw_reason = first.get("finish_reason")
        reason = _finish_reason(raw_reason) if raw_reason else None
        if not text and reason is None:
            return None
        return StreamChunk(
            delta=text,
            provider=self.name,
            model=str(event.get("model") or model_name),
            finish_reason=reason,
            index=index,
        )


def _finish_reason(value: object) -> FinishReason:
    """Map a vendor finish reason onto the neutral enum."""
    if isinstance(value, str):
        return _FINISH_REASONS.get(value, FinishReason.STOP)
    return FinishReason.STOP


def _as_int(value: object) -> int:
    """Coerce a vendor token count to ``int``, defaulting to zero."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
