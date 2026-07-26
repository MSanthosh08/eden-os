"""Adapter for Google's Gemini ``generateContent`` API.

Gemini names the assistant role ``model``, nests text inside ``parts``, and
carries the model name in the URL path rather than the body. Those three
translations are the entire contents of this module.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from eden.config.enums import FinishReason, Role
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

GENERATE_PATH = "/v1beta/models/{model}:generateContent"
STREAM_PATH = "/v1beta/models/{model}:streamGenerateContent?alt=sse"
EMBED_PATH = "/v1beta/models/{model}:batchEmbedContents"
STREAM_PREFIX = "data: "

_ROLE_MAP: Mapping[Role, str] = {
    Role.USER: "user",
    Role.ASSISTANT: "model",
    Role.TOOL: "user",
    Role.SYSTEM: "user",
}

_FINISH_REASONS: Mapping[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
}


class GeminiProvider(BaseProvider):
    """Talks to the Google Gemini generative language API."""

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

    def _endpoint(self, model_name: str, *, stream: bool) -> str:
        """Return the absolute URL for ``model_name``."""
        template = STREAM_PATH if stream else GENERATE_PATH
        return f"{self._config.base_url}{template.format(model=model_name)}"

    def _headers(self) -> dict[str, str]:
        """Return request headers including the API key."""
        headers = {"content-type": "application/json", **self.base_headers()}
        secret = self.credential(required=bool(self._config.api_key_env))
        if secret is not None:
            headers["x-goog-api-key"] = secret.reveal()
        return headers

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        """Build the vendor request body."""
        system_parts = [
            message.content for message in request.messages if message.role is Role.SYSTEM
        ]
        contents = [
            {"role": _ROLE_MAP[message.role], "parts": [{"text": message.content}]}
            for message in request.messages
            if message.role is not Role.SYSTEM
        ]
        generation: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            generation["maxOutputTokens"] = request.max_tokens
        if request.stop:
            generation["stopSequences"] = list(request.stop)
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return payload

    async def _perform_chat(self, request: ChatRequest, model_name: str) -> ChatResponse:
        """Send one request and parse the candidate.

        Raises:
            ProviderError: Subclass appropriate to the returned status.
        """
        response = await self._transport.post_json(
            self._endpoint(model_name, stream=False),
            headers=self._headers(),
            payload=self._payload(request),
            timeout=self._config.timeout_seconds,
        )
        self._raise_for_status(response)
        return self._parse_candidate(response.json(), model_name)

    async def _perform_stream(
        self,
        request: ChatRequest,
        model_name: str,
    ) -> AsyncIterator[StreamChunk]:
        """Send one request and yield candidate deltas."""
        index = 0
        lines = self._transport.stream_json(
            self._endpoint(model_name, stream=True),
            headers=self._headers(),
            payload=self._payload(request),
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
            text, reason = _extract_candidate(event)
            if not text and reason is None:
                continue
            yield StreamChunk(
                delta=text,
                provider=self.name,
                model=model_name,
                finish_reason=reason,
                index=index,
            )
            index += 1

    async def _perform_embed(
        self,
        request: EmbeddingRequest,
        model_name: str,
    ) -> EmbeddingResponse:
        """Send one batched embedding request and parse the vectors.

        Raises:
            ProviderResponseError: If the payload holds no usable vectors.
        """
        payload: dict[str, Any] = {
            "requests": [
                {
                    "model": f"models/{model_name}",
                    "content": {"parts": [{"text": text}]},
                }
                for text in request.texts
            ]
        }
        response = await self._transport.post_json(
            f"{self._config.base_url}{EMBED_PATH.format(model=model_name)}",
            headers=self._headers(),
            payload=payload,
            timeout=self._config.timeout_seconds,
        )
        self._raise_for_status(response)
        body = response.json()
        if not isinstance(body, dict):
            raise ProviderResponseError(
                "Embedding payload was not a JSON object.", provider=self.name
            )
        entries = body.get("embeddings")
        if not isinstance(entries, list) or not entries:
            raise ProviderResponseError(
                "Embedding payload contained no vectors.", provider=self.name
            )
        vectors: list[tuple[float, ...]] = []
        for entry in entries:
            values = entry.get("values") if isinstance(entry, dict) else None
            if not isinstance(values, list):
                raise ProviderResponseError("Embedding entry was malformed.", provider=self.name)
            vectors.append(tuple(float(value) for value in values))
        return EmbeddingResponse(vectors=tuple(vectors), model=model_name, provider=self.name)

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

    def _parse_candidate(
        self,
        payload: Any,  # noqa: ANN401 - vendor payloads are heterogeneous
        model_name: str,
    ) -> ChatResponse:
        """Convert a vendor candidate into a neutral response.

        Raises:
            ProviderResponseError: If the payload contains no candidate.
        """
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                "Candidate payload was not a JSON object.", provider=self.name
            )
        text, reason = _extract_candidate(payload)
        if not text and reason is None:
            raise ProviderResponseError(
                "Candidate payload contained no content.", provider=self.name
            )
        usage_raw = payload.get("usageMetadata")
        usage = Usage()
        if isinstance(usage_raw, dict):
            usage = Usage(
                prompt_tokens=_as_int(usage_raw.get("promptTokenCount")),
                completion_tokens=_as_int(usage_raw.get("candidatesTokenCount")),
            )
        return ChatResponse(
            content=text,
            model=model_name,
            provider=self.name,
            usage=usage,
            finish_reason=reason or FinishReason.STOP,
        )


def _extract_candidate(payload: object) -> tuple[str, FinishReason | None]:
    """Return the concatenated text and finish reason of the first candidate."""
    if not isinstance(payload, dict):
        return "", None
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return "", None
    first = candidates[0]
    if not isinstance(first, dict):
        return "", None
    content = first.get("content")
    text = ""
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    raw_reason = first.get("finishReason")
    reason = _FINISH_REASONS.get(raw_reason) if isinstance(raw_reason, str) else None
    return text, reason


def _as_int(value: object) -> int:
    """Coerce a vendor token count to ``int``, defaulting to zero."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
