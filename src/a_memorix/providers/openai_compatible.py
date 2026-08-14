"""OpenAI-compatible Embedding and chat-completions providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import asyncio
import json
import math
import time

from a_memorix.ports import LLMRequest, LLMResult


_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 429, 502, 503, 504})


@dataclass(frozen=True)
class LLMTaskConfig:
    """Task model settings understood by A_memorix model routing."""

    model_list: tuple[str, ...]
    max_tokens: int = 8192
    temperature: float = 0.0
    selection_strategy: str = "fixed"
    slow_threshold: float = 0.0
    hard_timeout: float = 0.0


class _OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_attempts: int,
        retry_delay_seconds: float,
        retry_max_delay_seconds: float,
        retry_backoff_multiplier: float,
        max_concurrent: int,
    ) -> None:
        self.endpoint = endpoint.strip()
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.retry_max_delay_seconds = max(
            self.retry_delay_seconds,
            float(retry_max_delay_seconds),
        )
        self.retry_backoff_multiplier = max(1.0, float(retry_backoff_multiplier))
        self.request_count = 0
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))
        self._last_error = ""
        self._last_success_at: float | None = None
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("provider endpoint must be an HTTP(S) URL")
        if not self.model:
            raise ValueError("provider model must be non-empty")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "A_memorix/2",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, url: str, payload: Mapping[str, object]) -> Mapping[str, Any]:
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = Request(
                url,
                data=request_body,
                headers=self._headers(),
                method="POST",
            )
            self.request_count += 1
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                decoded = json.loads(body)
                if not isinstance(decoded, Mapping):
                    raise RuntimeError("provider response must be a JSON object")
                self._last_error = ""
                self._last_success_at = time.time()
                return decoded
            except HTTPError as exc:
                detail = self._redact(
                    exc.read(2048).decode("utf-8", errors="replace")
                )
                last_error = RuntimeError(
                    f"provider request failed with HTTP {exc.code}: {detail}"
                )
                retryable = exc.code in _RETRYABLE_HTTP_STATUSES
            except (URLError, TimeoutError, OSError) as exc:
                last_error = RuntimeError(
                    f"provider request failed: {type(exc).__name__}"
                )
                retryable = True
            except (TypeError, ValueError, json.JSONDecodeError):
                last_error = RuntimeError("provider response is not valid JSON")
                retryable = False
            if not retryable or attempt >= self.max_attempts:
                break
            delay = min(
                self.retry_max_delay_seconds,
                self.retry_delay_seconds
                * self.retry_backoff_multiplier ** (attempt - 1),
            )
            if delay:
                time.sleep(delay)
        message = self._redact(str(last_error or "provider request failed"))
        self._last_error = message[:500]
        raise RuntimeError(message) from last_error

    def _redact(self, value: str) -> str:
        if self.api_key:
            return value.replace(self.api_key, "[redacted]")
        return value

    def _public_fingerprint(self, url: str) -> dict[str, object]:
        identity = {
            "provider": "openai-compatible",
            "model": self.model,
            "endpoint_sha256": sha256(url.encode("utf-8")).hexdigest(),
        }
        digest = sha256(
            json.dumps(
                identity,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {**identity, "hash": f"sha256:{digest}"}

    def health_status(self) -> Mapping[str, object]:
        return {
            "available": not self._last_error,
            "last_error": self._last_error,
            "last_success_at": self._last_success_at,
            "request_count": self.request_count,
        }


class OpenAICompatibleEmbeddingProvider(_OpenAICompatibleProvider):
    """Call an OpenAI-compatible Embeddings endpoint with validated ordering."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.embeddings_url = _operation_url(self.endpoint, "embeddings")
        self._observed_dimension: int | None = None

    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        normalized = [str(text) for text in texts]
        if not normalized:
            return []
        payload: dict[str, object] = {
            "model": self.model,
            "input": normalized,
        }
        if dimensions is not None:
            payload["dimensions"] = max(1, int(dimensions))
        async with self._semaphore:
            decoded = await asyncio.to_thread(
                self._post_json,
                self.embeddings_url,
                payload,
            )
        try:
            rows = list(decoded["data"])
            ordered = sorted(rows, key=lambda item: int(item["index"]))
            indexes = [int(item["index"]) for item in ordered]
            vectors = [list(item["embedding"]) for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "embedding response does not match the OpenAI schema"
            ) from exc
        if indexes != list(range(len(normalized))) or len(vectors) != len(normalized):
            raise RuntimeError("embedding response indexes do not match the request")
        observed_dimensions = set()
        for vector in vectors:
            if not vector or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in vector
            ):
                raise RuntimeError("embedding response contains an invalid vector")
            observed_dimensions.add(len(vector))
        if len(observed_dimensions) != 1:
            raise RuntimeError("embedding response contains mixed vector dimensions")
        self._observed_dimension = observed_dimensions.pop()
        return vectors

    async def probe(self, *, dimensions: int | None = None) -> int:
        vectors = await self.embed(
            ["A_memorix provider readiness probe"],
            dimensions=dimensions,
        )
        return len(vectors[0])

    def fingerprint(self) -> Mapping[str, object]:
        fingerprint = self._public_fingerprint(self.embeddings_url)
        if self._observed_dimension is not None:
            fingerprint.update(
                {
                    "dimension": self._observed_dimension,
                    "dimension_verified": True,
                }
            )
        return fingerprint


class OpenAICompatibleLLMProvider(_OpenAICompatibleProvider):
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
        thinking_mode: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max(1, int(max_tokens))
        self.temperature = min(2.0, max(0.0, float(temperature)))
        self.enable_thinking = enable_thinking
        self.thinking_mode = thinking_mode
        self.chat_completions_url = _operation_url(
            self.endpoint,
            "chat/completions",
        )

    def get_available_models(self) -> Mapping[str, object]:
        task = LLMTaskConfig(
            model_list=(self.model,),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            hard_timeout=self.timeout_seconds,
        )
        return {
            "memory": task,
            "utils": task,
            "planner": task,
        }

    async def generate(self, request: LLMRequest) -> LLMResult:
        payload: dict[str, object] = {
            "model": request.model or self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": (
                self.temperature
                if request.temperature is None
                else min(2.0, max(0.0, float(request.temperature)))
            ),
            "max_tokens": max(1, int(request.max_tokens or self.max_tokens)),
        }
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.thinking_mode is not None:
            payload["thinking"] = {"type": self.thinking_mode}
        try:
            async with self._semaphore:
                decoded = await asyncio.to_thread(
                    self._post_json,
                    self.chat_completions_url,
                    payload,
                )
            choice = list(decoded["choices"])[0]
            message = choice["message"]
            content = _message_content(message.get("content"))
            finish_reason = str(choice.get("finish_reason", "") or "")
            if not content:
                raise RuntimeError(
                    "LLM response contained no final content"
                    + (
                        f" (finish_reason={finish_reason})"
                        if finish_reason
                        else ""
                    )
                )
        except (KeyError, IndexError, TypeError, ValueError):
            error = "LLM response does not match the OpenAI chat-completions schema"
            self._last_error = error
            return LLMResult.from_error(error)
        except RuntimeError as exc:
            error = self._redact(str(exc))
            self._last_error = error[:500]
            return LLMResult.from_error(error)
        return LLMResult(
            success=True,
            content=content,
            metadata={
                "model": str(decoded.get("model", self.model)),
                "finish_reason": finish_reason,
                "usage": decoded.get("usage", {}),
            },
        )

    async def probe(self) -> None:
        result = await self.generate(
            LLMRequest(
                prompt="Reply with OK.",
                request_type="health_probe",
                task_name="memory",
                model=self.model,
                temperature=0.0,
                max_tokens=8,
            )
        )
        if not result.success or not result.content.strip():
            raise RuntimeError(result.error or "LLM readiness probe failed")

    def fingerprint(self) -> Mapping[str, object]:
        return self._public_fingerprint(self.chat_completions_url)


def _operation_url(endpoint: str, operation: str) -> str:
    parsed = urlparse(endpoint.strip())
    path = parsed.path.rstrip("/")
    operation_suffix = f"/{operation}"
    if path.endswith(operation_suffix):
        effective_path = path
    elif path.endswith("/v1"):
        effective_path = f"{path}{operation_suffix}"
    elif not path:
        effective_path = f"/v1{operation_suffix}"
    else:
        effective_path = f"{path}{operation_suffix}"
    return urlunparse(parsed._replace(path=effective_path))


def _message_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Sequence):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}:
            text = str(item.get("text", "") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)
