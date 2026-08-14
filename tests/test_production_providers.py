from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request

import json

import pytest

from a_memorix.ports import LLMRequest
from a_memorix.providers import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _base_settings(**overrides: object) -> dict[str, object]:
    return {
        "endpoint": "https://provider.example/v1",
        "api_key": "super-secret-provider-key",
        "model": "test-model",
        "timeout_seconds": 5.0,
        "max_attempts": 2,
        "retry_delay_seconds": 0.0,
        "retry_max_delay_seconds": 0.0,
        "retry_backoff_multiplier": 2.0,
        "max_concurrent": 2,
        **overrides,
    }


@pytest.mark.asyncio
async def test_embedding_provider_retries_orders_rows_and_redacts_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Request] = []

    def fake_urlopen(request: Request, *, timeout: float) -> _Response:
        assert timeout == 5.0
        requests.append(request)
        if len(requests) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "retry",
                hdrs=None,
                fp=BytesIO(b"retry later"),
            )
        return _Response(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            }
        )

    monkeypatch.setattr(
        "a_memorix.providers.openai_compatible.urlopen",
        fake_urlopen,
    )
    provider = OpenAICompatibleEmbeddingProvider(**_base_settings())

    vectors = await provider.embed(["alpha", "beta"], dimensions=3)

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert provider.request_count == 2
    assert requests[0].full_url == "https://provider.example/v1/embeddings"
    payload = json.loads(bytes(requests[-1].data or b"{}"))
    assert payload == {
        "model": "test-model",
        "input": ["alpha", "beta"],
        "dimensions": 3,
    }
    public_state = json.dumps(
        {
            "fingerprint": provider.fingerprint(),
            "health": provider.health_status(),
            "repr": repr(provider),
        }
    )
    assert "super-secret-provider-key" not in public_state


@pytest.mark.asyncio
async def test_llm_provider_returns_routable_models_and_validated_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Request] = []

    def fake_urlopen(request: Request, *, timeout: float) -> _Response:
        requests.append(request)
        return _Response(
            {
                "model": "chat-model-effective",
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "  valid answer  "}
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 4},
            }
        )

    monkeypatch.setattr(
        "a_memorix.providers.openai_compatible.urlopen",
        fake_urlopen,
    )
    provider = OpenAICompatibleLLMProvider(
        **_base_settings(model="chat-model"),
        max_tokens=128,
        temperature=0.2,
        enable_thinking=False,
    )

    result = await provider.generate(
        LLMRequest(
            prompt="test prompt",
            request_type="relation_extraction",
            task_name="memory",
        )
    )

    assert result.success
    assert result.content == "valid answer"
    assert result.metadata["model"] == "chat-model-effective"
    assert set(provider.get_available_models()) >= {"memory", "utils", "planner"}
    assert requests[0].full_url == "https://provider.example/v1/chat/completions"
    payload = json.loads(bytes(requests[0].data or b"{}"))
    assert payload["enable_thinking"] is False


@pytest.mark.asyncio
async def test_llm_provider_converts_schema_and_transport_failures_to_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[object] = [
        _Response({"choices": []}),
        OSError("super-secret-provider-key must not leak"),
        OSError("super-secret-provider-key must not leak"),
    ]

    def fake_urlopen(request: Request, *, timeout: float) -> _Response:
        value = responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, _Response)
        return value

    monkeypatch.setattr(
        "a_memorix.providers.openai_compatible.urlopen",
        fake_urlopen,
    )
    provider = OpenAICompatibleLLMProvider(
        **_base_settings(),
        max_tokens=32,
    )

    malformed = await provider.generate(
        LLMRequest(prompt="one", request_type="test")
    )
    failed = await provider.generate(
        LLMRequest(prompt="two", request_type="test")
    )

    assert not malformed.success
    assert "schema" in malformed.error
    assert not failed.success
    assert "super-secret-provider-key" not in failed.error
