from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from a_memorix.core.embedding.api_adapter import EmbeddingAPIAdapter
from a_memorix.core.utils.runtime_self_check import run_embedding_runtime_self_check


class FakeEmbeddingProvider:
    def __init__(self, natural_dimension: int = 12) -> None:
        self.natural_dimension = natural_dimension
        self.calls: list[tuple[list[str], int | None]] = []

    async def embed(self, texts, *, dimensions=None):
        normalized = list(texts)
        self.calls.append((normalized, dimensions))
        dimension = int(dimensions or self.natural_dimension)
        return [[float(ord(text[0])), *([1.0] * (dimension - 1))] for text in normalized]

    def fingerprint(self):
        return {"provider": "fake", "model": "fake-embedding"}


@pytest.mark.asyncio
async def test_explicit_mode_uses_natural_dimension_without_override() -> None:
    provider = FakeEmbeddingProvider(natural_dimension=12)
    adapter = EmbeddingAPIAdapter(provider, default_dimension=1024)

    embedding = await adapter.encode("A")

    assert provider.calls[-1][1] is None
    assert embedding.shape == (12,)
    assert adapter.get_embedding_dimension() == 12


@pytest.mark.asyncio
async def test_explicit_dimension_is_forwarded_and_verified() -> None:
    provider = FakeEmbeddingProvider()
    adapter = EmbeddingAPIAdapter(provider, default_dimension=1024)

    embedding = await adapter.encode("A", dimensions=256)

    assert provider.calls[-1][1] == 256
    assert embedding.shape == (256,)


@pytest.mark.asyncio
async def test_always_mode_uses_configured_dimension() -> None:
    provider = FakeEmbeddingProvider()
    adapter = EmbeddingAPIAdapter(provider, default_dimension=32, dimension_request_mode="always")

    embedding = await adapter.encode("A")

    assert provider.calls[-1][1] == 32
    assert embedding.shape == (32,)


@pytest.mark.asyncio
async def test_never_mode_ignores_explicit_dimension() -> None:
    provider = FakeEmbeddingProvider(natural_dimension=7)
    adapter = EmbeddingAPIAdapter(provider, default_dimension=32, dimension_request_mode="never")

    embedding = await adapter.encode("A", dimensions=3)

    assert provider.calls[-1][1] is None
    assert embedding.shape == (7,)


@pytest.mark.asyncio
async def test_batch_cache_preserves_input_order() -> None:
    provider = FakeEmbeddingProvider(natural_dimension=4)
    adapter = EmbeddingAPIAdapter(provider, default_dimension=4, enable_cache=True)

    first = await adapter.encode(["A", "B"], batch_size=2)
    second = await adapter.encode(["A", "C"], batch_size=2)

    assert np.array_equal(first[0], second[0])
    assert second[:, 0].tolist() == [float(ord("A")), float(ord("C"))]
    assert provider.calls[-1][0] == ["C"]


def test_fingerprint_uses_provider_identity_and_observed_dimension() -> None:
    provider = FakeEmbeddingProvider()
    adapter = EmbeddingAPIAdapter(provider, default_dimension=8)

    fingerprint = adapter.get_embedding_fingerprint()

    assert fingerprint["provider"] == "fake"
    assert fingerprint["model"] == "fake-embedding"
    assert fingerprint["dimension"] == 8
    assert fingerprint["hash"].startswith("sha256:")


@pytest.mark.asyncio
async def test_runtime_self_check_reports_observed_dimension() -> None:
    class FakeEmbeddingManager:
        async def _detect_dimension(self) -> int:
            return 384

        def get_requested_dimension(self) -> int:
            return 384

        async def encode(self, text):
            assert text == "A_Memorix runtime self check"
            return np.ones(384, dtype=np.float32)

    report = await run_embedding_runtime_self_check(
        config={"embedding": {"dimension": 1024}},
        vector_store=SimpleNamespace(dimension=384),
        embedding_manager=FakeEmbeddingManager(),
    )

    assert report["ok"] is True
    assert report["configured_dimension"] == 1024
    assert report["detected_dimension"] == 384
