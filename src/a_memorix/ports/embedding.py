"""Embedding provider contract."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generate vectors in one stable embedding space."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        """Embed a non-empty text batch."""

    def fingerprint(self) -> Mapping[str, object]:
        """Describe the provider, model and observed vector space."""
