"""Generic adapter around an injected embedding provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from a_memorix.logging import get_logger
from a_memorix.ports import EmbeddingProvider

logger = get_logger("a_memorix.embedding")


class EmbeddingAPIAdapter:
    """Provide the array and batching API used by the storage runtime."""

    _GLOBAL_DIMENSION_CACHE: Dict[str, int] = {}
    _GLOBAL_TEXT_EMBEDDING_CACHE: Dict[Tuple[str, int, str], np.ndarray] = {}

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 32,
        max_concurrent: int = 5,
        default_dimension: int = 1024,
        enable_cache: bool = False,
        model_name: str = "auto",
        dimension_request_mode: str = "explicit",
        retry_config: Optional[dict] = None,
    ) -> None:
        if provider is None:
            raise ValueError("embedding provider is required")
        self.provider = provider
        self.batch_size = max(1, int(batch_size))
        self.max_concurrent = max(1, int(max_concurrent))
        self.default_dimension = max(1, int(default_dimension))
        self.enable_cache = bool(enable_cache)
        self.model_name = str(model_name or "auto")
        self.dimension_request_mode = self._normalize_dimension_request_mode(dimension_request_mode)
        self.retry_config = dict(retry_config or {})
        self.max_attempts = max(1, int(self.retry_config.get("max_attempts", 3)))
        self.max_wait_seconds = max(0.1, float(self.retry_config.get("max_wait_seconds", 20)))
        self.min_wait_seconds = max(0.0, float(self.retry_config.get("min_wait_seconds", 0.5)))
        self.backoff_multiplier = max(1.0, float(self.retry_config.get("backoff_multiplier", 2)))
        self._dimension: Optional[int] = None
        self._dimension_detected = False
        provider_fingerprint = self._provider_fingerprint()
        if provider_fingerprint.get("dimension_verified") is True:
            observed_dimension = int(provider_fingerprint.get("dimension", 0) or 0)
            if observed_dimension <= 0:
                raise ValueError(
                    "embedding provider declared an invalid verified dimension"
                )
            if observed_dimension != self.default_dimension:
                raise ValueError(
                    "verified embedding dimension does not match runtime config: "
                    f"configured={self.default_dimension}, observed={observed_dimension}"
                )
            self._dimension = observed_dimension
            self._dimension_detected = True
        self._total_encoded = 0
        self._total_errors = 0
        self._total_time = 0.0

    @staticmethod
    def _normalize_dimension_request_mode(mode: str) -> str:
        normalized = str(mode or "explicit").strip().lower()
        if normalized in {"auto", "explicit", "on_demand", "on-demand"}:
            return "explicit"
        if normalized in {"always", "force", "forced"}:
            return "always"
        if normalized in {"never", "none", "natural"}:
            return "never"
        raise ValueError(f"invalid embedding dimension request mode: {mode}")

    @staticmethod
    def _validate_embedding_vector(embedding: Any, *, source: str) -> np.ndarray:
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1 or array.size <= 0:
            raise RuntimeError(f"{source} returned an invalid embedding shape: {array.shape}")
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{source} returned non-finite embedding values")
        return array

    def _provider_fingerprint(self) -> Dict[str, Any]:
        raw = self.provider.fingerprint()
        return dict(raw) if raw is not None else {}

    def _dimension_cache_key(self) -> str:
        payload = {
            "provider": self._provider_fingerprint(),
            "configured_dimension": self.default_dimension,
            "dimension_request_mode": self.dimension_request_mode,
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)

    def get_embedding_fingerprint(self, *, dimension: Optional[int] = None) -> Dict[str, Any]:
        effective_dimension = max(1, int(dimension or self.get_embedding_dimension()))
        provider_data = self._provider_fingerprint()
        compare_payload: Dict[str, Any] = {
            "provider": str(provider_data.get("provider", provider_data.get("name", "")) or ""),
            "model": str(provider_data.get("model", self.model_name) or self.model_name),
            "dimension": effective_dimension,
            "dimension_request_mode": self.dimension_request_mode,
        }
        digest = hashlib.sha256(
            json.dumps(compare_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "version": 1,
            "hash": f"sha256:{digest}",
            **compare_payload,
            "source": "observed" if self._dimension_detected else "configured",
        }

    def get_requested_dimension(self) -> int:
        return int(self._dimension or self.default_dimension)

    def _requested_dimension(self, dimensions: Optional[int]) -> Optional[int]:
        if self.dimension_request_mode == "never":
            return None
        if dimensions is not None:
            return max(1, int(dimensions))
        if self.dimension_request_mode == "always":
            return self.get_requested_dimension()
        return None

    async def _embed_with_retry(
        self,
        texts: List[str],
        *,
        dimensions: Optional[int],
    ) -> List[np.ndarray]:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_vectors = await self.provider.embed(texts, dimensions=dimensions)
                vectors = list(raw_vectors)
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"embedding provider returned {len(vectors)} vectors for {len(texts)} texts"
                    )
                return [
                    self._validate_embedding_vector(vector, source=f"embedding item {index}")
                    for index, vector in enumerate(vectors)
                ]
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                wait_seconds = min(
                    self.max_wait_seconds,
                    self.min_wait_seconds * self.backoff_multiplier ** (attempt - 1),
                )
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
        raise RuntimeError(f"embedding provider failed: {last_error}") from last_error

    async def _detect_dimension(self) -> int:
        if self._dimension_detected and self._dimension is not None:
            return self._dimension
        cache_key = self._dimension_cache_key()
        cached = self._GLOBAL_DIMENSION_CACHE.get(cache_key)
        if cached is not None:
            self._dimension = int(cached)
            self._dimension_detected = True
            return self._dimension
        requested = self.default_dimension if self.dimension_request_mode == "always" else None
        vector = (await self._embed_with_retry(["A_memorix dimension probe"], dimensions=requested))[0]
        self._dimension = int(vector.size)
        self._dimension_detected = True
        self._GLOBAL_DIMENSION_CACHE[cache_key] = self._dimension
        return self._dimension

    def _embedding_cache_key(self, text: str, dimensions: Optional[int]) -> Tuple[str, int, str]:
        requested = int(dimensions or self.get_requested_dimension())
        return self._dimension_cache_key(), requested, str(text or "")

    async def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        del show_progress, normalize
        single_input = isinstance(texts, str)
        normalized_texts = [texts] if single_input else list(texts or [])
        if not normalized_texts:
            dimension = self._dimension or self.default_dimension
            return np.zeros((0, dimension), dtype=np.float32)

        requested_dimension = self._requested_dimension(dimensions)
        started_at = time.monotonic()
        try:
            vectors = await self._encode_batches(
                normalized_texts,
                batch_size=max(1, int(batch_size or self.batch_size)),
                dimensions=requested_dimension,
            )
            observed_dimensions = {int(vector.size) for vector in vectors}
            if len(observed_dimensions) != 1:
                raise RuntimeError(f"embedding provider returned mixed dimensions: {sorted(observed_dimensions)}")
            observed_dimension = observed_dimensions.pop()
            if dimensions is not None and self.dimension_request_mode != "never" and observed_dimension != int(dimensions):
                raise RuntimeError(
                    f"embedding provider ignored requested dimension: requested={dimensions}, observed={observed_dimension}"
                )
            self._dimension = observed_dimension
            self._dimension_detected = True
            self._GLOBAL_DIMENSION_CACHE[self._dimension_cache_key()] = observed_dimension
            self._total_encoded += len(vectors)
            self._total_time += time.monotonic() - started_at
            matrix = np.asarray(vectors, dtype=np.float32)
            return matrix[0] if single_input else matrix
        except Exception:
            self._total_errors += 1
            raise

    async def _encode_batches(
        self,
        texts: List[str],
        *,
        batch_size: int,
        dimensions: Optional[int],
    ) -> List[np.ndarray]:
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def encode_batch(start: int, batch: List[str]) -> None:
            uncached: List[Tuple[int, str]] = []
            for offset, text in enumerate(batch):
                index = start + offset
                cached = None
                if self.enable_cache:
                    cached = self._GLOBAL_TEXT_EMBEDDING_CACHE.get(
                        self._embedding_cache_key(text, dimensions)
                    )
                if cached is None:
                    uncached.append((index, text))
                else:
                    results[index] = cached.copy()
            if not uncached:
                return
            async with semaphore:
                vectors = await self._embed_with_retry(
                    [text for _, text in uncached],
                    dimensions=dimensions,
                )
            for (index, text), vector in zip(uncached, vectors, strict=True):
                results[index] = vector
                if self.enable_cache:
                    self._GLOBAL_TEXT_EMBEDDING_CACHE[
                        self._embedding_cache_key(text, dimensions)
                    ] = vector.copy()

        tasks = [
            encode_batch(start, texts[start : start + batch_size])
            for start in range(0, len(texts), batch_size)
        ]
        await asyncio.gather(*tasks)
        if any(vector is None for vector in results):
            raise RuntimeError("embedding batch completed with missing results")
        return [vector for vector in results if vector is not None]

    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        del show_progress
        previous = self.max_concurrent
        if num_workers is not None:
            self.max_concurrent = max(1, int(num_workers))
        try:
            return await self.encode(texts, batch_size=batch_size, dimensions=dimensions)
        finally:
            self.max_concurrent = previous

    def get_embedding_dimension(self) -> int:
        return int(self._dimension or self.default_dimension)

    def get_model_info(self) -> dict:
        return {
            "model_name": self.get_embedding_fingerprint()["model"],
            "dimension": self.get_embedding_dimension(),
            "configured_dimension": self.default_dimension,
            "requested_dimension": self.get_requested_dimension(),
            "detected_dimension": int(self._dimension or 0),
            "dimension_detected": self._dimension_detected,
            "dimension_request_mode": self.dimension_request_mode,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "total_encoded": self._total_encoded,
            "total_errors": self._total_errors,
            "avg_time_per_text": self._total_time / self._total_encoded if self._total_encoded else 0.0,
        }

    def get_statistics(self) -> dict:
        return self.get_model_info()

    @property
    def is_model_loaded(self) -> bool:
        return True


def create_embedding_api_adapter(
    provider: EmbeddingProvider,
    **kwargs: Any,
) -> EmbeddingAPIAdapter:
    return EmbeddingAPIAdapter(provider, **kwargs)
