"""
OpenAI-compatible embedding adapter.

This adapter keeps the old EmbeddingAPIAdapter interface while removing MaiBot
runtime dependencies.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List, Optional, Union

import numpy as np
from openai import AsyncOpenAI

from amemorix.common.logging import get_logger

logger = get_logger("A_Memorix.EmbeddingAPIAdapter")


def _first_env(*keys: str) -> str:
    for key in keys:
        value = str(os.getenv(key, "") or "").strip()
        if value:
            return value
    return ""


class EmbeddingAPIAdapter:
    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent: int = 5,
        default_dimension: int = 1024,
        enable_cache: bool = False,
        model_name: str = "auto",
        retry_config: Optional[dict] = None,
        base_url: str = "",
        api_key: str = "",
        openai_model: str = "",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        self.batch_size = max(1, int(batch_size))
        self.max_concurrent = max(1, int(max_concurrent))
        self.default_dimension = max(1, int(default_dimension))
        self.enable_cache = bool(enable_cache)
        self.model_name = str(model_name or "auto")
        self.timeout_seconds = float(timeout_seconds or 30.0)
        self.max_retries = max(1, int(max_retries))

        self.base_url = str(base_url or _first_env("OPENAPI_BASE_URL", "OPENAI_BASE_URL")).strip()
        self.api_key = str(api_key or _first_env("OPENAPI_API_KEY", "OPENAI_API_KEY")).strip()
        if openai_model:
            self.openai_model = str(openai_model).strip()
        elif self.model_name and self.model_name.lower() != "auto":
            self.openai_model = self.model_name
        else:
            self.openai_model = str(
                _first_env(
                    "OPENAPI_EMBEDDING_MODEL",
                    "OPENAI_EMBEDDING_MODEL",
                    "OPENAPI_MODEL",
                    "OPENAI_MODEL",
                )
                or "text-embedding-3-large"
            ).strip()

        self.retry_config = retry_config or {}
        self.max_attempts = max(1, int(self.retry_config.get("max_attempts", self.max_retries)))
        self.max_wait_seconds = max(0.1, float(self.retry_config.get("max_wait_seconds", 30)))
        self.min_wait_seconds = max(0.1, float(self.retry_config.get("min_wait_seconds", 1)))

        self._dimension: Optional[int] = None
        self._dimension_detected = False
        self._client: Optional[AsyncOpenAI] = None

        self._total_encoded = 0
        self._total_errors = 0
        self._total_time = 0.0

        logger.info(
            "Embedding adapter initialized: model=%s, default_dim=%s, base_url=%s",
            self.openai_model,
            self.default_dimension,
            self.base_url or "<default>",
        )

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            kwargs = {
                "api_key": self.api_key or "EMPTY",
                "timeout": self.timeout_seconds,
                "max_retries": 0,  # retries are handled by adapter policy
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def _request_embeddings(
        self,
        inputs: Union[str, List[str]],
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        client = self._get_client()
        payload = {"model": self.openai_model, "input": inputs}
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = await client.embeddings.create(**payload)
                return [list(item.embedding) for item in resp.data]
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                wait_s = min(
                    self.max_wait_seconds,
                    self.min_wait_seconds * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "Embedding request failed (attempt %s/%s), retry in %.1fs: %s",
                    attempt,
                    self.max_attempts,
                    wait_s,
                    exc,
                )
                await asyncio.sleep(wait_s)

        assert last_error is not None
        raise last_error

    async def _detect_dimension(self) -> int:
        if self._dimension_detected and self._dimension is not None:
            return self._dimension

        # Probe with requested dimension first.
        try:
            probed = await self._request_embeddings("dimension_probe", dimensions=self.default_dimension)
            if probed and probed[0]:
                self._dimension = len(probed[0])
                self._dimension_detected = True
                return self._dimension
        except Exception as exc:
            logger.debug("Dimension probe with requested dimension failed: %s", exc)

        # Probe with natural model dimension.
        try:
            probed = await self._request_embeddings("dimension_probe", dimensions=None)
            if probed and probed[0]:
                self._dimension = len(probed[0])
                self._dimension_detected = True
                return self._dimension
        except Exception as exc:
            logger.warning("Dimension detection failed, fallback to default: %s", exc)

        self._dimension = self.default_dimension
        self._dimension_detected = True
        return self.default_dimension

    async def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        del show_progress  # kept for compatibility
        del normalize  # API already returns normalized-ish vectors by model behavior
        start = time.time()

        if isinstance(texts, str):
            input_texts = [texts]
            single = True
        else:
            input_texts = list(texts)
            single = False

        target_dim = dimensions
        if target_dim is None:
            if not self._dimension_detected:
                await self._detect_dimension()
            target_dim = self._dimension or self.default_dimension
        target_dim = int(target_dim)

        if not input_texts:
            empty = np.zeros((0, target_dim), dtype=np.float32)
            return empty[0] if single else empty

        use_batch = max(1, int(batch_size or self.batch_size))
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _encode_chunk(chunk: List[str]) -> np.ndarray:
            async with semaphore:
                try:
                    # Always send the effective target dimension so providers that
                    # support OpenAI-compatible `dimensions` return stable vector size.
                    vectors = await self._request_embeddings(chunk, dimensions=target_dim)
                    arr = np.asarray(vectors, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return arr
                except Exception as exc:
                    self._total_errors += len(chunk)
                    logger.error("Embedding chunk failed: %s", exc)
                    return np.full((len(chunk), target_dim), np.nan, dtype=np.float32)

        tasks = []
        for idx in range(0, len(input_texts), use_batch):
            tasks.append(_encode_chunk(input_texts[idx : idx + use_batch]))
        chunks = await asyncio.gather(*tasks)
        out = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, target_dim), dtype=np.float32)

        self._total_encoded += len(input_texts)
        self._total_time += max(0.0, time.time() - start)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out[0] if single else out

    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        old = self.max_concurrent
        if num_workers is not None:
            self.max_concurrent = max(1, int(num_workers))
        try:
            return await self.encode(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress,
                dimensions=dimensions,
            )
        finally:
            self.max_concurrent = old

    def get_embedding_dimension(self) -> int:
        if self._dimension is not None:
            return int(self._dimension)
        return int(self.default_dimension)

    def get_model_info(self) -> dict:
        avg_time = self._total_time / self._total_encoded if self._total_encoded > 0 else 0.0
        return {
            "model_name": self.openai_model,
            "dimension": self.get_embedding_dimension(),
            "dimension_detected": self._dimension_detected,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "base_url": self.base_url,
            "total_encoded": self._total_encoded,
            "total_errors": self._total_errors,
            "avg_time_per_text": avg_time,
        }

    @property
    def is_model_loaded(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            "EmbeddingAPIAdapter("
            f"model={self.openai_model}, "
            f"dim={self.get_embedding_dimension()}, "
            f"encoded={self._total_encoded})"
        )


def create_embedding_api_adapter(
    batch_size: int = 32,
    max_concurrent: int = 5,
    default_dimension: int = 1024,
    model_name: str = "auto",
    retry_config: Optional[dict] = None,
    base_url: str = "",
    api_key: str = "",
    openai_model: str = "",
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
) -> EmbeddingAPIAdapter:
    return EmbeddingAPIAdapter(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        default_dimension=default_dimension,
        model_name=model_name,
        retry_config=retry_config,
        base_url=base_url,
        api_key=api_key,
        openai_model=openai_model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
