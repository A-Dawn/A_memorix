from __future__ import annotations

import numpy as np


class FakeEmbedding:
    def __init__(self, dimension: int = 4):
        self.dimension = int(dimension)

    async def _detect_dimension(self) -> int:
        return self.dimension

    async def encode(self, texts, *args, **kwargs):
        del args, kwargs
        if isinstance(texts, str):
            return self._vector_for(texts)
        return np.vstack([self._vector_for(str(item)) for item in texts])

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def _vector_for(self, text: str) -> np.ndarray:
        raw = str(text or "")
        values = [0.0] * self.dimension
        for idx, ch in enumerate(raw.encode("utf-8") or b"x"):
            values[idx % self.dimension] += float((ch % 31) + 1)
        arr = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
