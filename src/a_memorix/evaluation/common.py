"""Shared configuration, Embedding cache and ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from math import log2
from pathlib import Path
from platform import machine, python_version, system
from statistics import mean, median
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import asyncio
import json
import os
import re
import sqlite3


DEFAULT_CUTOFFS = (1, 3, 5, 10, 20, 50)
EVALUATION_SCHEMA_VERSION = 1


@contextmanager
def evaluation_run_lock(output_dir: str | Path) -> Iterator[None]:
    """Prevent two evaluation processes from sharing one output directory."""

    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".run.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            _lock_file(handle)
        except OSError as exc:
            raise RuntimeError(
                f"another evaluation is using the output directory: {directory}"
            ) from exc
        try:
            yield
        finally:
            _unlock_file(handle)


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(  # type: ignore[attr-defined]
        handle.fileno(),
        fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
    )


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(  # type: ignore[attr-defined]
        handle.fileno(),
        fcntl.LOCK_UN,  # type: ignore[attr-defined]
    )


def runtime_fingerprint() -> dict[str, object]:
    """Return comparable runtime details without hostnames or local paths."""

    packages: dict[str, str] = {}
    for package in ("numpy", "scipy", "faiss-cpu"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            continue
    from a_memorix import __version__

    packages["a-memorix"] = __version__
    return {
        "python_version": python_version(),
        "system": system(),
        "machine": machine(),
        "packages": packages,
    }


@dataclass(frozen=True)
class EmbeddingConfig:
    """OpenAI-compatible Embedding settings loaded without exposing secrets."""

    endpoint: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float = 60.0

    @classmethod
    def from_file(cls, path: str | Path) -> "EmbeddingConfig":
        config_path = Path(path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        lines = [
            line.strip()
            for line in config_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith(("#", ";"))
        ]
        keyed: dict[str, str] = {}
        key_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]\s*(.*)$")
        for line in lines:
            match = key_pattern.match(line)
            if match:
                keyed[match.group(1).lower()] = match.group(2).strip()
        recognized_keys = {
            "endpoint",
            "base_url",
            "url",
            "api_key",
            "key",
            "token",
            "model",
            "model_id",
            "model_name",
        }
        if recognized_keys.intersection(keyed):
            endpoint = _first_value(keyed, "endpoint", "base_url", "url")
            api_key = _first_value(keyed, "api_key", "key", "token")
            model = _first_value(keyed, "model", "model_id", "model_name")
        elif len(lines) == 3:
            endpoint, api_key, model = lines
        else:
            raise ValueError(
                "config.txt must contain endpoint, API key and model on three lines, "
                "or use endpoint/api_key/model key-value entries"
            )
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Embedding endpoint must be an HTTP(S) URL")
        if not api_key or not model:
            raise ValueError("Embedding API key and model must be non-empty")
        return cls(endpoint=endpoint, api_key=api_key, model=model)

    @property
    def embeddings_url(self) -> str:
        parsed = urlparse(self.endpoint.strip())
        path = parsed.path.rstrip("/")
        if path.endswith("/embeddings"):
            effective_path = path
        elif path.endswith("/v1"):
            effective_path = f"{path}/embeddings"
        elif not path:
            effective_path = "/v1/embeddings"
        else:
            effective_path = f"{path}/embeddings"
        return urlunparse(parsed._replace(path=effective_path))

    def public_fingerprint(self) -> Mapping[str, object]:
        return {
            "provider": "openai-compatible",
            "model": self.model,
            "endpoint_sha256": sha256(self.embeddings_url.encode("utf-8")).hexdigest(),
        }


def _first_value(values: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = values.get(key, "").strip()
        if value:
            return value
    return ""


class OpenAICompatibleEmbeddingProvider:
    """Minimal provider that keeps the evaluation dependency surface small."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.request_count = 0

    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        normalized = [str(text) for text in texts]
        if not normalized:
            return []
        return await asyncio.to_thread(
            self._request,
            normalized,
            dimensions,
        )

    def _request(
        self,
        texts: list[str],
        dimensions: int | None,
    ) -> list[list[float]]:
        payload: dict[str, object] = {
            "model": self.config.model,
            "input": texts,
        }
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        request = Request(
            self.config.embeddings_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "A_memorix-evaluation/1.0",
            },
            method="POST",
        )
        self.request_count += 1
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Embedding request failed: {type(exc).__name__}"
            ) from exc
        try:
            decoded = json.loads(body)
            rows = sorted(decoded["data"], key=lambda item: int(item["index"]))
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Embedding response does not match the OpenAI schema"
            ) from exc
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding response count mismatch: {len(vectors)} != {len(texts)}"
            )
        return vectors

    def fingerprint(self) -> Mapping[str, object]:
        return self.config.public_fingerprint()


class SQLiteEmbeddingCache:
    """Persistent float32 cache keyed by provider fingerprint and text hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=30.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                namespace TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (namespace, text_hash)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_dimensions (
                provider_hash TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                observed_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def text_hash(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def get_many(
        self,
        namespace: str,
        texts: Sequence[str],
        dimension: int,
    ) -> dict[str, list[float]]:
        import numpy as np

        hashes = [self.text_hash(text) for text in texts]
        found: dict[str, list[float]] = {}
        for offset in range(0, len(hashes), 400):
            chunk = hashes[offset : offset + 400]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                "SELECT text_hash, dimension, vector FROM embeddings "
                f"WHERE namespace = ? AND text_hash IN ({placeholders})",
                (namespace, *chunk),
            ).fetchall()
            for text_hash, stored_dimension, blob in rows:
                if int(stored_dimension) != int(dimension):
                    continue
                vector = np.frombuffer(blob, dtype="<f4")
                if vector.size == dimension:
                    found[str(text_hash)] = vector.astype(float).tolist()
        return found

    def put_many(
        self,
        namespace: str,
        items: Sequence[tuple[str, Sequence[float]]],
        dimension: int,
    ) -> None:
        import numpy as np

        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                namespace,
                self.text_hash(text),
                int(dimension),
                np.asarray(vector, dtype="<f4").tobytes(),
                now,
            )
            for text, vector in items
        ]
        self._connection.executemany(
            """
            INSERT INTO embeddings(namespace, text_hash, dimension, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(namespace, text_hash) DO UPDATE SET
                dimension = excluded.dimension,
                vector = excluded.vector,
                created_at = excluded.created_at
            """,
            rows,
        )
        self._connection.commit()

    def count(self, namespace: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE namespace = ?",
            (namespace,),
        ).fetchone()
        return int(row[0]) if row else 0

    def get_provider_dimension(self, provider_hash: str) -> int | None:
        row = self._connection.execute(
            "SELECT dimension FROM provider_dimensions WHERE provider_hash = ?",
            (provider_hash,),
        ).fetchone()
        if row:
            return int(row[0])
        rows = self._connection.execute(
            "SELECT DISTINCT dimension FROM embeddings WHERE namespace LIKE ? LIMIT 2",
            (f"{provider_hash}:%",),
        ).fetchall()
        if len(rows) != 1:
            return None
        dimension = int(rows[0][0])
        self.put_provider_dimension(provider_hash, dimension)
        return dimension

    def put_provider_dimension(self, provider_hash: str, dimension: int) -> None:
        self._connection.execute(
            """
            INSERT INTO provider_dimensions(provider_hash, dimension, observed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(provider_hash) DO UPDATE SET
                dimension = excluded.dimension,
                observed_at = excluded.observed_at
            """,
            (
                provider_hash,
                int(dimension),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class CachedEmbeddingProvider:
    """EmbeddingProvider wrapper with a stable SQLite cache."""

    def __init__(
        self,
        provider: OpenAICompatibleEmbeddingProvider,
        cache_path: str | Path,
    ) -> None:
        self.provider = provider
        self.cache = SQLiteEmbeddingCache(cache_path)
        self.dimension = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._namespace = ""
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> int:
        if self.dimension:
            return self.dimension
        async with self._initialize_lock:
            await self._initialize_dimension()
        return self.dimension

    async def _initialize_dimension(self) -> None:
        if self.dimension:
            return
        base = json.dumps(
            self.provider.fingerprint(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        provider_hash = sha256(base.encode("utf-8")).hexdigest()
        cached_dimension = self.cache.get_provider_dimension(provider_hash)
        if cached_dimension is not None:
            self.dimension = cached_dimension
            self._namespace = f"{provider_hash}:{self.dimension}"
            return
        probe = await self.provider.embed(["A_memorix embedding dimension probe"])
        if len(probe) != 1 or not probe[0]:
            raise RuntimeError("Embedding dimension probe returned no vector")
        self.dimension = len(probe[0])
        self.cache.put_provider_dimension(provider_hash, self.dimension)
        self._namespace = f"{provider_hash}:{self.dimension}"

    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        dimension = await self.initialize()
        if dimensions is not None and int(dimensions) != dimension:
            raise ValueError(
                f"Embedding dimension is {dimension}, requested {dimensions}"
            )
        normalized = [str(text) for text in texts]
        if not normalized:
            return []
        unique = list(dict.fromkeys(normalized))
        cached = self.cache.get_many(self._namespace, unique, dimension)
        self.cache_hits += sum(
            int(self.cache.text_hash(text) in cached) for text in normalized
        )
        missing = [text for text in unique if self.cache.text_hash(text) not in cached]
        self.cache_misses += len(missing)
        if missing:
            vectors = list(await self.provider.embed(missing))
            if any(len(vector) != dimension for vector in vectors):
                raise RuntimeError("Embedding dimension changed during evaluation")
            self.cache.put_many(
                self._namespace,
                list(zip(missing, vectors, strict=True)),
                dimension,
            )
            for text, vector in zip(missing, vectors, strict=True):
                cached[self.cache.text_hash(text)] = list(vector)
        return [cached[self.cache.text_hash(text)] for text in normalized]

    async def prewarm(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
        max_concurrent: int = 3,
    ) -> int:
        unique = list(dict.fromkeys(str(text) for text in texts))
        size = max(1, int(batch_size))
        semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        async def fill(batch: list[str]) -> None:
            async with semaphore:
                for attempt in range(5):
                    try:
                        await self.embed(batch)
                        return
                    except RuntimeError:
                        if attempt >= 4:
                            raise
                        await asyncio.sleep(min(8.0, 2.0**attempt))

        await asyncio.gather(
            *(
                fill(unique[offset : offset + size])
                for offset in range(0, len(unique), size)
            )
        )
        return len(unique)

    def fingerprint(self) -> Mapping[str, object]:
        return {
            **self.provider.fingerprint(),
            "dimension": self.dimension,
            "cache_namespace": self._namespace,
        }

    def stats(self) -> Mapping[str, object]:
        return {
            "dimension": self.dimension,
            "cache_path": str(self.cache.path),
            "cache_entries": self.cache.count(self._namespace)
            if self._namespace
            else 0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "request_count": self.provider.request_count,
            "fingerprint": dict(self.fingerprint()),
        }

    def close(self) -> None:
        self.cache.close()


def evaluate_ranking(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    """Compute retrieval metrics from a ranked list and binary relevance labels."""

    ranked = list(dict.fromkeys(str(item) for item in ranked_ids if str(item)))
    gold = list(dict.fromkeys(str(item) for item in gold_ids if str(item)))
    gold_set = set(gold)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        k = int(cutoff)
        retrieved = ranked[:k]
        hit_count = len(set(retrieved) & gold_set)
        metrics[f"recall_any@{k}"] = float(hit_count > 0)
        metrics[f"recall_all@{k}"] = float(
            bool(gold_set) and gold_set.issubset(retrieved)
        )
        metrics[f"recall_fraction@{k}"] = (
            float(hit_count / len(gold_set)) if gold_set else 0.0
        )
        metrics[f"precision@{k}"] = float(hit_count / k)
        relevances = [1 if item in gold_set else 0 for item in retrieved]
        dcg = sum(
            relevance / log2(rank + 1)
            for rank, relevance in enumerate(relevances, start=1)
        )
        ideal = sum(
            1.0 / log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1)
        )
        metrics[f"ndcg@{k}"] = float(dcg / ideal) if ideal else 0.0
    first_rank = next(
        (rank for rank, item in enumerate(ranked, start=1) if item in gold_set),
        0,
    )
    metrics["mrr"] = 1.0 / first_rank if first_rank else 0.0
    metrics["first_relevant_rank"] = float(first_rank)
    metrics["returned_count"] = float(len(ranked))
    metrics["gold_count"] = float(len(gold_set))
    return metrics


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate completed case metrics and latency percentiles."""

    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    collected: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    collected[str(key)].append(float(value))
        by_type[str(row.get("case_type", "unknown"))].append(row)
    timings = {
        key: [
            float(row.get("timing_ms", {})[key])
            for row in completed
            if isinstance(row.get("timing_ms"), Mapping)
            and key in row.get("timing_ms", {})
        ]
        for key in ("embedding_prewarm", "initialize", "ingest", "search", "total")
    }
    return {
        "case_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "failed_case_ids": [str(row.get("case_id", "")) for row in failed],
        "metrics": {
            key: round(mean(values), 6)
            for key, values in sorted(collected.items())
            if values
        },
        "case_type_counts": {
            key: len(values) for key, values in sorted(by_type.items())
        },
        "timing_ms": {
            f"{key}_{stat}": round(value, 3)
            for key, values in timings.items()
            if values
            for stat, value in (
                ("mean", mean(values)),
                ("median", median(values)),
                ("p95", percentile(values, 0.95)),
            )
        },
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_result_log(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> list[dict[str, Any]]:
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "run-state.json"
    results_path = directory / "results.jsonl"
    normalized_manifest = json.loads(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True)
    )
    if not resume:
        write_json(state_path, normalized_manifest)
        write_jsonl(results_path, [])
        return []
    if not state_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            "resume requires run-state.json and results.jsonl in the output directory"
        )
    existing_manifest = json.loads(state_path.read_text(encoding="utf-8"))
    if existing_manifest != normalized_manifest:
        raise ValueError("resume state does not match the requested evaluation run")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_ids = set(str(item) for item in manifest.get("selected_case_ids", []))
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"result row {line_number} must be a JSON object")
            case_id = str(row.get("case_id", ""))
            if not case_id or case_id not in allowed_ids:
                raise ValueError(f"result row {line_number} has an unexpected case ID")
            if row.get("status") != "completed":
                continue
            if case_id in seen:
                raise ValueError(f"result row {line_number} duplicates case {case_id}")
            seen.add(case_id)
            rows.append(row)
    write_jsonl(results_path, rows)
    return rows
