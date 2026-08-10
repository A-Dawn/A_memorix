"""A_memorix application-API backend shared by public evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import gc
import hashlib
import shutil
import time

from a_memorix.contracts import (
    BatchIngestTextRequest,
    CreateNamespaceRequest,
    IngestTextInput,
    NamespaceConfig,
    NamespaceFeatureConfig,
    ProviderReference,
    RequestContext,
    SearchMemoryRequest,
)
from a_memorix.engine import AMemorixEngine
from a_memorix.core.utils.hash import normalize_text
from a_memorix.ports import NamespaceHostPorts

from .common import CachedEmbeddingProvider, evaluate_ranking


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    text: str
    metric_id: str = ""
    timestamp: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def effective_metric_id(self) -> str:
        return self.metric_id or self.document_id


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    case_type: str
    query: str
    documents: tuple[RetrievalDocument, ...]
    gold_ids: tuple[str, ...]


@dataclass(frozen=True)
class BackendOptions:
    work_root: Path
    top_k: int = 50
    embedding_batch_size: int = 16
    embedding_concurrency: int = 3
    keep_case_data: bool = False


class AMemorixEvaluationBackend:
    """Run isolated retrieval cases through the public in-process API."""

    def __init__(
        self,
        provider: CachedEmbeddingProvider,
        options: BackendOptions,
    ) -> None:
        self.provider = provider
        self.options = options
        self.work_root = Path(options.work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)

    async def run_case(self, case: RetrievalCase) -> dict[str, Any]:
        if not case.documents:
            raise ValueError(f"evaluation case has no documents: {case.case_id}")
        if not case.query.strip():
            raise ValueError(f"evaluation case has no query: {case.case_id}")
        started = perf_counter()
        dimension = await self.provider.initialize()
        case_dir = self._reset_case_dir(case.case_id)
        namespace_id = "benchmark"
        engine = AMemorixEngine(
            data_dir=case_dir,
            idle_timeout_seconds=0,
            max_active_namespaces=1,
            config_factory=lambda _namespace: self._runtime_config(dimension),
            host_port_factory=lambda _namespace: NamespaceHostPorts(
                embedding_provider=self.provider
            ),
        )
        timing: dict[str, float] = {}
        id_by_memory_hash: dict[str, str] = {}
        try:
            documents = list(case.documents)
            prewarm_started = perf_counter()
            await self.provider.prewarm(
                [normalize_text(document.text) for document in documents],
                batch_size=self.options.embedding_batch_size,
                max_concurrent=self.options.embedding_concurrency,
            )
            timing["embedding_prewarm"] = (perf_counter() - prewarm_started) * 1000.0

            init_started = perf_counter()
            await engine.initialize()
            await engine.create_namespace(
                CreateNamespaceRequest(
                    namespace_id=namespace_id,
                    config=NamespaceConfig(
                        embedding=ProviderReference(
                            provider_id="openai-compatible",
                            model_id=str(self.provider.fingerprint().get("model", "")),
                        ),
                        features=NamespaceFeatureConfig(
                            episodes=False,
                            person_profiles=False,
                            sparse_retrieval=True,
                            relation_vectors=False,
                            allow_metadata_only_write=False,
                        ),
                    ),
                )
            )
            timing["initialize"] = (perf_counter() - init_started) * 1000.0

            ingest_started = perf_counter()
            for batch_start in range(0, len(documents), 100):
                batch_documents = documents[batch_start : batch_start + 100]
                ingest_response = await engine.batch_ingest_text(
                    BatchIngestTextRequest(
                        context=RequestContext(
                            namespace_id=namespace_id,
                            agent_id="a-memorix-evaluation",
                            idempotency_key=f"ingest-batch-{batch_start // 100}",
                        ),
                        items=tuple(
                            self._ingest_input(document)
                            for document in batch_documents
                        ),
                    )
                )
                for document, item_result in zip(
                    batch_documents,
                    ingest_response.results,
                    strict=True,
                ):
                    if item_result.error is not None:
                        raise RuntimeError(
                            "batch ingestion failed for "
                            f"{document.document_id}: {item_result.error.code}"
                        )
                    assert item_result.response is not None
                    memory_ids = (
                        *item_result.response.stored_ids,
                        *item_result.response.skipped_ids,
                    )
                    for memory_id in memory_ids:
                        id_by_memory_hash[str(memory_id)] = (
                            document.effective_metric_id
                        )
            timing["ingest"] = (perf_counter() - ingest_started) * 1000.0

            search_started = perf_counter()
            search_response = await engine.search_memory(
                SearchMemoryRequest(
                    context=RequestContext(
                        namespace_id=namespace_id,
                        agent_id="a-memorix-evaluation",
                    ),
                    query=case.query,
                    limit=max(1, min(100, int(self.options.top_k))),
                    respect_filter=False,
                )
            )
            timing["search"] = (perf_counter() - search_started) * 1000.0
            ranked_ids: list[str] = []
            ranked_hits: list[dict[str, object]] = []
            seen: set[str] = set()
            for hit in search_response.hits:
                metric_id = str(
                    hit.metadata.get("evaluation_metric_id", "")
                    or id_by_memory_hash.get(hit.memory_id, "")
                )
                if not metric_id or metric_id in seen:
                    continue
                seen.add(metric_id)
                ranked_ids.append(metric_id)
                ranked_hits.append(
                    {
                        "metric_id": metric_id,
                        "memory_id": hit.memory_id,
                        "kind": hit.kind,
                        "score": hit.score,
                        "source": hit.source,
                    }
                )
            timing["total"] = (perf_counter() - started) * 1000.0
            return {
                "status": "completed",
                "case_id": case.case_id,
                "case_type": case.case_type,
                "document_count": len(case.documents),
                "gold_ids": list(case.gold_ids),
                "ranked_ids": ranked_ids,
                "ranked_hits": ranked_hits,
                "metrics": evaluate_ranking(ranked_ids, case.gold_ids),
                "timing_ms": {key: round(value, 3) for key, value in timing.items()},
                "runtime": {
                    "retrieval_ready": search_response.retrieval_ready,
                    "retrieval_mode": search_response.retrieval_mode,
                    "degraded": search_response.degraded,
                    "available_channels": list(search_response.available_channels),
                    "unavailable_channels": list(search_response.unavailable_channels),
                },
            }
        except Exception as exc:
            timing["total"] = (perf_counter() - started) * 1000.0
            return {
                "status": "failed",
                "case_id": case.case_id,
                "case_type": case.case_type,
                "document_count": len(case.documents),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timing_ms": {key: round(value, 3) for key, value in timing.items()},
            }
        finally:
            await engine.shutdown()
            if not self.options.keep_case_data:
                self._remove_case_dir(case_dir)

    def _runtime_config(self, dimension: int) -> dict[str, object]:
        return {
            "embedding": {
                "dimension": int(dimension),
                "dimension_request_mode": "never",
                "batch_size": max(1, int(self.options.embedding_batch_size)),
                "max_concurrent": max(1, int(self.options.embedding_concurrency)),
                "enable_cache": False,
                "fallback": {
                    "enabled": False,
                    "allow_metadata_only_write": False,
                },
                "paragraph_vector_backfill": {"enabled": False},
            },
            "retrieval": {
                "vector_pools": {"mode": "single"},
                "sparse": {"enabled": True, "mode": "hybrid"},
            },
            "episode": {"enabled": False, "generation_enabled": False},
            "person_profile": {"enabled": False},
        }

    @staticmethod
    def _ingest_input(
        document: RetrievalDocument,
    ) -> IngestTextInput:
        metadata = dict(document.metadata)
        metadata.update(
            {
                "evaluation_document_id": document.document_id,
                "evaluation_metric_id": document.effective_metric_id,
            }
        )
        observed_at = (
            datetime.fromtimestamp(document.timestamp, tz=timezone.utc)
            if document.timestamp is not None
            else None
        )
        return IngestTextInput(
            external_id=document.document_id,
            source_type="public_benchmark",
            text=document.text,
            observed_at=observed_at,
            metadata=metadata,
            respect_filter=False,
        )

    def _case_dir(self, case_id: str) -> Path:
        digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
        path = (self.work_root / digest).resolve()
        if path.parent != self.work_root:
            raise RuntimeError("evaluation case path escaped the work root")
        return path

    def _reset_case_dir(self, case_id: str) -> Path:
        path = self._case_dir(case_id)
        if path.exists():
            self._remove_case_dir(path)
        path.mkdir(parents=True)
        return path

    def _remove_case_dir(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.parent != self.work_root:
            raise RuntimeError("refusing to remove a path outside the evaluation root")
        last_error: OSError | None = None
        for attempt in range(6):
            if not resolved.exists():
                return
            gc.collect()
            try:
                shutil.rmtree(resolved)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(
            f"evaluation case directory could not be removed: {resolved}"
        ) from last_error
