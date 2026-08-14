"""A_memorix application-API backend shared by public evaluations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import gc
import hashlib
import shutil
import time
import asyncio

from a_memorix.contracts import (
    BatchIngestTextRequest,
    CreateNamespaceRequest,
    IngestTextInput,
    JobStatus,
    NamespaceConfig,
    NamespaceFeatureConfig,
    ProviderReference,
    RelationExtractionConfig,
    RelationExtractionMode,
    RequestContext,
    SearchMemoryRequest,
)
from a_memorix.engine import AMemorixEngine
from a_memorix.core.utils.hash import compute_paragraph_hash, normalize_text
from a_memorix.ports import LLMProvider, NamespaceHostPorts

from .common import CachedEmbeddingProvider, evaluate_ranking


DEFAULT_INGEST_BATCH_SIZE = 100
RELATION_EXTRACTION_INGEST_BATCH_SIZE = 16
RELATION_EXTRACTION_PENDING_LOW_WATERMARK = 32
RELATION_EXTRACTION_PENDING_HIGH_WATERMARK = 48


def _ingest_batch_size(relation_extraction: bool) -> int:
    return (
        RELATION_EXTRACTION_INGEST_BATCH_SIZE
        if relation_extraction
        else DEFAULT_INGEST_BATCH_SIZE
    )


async def _wait_for_extraction_jobs(
    engine: AMemorixEngine,
    namespace_id: str,
    jobs: Sequence[tuple[str, str]],
    metric_id_by_memory_id: dict[str, str],
) -> Counter[str]:
    pending = {job_id: metric_id for job_id, metric_id in jobs if job_id}
    report: Counter[str] = Counter(jobs=len(pending))
    report.update(
        await _drain_extraction_jobs(
            engine,
            namespace_id,
            pending,
            metric_id_by_memory_id,
        )
    )
    return report


async def _drain_extraction_jobs(
    engine: AMemorixEngine,
    namespace_id: str,
    pending: dict[str, str],
    metric_id_by_memory_id: dict[str, str],
    *,
    target_pending: int = 0,
) -> Counter[str]:
    report: Counter[str] = Counter()
    target = max(0, int(target_pending))
    while len(pending) > target:
        completed: list[str] = []
        for job_id, metric_id in pending.items():
            job = await engine.get_job(namespace_id, job_id)
            if job.status in {JobStatus.PENDING, JobStatus.RUNNING}:
                continue
            if job.status is not JobStatus.SUCCEEDED:
                message = (
                    job.error.message if job.error is not None else job.status.value
                )
                raise RuntimeError(f"relation extraction job failed: {message}")
            report["entities"] += int(job.result.get("entity_count", 0) or 0)
            report["relations"] += int(job.result.get("relation_count", 0) or 0)
            report["cache_hits"] += int(bool(job.result.get("cached", False)))
            relation_ids = job.result.get("relation_ids", ())
            if isinstance(relation_ids, Sequence) and not isinstance(
                relation_ids, (str, bytes)
            ):
                for relation_id in relation_ids:
                    token = str(relation_id or "").strip()
                    if token:
                        metric_id_by_memory_id[token] = metric_id
            completed.append(job_id)
        for job_id in completed:
            pending.pop(job_id, None)
        if len(pending) > target:
            await asyncio.sleep(1.0)
    return report


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
    relation_extraction: bool = False
    relation_extraction_profile: str = "general-v1"
    llm_provider: LLMProvider | None = None


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
                embedding_provider=self.provider,
                llm_provider=self.options.llm_provider,
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
                            relation_vectors=self.options.relation_extraction,
                            allow_metadata_only_write=False,
                        ),
                        llm=self._llm_reference(),
                        relation_extraction=RelationExtractionConfig(
                            enabled=self.options.relation_extraction,
                            default_enabled=False,
                            profile=self.options.relation_extraction_profile,
                        ),
                    ),
                )
            )
            timing["initialize"] = (perf_counter() - init_started) * 1000.0

            ingest_started = perf_counter()
            extraction_counts: Counter[str] = Counter()
            ingest_batch_size = _ingest_batch_size(self.options.relation_extraction)
            for batch_start in range(0, len(documents), ingest_batch_size):
                batch_documents = documents[
                    batch_start : batch_start + ingest_batch_size
                ]
                ingest_response = await engine.batch_ingest_text(
                    BatchIngestTextRequest(
                        context=RequestContext(
                            namespace_id=namespace_id,
                            agent_id="a-memorix-evaluation",
                            idempotency_key=(
                                f"ingest-batch-{batch_start // ingest_batch_size}"
                            ),
                        ),
                        items=tuple(
                            self._ingest_input(document) for document in batch_documents
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
                        id_by_memory_hash[str(memory_id)] = document.effective_metric_id
                extraction_jobs = [
                    (
                        item_result.response.relation_extraction_job_id,
                        document.effective_metric_id,
                    )
                    for document, item_result in zip(
                        batch_documents,
                        ingest_response.results,
                        strict=True,
                    )
                    if item_result.response is not None
                    and item_result.response.relation_extraction_job_id
                ]
                report = await _wait_for_extraction_jobs(
                    engine,
                    namespace_id,
                    extraction_jobs,
                    id_by_memory_hash,
                )
                extraction_counts.update(report)
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
                "relation_extraction": dict(extraction_counts),
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
            # Evaluation reuses one live runtime and flushes it during shutdown.
            # Avoid rewriting growing vector and graph snapshots after every job.
            "runtime": {"defer_relation_extraction_persist": True},
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

    def _ingest_input(
        self,
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
            relation_extraction=(
                RelationExtractionMode.ENABLED
                if self.options.relation_extraction
                else RelationExtractionMode.INHERIT
            ),
            respect_filter=False,
        )

    def _llm_reference(self) -> ProviderReference | None:
        provider = self.options.llm_provider
        if provider is None:
            return None
        fingerprint = getattr(provider, "fingerprint", None)
        raw = fingerprint() if callable(fingerprint) else {}
        model = str(raw.get("model", "") if isinstance(raw, Mapping) else "")
        return ProviderReference(provider_id="openai-compatible", model_id=model)

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


class AMemorixSharedNamespaceBackend(AMemorixEvaluationBackend):
    """Build one corpus and run every retrieval case in the same namespace."""

    def __init__(
        self,
        provider: CachedEmbeddingProvider,
        options: BackendOptions,
    ) -> None:
        super().__init__(provider, options)
        self._engine: AMemorixEngine | None = None
        self._namespace_id = "benchmark-full"
        self._shared_dir: Path | None = None
        self._metric_ids_by_memory_id: dict[str, set[str]] = defaultdict(set)

    async def prepare(self, cases: Sequence[RetrievalCase]) -> dict[str, Any]:
        if self._engine is not None:
            raise RuntimeError("shared evaluation backend is already prepared")
        documents, corpus = self._build_corpus(cases)
        if not documents:
            raise ValueError("shared evaluation corpus has no documents")

        dimension = await self.provider.initialize()
        shared_dir = self._reset_case_dir("longmemeval-full-namespace")
        self._shared_dir = shared_dir
        engine = AMemorixEngine(
            data_dir=shared_dir,
            idle_timeout_seconds=0,
            max_active_namespaces=1,
            config_factory=lambda _namespace: self._runtime_config(dimension),
            host_port_factory=lambda _namespace: NamespaceHostPorts(
                embedding_provider=self.provider,
                llm_provider=self.options.llm_provider,
            ),
        )
        self._engine = engine
        timing: dict[str, float] = {}
        try:
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
                    namespace_id=self._namespace_id,
                    config=NamespaceConfig(
                        embedding=ProviderReference(
                            provider_id="openai-compatible",
                            model_id=str(self.provider.fingerprint().get("model", "")),
                        ),
                        features=NamespaceFeatureConfig(
                            episodes=False,
                            person_profiles=False,
                            sparse_retrieval=True,
                            relation_vectors=self.options.relation_extraction,
                            allow_metadata_only_write=False,
                        ),
                        llm=self._llm_reference(),
                        relation_extraction=RelationExtractionConfig(
                            enabled=self.options.relation_extraction,
                            default_enabled=False,
                            profile=self.options.relation_extraction_profile,
                        ),
                    ),
                )
            )
            timing["initialize"] = (perf_counter() - init_started) * 1000.0

            ingest_started = perf_counter()
            extraction_counts: Counter[str] = Counter()
            pending_extraction_jobs: dict[str, str] = {}
            relation_mapping: dict[str, str] = {}
            ingest_batch_size = _ingest_batch_size(self.options.relation_extraction)
            for batch_start in range(0, len(documents), ingest_batch_size):
                batch_documents = documents[
                    batch_start : batch_start + ingest_batch_size
                ]
                response = await engine.batch_ingest_text(
                    BatchIngestTextRequest(
                        context=RequestContext(
                            namespace_id=self._namespace_id,
                            agent_id="a-memorix-evaluation",
                            idempotency_key=(
                                f"full-ingest-batch-{batch_start // ingest_batch_size}"
                            ),
                        ),
                        items=tuple(
                            self._ingest_input(document) for document in batch_documents
                        ),
                    )
                )
                for document, item_result in zip(
                    batch_documents,
                    response.results,
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
                    if not memory_ids:
                        raise RuntimeError(
                            "batch ingestion returned no memory ID for "
                            f"{document.document_id}"
                        )
                    for memory_id in memory_ids:
                        self._metric_ids_by_memory_id[str(memory_id)].add(
                            document.effective_metric_id
                        )
                extraction_jobs = [
                    (
                        item_result.response.relation_extraction_job_id,
                        document.effective_metric_id,
                    )
                    for document, item_result in zip(
                        batch_documents,
                        response.results,
                        strict=True,
                    )
                    if item_result.response is not None
                    and item_result.response.relation_extraction_job_id
                ]
                pending_extraction_jobs.update(
                    (job_id, metric_id)
                    for job_id, metric_id in extraction_jobs
                    if job_id
                )
                extraction_counts["jobs"] += len(extraction_jobs)
                if (
                    len(pending_extraction_jobs)
                    >= RELATION_EXTRACTION_PENDING_HIGH_WATERMARK
                ):
                    extraction_counts.update(
                        await _drain_extraction_jobs(
                            engine,
                            self._namespace_id,
                            pending_extraction_jobs,
                            relation_mapping,
                            target_pending=(RELATION_EXTRACTION_PENDING_LOW_WATERMARK),
                        )
                    )
            extraction_counts.update(
                await _drain_extraction_jobs(
                    engine,
                    self._namespace_id,
                    pending_extraction_jobs,
                    relation_mapping,
                )
            )
            for relation_id, metric_id in relation_mapping.items():
                self._metric_ids_by_memory_id[relation_id].add(metric_id)
            timing["ingest"] = (perf_counter() - ingest_started) * 1000.0
        except Exception:
            await self.close()
            raise

        return {
            **corpus,
            "relation_extraction": dict(extraction_counts),
            "timing_ms": {key: round(value, 3) for key, value in timing.items()},
        }

    async def run_case(self, case: RetrievalCase) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("shared evaluation backend has not been prepared")
        if not case.query.strip():
            raise ValueError(f"evaluation case has no query: {case.case_id}")
        started = perf_counter()
        timing: dict[str, float] = {}
        try:
            search_started = perf_counter()
            response = await self._engine.search_memory(
                SearchMemoryRequest(
                    context=RequestContext(
                        namespace_id=self._namespace_id,
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
            seen_metric_ids: set[str] = set()
            gold_ids = set(case.gold_ids)
            for hit in response.hits:
                candidates = self._metric_ids_by_memory_id.get(hit.memory_id, set())
                unseen = sorted(candidates - seen_metric_ids)
                if not unseen:
                    continue
                gold_candidates = [item for item in unseen if item in gold_ids]
                metric_id = gold_candidates[0] if gold_candidates else unseen[0]
                seen_metric_ids.add(metric_id)
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
                    "retrieval_ready": response.retrieval_ready,
                    "retrieval_mode": response.retrieval_mode,
                    "degraded": response.degraded,
                    "available_channels": list(response.available_channels),
                    "unavailable_channels": list(response.unavailable_channels),
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

    async def close(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            await engine.shutdown()
        if self._shared_dir is not None and not self.options.keep_case_data:
            self._remove_case_dir(self._shared_dir)
        self._shared_dir = None
        self._metric_ids_by_memory_id.clear()

    @staticmethod
    def _build_corpus(
        cases: Sequence[RetrievalCase],
    ) -> tuple[list[RetrievalDocument], dict[str, int]]:
        documents_by_id: dict[str, RetrievalDocument] = {}
        occurrence_counts: Counter[str] = Counter()
        timestamps_by_id: dict[str, set[float]] = defaultdict(set)
        occurrence_count = 0
        for case in cases:
            for document in case.documents:
                occurrence_count += 1
                occurrence_counts[document.document_id] += 1
                if document.timestamp is not None:
                    timestamps_by_id[document.document_id].add(document.timestamp)
                existing = documents_by_id.get(document.document_id)
                if existing is None:
                    documents_by_id[document.document_id] = document
                    continue
                if normalize_text(existing.text) != normalize_text(document.text):
                    raise ValueError(
                        "shared corpus document ID has conflicting content: "
                        f"{document.document_id}"
                    )
                if existing.effective_metric_id != document.effective_metric_id:
                    raise ValueError(
                        "shared corpus document ID has conflicting metric ID: "
                        f"{document.document_id}"
                    )
                if document.timestamp is not None and (
                    existing.timestamp is None
                    or document.timestamp < existing.timestamp
                ):
                    documents_by_id[document.document_id] = replace(
                        existing,
                        timestamp=document.timestamp,
                    )

        documents: list[RetrievalDocument] = []
        for document_id, document in documents_by_id.items():
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "evaluation_occurrence_count": occurrence_counts[document_id],
                    "evaluation_date_variant_count": len(
                        timestamps_by_id.get(document_id, set())
                    ),
                }
            )
            documents.append(replace(document, metadata=metadata))
        unique_memory_count = len(
            {compute_paragraph_hash(document.text) for document in documents}
        )
        return documents, {
            "occurrence_document_count": occurrence_count,
            "logical_document_count": len(documents),
            "unique_memory_count": unique_memory_count,
            "duplicate_occurrence_count": occurrence_count - len(documents),
            "shared_content_document_count": len(documents) - unique_memory_count,
            "date_variant_document_count": sum(
                len(timestamps) > 1 for timestamps in timestamps_by_id.values()
            ),
        }
