from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import asyncio
import json
import math

import pytest

from a_memorix import (
    AMemorixEngine,
    CreateNamespaceRequest,
    DeleteMemoryRequest,
    IngestTextRequest,
    JobStatus,
    LLMRequest,
    LLMResult,
    NamespaceHostPorts,
    RequestContext,
    SearchMemoryRequest,
    SDKMemoryKernel,
)


class _SemanticEmbeddingProvider:
    dimension = 8

    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        assert dimensions in {None, self.dimension}
        return [self._vector(text) for text in texts]

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "provider": "deterministic-semantic-test",
            "model": "concept-space-v1",
            "dimension": self.dimension,
            "dimension_verified": True,
        }

    def _vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        values = [0.0] * self.dimension
        if "cerulean-47" in lowered or "chromatic preference" in lowered:
            values[0] = 1.0
        elif "payroll" in lowered or "compensation schedule" in lowered:
            values[1] = 1.0
        elif "control surfaces" in lowered:
            values[2] = 1.0
        else:
            values[7] = 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class _RelationLLMProvider:
    def get_available_models(self) -> Mapping[str, object]:
        class _Config:
            model_list = ("deterministic-relation-v1",)
            max_tokens = 1024
            temperature = 0.0

        return {"memory": _Config()}

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "provider": "deterministic-relation-test",
            "model": "deterministic-relation-v1",
        }

    async def generate(self, request: LLMRequest) -> LLMResult:
        if "CERULEAN-47" in request.prompt:
            payload = {
                "entities": [
                    {"name": "control surfaces", "type": "interface"},
                    {"name": "CERULEAN-47", "type": "palette"},
                ],
                "relations": [
                    {
                        "subject": "control surfaces",
                        "predicate": "uses_palette",
                        "object": "CERULEAN-47",
                        "confidence": 0.99,
                    }
                ],
            }
        else:
            payload = {
                "entities": [
                    {"name": "payroll", "type": "process"},
                    {"name": "Friday", "type": "time"},
                ],
                "relations": [
                    {
                        "subject": "payroll",
                        "predicate": "runs_on",
                        "object": "Friday",
                        "confidence": 0.98,
                    }
                ],
            }
        return LLMResult(success=True, content=json.dumps(payload))


def _engine(data_dir: Path) -> AMemorixEngine:
    embedding = _SemanticEmbeddingProvider()
    llm = _RelationLLMProvider()
    return AMemorixEngine(
        data_dir=data_dir,
        config_factory=lambda _namespace: {
            "embedding": {
                "dimension": embedding.dimension,
                "dimension_request_mode": "always",
                "batch_size": 8,
                "max_concurrent": 2,
            },
            "retrieval": {
                "sparse": {"enabled": True},
                "relation_vectorization": {"enabled": True},
                "vector_pools": {"mode": "dual"},
            },
        },
        host_port_factory=lambda _namespace: NamespaceHostPorts(
            embedding_provider=embedding,
            llm_provider=llm,
        ),
        idle_timeout_seconds=900,
    )


async def _wait_for_job(
    engine: AMemorixEngine,
    namespace_id: str,
    job_id: str,
) -> None:
    for _ in range(200):
        job = await engine.get_job(namespace_id, job_id)
        if job.status not in {JobStatus.PENDING, JobStatus.RUNNING}:
            assert job.status is JobStatus.SUCCEEDED, job
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"job did not finish: {job_id}")


@pytest.mark.asyncio
async def test_standard_stack_writes_retrieves_restarts_and_deletes_all_channels(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "service"
    namespace_id = "standard-stack"
    context = RequestContext(
        namespace_id=namespace_id,
        agent_id="deepseek-harness",
        principal_id="test",
    )
    engine = _engine(data_dir)
    await engine.initialize()
    try:
        await engine.create_namespace(
            CreateNamespaceRequest(namespace_id=namespace_id)
        )
        relevant = await engine.ingest_text(
            IngestTextRequest(
                context=context,
                external_id="dsh:palette",
                source_type="deepseek-harness",
                text="Use CERULEAN-47 for all control surfaces.",
            )
        )
        noise = await engine.ingest_text(
            IngestTextRequest(
                context=context,
                external_id="dsh:payroll",
                source_type="deepseek-harness",
                text="Payroll processing runs every Friday.",
            )
        )
        await _wait_for_job(
            engine,
            namespace_id,
            relevant.relation_extraction_job_id,
        )
        await _wait_for_job(
            engine,
            namespace_id,
            noise.relation_extraction_job_id,
        )
        relevant_id = relevant.stored_ids[0]

        async with engine.runtime(context) as runtime:
            assert isinstance(runtime, SDKMemoryKernel)
            assert runtime.metadata_store is not None
            assert runtime.sparse_index is not None
            assert runtime.graph_store is not None
            assert runtime.paragraph_vector_store is not None
            assert runtime.graph_vector_store is not None
            assert runtime.metadata_store.count_paragraphs() == 2
            assert runtime.metadata_store.count_relations() == 2
            assert relevant_id in runtime.paragraph_vector_store
            relations = runtime.metadata_store.get_relations()
            relation_ids = {
                f"relation:{item['hash']}"
                for item in relations
            }
            relevant_relation_ids = {
                f"relation:{item['hash']}"
                for item in relations
                if item["predicate"] == "uses_palette"
            }
            assert relation_ids
            assert relevant_relation_ids
            assert all(
                relation_id in runtime.graph_vector_store
                for relation_id in relation_ids
            )
            status = runtime.runtime_capability_status()
            assert status["paragraph_vector_pool_ready"] is True
            assert status["relation_vector_pool_ready"] is True

        health = await engine.namespace_health(namespace_id)
        assert health.healthy
        assert not health.degraded
        assert health.embedding.available
        assert health.embedding.dimension == 8
        assert health.llm.available
        assert health.paragraph_vector_pool_ready
        assert health.relation_vector_pool_ready

        result = await engine.search_memory(
            SearchMemoryRequest(
                context=context,
                query="Which chromatic preference was selected?",
                limit=2,
            )
        )
        assert result.retrieval_ready
        assert result.retrieval_mode == "hybrid"
        assert result.hits
        assert result.hits[0].memory_id == relevant_id
        assert "vector_read" in result.available_channels
    finally:
        await engine.shutdown()

    restarted = _engine(data_dir)
    await restarted.initialize()
    try:
        result = await restarted.search_memory(
            SearchMemoryRequest(
                context=context,
                query="Which chromatic preference was selected?",
                limit=2,
            )
        )
        assert result.hits[0].memory_id == relevant_id

        deleted = await restarted.delete_memory(
            DeleteMemoryRequest(context=context, memory_id=relevant_id)
        )
        assert deleted.deleted_count == 1

        async with restarted.runtime(context) as runtime:
            assert isinstance(runtime, SDKMemoryKernel)
            assert runtime.metadata_store is not None
            assert runtime.paragraph_vector_store is not None
            assert runtime.graph_vector_store is not None
            deleted_paragraph = runtime.metadata_store.get_paragraph(relevant_id)
            assert deleted_paragraph is not None
            assert deleted_paragraph["is_deleted"] == 1
            assert relevant_id not in runtime.paragraph_vector_store
            assert runtime.metadata_store.count_relations() == 1
            assert all(
                "CERULEAN-47" not in str(item)
                for item in runtime.metadata_store.get_relations()
            )
            assert all(
                relation_id not in runtime.graph_vector_store
                for relation_id in relevant_relation_ids
            )

        after_delete = await restarted.search_memory(
            SearchMemoryRequest(
                context=context,
                query="Which chromatic preference was selected?",
                limit=2,
            )
        )
        assert all(hit.memory_id != relevant_id for hit in after_delete.hits)
    finally:
        await restarted.shutdown()
