from __future__ import annotations

import pytest

from core.utils.aggregate_query_service import AggregateQueryService
from core.utils.runtime_self_check import run_embedding_runtime_self_check

from fakes import FakeEmbedding


@pytest.mark.asyncio
async def test_aggregate_query_mixes_branch_results():
    service = AggregateQueryService({"retrieval": {"aggregate": {"rrf_k": 10}}})

    async def search_runner():
        return {
            "success": True,
            "results": [{"hash": "p1", "type": "paragraph", "content": "alpha"}],
            "count": 1,
        }

    async def episode_runner():
        return {
            "success": True,
            "results": [{"episode_id": "e1", "type": "episode", "title": "Episode"}],
            "count": 1,
        }

    payload = await service.execute(
        query="alpha",
        top_k=3,
        mix=True,
        mix_top_k=3,
        time_from=None,
        time_to=None,
        search_runner=search_runner,
        time_runner=None,
        episode_runner=episode_runner,
    )

    assert payload["success"] is True
    assert payload["summary"]["search"]["status"] == "success"
    assert payload["summary"]["episode"]["status"] == "success"
    assert {item["type"] for item in payload["mixed_results"]} == {"paragraph", "episode"}


@pytest.mark.asyncio
async def test_runtime_self_check_detects_embedding_dimension_mismatch():
    class Vector:
        dimension = 8

    report = await run_embedding_runtime_self_check(
        config={"embedding": {"dimension": 8}},
        vector_store=Vector(),
        embedding_manager=FakeEmbedding(4),
        sample_text="dimension mismatch",
    )

    assert report["ok"] is False
    assert report["code"] == "embedding_dimension_mismatch"
