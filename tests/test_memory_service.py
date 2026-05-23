from __future__ import annotations

from types import SimpleNamespace

import pytest

from amemorix.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_memory_freeze_accepts_plain_query_without_semantic_lookup(stores):
    metadata, graph, _vector = stores
    relation_hash = metadata.add_relation("Alice", "协作", "Bob", confidence=0.9)
    graph.add_nodes(["Alice", "Bob"])
    graph.add_edges([("Alice", "Bob")], weights=[0.9], relation_hashes=[relation_hash])

    ctx = SimpleNamespace(
        metadata_store=metadata,
        graph_store=graph,
        config={"memory": {"max_weight": 10.0, "auto_protect_ttl_hours": 24.0}},
        retriever=None,
        threshold_filter=None,
    )
    ctx.get_config = lambda key, default=None: {
        "memory.max_weight": 10.0,
        "memory.auto_protect_ttl_hours": 24.0,
    }.get(key, default)

    result = await MemoryService(ctx).freeze("Alice")

    assert result["success"] is True
    assert result["count"] == 1
    assert metadata.get_relation_status_batch([relation_hash])[relation_hash]["is_inactive"] is True
    assert graph.get_edge_weight("Alice", "Bob") == 0.0
