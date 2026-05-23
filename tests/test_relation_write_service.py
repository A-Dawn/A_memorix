from __future__ import annotations

import pytest

from core.utils.relation_write_service import RelationWriteService

from fakes import FakeEmbedding


@pytest.mark.asyncio
async def test_relation_write_service_tracks_vector_state(stores):
    metadata, graph, vector = stores
    service = RelationWriteService(
        metadata_store=metadata,
        graph_store=graph,
        vector_store=vector,
        embedding_manager=FakeEmbedding(4),
    )

    result = await service.upsert_relation_with_vector(
        "Alice",
        "knows",
        "Bob",
        confidence=0.8,
        source_paragraph="",
    )

    assert result.vector_state == "ready"
    assert result.hash_value in vector
    assert graph.has_node("Alice")
    assert graph.has_node("Bob")

    stored = metadata.get_relation(result.hash_value)
    assert stored is not None
    assert stored["vector_state"] == "ready"
    assert metadata.count_relations_by_vector_state()["ready"] == 1
