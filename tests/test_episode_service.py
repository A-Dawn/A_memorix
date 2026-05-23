from __future__ import annotations

import pytest

from core.utils.episode_retrieval_service import EpisodeRetrievalService
from core.utils.episode_service import EpisodeService


@pytest.mark.asyncio
async def test_episode_rebuild_uses_deterministic_fallback(stores):
    metadata, _graph, _vector = stores
    first = metadata.add_paragraph(
        "Alice discussed the roadmap with Bob.",
        source="chat:room-1",
        time_meta={"event_time": "2026-05-21 10:00"},
    )
    second = metadata.add_paragraph(
        "Bob agreed to prepare the release checklist.",
        source="chat:room-1",
        time_meta={"event_time": "2026-05-21 10:05"},
    )

    service = EpisodeService(metadata_store=metadata, plugin_config={"episode": {"source_time_window_hours": 24}})
    result = await service.rebuild_source("chat:room-1")

    assert result["episode_count"] == 1
    assert result["fallback_count"] == 1

    assert metadata.query_episodes(query="roadmap", source="chat:room-1", limit=5) == []
    metadata.mark_episode_source_done("chat:room-1")

    rows = metadata.query_episodes(query="roadmap", source="chat:room-1", limit=5)
    assert len(rows) == 1
    assert rows[0]["paragraph_count"] == 2

    paragraphs = metadata.get_episode_paragraphs(rows[0]["episode_id"])
    assert [item["hash"] for item in paragraphs] == [first, second]


@pytest.mark.asyncio
async def test_episode_retrieval_projects_paragraph_evidence(stores):
    metadata, _graph, _vector = stores
    paragraph_hash = metadata.add_paragraph("Release planning happened here.", source="notes")
    service = EpisodeService(metadata_store=metadata, plugin_config={})
    await service.rebuild_source("notes")
    metadata.mark_episode_source_done("notes")

    class DummyResult:
        hash_value = paragraph_hash
        result_type = "paragraph"

    class DummyRetriever:
        async def retrieve(self, **kwargs):
            del kwargs
            return [DummyResult()]

    retrieval = EpisodeRetrievalService(metadata_store=metadata, retriever=DummyRetriever())
    results = await retrieval.query(query="release planning", top_k=3)

    assert len(results) == 1
    assert results[0]["type"] == "episode"
    assert results[0]["episode_id"]


def test_episode_done_preserves_newer_pending_rebuild(stores):
    metadata, _graph, _vector = stores
    metadata.add_paragraph("First source version.", source="notes")
    row = metadata.get_episode_source_rebuild("notes")
    assert row is not None

    assert metadata.mark_episode_source_running("notes", requested_at=row["requested_at"])
    metadata.add_paragraph("Second source version.", source="notes")

    assert metadata.mark_episode_source_done("notes", requested_at=row["requested_at"])
    latest = metadata.get_episode_source_rebuild("notes")
    assert latest is not None
    assert latest["status"] == "pending"
