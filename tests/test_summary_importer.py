from __future__ import annotations

import numpy as np
import pytest

from core.utils.summary_importer import SummaryImporter


class FakeEmbeddingManager:
    async def encode(self, text: str):
        del text
        return np.ones(4, dtype=np.float32)


def test_transcript_messages_are_stored_in_chronological_order(stores):
    metadata, _graph, _vector = stores

    session = metadata.upsert_transcript_session(
        session_id="session-1",
        source="summary:test",
        metadata={"kind": "test"},
    )
    assert session["session_id"] == "session-1"

    inserted = metadata.append_transcript_messages(
        session_id="session-1",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": ""},
        ],
    )

    assert inserted == 2
    assert [m["content"] for m in metadata.get_transcript_messages("session-1", limit=10)] == ["first", "second"]
    assert [m["content"] for m in metadata.get_transcript_messages("session-1", limit=1)] == ["second"]


@pytest.mark.asyncio
async def test_summary_importer_falls_back_without_llm(stores):
    metadata, graph, vector = stores
    importer = SummaryImporter(
        vector_store=vector,
        graph_store=graph,
        metadata_store=metadata,
        embedding_manager=FakeEmbeddingManager(),
        plugin_config={"summarization": {"default_knowledge_type": "narrative"}},
        llm_client=None,
    )

    ok, message = await importer.import_from_transcript(
        session_id="session-2",
        source="summary:test",
        messages=[
            {"role": "user", "content": "Alice 讨论 A_Memorix 测试。"},
            {"role": "assistant", "content": "Bob 负责回归验证。"},
        ],
        context_length=10,
    )

    assert ok is True
    assert "session=session-2" in message
    assert metadata.get_transcript_messages("session-2", limit=10)
    paragraphs = metadata.get_paragraphs_by_source("chat_summary:session-2")
    assert len(paragraphs) == 1
    assert paragraphs[0]["knowledge_type"] == "narrative"
