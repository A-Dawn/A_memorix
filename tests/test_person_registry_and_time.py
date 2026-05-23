from __future__ import annotations

import pytest

from core.utils.person_profile_service import PersonProfileService
from core.utils.time_parser import (
    normalize_time_meta,
    parse_query_time_range,
)


def test_person_registry_upsert_list_and_resolve(stores):
    metadata, _graph, _vector = stores

    record = metadata.upsert_person_registry(
        person_name="Alice",
        nickname="A",
        user_id="alice-01",
        platform="chat",
        group_nick_name=["产品负责人"],
        memory_points=["负责路线图"],
    )

    assert len(record["person_id"]) == 32
    assert record["display_name"] == "Alice"
    assert metadata.resolve_person_registry("Alice") == record["person_id"]
    assert metadata.resolve_person_registry("产品负责人") == record["person_id"]

    listed = metadata.list_person_registry(keyword="alice", page=1, page_size=10)
    assert listed["success"] is True
    assert listed["total"] == 1
    assert listed["items"][0]["aliases"][:3] == ["Alice", "A", "alice-01"]


def test_query_time_parser_accepts_dash_and_slash_dates():
    dash_from, dash_to = parse_query_time_range("2026-05-01", "2026-05-23")
    slash_from, slash_to = parse_query_time_range("2026/05/01", "2026/05/23")

    assert dash_from == slash_from
    assert dash_to == slash_to


def test_ingest_time_parser_accepts_iso8601_z_and_offsets():
    utc_meta = normalize_time_meta({"event_time": "2026-05-23T00:00:00Z"})
    offset_meta = normalize_time_meta({"event_time": "2026-05-23T08:00:00+08:00"})
    compact_offset_meta = normalize_time_meta({"event_time": "2026-05-23T080000+0800"})

    assert utc_meta["event_time"] == offset_meta["event_time"]
    assert utc_meta["event_time"] == compact_offset_meta["event_time"]
    assert utc_meta["time_granularity"] == "minute"


@pytest.mark.asyncio
async def test_person_profile_query_accepts_raw_keyword_when_registry_empty(stores):
    metadata, _graph, _vector = stores
    service = PersonProfileService(metadata_store=metadata)

    profile = await service.query_person_profile(person_keyword="Alice", top_k=4)

    assert profile["success"] is True
    assert profile["person_id"] == "Alice"
    assert profile["person_name"] == "Alice"


@pytest.mark.asyncio
async def test_person_profile_vector_evidence_timeout_does_not_block_profile(stores):
    metadata, _graph, _vector = stores

    class HangingRetriever:
        async def retrieve(self, *args, **kwargs):
            import asyncio

            await asyncio.sleep(10)
            return []

    service = PersonProfileService(
        metadata_store=metadata,
        retriever=HangingRetriever(),
        plugin_config={"person_profile": {"vector_evidence_timeout_seconds": 0.01}},
    )

    profile = await service.query_person_profile(person_keyword="Alice", top_k=4)

    assert profile["success"] is True
    assert profile["vector_evidence"] == []
