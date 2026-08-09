"""Typed in-process memory application contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .context import RequestContext


class SearchMode(StrEnum):
    SEARCH = "search"
    TIME = "time"
    HYBRID = "hybrid"
    EPISODE = "episode"
    AGGREGATE = "aggregate"


class RelationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_value: str = Field(alias="object", serialization_alias="object", min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Mapping[str, object] = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: RequestContext
    external_id: str = Field(default="", max_length=512)
    source_type: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    person_ids: tuple[str, ...] = ()
    participants: tuple[str, ...] = ()
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = Field(default_factory=dict)
    entities: tuple[str, ...] = ()
    relations: tuple[RelationInput, ...] = ()
    respect_filter: bool = True


class IngestTextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stored_ids: tuple[str, ...] = ()
    skipped_ids: tuple[str, ...] = ()
    fact_claim_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    detail: str = ""


class SearchMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: RequestContext
    query: str = ""
    limit: int = Field(default=5, ge=1, le=100)
    mode: SearchMode = SearchMode.SEARCH
    shared_conversation_ids: tuple[str, ...] = ()
    person_id: str = ""
    time_start: datetime | None = None
    time_end: datetime | None = None
    respect_filter: bool = True


class MemoryHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str = ""
    kind: str = ""
    title: str = ""
    content: str = ""
    score: float = 0.0
    source: str = ""
    metadata: Mapping[str, object] = Field(default_factory=dict)


class SearchMemoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = ""
    hits: tuple[MemoryHit, ...] = ()
    filtered: bool = False
    degraded: bool = False
    retrieval_ready: bool = False
    retrieval_mode: str = ""
    available_channels: tuple[str, ...] = ()
    unavailable_channels: tuple[str, ...] = ()
