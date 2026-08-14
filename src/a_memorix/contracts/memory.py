"""Typed in-process memory application contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .context import RequestContext
from .errors import ErrorEnvelope


class SearchMode(StrEnum):
    SEARCH = "search"
    TIME = "time"
    HYBRID = "hybrid"
    EPISODE = "episode"
    AGGREGATE = "aggregate"


class RelationExtractionMode(StrEnum):
    INHERIT = "inherit"
    ENABLED = "enabled"
    DISABLED = "disabled"


class RelationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_value: str = Field(
        alias="object", serialization_alias="object", min_length=1
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Mapping[str, object] = Field(default_factory=dict)


class IngestTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
    relation_extraction: RelationExtractionMode = RelationExtractionMode.INHERIT
    respect_filter: bool = True


class IngestTextRequest(IngestTextInput):
    context: RequestContext


class IngestTextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stored_ids: tuple[str, ...] = ()
    skipped_ids: tuple[str, ...] = ()
    fact_claim_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    detail: str = ""
    relation_extraction_job_id: str = ""


class BatchIngestTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: RequestContext
    items: tuple[IngestTextInput, ...] = Field(min_length=1, max_length=100)


class BatchIngestItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    response: IngestTextResponse | None = None
    error: ErrorEnvelope | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "BatchIngestItemResult":
        if (self.response is None) == (self.error is None):
            raise ValueError("exactly one of response or error must be set")
        return self


class BatchIngestTextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[BatchIngestItemResult, ...] = ()
    succeeded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class GetMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: RequestContext
    memory_id: str = Field(default="", max_length=512)
    external_id: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def validate_selector(self) -> "GetMemoryRequest":
        if bool(self.memory_id.strip()) == bool(self.external_id.strip()):
            raise ValueError("exactly one of memory_id or external_id must be set")
        return self


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    external_id: str = ""
    source_type: str = ""
    source: str = ""
    content: str = ""
    metadata: Mapping[str, object] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class GetMemoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory: MemoryRecord


class DeleteMemoryRequest(GetMemoryRequest):
    reason: str = Field(default="user_delete", min_length=1, max_length=256)


class DeleteMemoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = ""
    deleted_count: int = Field(default=0, ge=0)
    deleted_memory_ids: tuple[str, ...] = ()


class DeleteBySourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: RequestContext
    source: str = Field(min_length=1, max_length=512)
    reason: str = Field(default="source_delete", min_length=1, max_length=256)


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
