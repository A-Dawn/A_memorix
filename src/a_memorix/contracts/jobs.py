"""Typed contracts for persistent background operations."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .errors import ErrorEnvelope


class JobType(StrEnum):
    DELETE_BY_SOURCE = "delete_by_source"
    RELATION_EXTRACTION = "relation_extraction"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    namespace_id: str
    job_type: JobType
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Mapping[str, object] = Field(default_factory=dict)
    error: ErrorEnvelope | None = None
