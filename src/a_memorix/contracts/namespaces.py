"""Typed namespace control-plane contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .context import NamespaceId


class NamespaceStatus(StrEnum):
    CREATING = "creating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    QUARANTINED = "quarantined"
    PURGING = "purging"


class NamespaceQuota(BaseModel):
    """Limits that can be enforced without coupling the core to one host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_concurrent_requests: int | None = Field(default=None, ge=1)
    max_storage_bytes: int | None = Field(default=None, ge=1)


class CreateNamespaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace_id: NamespaceId
    quota: NamespaceQuota = Field(default_factory=NamespaceQuota)


class NamespaceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace_id: NamespaceId
    status: NamespaceStatus
    created_at: datetime
    updated_at: datetime
    last_active_at: datetime | None = None
    version: int = Field(ge=1)
    quota: NamespaceQuota = Field(default_factory=NamespaceQuota)
    purge_after: datetime | None = None


class NamespaceRuntimeState(StrEnum):
    CLOSED = "closed"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class NamespaceResourceUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_requests: int = Field(default=0, ge=0)
    storage_bytes: int = Field(default=0, ge=0)


class NamespaceHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: NamespaceInfo
    runtime_state: NamespaceRuntimeState
    healthy: bool
    resource_usage: NamespaceResourceUsage
    last_error: str | None = None
