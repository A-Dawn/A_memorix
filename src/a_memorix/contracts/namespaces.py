"""Typed namespace control-plane contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Mapping

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


class ProviderReference(BaseModel):
    """Non-secret reference resolved by the host into one provider instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(default="", max_length=256)
    secret_ref: str = Field(default="", max_length=256)


class NamespaceFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episodes: bool = True
    person_profiles: bool = True
    sparse_retrieval: bool = True
    relation_vectors: bool = False
    allow_metadata_only_write: bool = True


class NamespaceConfig(BaseModel):
    """Persisted, non-secret configuration for one namespace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding: ProviderReference | None = None
    llm: ProviderReference | None = None
    identity_resolver: ProviderReference | None = None
    message_source: ProviderReference | None = None
    features: NamespaceFeatureConfig = Field(default_factory=NamespaceFeatureConfig)


class CreateNamespaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace_id: NamespaceId
    quota: NamespaceQuota = Field(default_factory=NamespaceQuota)
    config: NamespaceConfig = Field(default_factory=NamespaceConfig)


class NamespaceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace_id: NamespaceId
    status: NamespaceStatus
    created_at: datetime
    updated_at: datetime
    last_active_at: datetime | None = None
    version: int = Field(ge=1)
    config_version: int = Field(default=1, ge=1)
    quota: NamespaceQuota = Field(default_factory=NamespaceQuota)
    config: NamespaceConfig = Field(default_factory=NamespaceConfig)
    purge_after: datetime | None = None


class UpdateNamespaceConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace_id: NamespaceId
    config: NamespaceConfig
    expected_config_version: int | None = Field(default=None, ge=1)


class NamespaceCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace_id: NamespaceId
    config_version: int = Field(ge=1)
    capabilities: Mapping[str, bool] = Field(default_factory=dict)
    operations: tuple[str, ...] = ()
    search_modes: tuple[str, ...] = ()
    degraded: bool = False
    unavailable: tuple[str, ...] = ()


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
