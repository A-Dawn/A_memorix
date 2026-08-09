"""Transport-independent API key metadata."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .context import NamespaceId


class ApiKeyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: str = Field(min_length=1, max_length=64)
    namespace_id: NamespaceId
    label: str = Field(default="", max_length=128)
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None


class CreatedApiKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: ApiKeyInfo
    secret: str = Field(min_length=32)
