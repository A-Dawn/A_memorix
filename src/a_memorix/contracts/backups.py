"""Typed namespace backup contracts."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .context import NamespaceId
from .namespaces import NamespaceConfig, NamespaceQuota


NAMESPACE_BACKUP_FORMAT: Final[Literal["a-memorix-namespace-backup"]] = (
    "a-memorix-namespace-backup"
)
NAMESPACE_BACKUP_FORMAT_VERSION: Final[Literal[1]] = 1


class NamespaceBackupFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("backup paths must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("backup paths must be normalized relative paths")
        return path.as_posix()


class NamespaceBackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["a-memorix-namespace-backup"] = NAMESPACE_BACKUP_FORMAT
    format_version: Literal[1] = NAMESPACE_BACKUP_FORMAT_VERSION
    backup_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_namespace_id: NamespaceId
    created_at: datetime
    producer_version: str = Field(min_length=1, max_length=64)
    source_config_version: int = Field(ge=1)
    quota: NamespaceQuota = Field(default_factory=NamespaceQuota)
    config: NamespaceConfig = Field(default_factory=NamespaceConfig)
    data_size_bytes: int = Field(ge=0)
    files: tuple[NamespaceBackupFile, ...] = ()

    @model_validator(mode="after")
    def validate_file_index(self) -> "NamespaceBackupManifest":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("backup manifest contains duplicate paths")
        if sum(item.size_bytes for item in self.files) != self.data_size_bytes:
            raise ValueError("backup manifest data size does not match its files")
        return self


class NamespaceBackupInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backup_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_namespace_id: NamespaceId
    created_at: datetime
    format_version: int = Field(ge=1)
    producer_version: str = Field(min_length=1, max_length=64)
    source_config_version: int = Field(ge=1)
    archive_size_bytes: int = Field(ge=0)
    data_size_bytes: int = Field(ge=0)
    file_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RestoreNamespaceBackupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backup_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    target_namespace_id: NamespaceId


class NamespaceBackupChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backup: NamespaceBackupInfo
    offset: int = Field(ge=0)
    data: bytes
    next_offset: int = Field(ge=0)
    complete: bool


class NamespaceBackupUpload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    next_offset: int = Field(default=0, ge=0)
