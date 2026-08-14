"""Public manifest contract for Agent adapters."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

import re
import tomllib

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ADAPTER_MANIFEST_SCHEMA_VERSION = 1
ADAPTER_PROTOCOL_VERSION = "1"

_SEMVER_PATTERN = (
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_ADAPTER_ID_PATTERN = (
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$"
)
_PACKAGE_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$"
_ENTRYPOINT_PATTERN = (
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")


class AdapterRuntime(StrEnum):
    IN_PROCESS = "in_process"
    REMOTE = "remote"


class AdapterTransport(StrEnum):
    IN_PROCESS = "in_process"
    GRPC = "grpc"
    HTTP_JSON = "http_json"
    MCP = "mcp"


class AdapterApiPermission(StrEnum):
    NAMESPACE_READ = "namespace.read"
    NAMESPACE_MANAGE = "namespace.manage"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    MEMORY_DELETE = "memory.delete"
    JOB_READ = "job.read"
    BACKUP_READ = "backup.read"
    BACKUP_MANAGE = "backup.manage"
    API_KEY_MANAGE = "api_key.manage"


class AdapterHostPort(StrEnum):
    EMBEDDING = "embedding"
    LLM = "llm"
    IDENTITY = "identity"
    MESSAGE_SOURCE = "message_source"


class FilesystemAccess(StrEnum):
    READ = "read"
    WRITE = "write"


class AdapterFilesystemPermission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=1024)
    access: FilesystemAccess

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("filesystem permission paths must be printable")
        return normalized


class AdapterPermissions(BaseModel):
    """Declared access; enforcement remains the responsibility of the host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api: tuple[AdapterApiPermission, ...] = Field(min_length=1)
    network: tuple[str, ...]
    filesystem: tuple[AdapterFilesystemPermission, ...]
    environment: tuple[str, ...]
    subprocess: bool

    @field_validator("api", mode="before")
    @classmethod
    def validate_unique_api_permissions(cls, value: object) -> object:
        _ensure_unique_sequence(value, "API permissions")
        return value

    @field_validator("network")
    @classmethod
    def validate_network_origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validate_network_origin(item) for item in value)
        _ensure_unique_sequence(normalized, "network origins")
        return normalized

    @field_validator("filesystem")
    @classmethod
    def validate_unique_filesystem_permissions(
        cls,
        value: tuple[AdapterFilesystemPermission, ...],
    ) -> tuple[AdapterFilesystemPermission, ...]:
        keys = tuple((item.path, item.access.value) for item in value)
        _ensure_unique_sequence(keys, "filesystem permissions")
        return value

    @field_validator("environment")
    @classmethod
    def validate_environment_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        for item in normalized:
            if not _ENVIRONMENT_NAME_PATTERN.fullmatch(item):
                raise ValueError(
                    "environment permissions must contain uppercase variable names"
                )
        _ensure_unique_sequence(normalized, "environment permissions")
        return normalized


class AdapterManifest(BaseModel):
    """Registry metadata shared by in-process and remote adapters."""

    model_config = ConfigDict(
        title="A_memorix Adapter Manifest v1",
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1]
    id: str = Field(min_length=3, max_length=128, pattern=_ADAPTER_ID_PATTERN)
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=5, max_length=64, pattern=_SEMVER_PATTERN)
    runtime: AdapterRuntime
    package: str | None = Field(default=None, pattern=_PACKAGE_PATTERN)
    entrypoint: str | None = Field(default=None, pattern=_ENTRYPOINT_PATTERN)
    core_version: str = Field(min_length=1, max_length=256)
    adapter_protocol: Literal["1"]
    transports: tuple[AdapterTransport, ...] = Field(min_length=1)
    host_ports: tuple[AdapterHostPort, ...]
    license: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.+() -]*$",
    )
    source: str = Field(min_length=1, max_length=2048)
    permissions: AdapterPermissions

    @field_validator("version")
    @classmethod
    def validate_semantic_version(cls, value: str) -> str:
        prerelease = value.partition("-")[2].partition("+")[0]
        for identifier in prerelease.split(".") if prerelease else ():
            if identifier.isdigit() and len(identifier) > 1 and identifier.startswith("0"):
                raise ValueError(
                    "numeric semantic-version prerelease identifiers cannot have leading zeros"
                )
        return value

    @field_validator("core_version")
    @classmethod
    def validate_core_version(cls, value: str) -> str:
        normalized = value.strip()
        try:
            SpecifierSet(normalized)
        except InvalidSpecifier as exc:
            raise ValueError("core_version must be a PEP 440 specifier set") from exc
        return normalized

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip()
        parsed = urlsplit(normalized)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("source contains an invalid port") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("source must be an HTTPS URL without credentials")
        return normalized

    @field_validator("transports", "host_ports", mode="before")
    @classmethod
    def validate_unique_capabilities(cls, value: object) -> object:
        _ensure_unique_sequence(value, "adapter capabilities")
        return value

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> Self:
        transports = set(self.transports)
        if self.runtime is AdapterRuntime.IN_PROCESS:
            if transports != {AdapterTransport.IN_PROCESS}:
                raise ValueError(
                    "in_process adapters must use only the in_process transport"
                )
            if self.package is None or self.entrypoint is None:
                raise ValueError(
                    "in_process adapters require package and entrypoint"
                )
            return self

        if AdapterTransport.IN_PROCESS in transports:
            raise ValueError("remote adapters cannot use the in_process transport")
        if self.entrypoint is not None:
            raise ValueError("remote adapters cannot declare a Python entrypoint")
        if self.host_ports:
            raise ValueError("remote adapters cannot provide in-process Host Ports")
        if transports & {AdapterTransport.GRPC, AdapterTransport.HTTP_JSON}:
            if "a-memorix" not in self.permissions.network:
                raise ValueError(
                    "remote gRPC and HTTP/JSON adapters must declare the a-memorix network origin"
                )
        return self


class AdapterCompatibilityError(ValueError):
    """The manifest does not support the selected A_memorix core version."""


def load_adapter_manifest(path: str | Path) -> AdapterManifest:
    manifest_path = Path(path).expanduser()
    with manifest_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return AdapterManifest.model_validate(payload)


def ensure_adapter_compatible(
    manifest: AdapterManifest,
    *,
    core_version: str,
) -> None:
    try:
        selected_version = Version(core_version)
    except InvalidVersion as exc:
        raise ValueError("core version must be a valid PEP 440 version") from exc
    supported = SpecifierSet(manifest.core_version)
    if selected_version not in supported:
        raise AdapterCompatibilityError(
            f"adapter {manifest.id} {manifest.version} requires A_memorix "
            f"{manifest.core_version}; selected version is {selected_version}"
        )


def adapter_manifest_json_schema() -> dict[str, object]:
    return AdapterManifest.model_json_schema()


def _ensure_unique_sequence(value: object, label: str) -> None:
    if not isinstance(value, (list, tuple)):
        return
    comparable = tuple(str(item) for item in value)
    if len(comparable) != len(set(comparable)):
        raise ValueError(f"{label} must not contain duplicates")


def _validate_network_origin(value: str) -> str:
    normalized = value.strip()
    if normalized in {"a-memorix", "embedding-provider", "llm-provider"}:
        return normalized
    parsed = urlsplit(normalized)
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("network permission contains an invalid port") from exc
    if (
        "*" in normalized
        or parsed.scheme not in {"http", "https", "grpc", "grpcs"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "network permissions must be a known service placeholder or an "
            "HTTP/gRPC origin without credentials or paths"
        )
    return normalized.rstrip("/")
