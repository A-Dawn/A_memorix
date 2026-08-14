"""Service and client configuration with deterministic precedence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import os
import tomllib

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServerTLSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    certificate: Path | None = None
    private_key: Path | None = None
    client_ca: Path | None = None
    require_client_auth: bool = False

    @model_validator(mode="after")
    def validate_files(self) -> "ServerTLSConfig":
        if (self.certificate is None) != (self.private_key is None):
            raise ValueError("server TLS certificate and private key must be set together")
        if self.client_ca is not None and self.certificate is None:
            raise ValueError("server TLS client CA requires TLS to be enabled")
        if self.require_client_auth and self.client_ca is None:
            raise ValueError("server TLS client CA is required for client authentication")
        return self


class ClientTLSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    ca_certificate: Path | None = None
    certificate: Path | None = None
    private_key: Path | None = None
    server_name: str = Field(default="", max_length=253)

    @model_validator(mode="after")
    def validate_files(self) -> "ClientTLSConfig":
        if (self.certificate is None) != (self.private_key is None):
            raise ValueError("client TLS certificate and private key must be set together")
        return self


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = Path("./data")
    host: str = Field(default="127.0.0.1", min_length=1, max_length=253)
    port: int = Field(default=50051, ge=0, le=65535)
    allow_unauthenticated: bool = False
    admin_token_file: Path | None = None
    maximum_message_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    max_active_namespaces: int = Field(default=8, ge=1)
    max_concurrent_requests_per_namespace: int = Field(default=64, ge=1)
    idle_timeout_seconds: float = Field(default=900.0, ge=0)
    quarantine_retention_days: float = Field(default=7.0, ge=0)
    idempotency_retention_seconds: float = Field(
        default=24 * 60 * 60,
        gt=0,
    )
    tls: ServerTLSConfig = Field(default_factory=ServerTLSConfig)


class ClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(default="127.0.0.1:50051", min_length=1, max_length=512)
    token_file: Path | None = None
    timeout_seconds: float | None = Field(default=30.0, gt=0)
    maximum_message_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    tls: ClientTLSConfig = Field(default_factory=ClientTLSConfig)


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    log_level: str = Field(default="INFO", min_length=1, max_length=16)
    log_format: Literal["json", "text"] = "json"
    access_log: bool = True
    service_name: str = Field(default="a-memorix", min_length=1, max_length=128)
    metrics_host: str = Field(default="127.0.0.1", min_length=1, max_length=253)
    metrics_port: int | None = Field(default=None, ge=1, le=65535)
    otlp_endpoint: str = Field(default="", max_length=2048)
    otlp_insecure: bool = False
    trace_sample_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class EmbeddingProviderConfig(BaseModel):
    """OpenAI-compatible Embedding configuration without inline secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai-compatible"] = "openai-compatible"
    endpoint: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=256)
    api_key_file: Path | None = None
    dimension: int = Field(default=1024, ge=1)
    dimension_request_mode: Literal["explicit", "always", "never"] = "explicit"
    batch_size: int = Field(default=32, ge=1)
    max_concurrent: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0)
    retry_max_delay_seconds: float = Field(default=20.0, ge=0)
    retry_backoff_multiplier: float = Field(default=2.0, ge=1.0)

    @model_validator(mode="after")
    def validate_provider(self) -> "EmbeddingProviderConfig":
        _validate_provider_location(self.endpoint, self.model, "Embedding")
        return self

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.model)


class LLMProviderConfig(BaseModel):
    """OpenAI-compatible chat-completions configuration without inline secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai-compatible"] = "openai-compatible"
    endpoint: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=256)
    api_key_file: Path | None = None
    max_concurrent: int = Field(default=3, ge=1)
    max_tokens: int = Field(default=8192, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0)
    retry_max_delay_seconds: float = Field(default=20.0, ge=0)
    retry_backoff_multiplier: float = Field(default=2.0, ge=1.0)
    enable_thinking: bool | None = None
    thinking_mode: Literal["enabled", "disabled"] | None = None

    @model_validator(mode="after")
    def validate_provider(self) -> "LLMProviderConfig":
        _validate_provider_location(self.endpoint, self.model, "LLM")
        if self.enable_thinking is not None and self.thinking_mode is not None:
            raise ValueError(
                "LLM enable_thinking and thinking_mode cannot both be set"
            )
        return self

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.model)


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding: EmbeddingProviderConfig = Field(
        default_factory=EmbeddingProviderConfig
    )
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["standard", "degraded"] = "standard"
    probe_llm: bool = True


class AMemorixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server: ServerConfig = Field(default_factory=ServerConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    def redacted(self) -> dict[str, object]:
        return self.model_dump(mode="json")


_ENV_FIELDS: dict[str, tuple[str, ...]] = {
    "A_MEMORIX_DATA_DIR": ("server", "data_dir"),
    "A_MEMORIX_GRPC_HOST": ("server", "host"),
    "A_MEMORIX_GRPC_PORT": ("server", "port"),
    "A_MEMORIX_ALLOW_UNAUTHENTICATED": ("server", "allow_unauthenticated"),
    "A_MEMORIX_ADMIN_TOKEN_FILE": ("server", "admin_token_file"),
    "A_MEMORIX_MAXIMUM_MESSAGE_BYTES": ("server", "maximum_message_bytes"),
    "A_MEMORIX_MAX_ACTIVE_NAMESPACES": ("server", "max_active_namespaces"),
    "A_MEMORIX_MAX_CONCURRENT_REQUESTS": (
        "server",
        "max_concurrent_requests_per_namespace",
    ),
    "A_MEMORIX_IDLE_TIMEOUT_SECONDS": ("server", "idle_timeout_seconds"),
    "A_MEMORIX_QUARANTINE_RETENTION_DAYS": (
        "server",
        "quarantine_retention_days",
    ),
    "A_MEMORIX_IDEMPOTENCY_RETENTION_SECONDS": (
        "server",
        "idempotency_retention_seconds",
    ),
    "A_MEMORIX_TLS_CERTIFICATE": ("server", "tls", "certificate"),
    "A_MEMORIX_TLS_PRIVATE_KEY": ("server", "tls", "private_key"),
    "A_MEMORIX_TLS_CLIENT_CA": ("server", "tls", "client_ca"),
    "A_MEMORIX_TLS_REQUIRE_CLIENT_AUTH": (
        "server",
        "tls",
        "require_client_auth",
    ),
    "A_MEMORIX_CLIENT_TARGET": ("client", "target"),
    "A_MEMORIX_CLIENT_TOKEN_FILE": ("client", "token_file"),
    "A_MEMORIX_CLIENT_TIMEOUT_SECONDS": ("client", "timeout_seconds"),
    "A_MEMORIX_CLIENT_MAXIMUM_MESSAGE_BYTES": (
        "client",
        "maximum_message_bytes",
    ),
    "A_MEMORIX_CLIENT_TLS_ENABLED": ("client", "tls", "enabled"),
    "A_MEMORIX_CLIENT_CA_CERTIFICATE": (
        "client",
        "tls",
        "ca_certificate",
    ),
    "A_MEMORIX_CLIENT_CERTIFICATE": ("client", "tls", "certificate"),
    "A_MEMORIX_CLIENT_PRIVATE_KEY": ("client", "tls", "private_key"),
    "A_MEMORIX_CLIENT_SERVER_NAME": ("client", "tls", "server_name"),
    "A_MEMORIX_LOG_LEVEL": ("observability", "log_level"),
    "A_MEMORIX_LOG_FORMAT": ("observability", "log_format"),
    "A_MEMORIX_ACCESS_LOG": ("observability", "access_log"),
    "A_MEMORIX_SERVICE_NAME": ("observability", "service_name"),
    "A_MEMORIX_METRICS_HOST": ("observability", "metrics_host"),
    "A_MEMORIX_METRICS_PORT": ("observability", "metrics_port"),
    "A_MEMORIX_OTLP_ENDPOINT": ("observability", "otlp_endpoint"),
    "A_MEMORIX_OTLP_INSECURE": ("observability", "otlp_insecure"),
    "A_MEMORIX_TRACE_SAMPLE_RATIO": ("observability", "trace_sample_ratio"),
    "A_MEMORIX_EMBEDDING_BASE_URL": ("providers", "embedding", "endpoint"),
    "A_MEMORIX_EMBEDDING_ENDPOINT": ("providers", "embedding", "endpoint"),
    "A_MEMORIX_EMBEDDING_MODEL": ("providers", "embedding", "model"),
    "A_MEMORIX_EMBEDDING_API_KEY_FILE": (
        "providers",
        "embedding",
        "api_key_file",
    ),
    "A_MEMORIX_EMBEDDING_DIMENSION": ("providers", "embedding", "dimension"),
    "A_MEMORIX_EMBEDDING_DIMENSION_REQUEST_MODE": (
        "providers",
        "embedding",
        "dimension_request_mode",
    ),
    "A_MEMORIX_EMBEDDING_BATCH_SIZE": ("providers", "embedding", "batch_size"),
    "A_MEMORIX_EMBEDDING_MAX_CONCURRENT": (
        "providers",
        "embedding",
        "max_concurrent",
    ),
    "A_MEMORIX_EMBEDDING_TIMEOUT_SECONDS": (
        "providers",
        "embedding",
        "timeout_seconds",
    ),
    "A_MEMORIX_EMBEDDING_MAX_ATTEMPTS": (
        "providers",
        "embedding",
        "max_attempts",
    ),
    "A_MEMORIX_EMBEDDING_RETRY_DELAY_SECONDS": (
        "providers",
        "embedding",
        "retry_delay_seconds",
    ),
    "A_MEMORIX_EMBEDDING_RETRY_MAX_DELAY_SECONDS": (
        "providers",
        "embedding",
        "retry_max_delay_seconds",
    ),
    "A_MEMORIX_EMBEDDING_RETRY_BACKOFF_MULTIPLIER": (
        "providers",
        "embedding",
        "retry_backoff_multiplier",
    ),
    "A_MEMORIX_LLM_BASE_URL": ("providers", "llm", "endpoint"),
    "A_MEMORIX_LLM_ENDPOINT": ("providers", "llm", "endpoint"),
    "A_MEMORIX_LLM_MODEL": ("providers", "llm", "model"),
    "A_MEMORIX_LLM_API_KEY_FILE": ("providers", "llm", "api_key_file"),
    "A_MEMORIX_LLM_MAX_CONCURRENT": ("providers", "llm", "max_concurrent"),
    "A_MEMORIX_LLM_MAX_TOKENS": ("providers", "llm", "max_tokens"),
    "A_MEMORIX_LLM_TEMPERATURE": ("providers", "llm", "temperature"),
    "A_MEMORIX_LLM_TIMEOUT_SECONDS": ("providers", "llm", "timeout_seconds"),
    "A_MEMORIX_LLM_MAX_ATTEMPTS": ("providers", "llm", "max_attempts"),
    "A_MEMORIX_LLM_RETRY_DELAY_SECONDS": (
        "providers",
        "llm",
        "retry_delay_seconds",
    ),
    "A_MEMORIX_LLM_RETRY_MAX_DELAY_SECONDS": (
        "providers",
        "llm",
        "retry_max_delay_seconds",
    ),
    "A_MEMORIX_LLM_RETRY_BACKOFF_MULTIPLIER": (
        "providers",
        "llm",
        "retry_backoff_multiplier",
    ),
    "A_MEMORIX_LLM_ENABLE_THINKING": ("providers", "llm", "enable_thinking"),
    "A_MEMORIX_LLM_THINKING_MODE": ("providers", "llm", "thinking_mode"),
    "A_MEMORIX_MCP_MODE": ("mcp", "mode"),
    "A_MEMORIX_MCP_PROBE_LLM": ("mcp", "probe_llm"),
}

_BOOLEAN_ENV_FIELDS = {
    "A_MEMORIX_ALLOW_UNAUTHENTICATED",
    "A_MEMORIX_TLS_REQUIRE_CLIENT_AUTH",
    "A_MEMORIX_CLIENT_TLS_ENABLED",
    "A_MEMORIX_ACCESS_LOG",
    "A_MEMORIX_OTLP_INSECURE",
    "A_MEMORIX_LLM_ENABLE_THINKING",
    "A_MEMORIX_MCP_PROBE_LLM",
}


def load_config(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AMemorixConfig:
    environment = dict(os.environ if environ is None else environ)
    selected_path = config_path or environment.get("A_MEMORIX_CONFIG")
    raw: dict[str, object] = {}
    if selected_path:
        path = Path(selected_path).expanduser().resolve()
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
        raw = dict(loaded)
        _resolve_config_paths(raw, path.parent)
    for variable, field_path in _ENV_FIELDS.items():
        if variable not in environment:
            continue
        value: object = environment[variable]
        if variable in _BOOLEAN_ENV_FIELDS:
            value = _parse_bool(str(value), variable)
        _set_nested(raw, field_path, value)
    return AMemorixConfig.model_validate(raw)


def read_secret(*, environment_name: str, file_path: Path | None) -> str:
    value = os.environ.get(environment_name, "").strip()
    if value:
        return value
    if file_path is None:
        return ""
    return file_path.expanduser().read_text(encoding="utf-8").strip()


def _resolve_config_paths(raw: dict[str, object], base: Path) -> None:
    for field_path in (
        ("server", "data_dir"),
        ("server", "admin_token_file"),
        ("server", "tls", "certificate"),
        ("server", "tls", "private_key"),
        ("server", "tls", "client_ca"),
        ("client", "token_file"),
        ("client", "tls", "ca_certificate"),
        ("client", "tls", "certificate"),
        ("client", "tls", "private_key"),
        ("providers", "embedding", "api_key_file"),
        ("providers", "llm", "api_key_file"),
    ):
        value = _get_nested(raw, field_path)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            _set_nested(raw, field_path, str((base / path).resolve()))


def _get_nested(values: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = values
    for field in path:
        if not isinstance(current, Mapping) or field not in current:
            return None
        current = current[field]
    return current


def _set_nested(
    values: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    current = values
    for field in path[:-1]:
        child = current.get(field)
        if not isinstance(child, dict):
            child = {}
            current[field] = child
        current = child
    current[path[-1]] = value


def _parse_bool(value: str, variable: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{variable} must be a boolean value")


def _validate_provider_location(endpoint: str, model: str, label: str) -> None:
    endpoint = endpoint.strip()
    model = model.strip()
    if bool(endpoint) != bool(model):
        raise ValueError(f"{label} endpoint and model must be set together")
    if endpoint and not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"{label} endpoint must be an HTTP(S) URL")
