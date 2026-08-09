"""Public multi-namespace in-process engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncio
import hashlib
import json
import secrets
import time

from a_memorix.contracts import (
    ApiKeyInfo,
    CapabilityUnavailableError,
    CreateNamespaceRequest,
    CreatedApiKey,
    IngestTextRequest,
    IngestTextResponse,
    InvalidArgumentError,
    MemoryHit,
    NamespaceHealth,
    NamespaceInfo,
    NamespaceCapacityError,
    NamespaceStatus,
    RequestContext,
    SearchMemoryRequest,
    SearchMemoryResponse,
)
from a_memorix.ports import Clock

from .core.runtime.namespace_control import NamespaceControlStore
from .core.runtime.namespace_registry import NamespaceRuntimeRegistry
from .core.runtime.namespace_runtime import (
    NamespaceConfigFactory,
    NamespaceHostPortFactory,
    NamespaceRuntime,
    NamespaceRuntimeFactory,
    SDKKernelRuntimeFactory,
)
from .core.runtime.namespace_storage import NamespaceStorageLayout
from .core.runtime.models import KernelSearchRequest


class AMemorixEngine:
    """Manage isolated namespace runtimes under one service data root."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        runtime_factory: NamespaceRuntimeFactory | None = None,
        config_factory: NamespaceConfigFactory | None = None,
        host_port_factory: NamespaceHostPortFactory | None = None,
        max_active_namespaces: int = 8,
        max_concurrent_requests_per_namespace: int = 64,
        idle_timeout_seconds: float = 900.0,
        quarantine_retention_days: float = 7.0,
        clock: Clock | None = None,
    ) -> None:
        if runtime_factory is not None and (
            config_factory is not None or host_port_factory is not None
        ):
            raise ValueError(
                "config_factory and host_port_factory cannot be combined with runtime_factory"
            )
        self._layout = NamespaceStorageLayout(data_dir)
        self._runtime_factory = runtime_factory or SDKKernelRuntimeFactory(
            config_factory=config_factory,
            host_port_factory=host_port_factory,
        )
        self._max_active_namespaces = max_active_namespaces
        self._max_concurrent_requests = max_concurrent_requests_per_namespace
        self._idle_timeout_seconds = idle_timeout_seconds
        self._quarantine_retention_seconds = quarantine_retention_days * 24 * 60 * 60
        self._clock = clock
        self._store: NamespaceControlStore | None = None
        self._registry: NamespaceRuntimeRegistry | None = None
        self._application_write_locks: dict[str, asyncio.Lock] = {}

    @property
    def data_dir(self) -> Path:
        return self._layout.data_root

    async def initialize(self) -> None:
        if self._registry is not None:
            return
        self._layout.initialize()
        store = NamespaceControlStore(self._layout.control_db_path)
        registry = NamespaceRuntimeRegistry(
            control_store=store,
            storage_layout=self._layout,
            runtime_factory=self._runtime_factory,
            max_active_namespaces=self._max_active_namespaces,
            max_concurrent_requests_per_namespace=self._max_concurrent_requests,
            idle_timeout_seconds=self._idle_timeout_seconds,
            quarantine_retention_seconds=self._quarantine_retention_seconds,
            clock=self._clock,
        )
        try:
            await registry.start()
        except BaseException:
            store.close()
            raise
        self._store = store
        self._registry = registry

    async def shutdown(self) -> None:
        registry = self._registry
        store = self._store
        if registry is None:
            return
        await registry.shutdown()
        if store is not None:
            store.close()
        self._registry = None
        self._store = None

    async def __aenter__(self) -> "AMemorixEngine":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        await self.shutdown()

    async def create_namespace(self, request: CreateNamespaceRequest) -> NamespaceInfo:
        return await self._require_registry().create_namespace(request)

    async def get_namespace(self, namespace_id: str) -> NamespaceInfo:
        return self._require_registry().get_namespace(namespace_id)

    async def list_namespaces(self) -> list[NamespaceInfo]:
        return self._require_registry().list_namespaces()

    async def disable_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().disable_namespace(namespace_id)

    async def enable_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().enable_namespace(namespace_id)

    async def delete_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().delete_namespace(namespace_id)

    async def restore_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().restore_namespace(namespace_id)

    async def purge_namespace(self, namespace_id: str) -> None:
        await self._require_registry().purge_namespace(namespace_id, force=True)

    async def purge_expired_namespaces(self) -> list[str]:
        return await self._require_registry().purge_expired_namespaces()

    async def namespace_health(self, namespace_id: str) -> NamespaceHealth:
        return await self._require_registry().get_health(namespace_id)

    async def create_api_key(
        self,
        namespace_id: str,
        *,
        label: str = "",
        expires_at: datetime | None = None,
    ) -> CreatedApiKey:
        namespace = await self.get_namespace(namespace_id)
        if namespace.status in {NamespaceStatus.QUARANTINED, NamespaceStatus.PURGING}:
            raise InvalidArgumentError(
                "cannot create an API key for a deleted namespace",
                details={"namespace_id": namespace_id, "status": namespace.status.value},
            )
        normalized_label = str(label or "").strip()
        if len(normalized_label) > 128:
            raise InvalidArgumentError("API key label exceeds 128 characters")
        now = self._time()
        expires_timestamp = _optional_timestamp(expires_at)
        if expires_timestamp is not None and expires_timestamp <= now:
            raise InvalidArgumentError("API key expiration must be in the future")
        key_id = uuid4().hex
        secret = f"amx_v1_{key_id}_{secrets.token_urlsafe(32)}"
        info = self._require_store().create_api_key(
            key_id=key_id,
            namespace_id=namespace_id,
            secret_hash=_hash_api_key(secret),
            label=normalized_label,
            created_at=now,
            expires_at=expires_timestamp,
        )
        return CreatedApiKey(api_key=info, secret=secret)

    async def list_api_keys(self, namespace_id: str) -> list[ApiKeyInfo]:
        await self.get_namespace(namespace_id)
        return self._require_store().list_api_keys(namespace_id)

    async def revoke_api_key(self, namespace_id: str, key_id: str) -> ApiKeyInfo:
        await self.get_namespace(namespace_id)
        return self._require_store().revoke_api_key(
            namespace_id,
            str(key_id or "").strip(),
            now=self._time(),
        )

    def authenticate_api_key(self, secret: str) -> ApiKeyInfo | None:
        token = str(secret or "").strip()
        if not token:
            return None
        return self._require_store().authenticate_api_key(
            _hash_api_key(token),
            now=self._time(),
        )

    async def ingest_text(self, request: IngestTextRequest) -> IngestTextResponse:
        context = request.context
        lock = self._application_write_locks.setdefault(
            context.namespace_id,
            asyncio.Lock(),
        )
        async with lock:
            async with self.runtime(context) as runtime:
                if not await _contains_external_memory(runtime, request.external_id):
                    await self._enforce_write_admission(request)
                ingest = getattr(runtime, "ingest_text", None)
                if not callable(ingest):
                    raise CapabilityUnavailableError(
                        "namespace runtime does not support text ingestion",
                        details={"namespace_id": context.namespace_id},
                    )

                async def execute() -> dict[str, Any]:
                    return await ingest(
                        external_id=request.external_id,
                        source_type=request.source_type,
                        text=request.text,
                        chat_id=context.conversation_id or "",
                        person_ids=request.person_ids,
                        participants=request.participants,
                        timestamp=_optional_timestamp(request.observed_at),
                        time_start=_optional_timestamp(request.valid_from),
                        time_end=_optional_timestamp(request.valid_to),
                        tags=request.tags,
                        metadata=dict(request.metadata),
                        entities=request.entities,
                        relations=[
                            item.model_dump(by_alias=True) for item in request.relations
                        ],
                        respect_filter=request.respect_filter,
                        user_id=context.user_id or "",
                        group_id=context.group_id or "",
                    )

                deduplicate = getattr(runtime, "execute_request_with_dedup", None)
                if context.idempotency_key and callable(deduplicate):
                    _, result = await deduplicate(context.idempotency_key, execute)
                else:
                    result = await execute()
        return IngestTextResponse(
            stored_ids=tuple(str(item) for item in result.get("stored_ids", [])),
            skipped_ids=tuple(str(item) for item in result.get("skipped_ids", [])),
            fact_claim_ids=tuple(str(item) for item in result.get("fact_claim_ids", [])),
            warnings=tuple(str(item) for item in result.get("warnings", [])),
            detail=str(result.get("detail", result.get("reason", "")) or ""),
        )

    async def search_memory(self, request: SearchMemoryRequest) -> SearchMemoryResponse:
        context = request.context
        async with self.runtime(context) as runtime:
            search = getattr(runtime, "search_memory", None)
            if not callable(search):
                raise CapabilityUnavailableError(
                    "namespace runtime does not support memory search",
                    details={"namespace_id": context.namespace_id},
                )
            result = await search(
                KernelSearchRequest(
                    query=request.query,
                    limit=request.limit,
                    mode=request.mode.value,
                    chat_id=context.conversation_id or "",
                    shared_chat_ids=request.shared_conversation_ids,
                    person_id=request.person_id,
                    time_start=_optional_timestamp(request.time_start),
                    time_end=_optional_timestamp(request.time_end),
                    respect_filter=request.respect_filter,
                    user_id=context.user_id or "",
                    group_id=context.group_id or "",
                )
            )
        error = str(result.get("error", "") or "").strip()
        if error:
            raise CapabilityUnavailableError(
                error,
                details={"namespace_id": context.namespace_id},
            )
        hits = tuple(_memory_hit(item) for item in result.get("hits", []) if isinstance(item, dict))
        return SearchMemoryResponse(
            summary=str(result.get("summary", "") or ""),
            hits=hits,
            filtered=bool(result.get("filtered", False)),
            degraded=bool(result.get("degraded", False)),
            retrieval_ready=bool(result.get("retrieval_ready", False)),
            retrieval_mode=str(result.get("retrieval_mode", "") or ""),
            available_channels=tuple(
                str(item) for item in result.get("available_channels", [])
            ),
            unavailable_channels=tuple(
                str(item) for item in result.get("unavailable_channels", [])
            ),
        )

    async def close_idle_runtimes(self) -> list[str]:
        return await self._require_registry().close_idle_runtimes()

    @asynccontextmanager
    async def runtime(self, context: RequestContext) -> AsyncIterator[NamespaceRuntime]:
        async with self._require_registry().lease(context) as runtime:
            yield runtime

    def _require_registry(self) -> NamespaceRuntimeRegistry:
        if self._registry is None:
            raise RuntimeError("AMemorixEngine has not been initialized")
        return self._registry

    def _require_store(self) -> NamespaceControlStore:
        if self._store is None:
            raise RuntimeError("AMemorixEngine has not been initialized")
        return self._store

    async def _enforce_write_admission(self, request: IngestTextRequest) -> None:
        health = await self.namespace_health(request.context.namespace_id)
        maximum = health.namespace.quota.max_storage_bytes
        if maximum is None:
            return
        serialized = json.dumps(
            request.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        reservation = max(4096, len(serialized))
        projected = health.resource_usage.storage_bytes + reservation
        if projected > maximum:
            raise NamespaceCapacityError(
                "namespace storage admission limit exceeded",
                details={
                    "namespace_id": request.context.namespace_id,
                    "storage_bytes": health.resource_usage.storage_bytes,
                    "reservation_bytes": reservation,
                    "max_storage_bytes": maximum,
                },
            )

    def _time(self) -> float:
        return self._clock.time() if self._clock is not None else time.time()


def _hash_api_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


def _optional_timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


async def _contains_external_memory(runtime: object, external_id: str) -> bool:
    token = str(external_id or "").strip()
    contains = getattr(runtime, "contains_external_memory", None)
    if not token or not callable(contains):
        return False
    return bool(await contains(token))


def _memory_hit(item: dict[str, Any]) -> MemoryHit:
    metadata = item.get("metadata")
    return MemoryHit(
        memory_id=str(item.get("hash", item.get("episode_id", "")) or ""),
        kind=str(item.get("type", "") or ""),
        title=str(item.get("title", "") or ""),
        content=str(item.get("content", "") or ""),
        score=float(item.get("score", 0.0) or 0.0),
        source=str(item.get("source", "") or ""),
        metadata=metadata if isinstance(metadata, dict) else {},
    )
