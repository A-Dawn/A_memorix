"""Public multi-namespace in-process engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncio
import base64
import binascii
import hashlib
import json
import secrets
import time

from a_memorix.contracts import (
    AMemorixError,
    ApiKeyInfo,
    BatchIngestItemResult,
    BatchIngestTextRequest,
    BatchIngestTextResponse,
    CapabilityUnavailableError,
    CreateNamespaceRequest,
    CreatedApiKey,
    DeleteBySourceRequest,
    DeleteMemoryRequest,
    DeleteMemoryResponse,
    ErrorCode,
    ErrorEnvelope,
    GetMemoryRequest,
    GetMemoryResponse,
    IngestTextRequest,
    IngestTextResponse,
    InvalidArgumentError,
    JobInfo,
    JobStatus,
    JobType,
    MemoryHit,
    MemoryRecord,
    NamespaceBackupChunk,
    NamespaceBackupInfo,
    NamespaceBackupUpload,
    NamespaceCapabilities,
    NamespaceHealth,
    NamespaceInfo,
    NamespaceCapacityError,
    NamespaceStatus,
    NotFoundError,
    RequestContext,
    SearchMemoryRequest,
    SearchMemoryResponse,
    RestoreNamespaceBackupRequest,
    UpdateNamespaceConfigRequest,
)
from a_memorix.ports import Clock

from .core.runtime.namespace_control import NamespaceControlStore
from .core.runtime.namespace_backup import NamespaceBackupStore
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
        idempotency_retention_seconds: float = 24 * 60 * 60,
        clock: Clock | None = None,
    ) -> None:
        if runtime_factory is not None and (
            config_factory is not None or host_port_factory is not None
        ):
            raise ValueError(
                "config_factory and host_port_factory cannot be combined with runtime_factory"
            )
        if idempotency_retention_seconds <= 0:
            raise ValueError("idempotency_retention_seconds must be positive")
        self._layout = NamespaceStorageLayout(data_dir)
        self._backup_store = NamespaceBackupStore(
            self._layout.backups_root,
            self._layout.backup_uploads_root,
        )
        self._runtime_factory = runtime_factory or SDKKernelRuntimeFactory(
            config_factory=config_factory,
            host_port_factory=host_port_factory,
        )
        self._max_active_namespaces = max_active_namespaces
        self._max_concurrent_requests = max_concurrent_requests_per_namespace
        self._idle_timeout_seconds = idle_timeout_seconds
        self._quarantine_retention_seconds = quarantine_retention_days * 24 * 60 * 60
        self._idempotency_retention_seconds = idempotency_retention_seconds
        self._clock = clock
        self._store: NamespaceControlStore | None = None
        self._registry: NamespaceRuntimeRegistry | None = None
        self._application_write_locks: dict[str, asyncio.Lock] = {}
        self._job_tasks: dict[str, asyncio.Task[None]] = {}
        self._shutting_down = False

    @property
    def data_dir(self) -> Path:
        return self._layout.data_root

    async def initialize(self) -> None:
        if self._registry is not None:
            return
        self._layout.initialize()
        self._backup_store.initialize()
        store = NamespaceControlStore(self._layout.control_db_path)
        store.fail_interrupted_jobs(now=self._time())
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
        self._shutting_down = True
        if self._job_tasks:
            await asyncio.gather(
                *tuple(self._job_tasks.values()), return_exceptions=True
            )
        await registry.shutdown()
        if store is not None:
            store.close()
        self._registry = None
        self._store = None
        self._job_tasks.clear()
        self._shutting_down = False

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

    async def list_namespaces_page(
        self,
        *,
        page_size: int = 50,
        page_token: str = "",
    ) -> tuple[list[NamespaceInfo], str]:
        return _paginate(
            await self.list_namespaces(),
            page_size=page_size,
            page_token=page_token,
            token_kind="namespaces",
            item_key=lambda item: item.namespace_id,
        )

    async def update_namespace_config(
        self,
        request: UpdateNamespaceConfigRequest,
    ) -> NamespaceInfo:
        return await self._require_registry().update_namespace_config(request)

    async def get_namespace_capabilities(
        self,
        namespace_id: str,
    ) -> NamespaceCapabilities:
        namespace = await self.get_namespace(namespace_id)
        context = RequestContext(namespace_id=namespace_id)
        async with self.runtime(context) as runtime:
            inspect_capabilities = getattr(runtime, "runtime_capability_status", None)
            raw_status = (
                inspect_capabilities() if callable(inspect_capabilities) else {}
            )
            status = raw_status if isinstance(raw_status, dict) else {}
            raw_capabilities = status.get("capabilities")
            capabilities = {
                str(key): bool(value)
                for key, value in (
                    raw_capabilities.items()
                    if isinstance(raw_capabilities, dict)
                    else ()
                )
            }
            capabilities.update(
                {
                    "ingest_text": callable(getattr(runtime, "ingest_text", None)),
                    "search_memory": callable(getattr(runtime, "search_memory", None)),
                    "get_memory": callable(getattr(runtime, "get_memory_record", None)),
                    "delete_memory": callable(
                        getattr(runtime, "memory_delete_admin", None)
                    ),
                    "delete_by_source": callable(
                        getattr(runtime, "memory_source_admin", None)
                    ),
                }
            )
        operations = tuple(
            name
            for name in (
                "ingest_text",
                "batch_ingest_text",
                "search_memory",
                "get_memory",
                "delete_memory",
                "delete_by_source",
            )
            if capabilities.get(
                "ingest_text" if name == "batch_ingest_text" else name,
                False,
            )
        )
        search_modes = ["search", "time", "hybrid", "aggregate"]
        if namespace.config.features.episodes:
            search_modes.append("episode")
        unavailable = tuple(
            sorted(name for name, available in capabilities.items() if not available)
        )
        return NamespaceCapabilities(
            namespace_id=namespace_id,
            config_version=namespace.config_version,
            capabilities=capabilities,
            operations=operations,
            search_modes=tuple(search_modes),
            degraded=bool(status.get("degraded", False)),
            unavailable=unavailable,
        )

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

    async def create_namespace_backup(
        self,
        namespace_id: str,
    ) -> NamespaceBackupInfo:
        registry = self._require_registry()
        async with registry.inactive_storage(namespace_id) as (namespace, source):
            return await asyncio.to_thread(
                self._backup_store.create_backup,
                namespace,
                source,
                created_at=datetime.fromtimestamp(self._time(), tz=timezone.utc),
            )

    async def get_namespace_backup(self, backup_id: str) -> NamespaceBackupInfo:
        self._require_registry()
        return await asyncio.to_thread(self._backup_store.get_backup, backup_id)

    async def list_namespace_backups(
        self,
        *,
        source_namespace_id: str = "",
    ) -> list[NamespaceBackupInfo]:
        self._require_registry()
        return await asyncio.to_thread(
            self._backup_store.list_backups,
            source_namespace_id,
        )

    async def list_namespace_backups_page(
        self,
        *,
        source_namespace_id: str = "",
        page_size: int = 50,
        page_token: str = "",
    ) -> tuple[list[NamespaceBackupInfo], str]:
        return _paginate(
            await self.list_namespace_backups(
                source_namespace_id=source_namespace_id
            ),
            page_size=page_size,
            page_token=page_token,
            token_kind=f"backups:{source_namespace_id}",
            item_key=lambda item: item.backup_id,
        )

    async def delete_namespace_backup(self, backup_id: str) -> None:
        self._require_registry()
        await asyncio.to_thread(self._backup_store.delete_backup, backup_id)

    async def download_namespace_backup(
        self,
        backup_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 256 * 1024,
    ) -> NamespaceBackupChunk:
        self._require_registry()
        info, data, next_offset, complete = await asyncio.to_thread(
            self._backup_store.read_chunk,
            backup_id,
            offset=offset,
            max_bytes=max_bytes,
        )
        return NamespaceBackupChunk(
            backup=info,
            offset=offset,
            data=data,
            next_offset=next_offset,
            complete=complete,
        )

    async def begin_namespace_backup_upload(self) -> NamespaceBackupUpload:
        self._require_registry()
        return await asyncio.to_thread(self._backup_store.begin_upload)

    async def upload_namespace_backup_chunk(
        self,
        upload_id: str,
        *,
        offset: int,
        data: bytes,
    ) -> NamespaceBackupUpload:
        self._require_registry()
        return await asyncio.to_thread(
            self._backup_store.append_upload,
            upload_id,
            offset=offset,
            data=data,
        )

    async def complete_namespace_backup_upload(
        self,
        upload_id: str,
        *,
        expected_sha256: str = "",
    ) -> NamespaceBackupInfo:
        self._require_registry()
        return await asyncio.to_thread(
            self._backup_store.complete_upload,
            upload_id,
            expected_sha256=expected_sha256,
        )

    async def abort_namespace_backup_upload(self, upload_id: str) -> None:
        self._require_registry()
        await asyncio.to_thread(self._backup_store.abort_upload, upload_id)

    async def restore_namespace_from_backup(
        self,
        request: RestoreNamespaceBackupRequest,
    ) -> NamespaceInfo:
        registry = self._require_registry()
        manifest, _ = await asyncio.to_thread(
            self._backup_store.inspect_backup,
            request.backup_id,
        )
        create_request = CreateNamespaceRequest(
            namespace_id=request.target_namespace_id,
            quota=manifest.quota,
            config=manifest.config,
        )
        return await registry.restore_inactive_namespace(
            create_request,
            populate=lambda destination: self._backup_store.extract_backup(
                request.backup_id,
                destination,
            ),
        )

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
                details={
                    "namespace_id": namespace_id,
                    "status": namespace.status.value,
                },
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

    async def list_api_keys_page(
        self,
        namespace_id: str,
        *,
        page_size: int = 50,
        page_token: str = "",
    ) -> tuple[list[ApiKeyInfo], str]:
        return _paginate(
            await self.list_api_keys(namespace_id),
            page_size=page_size,
            page_token=page_token,
            token_kind=f"api_keys:{namespace_id}",
            item_key=lambda item: item.key_id,
        )

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
            payload = await self._execute_durable_write(
                context=context,
                operation="ingest_text",
                request=request,
                execute=lambda: self._ingest_text_locked(request),
            )
        return IngestTextResponse.model_validate(payload)

    async def _ingest_text_locked(
        self,
        request: IngestTextRequest,
    ) -> dict[str, object]:
        context = request.context
        async with self.runtime(context) as runtime:
            if not await _contains_external_memory(runtime, request.external_id):
                await self._enforce_write_admission(request)
            ingest = getattr(runtime, "ingest_text", None)
            if not callable(ingest):
                raise CapabilityUnavailableError(
                    "namespace runtime does not support text ingestion",
                    details={"namespace_id": context.namespace_id},
                )
            result = await ingest(**self._runtime_ingest_payload(request))
        return self._runtime_ingest_response(result)

    @staticmethod
    def _runtime_ingest_response(result: dict[str, object]) -> dict[str, object]:
        response = IngestTextResponse(
            stored_ids=tuple(str(item) for item in result.get("stored_ids", [])),
            skipped_ids=tuple(str(item) for item in result.get("skipped_ids", [])),
            fact_claim_ids=tuple(
                str(item) for item in result.get("fact_claim_ids", [])
            ),
            warnings=tuple(str(item) for item in result.get("warnings", [])),
            detail=str(result.get("detail", result.get("reason", "")) or ""),
        )
        return response.model_dump(mode="json")

    async def batch_ingest_text(
        self,
        request: BatchIngestTextRequest,
    ) -> BatchIngestTextResponse:
        context = request.context
        lock = self._application_write_locks.setdefault(
            context.namespace_id,
            asyncio.Lock(),
        )
        async with lock:
            payload = await self._execute_durable_write(
                context=context,
                operation="batch_ingest_text",
                request=request,
                execute=lambda: self._batch_ingest_text_locked(request),
            )
        return BatchIngestTextResponse.model_validate(payload)

    async def _batch_ingest_text_locked(
        self,
        request: BatchIngestTextRequest,
    ) -> dict[str, object]:
        context = request.context
        namespace = await self.get_namespace(context.namespace_id)
        async with self.runtime(context) as runtime:
            batch_ingest = getattr(runtime, "batch_ingest_text", None)
            if (
                callable(batch_ingest)
                and namespace.quota.max_storage_bytes is None
            ):
                raw_results = await batch_ingest(
                    items=[
                        self._runtime_ingest_payload(
                            IngestTextRequest(context=context, **item.model_dump())
                        )
                        for item in request.items
                    ]
                )
                if len(raw_results) != len(request.items):
                    raise RuntimeError(
                        "namespace runtime returned an invalid batch result count"
                    )
                return self._batch_ingest_response(raw_results, context=context)

        raw_results: list[dict[str, object] | Exception] = []
        for item in request.items:
            ingest_request = IngestTextRequest(
                context=request.context,
                **item.model_dump(),
            )
            try:
                raw_results.append(await self._ingest_text_locked(ingest_request))
            except Exception as exc:
                raw_results.append(exc)
        return self._batch_ingest_response(
            raw_results,
            context=context,
        )

    @staticmethod
    def _runtime_ingest_payload(request: IngestTextRequest) -> dict[str, object]:
        context = request.context
        return {
            "external_id": request.external_id,
            "source_type": request.source_type,
            "text": request.text,
            "chat_id": context.conversation_id or "",
            "person_ids": request.person_ids,
            "participants": request.participants,
            "timestamp": _optional_timestamp(request.observed_at),
            "time_start": _optional_timestamp(request.valid_from),
            "time_end": _optional_timestamp(request.valid_to),
            "tags": request.tags,
            "metadata": dict(request.metadata),
            "entities": request.entities,
            "relations": [item.model_dump(by_alias=True) for item in request.relations],
            "respect_filter": request.respect_filter,
            "user_id": context.user_id or "",
            "group_id": context.group_id or "",
        }

    @staticmethod
    def _batch_ingest_response(
        raw_results: list[dict[str, object] | Exception],
        *,
        context: RequestContext,
    ) -> dict[str, object]:
        results: list[BatchIngestItemResult] = []
        succeeded = 0
        for index, item in enumerate(raw_results):
            if isinstance(item, AMemorixError):
                results.append(BatchIngestItemResult(index=index, error=item.to_envelope()))
            elif isinstance(item, Exception):
                results.append(
                    BatchIngestItemResult(
                        index=index,
                        error=ErrorEnvelope(
                            code=ErrorCode.INTERNAL_ERROR,
                            message="batch item failed",
                            request_id=context.request_id,
                            trace_id=context.trace_id,
                            retryable=True,
                        ),
                    )
                )
            else:
                results.append(
                    BatchIngestItemResult(
                        index=index,
                        response=IngestTextResponse.model_validate(
                            AMemorixEngine._runtime_ingest_response(item)
                        ),
                    )
                )
                succeeded += 1
        response = BatchIngestTextResponse(
            results=tuple(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
        )
        return response.model_dump(mode="json")

    async def get_memory(self, request: GetMemoryRequest) -> GetMemoryResponse:
        async with self.runtime(request.context) as runtime:
            get_record = getattr(runtime, "get_memory_record", None)
            if not callable(get_record):
                raise CapabilityUnavailableError(
                    "namespace runtime does not support direct memory reads",
                    details={"namespace_id": request.context.namespace_id},
                )
            result = await get_record(
                memory_id=request.memory_id,
                external_id=request.external_id,
            )
        if not isinstance(result, dict):
            raise NotFoundError(
                "memory not found",
                details={
                    "namespace_id": request.context.namespace_id,
                    "memory_id": request.memory_id,
                    "external_id": request.external_id,
                },
            )
        return GetMemoryResponse(memory=_memory_record(result))

    async def delete_memory(
        self,
        request: DeleteMemoryRequest,
    ) -> DeleteMemoryResponse:
        context = request.context
        lock = self._application_write_locks.setdefault(
            context.namespace_id,
            asyncio.Lock(),
        )
        async with lock:
            selected = await self.get_memory(
                GetMemoryRequest(
                    context=context,
                    memory_id=request.memory_id,
                    external_id=request.external_id,
                )
            )
            memory_id = selected.memory.memory_id
            async with self.runtime(context) as runtime:
                delete = getattr(runtime, "memory_delete_admin", None)
                if not callable(delete):
                    raise CapabilityUnavailableError(
                        "namespace runtime does not support memory deletion",
                        details={"namespace_id": context.namespace_id},
                    )
                result = await delete(
                    action="execute",
                    mode="paragraph",
                    selector={"hash": memory_id},
                    requested_by=context.principal_id or context.agent_id or "api",
                    reason=request.reason,
                )
        error = str(result.get("error", "") or "").strip()
        deleted_count = int(result.get("deleted_paragraph_count", 0) or 0)
        if error and deleted_count == 0:
            raise NotFoundError(
                error,
                details={"namespace_id": context.namespace_id, "memory_id": memory_id},
            )
        return DeleteMemoryResponse(
            operation_id=str(result.get("operation_id", "") or ""),
            deleted_count=deleted_count,
            deleted_memory_ids=(memory_id,) if deleted_count else (),
        )

    async def submit_delete_by_source(
        self,
        request: DeleteBySourceRequest,
    ) -> JobInfo:
        if self._shutting_down:
            raise RuntimeError("AMemorixEngine is shutting down")
        namespace = await self.get_namespace(request.context.namespace_id)
        if namespace.status is not NamespaceStatus.ACTIVE:
            raise InvalidArgumentError(
                "source deletion requires an active namespace",
                details={
                    "namespace_id": namespace.namespace_id,
                    "status": namespace.status.value,
                },
            )
        job_id = uuid4().hex
        job = self._require_store().create_job(
            job_id=job_id,
            namespace_id=namespace.namespace_id,
            job_type=JobType.DELETE_BY_SOURCE,
            payload=request.model_dump(mode="json"),
            now=self._time(),
        )
        task = asyncio.create_task(
            self._run_delete_by_source(job_id, request),
            name=f"a-memorix-job-{job_id}",
        )
        self._job_tasks[job_id] = task
        task.add_done_callback(lambda _task: self._job_tasks.pop(job_id, None))
        return job

    async def get_job(self, namespace_id: str, job_id: str) -> JobInfo:
        await self.get_namespace(namespace_id)
        return self._require_store().get_job(namespace_id, job_id)

    async def list_jobs(self, namespace_id: str) -> list[JobInfo]:
        await self.get_namespace(namespace_id)
        return self._require_store().list_jobs(namespace_id)

    async def list_jobs_page(
        self,
        namespace_id: str,
        *,
        page_size: int = 50,
        page_token: str = "",
    ) -> tuple[list[JobInfo], str]:
        return _paginate(
            await self.list_jobs(namespace_id),
            page_size=page_size,
            page_token=page_token,
            token_kind=f"jobs:{namespace_id}",
            item_key=lambda item: item.job_id,
        )

    async def cancel_job(self, namespace_id: str, job_id: str) -> JobInfo:
        await self.get_namespace(namespace_id)
        return self._require_store().cancel_job(
            namespace_id,
            job_id,
            now=self._time(),
        )

    async def _run_delete_by_source(
        self,
        job_id: str,
        request: DeleteBySourceRequest,
    ) -> None:
        namespace_id = request.context.namespace_id
        store = self._require_store()
        started = store.start_job(namespace_id, job_id, now=self._time())
        if started.status is not JobStatus.RUNNING:
            return
        try:
            lock = self._application_write_locks.setdefault(
                namespace_id,
                asyncio.Lock(),
            )
            async with lock:
                async with self.runtime(request.context) as runtime:
                    delete = getattr(runtime, "memory_source_admin", None)
                    if not callable(delete):
                        raise CapabilityUnavailableError(
                            "namespace runtime does not support source deletion",
                            details={"namespace_id": namespace_id},
                        )
                    result = await delete(
                        action="delete",
                        source=request.source,
                        requested_by=request.context.principal_id
                        or request.context.agent_id
                        or "api",
                        reason=request.reason,
                    )
            error = str(result.get("error", "") or "").strip()
            if error and error != "未命中可删除内容":
                raise CapabilityUnavailableError(
                    "source deletion failed",
                    details={"namespace_id": namespace_id, "source": request.source},
                )
            store.complete_job(
                namespace_id,
                job_id,
                result={
                    "source": request.source,
                    "operation_id": str(result.get("operation_id", "") or ""),
                    "deleted_count": int(result.get("deleted_count", 0) or 0),
                    "deleted_memory_count": int(
                        result.get("deleted_paragraph_count", 0) or 0
                    ),
                },
                now=self._time(),
            )
        except AMemorixError as exc:
            store.fail_job(
                namespace_id,
                job_id,
                error=exc.to_envelope(),
                now=self._time(),
            )
        except Exception:
            store.fail_job(
                namespace_id,
                job_id,
                error=ErrorEnvelope(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="source deletion job failed",
                    request_id=request.context.request_id,
                    trace_id=request.context.trace_id,
                    retryable=True,
                ),
                now=self._time(),
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
        hits = tuple(
            _memory_hit(item)
            for item in result.get("hits", [])
            if isinstance(item, dict)
        )
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

    async def _execute_durable_write(
        self,
        *,
        context: RequestContext,
        operation: str,
        request: object,
        execute: Any,
    ) -> dict[str, object]:
        key = str(context.idempotency_key or "").strip()
        if not key:
            return dict(await execute())
        request_hash = _request_hash(request)
        now = self._time()
        expires_at = now + self._idempotency_retention_seconds
        store = self._require_store()
        decision = store.claim_idempotency(
            namespace_id=context.namespace_id,
            operation=operation,
            idempotency_key=key,
            request_hash=request_hash,
            now=now,
            expires_at=expires_at,
        )
        if not decision.execute:
            return dict(decision.response or {})
        try:
            response = dict(await execute())
        except BaseException:
            store.abandon_idempotency(
                namespace_id=context.namespace_id,
                operation=operation,
                idempotency_key=key,
                request_hash=request_hash,
            )
            raise
        completed_at = self._time()
        store.complete_idempotency(
            namespace_id=context.namespace_id,
            operation=operation,
            idempotency_key=key,
            request_hash=request_hash,
            response=response,
            now=completed_at,
            expires_at=completed_at + self._idempotency_retention_seconds,
        )
        return response

    async def _enforce_write_admission(self, request: IngestTextRequest) -> None:
        serialized = json.dumps(
            request.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        reservation = max(4096, len(serialized))
        await self._enforce_storage_reservation(request.context, reservation)

    async def _enforce_storage_reservation(
        self,
        context: RequestContext,
        reservation: int,
    ) -> None:
        namespace = await self.get_namespace(context.namespace_id)
        maximum = namespace.quota.max_storage_bytes
        if maximum is None:
            return
        health = await self.namespace_health(context.namespace_id)
        projected = health.resource_usage.storage_bytes + reservation
        if projected > maximum:
            raise NamespaceCapacityError(
                "namespace storage admission limit exceeded",
                details={
                    "namespace_id": context.namespace_id,
                    "storage_bytes": health.resource_usage.storage_bytes,
                    "reservation_bytes": reservation,
                    "max_storage_bytes": maximum,
                },
            )

    def _time(self) -> float:
        return self._clock.time() if self._clock is not None else time.time()


def _hash_api_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


def _request_hash(request: object) -> str:
    model_dump = getattr(request, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("durable request must be a Pydantic model")
    payload = model_dump(mode="json")
    context = payload.get("context") if isinstance(payload, dict) else None
    if isinstance(context, dict):
        for field in ("request_id", "trace_id", "principal_id", "idempotency_key"):
            context.pop(field, None)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


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


def _memory_record(item: dict[str, Any]) -> MemoryRecord:
    metadata = item.get("metadata")
    return MemoryRecord(
        memory_id=str(item.get("memory_id", "") or ""),
        external_id=str(item.get("external_id", "") or ""),
        source_type=str(item.get("source_type", "") or ""),
        source=str(item.get("source", "") or ""),
        content=str(item.get("content", "") or ""),
        metadata=metadata if isinstance(metadata, dict) else {},
        created_at=_optional_datetime(item.get("created_at")),
        updated_at=_optional_datetime(item.get("updated_at")),
        observed_at=_optional_datetime(item.get("observed_at")),
        valid_from=_optional_datetime(item.get("valid_from")),
        valid_to=_optional_datetime(item.get("valid_to")),
    )


def _optional_datetime(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _paginate(
    items: list[Any],
    *,
    page_size: int,
    page_token: str,
    token_kind: str,
    item_key: Any,
) -> tuple[list[Any], str]:
    if not 1 <= page_size <= 100:
        raise InvalidArgumentError("page_size must be between 1 and 100")
    start = 0
    token = str(page_token or "").strip()
    if token:
        after = _decode_page_token(token, token_kind)
        positions = [
            index for index, item in enumerate(items) if str(item_key(item)) == after
        ]
        if len(positions) != 1:
            raise InvalidArgumentError("page_token is stale or invalid")
        start = positions[0] + 1
    page = items[start : start + page_size]
    next_token = ""
    if page and start + len(page) < len(items):
        next_token = _encode_page_token(token_kind, str(item_key(page[-1])))
    return page, next_token


def _encode_page_token(kind: str, after: str) -> str:
    payload = json.dumps(
        {"kind": kind, "after": after},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_page_token(token: str, expected_kind: str) -> str:
    try:
        padding = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode((token + padding).encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise InvalidArgumentError("page_token is invalid") from exc
    if not isinstance(payload, dict) or payload.get("kind") != expected_kind:
        raise InvalidArgumentError("page_token does not belong to this collection")
    after = payload.get("after")
    if not isinstance(after, str) or not after:
        raise InvalidArgumentError("page_token is invalid")
    return after
