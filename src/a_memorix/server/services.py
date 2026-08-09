"""Generated gRPC service implementations backed by AMemorixEngine."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timezone
from typing import TypeVar

import grpc

from a_memorix.api.v1 import (
    auth_pb2,
    auth_pb2_grpc,
    backup_pb2,
    backup_pb2_grpc,
    job_pb2,
    job_pb2_grpc,
    memory_pb2,
    memory_pb2_grpc,
    namespace_pb2,
    namespace_pb2_grpc,
)
from a_memorix.contracts import (
    InvalidArgumentError,
    RequestContext,
    RestoreNamespaceBackupRequest,
)
from a_memorix.engine import AMemorixEngine

from .auth import AuthPrincipal, GrpcAuthPolicy
from .errors import abort_for_exception
from .mapping import (
    api_key_info_to_proto,
    batch_ingest_request_from_proto,
    batch_ingest_response_to_proto,
    create_namespace_request_from_proto,
    delete_by_source_request_from_proto,
    delete_memory_request_from_proto,
    delete_memory_response_to_proto,
    get_memory_request_from_proto,
    get_memory_response_to_proto,
    ingest_request_from_proto,
    ingest_response_to_proto,
    job_info_to_proto,
    namespace_backup_info_to_proto,
    namespace_capabilities_to_proto,
    namespace_health_to_proto,
    namespace_info_to_proto,
    search_request_from_proto,
    search_response_to_proto,
    update_namespace_config_request_from_proto,
)


ResponseT = TypeVar("ResponseT")


class _ServiceBase:
    def __init__(self, engine: AMemorixEngine, auth: GrpcAuthPolicy) -> None:
        self._engine = engine
        self._auth = auth

    @staticmethod
    async def _invoke(
        context: grpc.aio.ServicerContext,
        operation: Callable[[], Awaitable[ResponseT]],
        *,
        request_context: (
            RequestContext | Callable[[], RequestContext | None] | None
        ) = None,
    ) -> ResponseT:
        try:
            return await operation()
        except Exception as error:
            await abort_for_exception(
                context,
                error,
                request_context=(
                    request_context()
                    if callable(request_context)
                    else request_context
                ),
            )
            raise RuntimeError("gRPC abort unexpectedly returned") from error


class NamespaceGrpcService(_ServiceBase, namespace_pb2_grpc.NamespaceServiceServicer):
    async def CreateNamespace(self, request, context):
        async def operation() -> namespace_pb2.CreateNamespaceResponse:
            self._auth.require_admin(context)
            namespace = await self._engine.create_namespace(
                create_namespace_request_from_proto(request)
            )
            return namespace_pb2.CreateNamespaceResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)

    async def GetNamespace(self, request, context):
        async def operation() -> namespace_pb2.GetNamespaceResponse:
            self._auth.require_namespace(context, request.namespace_id)
            namespace = await self._engine.get_namespace(request.namespace_id)
            return namespace_pb2.GetNamespaceResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)

    async def ListNamespaces(self, request, context):
        async def operation() -> namespace_pb2.ListNamespacesResponse:
            self._auth.require_admin(context)
            namespaces, next_page_token = await self._engine.list_namespaces_page(
                page_size=request.page_size if request.HasField("page_size") else 50,
                page_token=request.page_token,
            )
            return namespace_pb2.ListNamespacesResponse(
                namespaces=[namespace_info_to_proto(item) for item in namespaces],
                next_page_token=next_page_token,
            )

        return await self._invoke(context, operation)

    async def UpdateNamespaceConfig(self, request, context):
        async def operation() -> namespace_pb2.UpdateNamespaceConfigResponse:
            self._auth.require_admin(context)
            namespace = await self._engine.update_namespace_config(
                update_namespace_config_request_from_proto(request)
            )
            return namespace_pb2.UpdateNamespaceConfigResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)

    async def GetNamespaceCapabilities(self, request, context):
        async def operation() -> namespace_pb2.GetNamespaceCapabilitiesResponse:
            self._auth.require_namespace(context, request.namespace_id)
            capabilities = await self._engine.get_namespace_capabilities(
                request.namespace_id
            )
            return namespace_pb2.GetNamespaceCapabilitiesResponse(
                capabilities=namespace_capabilities_to_proto(capabilities)
            )

        return await self._invoke(context, operation)

    async def DisableNamespace(self, request, context):
        async def operation() -> namespace_pb2.DisableNamespaceResponse:
            self._auth.require_admin(context)
            namespace = await self._engine.disable_namespace(request.namespace_id)
            return namespace_pb2.DisableNamespaceResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)

    async def EnableNamespace(self, request, context):
        async def operation() -> namespace_pb2.EnableNamespaceResponse:
            self._auth.require_admin(context)
            namespace = await self._engine.enable_namespace(request.namespace_id)
            return namespace_pb2.EnableNamespaceResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)

    async def DeleteNamespace(self, request, context):
        async def operation() -> namespace_pb2.DeleteNamespaceResponse:
            self._auth.require_admin(context)
            namespace = await self._engine.delete_namespace(request.namespace_id)
            return namespace_pb2.DeleteNamespaceResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)

    async def RestoreNamespace(self, request, context):
        async def operation() -> namespace_pb2.RestoreNamespaceResponse:
            self._auth.require_admin(context)
            namespace = await self._engine.restore_namespace(request.namespace_id)
            return namespace_pb2.RestoreNamespaceResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)

    async def PurgeNamespace(self, request, context):
        async def operation() -> namespace_pb2.PurgeNamespaceResponse:
            self._auth.require_admin(context)
            await self._engine.purge_namespace(request.namespace_id)
            return namespace_pb2.PurgeNamespaceResponse()

        return await self._invoke(context, operation)

    async def GetNamespaceHealth(self, request, context):
        async def operation() -> namespace_pb2.GetNamespaceHealthResponse:
            self._auth.require_namespace(context, request.namespace_id)
            health = await self._engine.namespace_health(request.namespace_id)
            return namespace_pb2.GetNamespaceHealthResponse(
                health=namespace_health_to_proto(health)
            )

        return await self._invoke(context, operation)


class BackupGrpcService(_ServiceBase, backup_pb2_grpc.BackupServiceServicer):
    async def CreateNamespaceBackup(self, request, context):
        async def operation() -> backup_pb2.CreateNamespaceBackupResponse:
            self._auth.require_admin(context)
            backup = await self._engine.create_namespace_backup(request.namespace_id)
            return backup_pb2.CreateNamespaceBackupResponse(
                backup=namespace_backup_info_to_proto(backup)
            )

        return await self._invoke(context, operation)

    async def GetNamespaceBackup(self, request, context):
        async def operation() -> backup_pb2.GetNamespaceBackupResponse:
            self._auth.require_admin(context)
            backup = await self._engine.get_namespace_backup(request.backup_id)
            return backup_pb2.GetNamespaceBackupResponse(
                backup=namespace_backup_info_to_proto(backup)
            )

        return await self._invoke(context, operation)

    async def ListNamespaceBackups(self, request, context):
        async def operation() -> backup_pb2.ListNamespaceBackupsResponse:
            self._auth.require_admin(context)
            backups, next_page_token = (
                await self._engine.list_namespace_backups_page(
                    source_namespace_id=request.source_namespace_id,
                    page_size=(
                        request.page_size if request.HasField("page_size") else 50
                    ),
                    page_token=request.page_token,
                )
            )
            return backup_pb2.ListNamespaceBackupsResponse(
                backups=[namespace_backup_info_to_proto(item) for item in backups],
                next_page_token=next_page_token,
            )

        return await self._invoke(context, operation)

    async def DeleteNamespaceBackup(self, request, context):
        async def operation() -> backup_pb2.DeleteNamespaceBackupResponse:
            self._auth.require_admin(context)
            await self._engine.delete_namespace_backup(request.backup_id)
            return backup_pb2.DeleteNamespaceBackupResponse()

        return await self._invoke(context, operation)

    async def DownloadNamespaceBackup(self, request, context):
        async def operation() -> backup_pb2.DownloadNamespaceBackupResponse:
            self._auth.require_admin(context)
            chunk = await self._engine.download_namespace_backup(
                request.backup_id,
                offset=request.offset if request.HasField("offset") else 0,
                max_bytes=(
                    request.max_bytes if request.HasField("max_bytes") else 256 * 1024
                ),
            )
            return backup_pb2.DownloadNamespaceBackupResponse(
                backup=namespace_backup_info_to_proto(chunk.backup),
                offset=chunk.offset,
                data=chunk.data,
                next_offset=chunk.next_offset,
                complete=chunk.complete,
            )

        return await self._invoke(context, operation)

    async def BeginNamespaceBackupUpload(self, request, context):
        del request

        async def operation() -> backup_pb2.BeginNamespaceBackupUploadResponse:
            self._auth.require_admin(context)
            upload = await self._engine.begin_namespace_backup_upload()
            return backup_pb2.BeginNamespaceBackupUploadResponse(
                upload_id=upload.upload_id,
                next_offset=upload.next_offset,
            )

        return await self._invoke(context, operation)

    async def UploadNamespaceBackupChunk(self, request, context):
        async def operation() -> backup_pb2.UploadNamespaceBackupChunkResponse:
            self._auth.require_admin(context)
            upload = await self._engine.upload_namespace_backup_chunk(
                request.upload_id,
                offset=request.offset,
                data=request.data,
            )
            return backup_pb2.UploadNamespaceBackupChunkResponse(
                upload_id=upload.upload_id,
                next_offset=upload.next_offset,
            )

        return await self._invoke(context, operation)

    async def CompleteNamespaceBackupUpload(self, request, context):
        async def operation() -> backup_pb2.CompleteNamespaceBackupUploadResponse:
            self._auth.require_admin(context)
            backup = await self._engine.complete_namespace_backup_upload(
                request.upload_id,
                expected_sha256=request.expected_sha256,
            )
            return backup_pb2.CompleteNamespaceBackupUploadResponse(
                backup=namespace_backup_info_to_proto(backup)
            )

        return await self._invoke(context, operation)

    async def AbortNamespaceBackupUpload(self, request, context):
        async def operation() -> backup_pb2.AbortNamespaceBackupUploadResponse:
            self._auth.require_admin(context)
            await self._engine.abort_namespace_backup_upload(request.upload_id)
            return backup_pb2.AbortNamespaceBackupUploadResponse()

        return await self._invoke(context, operation)

    async def RestoreNamespaceFromBackup(self, request, context):
        async def operation() -> backup_pb2.RestoreNamespaceFromBackupResponse:
            self._auth.require_admin(context)
            namespace = await self._engine.restore_namespace_from_backup(
                RestoreNamespaceBackupRequest(
                    backup_id=request.backup_id,
                    target_namespace_id=request.target_namespace_id,
                )
            )
            return backup_pb2.RestoreNamespaceFromBackupResponse(
                namespace=namespace_info_to_proto(namespace)
            )

        return await self._invoke(context, operation)


class AuthGrpcService(_ServiceBase, auth_pb2_grpc.AuthServiceServicer):
    async def CreateApiKey(self, request, context):
        async def operation() -> auth_pb2.CreateApiKeyResponse:
            self._auth.require_admin(context)
            expires_at = (
                request.expires_at.ToDatetime(tzinfo=timezone.utc)
                if request.HasField("expires_at")
                else None
            )
            created = await self._engine.create_api_key(
                request.namespace_id,
                label=request.label,
                expires_at=expires_at,
            )
            return auth_pb2.CreateApiKeyResponse(
                api_key=api_key_info_to_proto(created.api_key),
                secret=created.secret,
            )

        return await self._invoke(context, operation)

    async def ListApiKeys(self, request, context):
        async def operation() -> auth_pb2.ListApiKeysResponse:
            self._auth.require_admin(context)
            keys, next_page_token = await self._engine.list_api_keys_page(
                request.namespace_id,
                page_size=request.page_size if request.HasField("page_size") else 50,
                page_token=request.page_token,
            )
            return auth_pb2.ListApiKeysResponse(
                api_keys=[api_key_info_to_proto(item) for item in keys],
                next_page_token=next_page_token,
            )

        return await self._invoke(context, operation)

    async def RevokeApiKey(self, request, context):
        async def operation() -> auth_pb2.RevokeApiKeyResponse:
            self._auth.require_admin(context)
            await self._engine.revoke_api_key(request.namespace_id, request.key_id)
            return auth_pb2.RevokeApiKeyResponse()

        return await self._invoke(context, operation)


class MemoryGrpcService(_ServiceBase, memory_pb2_grpc.MemoryServiceServicer):
    async def IngestText(self, request, context):
        request_context: RequestContext | None = None

        async def operation() -> memory_pb2.IngestTextResponse:
            nonlocal request_context
            application_request = ingest_request_from_proto(request)
            request_context = application_request.context
            principal = self._auth.require_namespace(
                context,
                application_request.context.namespace_id,
            )
            request_context = _bind_authenticated_context(
                application_request.context,
                context,
                principal,
                wire_context=request.context,
            )
            application_request = application_request.model_copy(
                update={"context": request_context}
            )
            result = await self._engine.ingest_text(application_request)
            return ingest_response_to_proto(result)

        return await self._invoke(
            context,
            operation,
            request_context=lambda: request_context,
        )

    async def BatchIngestText(self, request, context):
        request_context: RequestContext | None = None

        async def operation() -> memory_pb2.BatchIngestTextResponse:
            nonlocal request_context
            application_request = batch_ingest_request_from_proto(request)
            request_context = application_request.context
            principal = self._auth.require_namespace(
                context,
                application_request.context.namespace_id,
            )
            request_context = _bind_authenticated_context(
                application_request.context,
                context,
                principal,
                wire_context=request.context,
            )
            application_request = application_request.model_copy(
                update={"context": request_context}
            )
            result = await self._engine.batch_ingest_text(application_request)
            return batch_ingest_response_to_proto(result)

        return await self._invoke(
            context,
            operation,
            request_context=lambda: request_context,
        )

    async def GetMemory(self, request, context):
        request_context: RequestContext | None = None

        async def operation() -> memory_pb2.GetMemoryResponse:
            nonlocal request_context
            application_request = get_memory_request_from_proto(request)
            request_context = application_request.context
            principal = self._auth.require_namespace(
                context,
                application_request.context.namespace_id,
            )
            request_context = _bind_authenticated_context(
                application_request.context,
                context,
                principal,
                wire_context=request.context,
            )
            application_request = application_request.model_copy(
                update={"context": request_context}
            )
            result = await self._engine.get_memory(application_request)
            return get_memory_response_to_proto(result)

        return await self._invoke(
            context,
            operation,
            request_context=lambda: request_context,
        )

    async def DeleteMemory(self, request, context):
        request_context: RequestContext | None = None

        async def operation() -> memory_pb2.DeleteMemoryResponse:
            nonlocal request_context
            application_request = delete_memory_request_from_proto(request)
            request_context = application_request.context
            principal = self._auth.require_namespace(
                context,
                application_request.context.namespace_id,
            )
            request_context = _bind_authenticated_context(
                application_request.context,
                context,
                principal,
                wire_context=request.context,
            )
            application_request = application_request.model_copy(
                update={"context": request_context}
            )
            result = await self._engine.delete_memory(application_request)
            return delete_memory_response_to_proto(result)

        return await self._invoke(
            context,
            operation,
            request_context=lambda: request_context,
        )

    async def SearchMemory(self, request, context):
        request_context: RequestContext | None = None

        async def operation() -> memory_pb2.SearchMemoryResponse:
            nonlocal request_context
            application_request = search_request_from_proto(request)
            request_context = application_request.context
            principal = self._auth.require_namespace(
                context,
                application_request.context.namespace_id,
            )
            request_context = _bind_authenticated_context(
                application_request.context,
                context,
                principal,
                wire_context=request.context,
            )
            application_request = application_request.model_copy(
                update={"context": request_context}
            )
            result = await self._engine.search_memory(application_request)
            return search_response_to_proto(result)

        return await self._invoke(
            context,
            operation,
            request_context=lambda: request_context,
        )


class JobGrpcService(_ServiceBase, job_pb2_grpc.JobServiceServicer):
    async def SubmitDeleteBySource(self, request, context):
        request_context: RequestContext | None = None

        async def operation() -> job_pb2.SubmitDeleteBySourceResponse:
            nonlocal request_context
            application_request = delete_by_source_request_from_proto(request)
            request_context = application_request.context
            principal = self._auth.require_namespace(
                context,
                application_request.context.namespace_id,
            )
            request_context = _bind_authenticated_context(
                application_request.context,
                context,
                principal,
                wire_context=request.context,
            )
            application_request = application_request.model_copy(
                update={"context": request_context}
            )
            job = await self._engine.submit_delete_by_source(application_request)
            return job_pb2.SubmitDeleteBySourceResponse(job=job_info_to_proto(job))

        return await self._invoke(
            context,
            operation,
            request_context=lambda: request_context,
        )

    async def GetJob(self, request, context):
        async def operation() -> job_pb2.GetJobResponse:
            self._auth.require_namespace(context, request.namespace_id)
            job = await self._engine.get_job(request.namespace_id, request.job_id)
            return job_pb2.GetJobResponse(job=job_info_to_proto(job))

        return await self._invoke(context, operation)

    async def ListJobs(self, request, context):
        async def operation() -> job_pb2.ListJobsResponse:
            self._auth.require_namespace(context, request.namespace_id)
            jobs, next_page_token = await self._engine.list_jobs_page(
                request.namespace_id,
                page_size=request.page_size if request.HasField("page_size") else 50,
                page_token=request.page_token,
            )
            return job_pb2.ListJobsResponse(
                jobs=[job_info_to_proto(item) for item in jobs],
                next_page_token=next_page_token,
            )

        return await self._invoke(context, operation)

    async def CancelJob(self, request, context):
        async def operation() -> job_pb2.CancelJobResponse:
            self._auth.require_namespace(context, request.namespace_id)
            job = await self._engine.cancel_job(request.namespace_id, request.job_id)
            return job_pb2.CancelJobResponse(job=job_info_to_proto(job))

        return await self._invoke(context, operation)


def register_services(
    server: grpc.aio.Server,
    engine: AMemorixEngine,
    auth: GrpcAuthPolicy,
) -> None:
    namespace_pb2_grpc.add_NamespaceServiceServicer_to_server(
        NamespaceGrpcService(engine, auth),
        server,
    )
    backup_pb2_grpc.add_BackupServiceServicer_to_server(
        BackupGrpcService(engine, auth),
        server,
    )
    auth_pb2_grpc.add_AuthServiceServicer_to_server(
        AuthGrpcService(engine, auth),
        server,
    )
    memory_pb2_grpc.add_MemoryServiceServicer_to_server(
        MemoryGrpcService(engine, auth),
        server,
    )
    job_pb2_grpc.add_JobServiceServicer_to_server(
        JobGrpcService(engine, auth),
        server,
    )


def _bind_authenticated_context(
    value: RequestContext,
    grpc_context: grpc.aio.ServicerContext,
    principal: AuthPrincipal,
    *,
    wire_context: object | None = None,
) -> RequestContext:
    metadata = grpc_context.invocation_metadata()
    updates = {"principal_id": principal.principal_id}
    for header, field in (
        ("idempotency-key", "idempotency_key"),
        ("x-request-id", "request_id"),
        ("x-trace-id", "trace_id"),
    ):
        values = [
            str(item.value).strip()
            for item in metadata
            if item.key.lower() == header
        ]
        if len(values) > 1:
            raise InvalidArgumentError(f"multiple {header} headers are not allowed")
        header_value = values[0] if values else ""
        body_value = str(getattr(wire_context, field, "") or "")
        if header_value and body_value and header_value != body_value:
            raise InvalidArgumentError(
                f"{header} header conflicts with request context",
                details={"namespace_id": value.namespace_id},
            )
        updates[field] = header_value or body_value or getattr(value, field)
    return value.model_copy(
        update=updates,
    )
