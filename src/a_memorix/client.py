"""Async Python client for the public A_memorix gRPC API."""

from __future__ import annotations

from typing import Any

import grpc
from google.rpc import status_pb2

from a_memorix.api.v1 import (
    auth_pb2,
    auth_pb2_grpc,
    common_pb2,
    job_pb2,
    job_pb2_grpc,
    memory_pb2,
    memory_pb2_grpc,
    namespace_pb2,
    namespace_pb2_grpc,
)
from a_memorix.contracts import ErrorCode, RemoteAMemorixError
from a_memorix.server.mapping import struct_to_mapping


class AMemorixClient:
    def __init__(
        self,
        target: str,
        *,
        api_key: str = "",
        credentials: grpc.ChannelCredentials | None = None,
        timeout: float | None = 30.0,
        maximum_message_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        options = (
            ("grpc.max_receive_message_length", maximum_message_bytes),
            ("grpc.max_send_message_length", maximum_message_bytes),
        )
        if credentials is None:
            self._channel = grpc.aio.insecure_channel(target, options=options)
        else:
            self._channel = grpc.aio.secure_channel(
                target,
                credentials,
                options=options,
            )
        self._metadata = (
            (("authorization", f"Bearer {api_key}"),) if api_key else ()
        )
        self._timeout = timeout
        self.namespaces = namespace_pb2_grpc.NamespaceServiceStub(self._channel)
        self.auth = auth_pb2_grpc.AuthServiceStub(self._channel)
        self.memory = memory_pb2_grpc.MemoryServiceStub(self._channel)
        self.jobs = job_pb2_grpc.JobServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> "AMemorixClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        await self.close()

    async def create_namespace(
        self,
        request: namespace_pb2.CreateNamespaceRequest,
    ) -> namespace_pb2.CreateNamespaceResponse:
        return await self._call(self.namespaces.CreateNamespace, request)

    async def get_namespace(
        self,
        request: namespace_pb2.GetNamespaceRequest,
    ) -> namespace_pb2.GetNamespaceResponse:
        return await self._call(self.namespaces.GetNamespace, request)

    async def list_namespaces(
        self,
        request: namespace_pb2.ListNamespacesRequest | None = None,
    ) -> namespace_pb2.ListNamespacesResponse:
        return await self._call(
            self.namespaces.ListNamespaces,
            request or namespace_pb2.ListNamespacesRequest(),
        )

    async def disable_namespace(self, request):
        return await self._call(self.namespaces.DisableNamespace, request)

    async def enable_namespace(self, request):
        return await self._call(self.namespaces.EnableNamespace, request)

    async def delete_namespace(self, request):
        return await self._call(self.namespaces.DeleteNamespace, request)

    async def restore_namespace(self, request):
        return await self._call(self.namespaces.RestoreNamespace, request)

    async def purge_namespace(self, request):
        return await self._call(self.namespaces.PurgeNamespace, request)

    async def get_namespace_health(self, request):
        return await self._call(self.namespaces.GetNamespaceHealth, request)

    async def update_namespace_config(self, request):
        return await self._call(self.namespaces.UpdateNamespaceConfig, request)

    async def get_namespace_capabilities(self, request):
        return await self._call(self.namespaces.GetNamespaceCapabilities, request)

    async def create_api_key(
        self,
        request: auth_pb2.CreateApiKeyRequest,
    ) -> auth_pb2.CreateApiKeyResponse:
        return await self._call(self.auth.CreateApiKey, request)

    async def list_api_keys(
        self,
        request: auth_pb2.ListApiKeysRequest,
    ) -> auth_pb2.ListApiKeysResponse:
        return await self._call(self.auth.ListApiKeys, request)

    async def revoke_api_key(
        self,
        request: auth_pb2.RevokeApiKeyRequest,
    ) -> auth_pb2.RevokeApiKeyResponse:
        return await self._call(self.auth.RevokeApiKey, request)

    async def ingest_text(
        self,
        request: memory_pb2.IngestTextRequest,
        *,
        idempotency_key: str = "",
    ) -> memory_pb2.IngestTextResponse:
        metadata = self._metadata
        if idempotency_key:
            metadata = (*metadata, ("idempotency-key", idempotency_key))
        return await self._call(self.memory.IngestText, request, metadata=metadata)

    async def search_memory(
        self,
        request: memory_pb2.SearchMemoryRequest,
    ) -> memory_pb2.SearchMemoryResponse:
        return await self._call(self.memory.SearchMemory, request)

    async def batch_ingest_text(
        self,
        request: memory_pb2.BatchIngestTextRequest,
        *,
        idempotency_key: str = "",
    ) -> memory_pb2.BatchIngestTextResponse:
        metadata = self._metadata
        if idempotency_key:
            metadata = (*metadata, ("idempotency-key", idempotency_key))
        return await self._call(
            self.memory.BatchIngestText,
            request,
            metadata=metadata,
        )

    async def get_memory(
        self,
        request: memory_pb2.GetMemoryRequest,
    ) -> memory_pb2.GetMemoryResponse:
        return await self._call(self.memory.GetMemory, request)

    async def delete_memory(
        self,
        request: memory_pb2.DeleteMemoryRequest,
    ) -> memory_pb2.DeleteMemoryResponse:
        return await self._call(self.memory.DeleteMemory, request)

    async def submit_delete_by_source(
        self,
        request: job_pb2.SubmitDeleteBySourceRequest,
    ) -> job_pb2.SubmitDeleteBySourceResponse:
        return await self._call(self.jobs.SubmitDeleteBySource, request)

    async def get_job(self, request: job_pb2.GetJobRequest) -> job_pb2.GetJobResponse:
        return await self._call(self.jobs.GetJob, request)

    async def list_jobs(
        self,
        request: job_pb2.ListJobsRequest,
    ) -> job_pb2.ListJobsResponse:
        return await self._call(self.jobs.ListJobs, request)

    async def cancel_job(
        self,
        request: job_pb2.CancelJobRequest,
    ) -> job_pb2.CancelJobResponse:
        return await self._call(self.jobs.CancelJob, request)

    async def _call(
        self,
        method: Any,
        request: Any,
        *,
        metadata: tuple[tuple[str, str], ...] | None = None,
    ) -> Any:
        try:
            return await method(
                request,
                metadata=self._metadata if metadata is None else metadata,
                timeout=self._timeout,
            )
        except grpc.aio.AioRpcError as error:
            raise _remote_error(error) from error


def _remote_error(error: grpc.aio.AioRpcError) -> RemoteAMemorixError:
    detail = _error_detail(error)
    if detail is None:
        return RemoteAMemorixError(
            ErrorCode.INTERNAL_ERROR,
            error.details() or "remote gRPC request failed",
            details={"grpc_status": error.code().name},
        )
    try:
        code = ErrorCode(detail.code)
    except ValueError:
        code = ErrorCode.INTERNAL_ERROR
    return RemoteAMemorixError(
        code,
        error.details() or "remote gRPC request failed",
        request_id=detail.request_id,
        trace_id=detail.trace_id,
        retryable=detail.retryable,
        details=struct_to_mapping(detail.details),
    )


def _error_detail(error: grpc.aio.AioRpcError) -> common_pb2.ErrorDetail | None:
    metadata = error.trailing_metadata() or ()
    for item in metadata:
        key = item.key if hasattr(item, "key") else item[0]
        value = item.value if hasattr(item, "value") else item[1]
        if key != "grpc-status-details-bin" or not isinstance(value, bytes):
            continue
        status = status_pb2.Status()
        status.ParseFromString(value)
        for packed in status.details:
            detail = common_pb2.ErrorDetail()
            if packed.Unpack(detail):
                return detail
    return None
