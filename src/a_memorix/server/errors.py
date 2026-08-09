"""Canonical A_memorix error mapping for gRPC and gRPC-Gateway."""

from __future__ import annotations

from collections.abc import Mapping

import grpc
from google.protobuf import any_pb2
from google.rpc import status_pb2
from grpc_status import rpc_status
from pydantic import ValidationError

from a_memorix.api.v1 import common_pb2
from a_memorix.contracts import (
    AMemorixError,
    ErrorCode,
    InvalidArgumentError,
    RequestContext,
)
from a_memorix.logging import get_logger

from .mapping import mapping_to_struct


logger = get_logger("A_Memorix.GrpcServer")

_GRPC_CODE = {
    ErrorCode.INVALID_ARGUMENT: grpc.StatusCode.INVALID_ARGUMENT,
    ErrorCode.UNAUTHORIZED: grpc.StatusCode.UNAUTHENTICATED,
    ErrorCode.FORBIDDEN: grpc.StatusCode.PERMISSION_DENIED,
    ErrorCode.NAMESPACE_NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    ErrorCode.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    ErrorCode.CONFLICT: grpc.StatusCode.ALREADY_EXISTS,
    ErrorCode.INTEGRITY_ERROR: grpc.StatusCode.DATA_LOSS,
    ErrorCode.MIGRATION_REQUIRED: grpc.StatusCode.FAILED_PRECONDITION,
    ErrorCode.CAPABILITY_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    ErrorCode.TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
    ErrorCode.CANCELLED: grpc.StatusCode.CANCELLED,
    ErrorCode.INTERNAL_ERROR: grpc.StatusCode.INTERNAL,
}


async def abort_for_exception(
    context: grpc.aio.ServicerContext,
    error: Exception,
    *,
    request_context: RequestContext | None = None,
) -> None:
    public_error = _public_error(error)
    request_id = public_error.request_id or (
        request_context.request_id if request_context is not None else ""
    ) or _metadata_value(context, "x-request-id")
    trace_id = public_error.trace_id or (
        request_context.trace_id if request_context is not None else ""
    ) or _metadata_value(context, "x-trace-id")
    detail = common_pb2.ErrorDetail(
        code=public_error.code.value,
        request_id=request_id,
        trace_id=trace_id,
        retryable=public_error.retryable,
        details=mapping_to_struct(_safe_details(public_error.details)),
        message=str(public_error),
    )
    packed = any_pb2.Any()
    packed.Pack(detail)
    status = status_pb2.Status(
        code=_GRPC_CODE[public_error.code].value[0],
        message=str(public_error),
        details=[packed],
    )
    await context.abort_with_status(rpc_status.to_status(status))


def _public_error(error: Exception) -> AMemorixError:
    if isinstance(error, AMemorixError):
        return error
    if isinstance(error, ValidationError):
        return InvalidArgumentError(
            "request validation failed",
            details={"errors": error.errors(include_url=False)},
        )
    if isinstance(error, ValueError):
        return InvalidArgumentError(str(error))
    logger.exception("unhandled gRPC service error", exc_info=error)
    return AMemorixError("internal server error")


def _safe_details(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): item for key, item in value.items()}


def _metadata_value(context: grpc.aio.ServicerContext, key: str) -> str:
    values = [
        str(item.value).strip()
        for item in context.invocation_metadata()
        if item.key.lower() == key
    ]
    return values[0] if len(values) == 1 else ""
