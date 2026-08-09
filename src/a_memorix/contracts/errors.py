"""Stable error codes and exception types for public boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(StrEnum):
    INVALID_ARGUMENT = "invalid_argument"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NAMESPACE_NOT_FOUND = "namespace_not_found"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INTEGRITY_ERROR = "integrity_error"
    MIGRATION_REQUIRED = "migration_required"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ErrorCode
    message: str
    request_id: str = ""
    trace_id: str = ""
    retryable: bool = False
    details: Mapping[str, object] = Field(default_factory=dict)


class AMemorixError(Exception):
    """Base exception carrying transport-independent error information."""

    code = ErrorCode.INTERNAL_ERROR
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        request_id: str = "",
        trace_id: str = "",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.trace_id = trace_id
        self.details = dict(details or {})

    def to_envelope(self) -> ErrorEnvelope:
        return ErrorEnvelope(
            code=self.code,
            message=str(self),
            request_id=self.request_id,
            trace_id=self.trace_id,
            retryable=self.retryable,
            details=self.details,
        )


class InvalidArgumentError(AMemorixError):
    code = ErrorCode.INVALID_ARGUMENT


class UnauthorizedError(AMemorixError):
    code = ErrorCode.UNAUTHORIZED


class ForbiddenError(AMemorixError):
    code = ErrorCode.FORBIDDEN


class NotFoundError(AMemorixError):
    code = ErrorCode.NOT_FOUND


class NamespaceNotFoundError(AMemorixError):
    code = ErrorCode.NAMESPACE_NOT_FOUND


class NamespaceConflictError(AMemorixError):
    code = ErrorCode.CONFLICT


class NamespaceStateError(AMemorixError):
    code = ErrorCode.CONFLICT


class NamespaceIntegrityError(AMemorixError):
    code = ErrorCode.INTEGRITY_ERROR


class MigrationRequiredError(AMemorixError):
    code = ErrorCode.MIGRATION_REQUIRED


class NamespaceCapacityError(AMemorixError):
    code = ErrorCode.CAPABILITY_UNAVAILABLE
    retryable = True


class NamespaceRuntimeError(AMemorixError):
    code = ErrorCode.CAPABILITY_UNAVAILABLE
    retryable = True


class CapabilityUnavailableError(AMemorixError):
    code = ErrorCode.CAPABILITY_UNAVAILABLE
    retryable = True


class RemoteAMemorixError(AMemorixError):
    """Error reconstructed from a remote ErrorDetail message."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        request_id: str = "",
        trace_id: str = "",
        retryable: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(
            message,
            request_id=request_id,
            trace_id=trace_id,
            details=details,
        )
