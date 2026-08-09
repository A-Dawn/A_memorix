"""Public request, response and error contracts."""

from .context import NamespaceId, RequestContext
from .errors import (
    AMemorixError,
    ErrorCode,
    ErrorEnvelope,
    InvalidArgumentError,
    MigrationRequiredError,
    NamespaceCapacityError,
    NamespaceConflictError,
    NamespaceIntegrityError,
    NamespaceNotFoundError,
    NamespaceRuntimeError,
    NamespaceStateError,
)
from .namespaces import (
    CreateNamespaceRequest,
    NamespaceHealth,
    NamespaceInfo,
    NamespaceQuota,
    NamespaceResourceUsage,
    NamespaceRuntimeState,
    NamespaceStatus,
)

__all__ = [
    "AMemorixError",
    "CreateNamespaceRequest",
    "ErrorCode",
    "ErrorEnvelope",
    "InvalidArgumentError",
    "MigrationRequiredError",
    "NamespaceCapacityError",
    "NamespaceConflictError",
    "NamespaceHealth",
    "NamespaceId",
    "NamespaceInfo",
    "NamespaceIntegrityError",
    "NamespaceNotFoundError",
    "NamespaceQuota",
    "NamespaceResourceUsage",
    "NamespaceRuntimeError",
    "NamespaceRuntimeState",
    "NamespaceStateError",
    "NamespaceStatus",
    "RequestContext",
]
