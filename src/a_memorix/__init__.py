"""Public API for the A_memorix memory engine."""

from .contracts import (
    AMemorixError,
    ApiKeyInfo,
    CapabilityUnavailableError,
    CreateNamespaceRequest,
    CreatedApiKey,
    ErrorCode,
    ErrorEnvelope,
    ForbiddenError,
    IngestTextRequest,
    IngestTextResponse,
    InvalidArgumentError,
    MigrationRequiredError,
    MemoryHit,
    NotFoundError,
    NamespaceCapacityError,
    NamespaceConflictError,
    NamespaceHealth,
    NamespaceInfo,
    NamespaceIntegrityError,
    NamespaceNotFoundError,
    NamespaceQuota,
    NamespaceRuntimeError,
    NamespaceStateError,
    NamespaceStatus,
    RequestContext,
    RelationInput,
    RemoteAMemorixError,
    SearchMemoryRequest,
    SearchMemoryResponse,
    SearchMode,
    UnauthorizedError,
)
from .core.runtime import KernelSearchRequest, SDKMemoryKernel
from .engine import AMemorixEngine
from .ports import (
    Clock,
    EmbeddingProvider,
    IdentityRecord,
    IdentityResolver,
    LLMProvider,
    LLMRequest,
    LLMResult,
    MessageRecord,
    MessageSource,
    NamespaceHostPorts,
    SystemClock,
)

__all__ = [
    "AMemorixEngine",
    "AMemorixError",
    "AMemorixClient",
    "ApiKeyInfo",
    "CapabilityUnavailableError",
    "Clock",
    "CreateNamespaceRequest",
    "CreatedApiKey",
    "create_fixed_namespace_mcp",
    "EmbeddingProvider",
    "ErrorCode",
    "ErrorEnvelope",
    "ForbiddenError",
    "IngestTextRequest",
    "IngestTextResponse",
    "InvalidArgumentError",
    "IdentityRecord",
    "IdentityResolver",
    "KernelSearchRequest",
    "LLMProvider",
    "LLMRequest",
    "LLMResult",
    "MessageRecord",
    "MessageSource",
    "MigrationRequiredError",
    "MemoryHit",
    "NamespaceCapacityError",
    "NamespaceConflictError",
    "NamespaceHealth",
    "NamespaceHostPorts",
    "NamespaceInfo",
    "NamespaceIntegrityError",
    "NamespaceNotFoundError",
    "NamespaceQuota",
    "NamespaceRuntimeError",
    "NamespaceStateError",
    "NamespaceStatus",
    "NotFoundError",
    "RequestContext",
    "RelationInput",
    "RemoteAMemorixError",
    "SDKMemoryKernel",
    "SearchMemoryRequest",
    "SearchMemoryResponse",
    "SearchMode",
    "SystemClock",
    "UnauthorizedError",
    "__version__",
]
__version__ = "2.0.0a1"


def __getattr__(name: str):
    if name == "AMemorixClient":
        try:
            from .client import AMemorixClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "AMemorixClient requires the 'rpc' extra: pip install 'a-memorix[rpc]'"
            ) from exc
        return AMemorixClient
    if name == "create_fixed_namespace_mcp":
        try:
            from .mcp_server import create_fixed_namespace_mcp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MCP support requires the 'mcp' extra: pip install 'a-memorix[mcp]'"
            ) from exc
        return create_fixed_namespace_mcp
    raise AttributeError(name)
