"""Public API for the A_memorix memory engine."""

from .core.runtime import KernelSearchRequest, SDKMemoryKernel
from .ports import (
    EmbeddingProvider,
    IdentityRecord,
    IdentityResolver,
    LLMProvider,
    LLMRequest,
    LLMResult,
    MessageRecord,
    MessageSource,
)

__all__ = [
    "EmbeddingProvider",
    "IdentityRecord",
    "IdentityResolver",
    "KernelSearchRequest",
    "LLMProvider",
    "LLMRequest",
    "LLMResult",
    "MessageRecord",
    "MessageSource",
    "SDKMemoryKernel",
    "__version__",
]
__version__ = "2.0.0a1"
