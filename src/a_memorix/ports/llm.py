"""Text generation provider contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    request_type: str
    task_name: str = ""
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class LLMResult:
    success: bool
    content: str = ""
    error: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, error: str) -> "LLMResult":
        return cls(success=False, error=str(error))


@runtime_checkable
class LLMProvider(Protocol):
    """Expose model discovery and asynchronous text generation."""

    def get_available_models(self) -> Mapping[str, Any]:
        """Return task names mapped to provider-specific task configurations."""

    async def generate(self, request: LLMRequest) -> LLMResult:
        """Generate text for one request."""
