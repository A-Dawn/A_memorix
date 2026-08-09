"""Host capabilities that may differ between namespaces."""

from __future__ import annotations

from dataclasses import dataclass

from .embedding import EmbeddingProvider
from .identity import IdentityResolver
from .llm import LLMProvider
from .message_source import MessageSource


@dataclass(frozen=True)
class NamespaceHostPorts:
    embedding_provider: EmbeddingProvider | None = None
    llm_provider: LLMProvider | None = None
    identity_resolver: IdentityResolver | None = None
    message_source: MessageSource | None = None
