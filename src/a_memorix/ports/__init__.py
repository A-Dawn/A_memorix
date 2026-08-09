"""Minimal interfaces supplied by a host application."""

from .embedding import EmbeddingProvider
from .identity import IdentityRecord, IdentityResolver
from .llm import LLMProvider, LLMRequest, LLMResult
from .message_source import MessageRecord, MessageSource

__all__ = [
    "EmbeddingProvider",
    "IdentityRecord",
    "IdentityResolver",
    "LLMProvider",
    "LLMRequest",
    "LLMResult",
    "MessageRecord",
    "MessageSource",
]
