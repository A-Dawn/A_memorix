"""Minimal interfaces supplied by a host application."""

from .embedding import EmbeddingProvider
from .clock import Clock, SystemClock
from .identity import IdentityRecord, IdentityResolver
from .llm import LLMProvider, LLMRequest, LLMResult
from .message_source import MessageRecord, MessageSource
from .namespace import NamespaceHostPorts

__all__ = [
    "Clock",
    "EmbeddingProvider",
    "IdentityRecord",
    "IdentityResolver",
    "LLMProvider",
    "LLMRequest",
    "LLMResult",
    "MessageRecord",
    "MessageSource",
    "NamespaceHostPorts",
    "SystemClock",
]
