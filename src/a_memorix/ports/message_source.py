"""Conversation message source contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class MessageRecord:
    content: str
    sender_id: str = ""
    sender_name: str = ""
    timestamp: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class MessageSource(Protocol):
    """Read messages from an Agent-owned conversation store."""

    async def list_messages(
        self,
        *,
        conversation_id: str,
        start_time: float,
        end_time: float,
        limit: int,
    ) -> Sequence[MessageRecord]:
        """Return messages ordered from oldest to newest."""
