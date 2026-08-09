"""Request-scoped identity shared by in-process and remote APIs."""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError


NamespaceId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]
_NAMESPACE_ID_ADAPTER = TypeAdapter(NamespaceId)


def validate_namespace_id(value: str) -> str:
    """Apply the same namespace identifier rules outside Pydantic models."""

    try:
        return _NAMESPACE_ID_ADAPTER.validate_python(value)
    except ValidationError as exc:
        from .errors import InvalidArgumentError

        raise InvalidArgumentError(
            f"invalid namespace_id: {value}",
            details={"namespace_id": value},
        ) from exc


class RequestContext(BaseModel):
    """Identity and correlation data carried by every namespace request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace_id: NamespaceId
    agent_id: str | None = None
    principal_id: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None
    group_id: str | None = None
    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=255)
    trace_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=255)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)
