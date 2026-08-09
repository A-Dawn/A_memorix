"""Fixed-namespace MCP adapter backed by AMemorixEngine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import BaseModel, ConfigDict, Field

from a_memorix.contracts import (
    CreateNamespaceRequest,
    IngestTextRequest,
    NamespaceNotFoundError,
    RelationInput,
    RequestContext,
    SearchMemoryRequest,
    SearchMode,
)
from a_memorix.engine import AMemorixEngine


class MCPRelationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_value: str = Field(alias="object", serialization_alias="object", min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)


def create_fixed_namespace_mcp(
    engine: AMemorixEngine,
    namespace_id: str,
    *,
    create_namespace: bool = False,
    manage_engine_lifecycle: bool = True,
) -> MCPServer:
    """Create an MCP server whose tools cannot address another namespace."""

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        if manage_engine_lifecycle:
            await engine.initialize()
        try:
            try:
                await engine.get_namespace(namespace_id)
            except NamespaceNotFoundError:
                if not create_namespace:
                    raise
                await engine.create_namespace(
                    CreateNamespaceRequest(namespace_id=namespace_id)
                )
            yield None
        finally:
            if manage_engine_lifecycle:
                await engine.shutdown()

    server = MCPServer("A_memorix", lifespan=lifespan)

    @server.tool()
    async def ingest_text(
        text: str,
        source_type: str,
        ctx: Context,
        external_id: str = "",
        conversation_id: str = "",
        user_id: str = "",
        group_id: str = "",
        person_ids: list[str] | None = None,
        participants: list[str] | None = None,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        relations: list[MCPRelationInput] | None = None,
        metadata: dict[str, object] | None = None,
        observed_at: datetime | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        idempotency_key: str = "",
    ) -> dict[str, object]:
        """Store one text memory in the bound namespace."""

        request_context = _request_context(
            namespace_id,
            ctx,
            conversation_id=conversation_id,
            user_id=user_id,
            group_id=group_id,
            idempotency_key=idempotency_key,
        )
        result = await engine.ingest_text(
            IngestTextRequest(
                context=request_context,
                external_id=external_id,
                source_type=source_type,
                text=text,
                person_ids=tuple(person_ids or ()),
                participants=tuple(participants or ()),
                tags=tuple(tags or ()),
                entities=tuple(entities or ()),
                relations=tuple(
                    RelationInput.model_validate(item.model_dump(by_alias=True))
                    for item in (relations or ())
                ),
                metadata=metadata or {},
                observed_at=observed_at,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        )
        return result.model_dump(mode="json")

    @server.tool()
    async def search_memory(
        ctx: Context,
        query: str = "",
        limit: int = 5,
        mode: SearchMode = SearchMode.SEARCH,
        conversation_id: str = "",
        shared_conversation_ids: list[str] | None = None,
        person_id: str = "",
        user_id: str = "",
        group_id: str = "",
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> dict[str, object]:
        """Search memories in the bound namespace."""

        result = await engine.search_memory(
            SearchMemoryRequest(
                context=_request_context(
                    namespace_id,
                    ctx,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    group_id=group_id,
                ),
                query=query,
                limit=limit,
                mode=mode,
                shared_conversation_ids=tuple(shared_conversation_ids or ()),
                person_id=person_id,
                time_start=time_start,
                time_end=time_end,
            )
        )
        return result.model_dump(mode="json")

    @server.tool()
    async def namespace_health() -> dict[str, object]:
        """Return health and resource usage for the bound namespace."""

        health = await engine.namespace_health(namespace_id)
        return health.model_dump(mode="json")

    return server


def _request_context(
    namespace_id: str,
    ctx: Context | None,
    **values: str,
) -> RequestContext:
    request_id = ""
    if ctx is not None:
        request_id = str(ctx.request_context.request_id)
    data: dict[str, object] = {
        "namespace_id": namespace_id,
        "agent_id": "mcp",
        "principal_id": "mcp:fixed-namespace",
    }
    data.update({key: value for key, value in values.items() if value})
    if request_id:
        data["request_id"] = request_id
    return RequestContext.model_validate(data)
