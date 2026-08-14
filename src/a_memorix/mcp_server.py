"""Fixed-namespace MCP adapter backed by AMemorixEngine."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import BaseModel, ConfigDict, Field

from a_memorix.contracts import (
    BatchIngestTextRequest,
    CreateNamespaceRequest,
    DeleteBySourceRequest,
    DeleteMemoryRequest,
    GetMemoryRequest,
    IngestTextInput,
    IngestTextRequest,
    NamespaceConfig,
    NamespaceNotFoundError,
    RelationExtractionMode,
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
    object_value: str = Field(
        alias="object", serialization_alias="object", min_length=1
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class MCPIngestTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    source_type: str = Field(min_length=1, max_length=128)
    external_id: str = ""
    person_ids: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    relations: list[MCPRelationInput] = Field(default_factory=list)
    relation_extraction: RelationExtractionMode = RelationExtractionMode.INHERIT
    metadata: dict[str, object] = Field(default_factory=dict)
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


def create_fixed_namespace_mcp(
    engine: AMemorixEngine,
    namespace_id: str,
    *,
    create_namespace: bool = False,
    manage_engine_lifecycle: bool = True,
    namespace_config: NamespaceConfig | None = None,
    required_capabilities: Sequence[str] = (),
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
                    CreateNamespaceRequest(
                        namespace_id=namespace_id,
                        config=namespace_config or NamespaceConfig(),
                    )
                )
            lease_context = RequestContext(
                namespace_id=namespace_id,
                agent_id="mcp",
                principal_id="mcp:fixed-namespace",
            )
            async with engine.runtime(lease_context) as runtime:
                _require_runtime_capabilities(runtime, required_capabilities)
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
        relation_extraction: RelationExtractionMode = RelationExtractionMode.INHERIT,
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
                relation_extraction=relation_extraction,
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
    async def batch_ingest_text(
        items: list[MCPIngestTextInput],
        ctx: Context,
        conversation_id: str = "",
        user_id: str = "",
        group_id: str = "",
        idempotency_key: str = "",
    ) -> dict[str, object]:
        """Store up to 100 text memories in the bound namespace."""

        result = await engine.batch_ingest_text(
            BatchIngestTextRequest(
                context=_request_context(
                    namespace_id,
                    ctx,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    group_id=group_id,
                    idempotency_key=idempotency_key,
                ),
                items=tuple(_mcp_ingest_input(item) for item in items),
            )
        )
        return result.model_dump(mode="json")

    @server.tool()
    async def get_memory(
        ctx: Context,
        memory_id: str = "",
        external_id: str = "",
    ) -> dict[str, object]:
        """Read one memory from the bound namespace."""

        result = await engine.get_memory(
            GetMemoryRequest(
                context=_request_context(namespace_id, ctx),
                memory_id=memory_id,
                external_id=external_id,
            )
        )
        return result.model_dump(mode="json")

    @server.tool()
    async def delete_memory(
        ctx: Context,
        memory_id: str = "",
        external_id: str = "",
        reason: str = "user_delete",
    ) -> dict[str, object]:
        """Delete one memory from the bound namespace."""

        result = await engine.delete_memory(
            DeleteMemoryRequest(
                context=_request_context(namespace_id, ctx),
                memory_id=memory_id,
                external_id=external_id,
                reason=reason,
            )
        )
        return result.model_dump(mode="json")

    @server.tool()
    async def delete_by_source(
        source: str,
        ctx: Context,
        reason: str = "source_delete",
    ) -> dict[str, object]:
        """Start a source deletion job in the bound namespace."""

        job = await engine.submit_delete_by_source(
            DeleteBySourceRequest(
                context=_request_context(namespace_id, ctx),
                source=source,
                reason=reason,
            )
        )
        return job.model_dump(mode="json")

    @server.tool()
    async def get_job(job_id: str) -> dict[str, object]:
        """Read one persistent job from the bound namespace."""

        job = await engine.get_job(namespace_id, job_id)
        return job.model_dump(mode="json")

    @server.tool()
    async def list_jobs(
        page_size: int = 50,
        page_token: str = "",
    ) -> dict[str, object]:
        """List persistent jobs in the bound namespace."""

        jobs, next_page_token = await engine.list_jobs_page(
            namespace_id,
            page_size=page_size,
            page_token=page_token,
        )
        return {
            "jobs": [item.model_dump(mode="json") for item in jobs],
            "next_page_token": next_page_token,
        }

    @server.tool()
    async def namespace_health() -> dict[str, object]:
        """Return health and resource usage for the bound namespace."""

        health = await engine.namespace_health(namespace_id)
        return health.model_dump(mode="json")

    return server


def _require_runtime_capabilities(
    runtime: object,
    required_capabilities: Sequence[str],
) -> None:
    required = tuple(str(item).strip() for item in required_capabilities if item)
    if not required:
        return
    inspect_capabilities = getattr(runtime, "runtime_capability_status", None)
    raw_status = inspect_capabilities() if callable(inspect_capabilities) else {}
    status = raw_status if isinstance(raw_status, dict) else {}
    raw_capabilities = status.get("capabilities")
    capabilities = (
        raw_capabilities if isinstance(raw_capabilities, dict) else {}
    )
    available = {
        **capabilities,
        "paragraph_vector_pool": bool(
            status.get("paragraph_vector_pool_ready", False)
        ),
        "relation_vector_pool": bool(
            status.get("relation_vector_pool_ready", False)
        ),
    }
    missing = [name for name in required if not available.get(name, False)]
    if missing:
        raise RuntimeError(
            "standard MCP runtime is missing required capabilities: "
            + ", ".join(sorted(set(missing)))
        )


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


def _mcp_ingest_input(value: MCPIngestTextInput) -> IngestTextInput:
    return IngestTextInput(
        external_id=value.external_id,
        source_type=value.source_type,
        text=value.text,
        person_ids=tuple(value.person_ids),
        participants=tuple(value.participants),
        tags=tuple(value.tags),
        entities=tuple(value.entities),
        relations=tuple(
            RelationInput.model_validate(item.model_dump(by_alias=True))
            for item in value.relations
        ),
        relation_extraction=value.relation_extraction,
        metadata=value.metadata,
        observed_at=value.observed_at,
        valid_from=value.valid_from,
        valid_to=value.valid_to,
    )
