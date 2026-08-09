"""Conversions between generated API messages and application contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from google.protobuf import json_format, struct_pb2, timestamp_pb2
from pydantic import TypeAdapter

from a_memorix.api.v1 import auth_pb2, common_pb2, memory_pb2, namespace_pb2
from a_memorix.contracts import (
    ApiKeyInfo,
    CreateNamespaceRequest,
    IngestTextRequest,
    IngestTextResponse,
    MemoryHit,
    NamespaceHealth,
    NamespaceInfo,
    NamespaceQuota,
    RelationInput,
    RequestContext,
    SearchMemoryRequest,
    SearchMemoryResponse,
    SearchMode,
)


_JSON_MAPPING = TypeAdapter(dict[str, object])

_NAMESPACE_STATUS_TO_PROTO = {
    "creating": namespace_pb2.NAMESPACE_STATUS_CREATING,
    "active": namespace_pb2.NAMESPACE_STATUS_ACTIVE,
    "inactive": namespace_pb2.NAMESPACE_STATUS_INACTIVE,
    "quarantined": namespace_pb2.NAMESPACE_STATUS_QUARANTINED,
    "purging": namespace_pb2.NAMESPACE_STATUS_PURGING,
}
_RUNTIME_STATE_TO_PROTO = {
    "closed": namespace_pb2.NAMESPACE_RUNTIME_STATE_CLOSED,
    "loading": namespace_pb2.NAMESPACE_RUNTIME_STATE_LOADING,
    "ready": namespace_pb2.NAMESPACE_RUNTIME_STATE_READY,
    "degraded": namespace_pb2.NAMESPACE_RUNTIME_STATE_DEGRADED,
    "failed": namespace_pb2.NAMESPACE_RUNTIME_STATE_FAILED,
}
_SEARCH_MODE_FROM_PROTO = {
    memory_pb2.SEARCH_MODE_UNSPECIFIED: SearchMode.SEARCH,
    memory_pb2.SEARCH_MODE_SEARCH: SearchMode.SEARCH,
    memory_pb2.SEARCH_MODE_TIME: SearchMode.TIME,
    memory_pb2.SEARCH_MODE_HYBRID: SearchMode.HYBRID,
    memory_pb2.SEARCH_MODE_EPISODE: SearchMode.EPISODE,
    memory_pb2.SEARCH_MODE_AGGREGATE: SearchMode.AGGREGATE,
}


def request_context_from_proto(value: common_pb2.RequestContext) -> RequestContext:
    kwargs: dict[str, object] = {"namespace_id": value.namespace_id}
    for field in (
        "agent_id",
        "principal_id",
        "conversation_id",
        "user_id",
        "group_id",
        "request_id",
        "trace_id",
        "idempotency_key",
    ):
        raw = getattr(value, field)
        if raw:
            kwargs[field] = raw
    return RequestContext.model_validate(kwargs)


def create_namespace_request_from_proto(
    value: namespace_pb2.CreateNamespaceRequest,
) -> CreateNamespaceRequest:
    quota = NamespaceQuota()
    if value.HasField("quota"):
        quota = NamespaceQuota(
            max_concurrent_requests=(
                value.quota.max_concurrent_requests
                if value.quota.HasField("max_concurrent_requests")
                else None
            ),
            max_storage_bytes=(
                value.quota.max_storage_bytes
                if value.quota.HasField("max_storage_bytes")
                else None
            ),
        )
    return CreateNamespaceRequest(namespace_id=value.namespace_id, quota=quota)


def ingest_request_from_proto(value: memory_pb2.IngestTextRequest) -> IngestTextRequest:
    return IngestTextRequest(
        context=request_context_from_proto(value.context),
        external_id=value.external_id,
        source_type=value.source_type,
        text=value.text,
        person_ids=tuple(value.person_ids),
        participants=tuple(value.participants),
        observed_at=_optional_datetime(value, "observed_at"),
        valid_from=_optional_datetime(value, "valid_from"),
        valid_to=_optional_datetime(value, "valid_to"),
        tags=tuple(value.tags),
        metadata=struct_to_mapping(value.metadata),
        entities=tuple(value.entities),
        relations=tuple(
            RelationInput(
                subject=item.subject,
                predicate=item.predicate,
                object=item.object,
                confidence=item.confidence if item.HasField("confidence") else 1.0,
                metadata=struct_to_mapping(item.metadata),
            )
            for item in value.relations
        ),
        respect_filter=(value.respect_filter if value.HasField("respect_filter") else True),
    )


def search_request_from_proto(value: memory_pb2.SearchMemoryRequest) -> SearchMemoryRequest:
    return SearchMemoryRequest(
        context=request_context_from_proto(value.context),
        query=value.query,
        limit=value.limit if value.HasField("limit") else 5,
        mode=_SEARCH_MODE_FROM_PROTO.get(value.mode, SearchMode.SEARCH),
        shared_conversation_ids=tuple(value.shared_conversation_ids),
        person_id=value.person_id,
        time_start=_optional_datetime(value, "time_start"),
        time_end=_optional_datetime(value, "time_end"),
        respect_filter=(value.respect_filter if value.HasField("respect_filter") else True),
    )


def namespace_info_to_proto(value: NamespaceInfo) -> namespace_pb2.NamespaceInfo:
    result = namespace_pb2.NamespaceInfo(
        namespace_id=value.namespace_id,
        status=_NAMESPACE_STATUS_TO_PROTO[value.status.value],
        version=value.version,
    )
    result.created_at.CopyFrom(timestamp_to_proto(value.created_at))
    result.updated_at.CopyFrom(timestamp_to_proto(value.updated_at))
    if value.last_active_at is not None:
        result.last_active_at.CopyFrom(timestamp_to_proto(value.last_active_at))
    if value.purge_after is not None:
        result.purge_after.CopyFrom(timestamp_to_proto(value.purge_after))
    if value.quota.max_concurrent_requests is not None:
        result.quota.max_concurrent_requests = value.quota.max_concurrent_requests
    if value.quota.max_storage_bytes is not None:
        result.quota.max_storage_bytes = value.quota.max_storage_bytes
    return result


def namespace_health_to_proto(value: NamespaceHealth) -> namespace_pb2.NamespaceHealth:
    return namespace_pb2.NamespaceHealth(
        namespace=namespace_info_to_proto(value.namespace),
        runtime_state=_RUNTIME_STATE_TO_PROTO[value.runtime_state.value],
        healthy=value.healthy,
        resource_usage=namespace_pb2.NamespaceResourceUsage(
            active_requests=value.resource_usage.active_requests,
            storage_bytes=value.resource_usage.storage_bytes,
        ),
        last_error=value.last_error or "",
    )


def api_key_info_to_proto(value: ApiKeyInfo) -> auth_pb2.ApiKeyInfo:
    result = auth_pb2.ApiKeyInfo(
        key_id=value.key_id,
        namespace_id=value.namespace_id,
        label=value.label,
    )
    result.created_at.CopyFrom(timestamp_to_proto(value.created_at))
    if value.expires_at is not None:
        result.expires_at.CopyFrom(timestamp_to_proto(value.expires_at))
    if value.revoked_at is not None:
        result.revoked_at.CopyFrom(timestamp_to_proto(value.revoked_at))
    if value.last_used_at is not None:
        result.last_used_at.CopyFrom(timestamp_to_proto(value.last_used_at))
    return result


def ingest_response_to_proto(value: IngestTextResponse) -> memory_pb2.IngestTextResponse:
    return memory_pb2.IngestTextResponse(
        stored_ids=value.stored_ids,
        skipped_ids=value.skipped_ids,
        fact_claim_ids=value.fact_claim_ids,
        warnings=value.warnings,
        detail=value.detail,
    )


def search_response_to_proto(value: SearchMemoryResponse) -> memory_pb2.SearchMemoryResponse:
    return memory_pb2.SearchMemoryResponse(
        summary=value.summary,
        hits=[memory_hit_to_proto(item) for item in value.hits],
        filtered=value.filtered,
        degraded=value.degraded,
        retrieval_ready=value.retrieval_ready,
        retrieval_mode=value.retrieval_mode,
        available_channels=value.available_channels,
        unavailable_channels=value.unavailable_channels,
    )


def memory_hit_to_proto(value: MemoryHit) -> memory_pb2.MemoryHit:
    return memory_pb2.MemoryHit(
        memory_id=value.memory_id,
        kind=value.kind,
        title=value.title,
        content=value.content,
        score=value.score,
        source=value.source,
        metadata=mapping_to_struct(value.metadata),
    )


def mapping_to_struct(value: Mapping[str, object]) -> struct_pb2.Struct:
    result = struct_pb2.Struct()
    json_format.ParseDict(
        _JSON_MAPPING.dump_python(dict(value), mode="json"),
        result,
    )
    return result


def struct_to_mapping(value: struct_pb2.Struct) -> dict[str, object]:
    return dict(json_format.MessageToDict(value, preserving_proto_field_name=True))


def timestamp_to_proto(value: datetime) -> timestamp_pb2.Timestamp:
    result = timestamp_pb2.Timestamp()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    result.FromDatetime(value)
    return result


def _optional_datetime(message: object, field: str) -> datetime | None:
    if not message.HasField(field):  # type: ignore[attr-defined]
        return None
    value = getattr(message, field)
    return value.ToDatetime(tzinfo=timezone.utc)
