"""Conversions between generated API messages and application contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from google.protobuf import json_format, struct_pb2, timestamp_pb2
from pydantic import TypeAdapter

from a_memorix.api.v1 import (
    auth_pb2,
    common_pb2,
    job_pb2,
    memory_pb2,
    namespace_pb2,
)
from a_memorix.contracts import (
    ApiKeyInfo,
    BatchIngestTextRequest,
    BatchIngestTextResponse,
    CreateNamespaceRequest,
    DeleteBySourceRequest,
    DeleteMemoryRequest,
    DeleteMemoryResponse,
    ErrorEnvelope,
    GetMemoryRequest,
    GetMemoryResponse,
    IngestTextInput,
    IngestTextRequest,
    IngestTextResponse,
    JobInfo,
    JobStatus,
    JobType,
    MemoryHit,
    MemoryRecord,
    NamespaceCapabilities,
    NamespaceConfig,
    NamespaceFeatureConfig,
    NamespaceHealth,
    NamespaceInfo,
    NamespaceQuota,
    ProviderReference,
    RelationInput,
    RequestContext,
    SearchMemoryRequest,
    SearchMemoryResponse,
    SearchMode,
    UpdateNamespaceConfigRequest,
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
_JOB_TYPE_TO_PROTO = {
    JobType.DELETE_BY_SOURCE: job_pb2.JOB_TYPE_DELETE_BY_SOURCE,
}
_JOB_STATUS_TO_PROTO = {
    JobStatus.PENDING: job_pb2.JOB_STATUS_PENDING,
    JobStatus.RUNNING: job_pb2.JOB_STATUS_RUNNING,
    JobStatus.SUCCEEDED: job_pb2.JOB_STATUS_SUCCEEDED,
    JobStatus.FAILED: job_pb2.JOB_STATUS_FAILED,
    JobStatus.CANCELLED: job_pb2.JOB_STATUS_CANCELLED,
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
    config = (
        namespace_config_from_proto(value.config)
        if value.HasField("config")
        else NamespaceConfig()
    )
    return CreateNamespaceRequest(
        namespace_id=value.namespace_id,
        quota=quota,
        config=config,
    )


def update_namespace_config_request_from_proto(
    value: namespace_pb2.UpdateNamespaceConfigRequest,
) -> UpdateNamespaceConfigRequest:
    return UpdateNamespaceConfigRequest(
        namespace_id=value.namespace_id,
        config=namespace_config_from_proto(value.config),
        expected_config_version=(
            value.expected_config_version
            if value.HasField("expected_config_version")
            else None
        ),
    )


def namespace_config_from_proto(
    value: namespace_pb2.NamespaceConfig,
) -> NamespaceConfig:
    features = NamespaceFeatureConfig()
    if value.HasField("features"):
        raw = value.features
        features = NamespaceFeatureConfig(
            episodes=raw.episodes if raw.HasField("episodes") else True,
            person_profiles=(
                raw.person_profiles if raw.HasField("person_profiles") else True
            ),
            sparse_retrieval=(
                raw.sparse_retrieval if raw.HasField("sparse_retrieval") else True
            ),
            relation_vectors=(
                raw.relation_vectors if raw.HasField("relation_vectors") else False
            ),
            allow_metadata_only_write=(
                raw.allow_metadata_only_write
                if raw.HasField("allow_metadata_only_write")
                else True
            ),
        )
    return NamespaceConfig(
        embedding=_provider_reference_from_proto(value, "embedding"),
        llm=_provider_reference_from_proto(value, "llm"),
        identity_resolver=_provider_reference_from_proto(value, "identity_resolver"),
        message_source=_provider_reference_from_proto(value, "message_source"),
        features=features,
    )


def _provider_reference_from_proto(
    value: namespace_pb2.NamespaceConfig,
    field: str,
) -> ProviderReference | None:
    if not value.HasField(field):
        return None
    raw = getattr(value, field)
    return ProviderReference(
        provider_id=raw.provider_id,
        model_id=raw.model_id,
        secret_ref=raw.secret_ref,
    )


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


def ingest_input_from_proto(value: memory_pb2.IngestTextInput) -> IngestTextInput:
    return IngestTextInput(
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
        respect_filter=(
            value.respect_filter if value.HasField("respect_filter") else True
        ),
    )


def batch_ingest_request_from_proto(
    value: memory_pb2.BatchIngestTextRequest,
) -> BatchIngestTextRequest:
    return BatchIngestTextRequest(
        context=request_context_from_proto(value.context),
        items=tuple(ingest_input_from_proto(item) for item in value.items),
    )


def get_memory_request_from_proto(
    value: memory_pb2.GetMemoryRequest,
) -> GetMemoryRequest:
    return GetMemoryRequest(
        context=request_context_from_proto(value.context),
        memory_id=value.memory_id if value.WhichOneof("selector") == "memory_id" else "",
        external_id=(
            value.external_id if value.WhichOneof("selector") == "external_id" else ""
        ),
    )


def delete_memory_request_from_proto(
    value: memory_pb2.DeleteMemoryRequest,
) -> DeleteMemoryRequest:
    return DeleteMemoryRequest(
        context=request_context_from_proto(value.context),
        memory_id=value.memory_id if value.WhichOneof("selector") == "memory_id" else "",
        external_id=(
            value.external_id if value.WhichOneof("selector") == "external_id" else ""
        ),
        reason=value.reason or "user_delete",
    )


def delete_by_source_request_from_proto(
    value: job_pb2.SubmitDeleteBySourceRequest,
) -> DeleteBySourceRequest:
    return DeleteBySourceRequest(
        context=request_context_from_proto(value.context),
        source=value.source,
        reason=value.reason or "source_delete",
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
        config_version=value.config_version,
        config=namespace_config_to_proto(value.config),
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


def namespace_config_to_proto(
    value: NamespaceConfig,
) -> namespace_pb2.NamespaceConfig:
    result = namespace_pb2.NamespaceConfig(
        features=namespace_pb2.NamespaceFeatureConfig(
            episodes=value.features.episodes,
            person_profiles=value.features.person_profiles,
            sparse_retrieval=value.features.sparse_retrieval,
            relation_vectors=value.features.relation_vectors,
            allow_metadata_only_write=value.features.allow_metadata_only_write,
        )
    )
    for field in ("embedding", "llm", "identity_resolver", "message_source"):
        reference = getattr(value, field)
        if reference is None:
            continue
        getattr(result, field).CopyFrom(
            namespace_pb2.ProviderReference(
                provider_id=reference.provider_id,
                model_id=reference.model_id,
                secret_ref=reference.secret_ref,
            )
        )
    return result


def namespace_capabilities_to_proto(
    value: NamespaceCapabilities,
) -> namespace_pb2.NamespaceCapabilities:
    return namespace_pb2.NamespaceCapabilities(
        namespace_id=value.namespace_id,
        config_version=value.config_version,
        capabilities=dict(value.capabilities),
        operations=value.operations,
        search_modes=value.search_modes,
        degraded=value.degraded,
        unavailable=value.unavailable,
    )


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


def batch_ingest_response_to_proto(
    value: BatchIngestTextResponse,
) -> memory_pb2.BatchIngestTextResponse:
    items: list[memory_pb2.BatchIngestItemResult] = []
    for item in value.results:
        result = memory_pb2.BatchIngestItemResult(index=item.index)
        if item.response is not None:
            result.response.CopyFrom(ingest_response_to_proto(item.response))
        elif item.error is not None:
            result.error.CopyFrom(error_envelope_to_proto(item.error))
        items.append(result)
    return memory_pb2.BatchIngestTextResponse(
        results=items,
        succeeded=value.succeeded,
        failed=value.failed,
    )


def get_memory_response_to_proto(
    value: GetMemoryResponse,
) -> memory_pb2.GetMemoryResponse:
    return memory_pb2.GetMemoryResponse(memory=memory_record_to_proto(value.memory))


def delete_memory_response_to_proto(
    value: DeleteMemoryResponse,
) -> memory_pb2.DeleteMemoryResponse:
    return memory_pb2.DeleteMemoryResponse(
        operation_id=value.operation_id,
        deleted_count=value.deleted_count,
        deleted_memory_ids=value.deleted_memory_ids,
    )


def memory_record_to_proto(value: MemoryRecord) -> memory_pb2.MemoryRecord:
    result = memory_pb2.MemoryRecord(
        memory_id=value.memory_id,
        external_id=value.external_id,
        source_type=value.source_type,
        source=value.source,
        content=value.content,
        metadata=mapping_to_struct(value.metadata),
    )
    for field in (
        "created_at",
        "updated_at",
        "observed_at",
        "valid_from",
        "valid_to",
    ):
        timestamp = getattr(value, field)
        if timestamp is not None:
            getattr(result, field).CopyFrom(timestamp_to_proto(timestamp))
    return result


def job_info_to_proto(value: JobInfo) -> job_pb2.JobInfo:
    result = job_pb2.JobInfo(
        job_id=value.job_id,
        namespace_id=value.namespace_id,
        job_type=_JOB_TYPE_TO_PROTO[value.job_type],
        status=_JOB_STATUS_TO_PROTO[value.status],
        progress=value.progress,
        result=mapping_to_struct(value.result),
    )
    result.created_at.CopyFrom(timestamp_to_proto(value.created_at))
    result.updated_at.CopyFrom(timestamp_to_proto(value.updated_at))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.completed_at is not None:
        result.completed_at.CopyFrom(timestamp_to_proto(value.completed_at))
    if value.error is not None:
        result.error.CopyFrom(error_envelope_to_proto(value.error))
    return result


def error_envelope_to_proto(value: ErrorEnvelope) -> common_pb2.ErrorDetail:
    return common_pb2.ErrorDetail(
        code=value.code.value,
        request_id=value.request_id,
        trace_id=value.trace_id,
        retryable=value.retryable,
        details=mapping_to_struct(value.details),
        message=value.message,
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
