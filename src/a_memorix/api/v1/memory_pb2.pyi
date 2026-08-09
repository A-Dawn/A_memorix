import datetime

from a_memorix.api.v1 import common_pb2 as _common_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SEARCH_MODE_UNSPECIFIED: _ClassVar[SearchMode]
    SEARCH_MODE_SEARCH: _ClassVar[SearchMode]
    SEARCH_MODE_TIME: _ClassVar[SearchMode]
    SEARCH_MODE_HYBRID: _ClassVar[SearchMode]
    SEARCH_MODE_EPISODE: _ClassVar[SearchMode]
    SEARCH_MODE_AGGREGATE: _ClassVar[SearchMode]
SEARCH_MODE_UNSPECIFIED: SearchMode
SEARCH_MODE_SEARCH: SearchMode
SEARCH_MODE_TIME: SearchMode
SEARCH_MODE_HYBRID: SearchMode
SEARCH_MODE_EPISODE: SearchMode
SEARCH_MODE_AGGREGATE: SearchMode

class RelationInput(_message.Message):
    __slots__ = ("subject", "predicate", "object", "confidence", "metadata")
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    subject: str
    predicate: str
    object: str
    confidence: float
    metadata: _struct_pb2.Struct
    def __init__(self, subject: _Optional[str] = ..., predicate: _Optional[str] = ..., object: _Optional[str] = ..., confidence: _Optional[float] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class IngestTextRequest(_message.Message):
    __slots__ = ("context", "external_id", "source_type", "text", "person_ids", "participants", "observed_at", "valid_from", "valid_to", "tags", "metadata", "entities", "relations", "respect_filter")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    PERSON_IDS_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    VALID_FROM_FIELD_NUMBER: _ClassVar[int]
    VALID_TO_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    RESPECT_FILTER_FIELD_NUMBER: _ClassVar[int]
    context: _common_pb2.RequestContext
    external_id: str
    source_type: str
    text: str
    person_ids: _containers.RepeatedScalarFieldContainer[str]
    participants: _containers.RepeatedScalarFieldContainer[str]
    observed_at: _timestamp_pb2.Timestamp
    valid_from: _timestamp_pb2.Timestamp
    valid_to: _timestamp_pb2.Timestamp
    tags: _containers.RepeatedScalarFieldContainer[str]
    metadata: _struct_pb2.Struct
    entities: _containers.RepeatedScalarFieldContainer[str]
    relations: _containers.RepeatedCompositeFieldContainer[RelationInput]
    respect_filter: bool
    def __init__(self, context: _Optional[_Union[_common_pb2.RequestContext, _Mapping]] = ..., external_id: _Optional[str] = ..., source_type: _Optional[str] = ..., text: _Optional[str] = ..., person_ids: _Optional[_Iterable[str]] = ..., participants: _Optional[_Iterable[str]] = ..., observed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., valid_from: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., valid_to: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., tags: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., entities: _Optional[_Iterable[str]] = ..., relations: _Optional[_Iterable[_Union[RelationInput, _Mapping]]] = ..., respect_filter: _Optional[bool] = ...) -> None: ...

class IngestTextResponse(_message.Message):
    __slots__ = ("stored_ids", "skipped_ids", "fact_claim_ids", "warnings", "detail")
    STORED_IDS_FIELD_NUMBER: _ClassVar[int]
    SKIPPED_IDS_FIELD_NUMBER: _ClassVar[int]
    FACT_CLAIM_IDS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    stored_ids: _containers.RepeatedScalarFieldContainer[str]
    skipped_ids: _containers.RepeatedScalarFieldContainer[str]
    fact_claim_ids: _containers.RepeatedScalarFieldContainer[str]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    detail: str
    def __init__(self, stored_ids: _Optional[_Iterable[str]] = ..., skipped_ids: _Optional[_Iterable[str]] = ..., fact_claim_ids: _Optional[_Iterable[str]] = ..., warnings: _Optional[_Iterable[str]] = ..., detail: _Optional[str] = ...) -> None: ...

class IngestTextInput(_message.Message):
    __slots__ = ("external_id", "source_type", "text", "person_ids", "participants", "observed_at", "valid_from", "valid_to", "tags", "metadata", "entities", "relations", "respect_filter")
    EXTERNAL_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    PERSON_IDS_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANTS_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    VALID_FROM_FIELD_NUMBER: _ClassVar[int]
    VALID_TO_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    RESPECT_FILTER_FIELD_NUMBER: _ClassVar[int]
    external_id: str
    source_type: str
    text: str
    person_ids: _containers.RepeatedScalarFieldContainer[str]
    participants: _containers.RepeatedScalarFieldContainer[str]
    observed_at: _timestamp_pb2.Timestamp
    valid_from: _timestamp_pb2.Timestamp
    valid_to: _timestamp_pb2.Timestamp
    tags: _containers.RepeatedScalarFieldContainer[str]
    metadata: _struct_pb2.Struct
    entities: _containers.RepeatedScalarFieldContainer[str]
    relations: _containers.RepeatedCompositeFieldContainer[RelationInput]
    respect_filter: bool
    def __init__(self, external_id: _Optional[str] = ..., source_type: _Optional[str] = ..., text: _Optional[str] = ..., person_ids: _Optional[_Iterable[str]] = ..., participants: _Optional[_Iterable[str]] = ..., observed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., valid_from: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., valid_to: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., tags: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., entities: _Optional[_Iterable[str]] = ..., relations: _Optional[_Iterable[_Union[RelationInput, _Mapping]]] = ..., respect_filter: _Optional[bool] = ...) -> None: ...

class BatchIngestTextRequest(_message.Message):
    __slots__ = ("context", "items")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    context: _common_pb2.RequestContext
    items: _containers.RepeatedCompositeFieldContainer[IngestTextInput]
    def __init__(self, context: _Optional[_Union[_common_pb2.RequestContext, _Mapping]] = ..., items: _Optional[_Iterable[_Union[IngestTextInput, _Mapping]]] = ...) -> None: ...

class BatchIngestItemResult(_message.Message):
    __slots__ = ("index", "response", "error")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    index: int
    response: IngestTextResponse
    error: _common_pb2.ErrorDetail
    def __init__(self, index: _Optional[int] = ..., response: _Optional[_Union[IngestTextResponse, _Mapping]] = ..., error: _Optional[_Union[_common_pb2.ErrorDetail, _Mapping]] = ...) -> None: ...

class BatchIngestTextResponse(_message.Message):
    __slots__ = ("results", "succeeded", "failed")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[BatchIngestItemResult]
    succeeded: int
    failed: int
    def __init__(self, results: _Optional[_Iterable[_Union[BatchIngestItemResult, _Mapping]]] = ..., succeeded: _Optional[int] = ..., failed: _Optional[int] = ...) -> None: ...

class GetMemoryRequest(_message.Message):
    __slots__ = ("context", "memory_id", "external_id")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_ID_FIELD_NUMBER: _ClassVar[int]
    context: _common_pb2.RequestContext
    memory_id: str
    external_id: str
    def __init__(self, context: _Optional[_Union[_common_pb2.RequestContext, _Mapping]] = ..., memory_id: _Optional[str] = ..., external_id: _Optional[str] = ...) -> None: ...

class MemoryRecord(_message.Message):
    __slots__ = ("memory_id", "external_id", "source_type", "source", "content", "metadata", "created_at", "updated_at", "observed_at", "valid_from", "valid_to")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    VALID_FROM_FIELD_NUMBER: _ClassVar[int]
    VALID_TO_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    external_id: str
    source_type: str
    source: str
    content: str
    metadata: _struct_pb2.Struct
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    observed_at: _timestamp_pb2.Timestamp
    valid_from: _timestamp_pb2.Timestamp
    valid_to: _timestamp_pb2.Timestamp
    def __init__(self, memory_id: _Optional[str] = ..., external_id: _Optional[str] = ..., source_type: _Optional[str] = ..., source: _Optional[str] = ..., content: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., observed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., valid_from: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., valid_to: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetMemoryResponse(_message.Message):
    __slots__ = ("memory",)
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    memory: MemoryRecord
    def __init__(self, memory: _Optional[_Union[MemoryRecord, _Mapping]] = ...) -> None: ...

class DeleteMemoryRequest(_message.Message):
    __slots__ = ("context", "memory_id", "external_id", "reason")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    context: _common_pb2.RequestContext
    memory_id: str
    external_id: str
    reason: str
    def __init__(self, context: _Optional[_Union[_common_pb2.RequestContext, _Mapping]] = ..., memory_id: _Optional[str] = ..., external_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class DeleteMemoryResponse(_message.Message):
    __slots__ = ("operation_id", "deleted_count", "deleted_memory_ids")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    DELETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    DELETED_MEMORY_IDS_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    deleted_count: int
    deleted_memory_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, operation_id: _Optional[str] = ..., deleted_count: _Optional[int] = ..., deleted_memory_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class SearchMemoryRequest(_message.Message):
    __slots__ = ("context", "query", "limit", "mode", "shared_conversation_ids", "person_id", "time_start", "time_end", "respect_filter")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    SHARED_CONVERSATION_IDS_FIELD_NUMBER: _ClassVar[int]
    PERSON_ID_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    RESPECT_FILTER_FIELD_NUMBER: _ClassVar[int]
    context: _common_pb2.RequestContext
    query: str
    limit: int
    mode: SearchMode
    shared_conversation_ids: _containers.RepeatedScalarFieldContainer[str]
    person_id: str
    time_start: _timestamp_pb2.Timestamp
    time_end: _timestamp_pb2.Timestamp
    respect_filter: bool
    def __init__(self, context: _Optional[_Union[_common_pb2.RequestContext, _Mapping]] = ..., query: _Optional[str] = ..., limit: _Optional[int] = ..., mode: _Optional[_Union[SearchMode, str]] = ..., shared_conversation_ids: _Optional[_Iterable[str]] = ..., person_id: _Optional[str] = ..., time_start: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., time_end: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., respect_filter: _Optional[bool] = ...) -> None: ...

class MemoryHit(_message.Message):
    __slots__ = ("memory_id", "kind", "title", "content", "score", "source", "metadata")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    kind: str
    title: str
    content: str
    score: float
    source: str
    metadata: _struct_pb2.Struct
    def __init__(self, memory_id: _Optional[str] = ..., kind: _Optional[str] = ..., title: _Optional[str] = ..., content: _Optional[str] = ..., score: _Optional[float] = ..., source: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SearchMemoryResponse(_message.Message):
    __slots__ = ("summary", "hits", "filtered", "degraded", "retrieval_ready", "retrieval_mode", "available_channels", "unavailable_channels")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    FILTERED_FIELD_NUMBER: _ClassVar[int]
    DEGRADED_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_READY_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_MODE_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    summary: str
    hits: _containers.RepeatedCompositeFieldContainer[MemoryHit]
    filtered: bool
    degraded: bool
    retrieval_ready: bool
    retrieval_mode: str
    available_channels: _containers.RepeatedScalarFieldContainer[str]
    unavailable_channels: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, summary: _Optional[str] = ..., hits: _Optional[_Iterable[_Union[MemoryHit, _Mapping]]] = ..., filtered: _Optional[bool] = ..., degraded: _Optional[bool] = ..., retrieval_ready: _Optional[bool] = ..., retrieval_mode: _Optional[str] = ..., available_channels: _Optional[_Iterable[str]] = ..., unavailable_channels: _Optional[_Iterable[str]] = ...) -> None: ...
