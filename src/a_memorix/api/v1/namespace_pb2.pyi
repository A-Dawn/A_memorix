import datetime

from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NamespaceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NAMESPACE_STATUS_UNSPECIFIED: _ClassVar[NamespaceStatus]
    NAMESPACE_STATUS_CREATING: _ClassVar[NamespaceStatus]
    NAMESPACE_STATUS_ACTIVE: _ClassVar[NamespaceStatus]
    NAMESPACE_STATUS_INACTIVE: _ClassVar[NamespaceStatus]
    NAMESPACE_STATUS_QUARANTINED: _ClassVar[NamespaceStatus]
    NAMESPACE_STATUS_PURGING: _ClassVar[NamespaceStatus]

class NamespaceRuntimeState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NAMESPACE_RUNTIME_STATE_UNSPECIFIED: _ClassVar[NamespaceRuntimeState]
    NAMESPACE_RUNTIME_STATE_CLOSED: _ClassVar[NamespaceRuntimeState]
    NAMESPACE_RUNTIME_STATE_LOADING: _ClassVar[NamespaceRuntimeState]
    NAMESPACE_RUNTIME_STATE_READY: _ClassVar[NamespaceRuntimeState]
    NAMESPACE_RUNTIME_STATE_DEGRADED: _ClassVar[NamespaceRuntimeState]
    NAMESPACE_RUNTIME_STATE_FAILED: _ClassVar[NamespaceRuntimeState]
NAMESPACE_STATUS_UNSPECIFIED: NamespaceStatus
NAMESPACE_STATUS_CREATING: NamespaceStatus
NAMESPACE_STATUS_ACTIVE: NamespaceStatus
NAMESPACE_STATUS_INACTIVE: NamespaceStatus
NAMESPACE_STATUS_QUARANTINED: NamespaceStatus
NAMESPACE_STATUS_PURGING: NamespaceStatus
NAMESPACE_RUNTIME_STATE_UNSPECIFIED: NamespaceRuntimeState
NAMESPACE_RUNTIME_STATE_CLOSED: NamespaceRuntimeState
NAMESPACE_RUNTIME_STATE_LOADING: NamespaceRuntimeState
NAMESPACE_RUNTIME_STATE_READY: NamespaceRuntimeState
NAMESPACE_RUNTIME_STATE_DEGRADED: NamespaceRuntimeState
NAMESPACE_RUNTIME_STATE_FAILED: NamespaceRuntimeState

class NamespaceQuota(_message.Message):
    __slots__ = ("max_concurrent_requests", "max_storage_bytes")
    MAX_CONCURRENT_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    MAX_STORAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    max_concurrent_requests: int
    max_storage_bytes: int
    def __init__(self, max_concurrent_requests: _Optional[int] = ..., max_storage_bytes: _Optional[int] = ...) -> None: ...

class ProviderReference(_message.Message):
    __slots__ = ("provider_id", "model_id", "secret_ref")
    PROVIDER_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SECRET_REF_FIELD_NUMBER: _ClassVar[int]
    provider_id: str
    model_id: str
    secret_ref: str
    def __init__(self, provider_id: _Optional[str] = ..., model_id: _Optional[str] = ..., secret_ref: _Optional[str] = ...) -> None: ...

class NamespaceFeatureConfig(_message.Message):
    __slots__ = ("episodes", "person_profiles", "sparse_retrieval", "relation_vectors", "allow_metadata_only_write")
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    PERSON_PROFILES_FIELD_NUMBER: _ClassVar[int]
    SPARSE_RETRIEVAL_FIELD_NUMBER: _ClassVar[int]
    RELATION_VECTORS_FIELD_NUMBER: _ClassVar[int]
    ALLOW_METADATA_ONLY_WRITE_FIELD_NUMBER: _ClassVar[int]
    episodes: bool
    person_profiles: bool
    sparse_retrieval: bool
    relation_vectors: bool
    allow_metadata_only_write: bool
    def __init__(self, episodes: _Optional[bool] = ..., person_profiles: _Optional[bool] = ..., sparse_retrieval: _Optional[bool] = ..., relation_vectors: _Optional[bool] = ..., allow_metadata_only_write: _Optional[bool] = ...) -> None: ...

class NamespaceConfig(_message.Message):
    __slots__ = ("embedding", "llm", "identity_resolver", "message_source", "features")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    LLM_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_RESOLVER_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    embedding: ProviderReference
    llm: ProviderReference
    identity_resolver: ProviderReference
    message_source: ProviderReference
    features: NamespaceFeatureConfig
    def __init__(self, embedding: _Optional[_Union[ProviderReference, _Mapping]] = ..., llm: _Optional[_Union[ProviderReference, _Mapping]] = ..., identity_resolver: _Optional[_Union[ProviderReference, _Mapping]] = ..., message_source: _Optional[_Union[ProviderReference, _Mapping]] = ..., features: _Optional[_Union[NamespaceFeatureConfig, _Mapping]] = ...) -> None: ...

class NamespaceInfo(_message.Message):
    __slots__ = ("namespace_id", "status", "created_at", "updated_at", "last_active_at", "version", "quota", "purge_after", "config_version", "config")
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVE_AT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    QUOTA_FIELD_NUMBER: _ClassVar[int]
    PURGE_AFTER_FIELD_NUMBER: _ClassVar[int]
    CONFIG_VERSION_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    status: NamespaceStatus
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    last_active_at: _timestamp_pb2.Timestamp
    version: int
    quota: NamespaceQuota
    purge_after: _timestamp_pb2.Timestamp
    config_version: int
    config: NamespaceConfig
    def __init__(self, namespace_id: _Optional[str] = ..., status: _Optional[_Union[NamespaceStatus, str]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., last_active_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., version: _Optional[int] = ..., quota: _Optional[_Union[NamespaceQuota, _Mapping]] = ..., purge_after: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., config_version: _Optional[int] = ..., config: _Optional[_Union[NamespaceConfig, _Mapping]] = ...) -> None: ...

class NamespaceResourceUsage(_message.Message):
    __slots__ = ("active_requests", "storage_bytes")
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    STORAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    active_requests: int
    storage_bytes: int
    def __init__(self, active_requests: _Optional[int] = ..., storage_bytes: _Optional[int] = ...) -> None: ...

class NamespaceHealth(_message.Message):
    __slots__ = ("namespace", "runtime_state", "healthy", "resource_usage", "last_error")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_STATE_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_USAGE_FIELD_NUMBER: _ClassVar[int]
    LAST_ERROR_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    runtime_state: NamespaceRuntimeState
    healthy: bool
    resource_usage: NamespaceResourceUsage
    last_error: str
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ..., runtime_state: _Optional[_Union[NamespaceRuntimeState, str]] = ..., healthy: _Optional[bool] = ..., resource_usage: _Optional[_Union[NamespaceResourceUsage, _Mapping]] = ..., last_error: _Optional[str] = ...) -> None: ...

class CreateNamespaceRequest(_message.Message):
    __slots__ = ("namespace_id", "quota", "config")
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    QUOTA_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    quota: NamespaceQuota
    config: NamespaceConfig
    def __init__(self, namespace_id: _Optional[str] = ..., quota: _Optional[_Union[NamespaceQuota, _Mapping]] = ..., config: _Optional[_Union[NamespaceConfig, _Mapping]] = ...) -> None: ...

class CreateNamespaceResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ...) -> None: ...

class GetNamespaceRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class GetNamespaceResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ...) -> None: ...

class ListNamespacesRequest(_message.Message):
    __slots__ = ("page_size", "page_token")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListNamespacesResponse(_message.Message):
    __slots__ = ("namespaces", "next_page_token")
    NAMESPACES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    namespaces: _containers.RepeatedCompositeFieldContainer[NamespaceInfo]
    next_page_token: str
    def __init__(self, namespaces: _Optional[_Iterable[_Union[NamespaceInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class UpdateNamespaceConfigRequest(_message.Message):
    __slots__ = ("namespace_id", "config", "expected_config_version")
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_CONFIG_VERSION_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    config: NamespaceConfig
    expected_config_version: int
    def __init__(self, namespace_id: _Optional[str] = ..., config: _Optional[_Union[NamespaceConfig, _Mapping]] = ..., expected_config_version: _Optional[int] = ...) -> None: ...

class UpdateNamespaceConfigResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ...) -> None: ...

class GetNamespaceCapabilitiesRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class NamespaceCapabilities(_message.Message):
    __slots__ = ("namespace_id", "config_version", "capabilities", "operations", "search_modes", "degraded", "unavailable")
    class CapabilitiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bool
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bool] = ...) -> None: ...
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_VERSION_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    SEARCH_MODES_FIELD_NUMBER: _ClassVar[int]
    DEGRADED_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    config_version: int
    capabilities: _containers.ScalarMap[str, bool]
    operations: _containers.RepeatedScalarFieldContainer[str]
    search_modes: _containers.RepeatedScalarFieldContainer[str]
    degraded: bool
    unavailable: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, namespace_id: _Optional[str] = ..., config_version: _Optional[int] = ..., capabilities: _Optional[_Mapping[str, bool]] = ..., operations: _Optional[_Iterable[str]] = ..., search_modes: _Optional[_Iterable[str]] = ..., degraded: _Optional[bool] = ..., unavailable: _Optional[_Iterable[str]] = ...) -> None: ...

class GetNamespaceCapabilitiesResponse(_message.Message):
    __slots__ = ("capabilities",)
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    capabilities: NamespaceCapabilities
    def __init__(self, capabilities: _Optional[_Union[NamespaceCapabilities, _Mapping]] = ...) -> None: ...

class DisableNamespaceRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class DisableNamespaceResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ...) -> None: ...

class EnableNamespaceRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class EnableNamespaceResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ...) -> None: ...

class DeleteNamespaceRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class DeleteNamespaceResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ...) -> None: ...

class RestoreNamespaceRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class RestoreNamespaceResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[NamespaceInfo, _Mapping]] = ...) -> None: ...

class PurgeNamespaceRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class PurgeNamespaceResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetNamespaceHealthRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class GetNamespaceHealthResponse(_message.Message):
    __slots__ = ("health",)
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    health: NamespaceHealth
    def __init__(self, health: _Optional[_Union[NamespaceHealth, _Mapping]] = ...) -> None: ...
