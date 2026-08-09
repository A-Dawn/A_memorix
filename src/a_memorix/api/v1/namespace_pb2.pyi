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

class NamespaceInfo(_message.Message):
    __slots__ = ("namespace_id", "status", "created_at", "updated_at", "last_active_at", "version", "quota", "purge_after")
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVE_AT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    QUOTA_FIELD_NUMBER: _ClassVar[int]
    PURGE_AFTER_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    status: NamespaceStatus
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    last_active_at: _timestamp_pb2.Timestamp
    version: int
    quota: NamespaceQuota
    purge_after: _timestamp_pb2.Timestamp
    def __init__(self, namespace_id: _Optional[str] = ..., status: _Optional[_Union[NamespaceStatus, str]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., last_active_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., version: _Optional[int] = ..., quota: _Optional[_Union[NamespaceQuota, _Mapping]] = ..., purge_after: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("namespace_id", "quota")
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    QUOTA_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    quota: NamespaceQuota
    def __init__(self, namespace_id: _Optional[str] = ..., quota: _Optional[_Union[NamespaceQuota, _Mapping]] = ...) -> None: ...

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
    __slots__ = ()
    def __init__(self) -> None: ...

class ListNamespacesResponse(_message.Message):
    __slots__ = ("namespaces",)
    NAMESPACES_FIELD_NUMBER: _ClassVar[int]
    namespaces: _containers.RepeatedCompositeFieldContainer[NamespaceInfo]
    def __init__(self, namespaces: _Optional[_Iterable[_Union[NamespaceInfo, _Mapping]]] = ...) -> None: ...

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
