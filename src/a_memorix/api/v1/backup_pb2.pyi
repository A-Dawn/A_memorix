import datetime

from a_memorix.api.v1 import namespace_pb2 as _namespace_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NamespaceBackupInfo(_message.Message):
    __slots__ = ("backup_id", "source_namespace_id", "created_at", "format_version", "producer_version", "source_config_version", "archive_size_bytes", "data_size_bytes", "file_count", "sha256")
    BACKUP_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    PRODUCER_VERSION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_CONFIG_VERSION_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    DATA_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    FILE_COUNT_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    backup_id: str
    source_namespace_id: str
    created_at: _timestamp_pb2.Timestamp
    format_version: int
    producer_version: str
    source_config_version: int
    archive_size_bytes: int
    data_size_bytes: int
    file_count: int
    sha256: str
    def __init__(self, backup_id: _Optional[str] = ..., source_namespace_id: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., format_version: _Optional[int] = ..., producer_version: _Optional[str] = ..., source_config_version: _Optional[int] = ..., archive_size_bytes: _Optional[int] = ..., data_size_bytes: _Optional[int] = ..., file_count: _Optional[int] = ..., sha256: _Optional[str] = ...) -> None: ...

class CreateNamespaceBackupRequest(_message.Message):
    __slots__ = ("namespace_id",)
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    def __init__(self, namespace_id: _Optional[str] = ...) -> None: ...

class CreateNamespaceBackupResponse(_message.Message):
    __slots__ = ("backup",)
    BACKUP_FIELD_NUMBER: _ClassVar[int]
    backup: NamespaceBackupInfo
    def __init__(self, backup: _Optional[_Union[NamespaceBackupInfo, _Mapping]] = ...) -> None: ...

class GetNamespaceBackupRequest(_message.Message):
    __slots__ = ("backup_id",)
    BACKUP_ID_FIELD_NUMBER: _ClassVar[int]
    backup_id: str
    def __init__(self, backup_id: _Optional[str] = ...) -> None: ...

class GetNamespaceBackupResponse(_message.Message):
    __slots__ = ("backup",)
    BACKUP_FIELD_NUMBER: _ClassVar[int]
    backup: NamespaceBackupInfo
    def __init__(self, backup: _Optional[_Union[NamespaceBackupInfo, _Mapping]] = ...) -> None: ...

class ListNamespaceBackupsRequest(_message.Message):
    __slots__ = ("source_namespace_id", "page_size", "page_token")
    SOURCE_NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    source_namespace_id: str
    page_size: int
    page_token: str
    def __init__(self, source_namespace_id: _Optional[str] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListNamespaceBackupsResponse(_message.Message):
    __slots__ = ("backups", "next_page_token")
    BACKUPS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    backups: _containers.RepeatedCompositeFieldContainer[NamespaceBackupInfo]
    next_page_token: str
    def __init__(self, backups: _Optional[_Iterable[_Union[NamespaceBackupInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class DeleteNamespaceBackupRequest(_message.Message):
    __slots__ = ("backup_id",)
    BACKUP_ID_FIELD_NUMBER: _ClassVar[int]
    backup_id: str
    def __init__(self, backup_id: _Optional[str] = ...) -> None: ...

class DeleteNamespaceBackupResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DownloadNamespaceBackupRequest(_message.Message):
    __slots__ = ("backup_id", "offset", "max_bytes")
    BACKUP_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    backup_id: str
    offset: int
    max_bytes: int
    def __init__(self, backup_id: _Optional[str] = ..., offset: _Optional[int] = ..., max_bytes: _Optional[int] = ...) -> None: ...

class DownloadNamespaceBackupResponse(_message.Message):
    __slots__ = ("backup", "offset", "data", "next_offset", "complete")
    BACKUP_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    NEXT_OFFSET_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    backup: NamespaceBackupInfo
    offset: int
    data: bytes
    next_offset: int
    complete: bool
    def __init__(self, backup: _Optional[_Union[NamespaceBackupInfo, _Mapping]] = ..., offset: _Optional[int] = ..., data: _Optional[bytes] = ..., next_offset: _Optional[int] = ..., complete: _Optional[bool] = ...) -> None: ...

class BeginNamespaceBackupUploadRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BeginNamespaceBackupUploadResponse(_message.Message):
    __slots__ = ("upload_id", "next_offset")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    NEXT_OFFSET_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    next_offset: int
    def __init__(self, upload_id: _Optional[str] = ..., next_offset: _Optional[int] = ...) -> None: ...

class UploadNamespaceBackupChunkRequest(_message.Message):
    __slots__ = ("upload_id", "offset", "data")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    offset: int
    data: bytes
    def __init__(self, upload_id: _Optional[str] = ..., offset: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class UploadNamespaceBackupChunkResponse(_message.Message):
    __slots__ = ("upload_id", "next_offset")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    NEXT_OFFSET_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    next_offset: int
    def __init__(self, upload_id: _Optional[str] = ..., next_offset: _Optional[int] = ...) -> None: ...

class CompleteNamespaceBackupUploadRequest(_message.Message):
    __slots__ = ("upload_id", "expected_sha256")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_SHA256_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    expected_sha256: str
    def __init__(self, upload_id: _Optional[str] = ..., expected_sha256: _Optional[str] = ...) -> None: ...

class CompleteNamespaceBackupUploadResponse(_message.Message):
    __slots__ = ("backup",)
    BACKUP_FIELD_NUMBER: _ClassVar[int]
    backup: NamespaceBackupInfo
    def __init__(self, backup: _Optional[_Union[NamespaceBackupInfo, _Mapping]] = ...) -> None: ...

class AbortNamespaceBackupUploadRequest(_message.Message):
    __slots__ = ("upload_id",)
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    def __init__(self, upload_id: _Optional[str] = ...) -> None: ...

class AbortNamespaceBackupUploadResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RestoreNamespaceFromBackupRequest(_message.Message):
    __slots__ = ("backup_id", "target_namespace_id")
    BACKUP_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    backup_id: str
    target_namespace_id: str
    def __init__(self, backup_id: _Optional[str] = ..., target_namespace_id: _Optional[str] = ...) -> None: ...

class RestoreNamespaceFromBackupResponse(_message.Message):
    __slots__ = ("namespace",)
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    namespace: _namespace_pb2.NamespaceInfo
    def __init__(self, namespace: _Optional[_Union[_namespace_pb2.NamespaceInfo, _Mapping]] = ...) -> None: ...
