from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RequestContext(_message.Message):
    __slots__ = ("namespace_id", "agent_id", "principal_id", "conversation_id", "user_id", "group_id", "request_id", "trace_id", "idempotency_key")
    NAMESPACE_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_KEY_FIELD_NUMBER: _ClassVar[int]
    namespace_id: str
    agent_id: str
    principal_id: str
    conversation_id: str
    user_id: str
    group_id: str
    request_id: str
    trace_id: str
    idempotency_key: str
    def __init__(self, namespace_id: _Optional[str] = ..., agent_id: _Optional[str] = ..., principal_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., user_id: _Optional[str] = ..., group_id: _Optional[str] = ..., request_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., idempotency_key: _Optional[str] = ...) -> None: ...

class ErrorDetail(_message.Message):
    __slots__ = ("code", "request_id", "trace_id", "retryable", "details")
    CODE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    code: str
    request_id: str
    trace_id: str
    retryable: bool
    details: _struct_pb2.Struct
    def __init__(self, code: _Optional[str] = ..., request_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., retryable: _Optional[bool] = ..., details: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...
