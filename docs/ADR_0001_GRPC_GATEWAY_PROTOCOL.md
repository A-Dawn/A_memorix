# ADR 0001：统一网络协议采用 gRPC 与 gRPC-Gateway

- 状态：已接受
- 日期：2026-08-09

## 背景

A_memorix 需要同时服务 Python 宿主、跨语言 Agent、普通 HTTP 客户端和 MCP 客户端。若分别维护 gRPC、JSON-RPC 和手写 HTTP contract，字段、错误码、认证边界和幂等语义会逐渐分叉。

## 决策

Protobuf 是唯一网络 IDL。核心服务使用 gRPC，HTTP/JSON 由 gRPC-Gateway 根据 `google.api.http` 注解转码，OpenAPI 也从同一份 Protobuf 生成。项目不维护第二套 JSON-RPC API，也不把 ConnectRPC 作为核心协议。

MCP 属于 Agent 工具适配层，不是新的业务 IDL。每个 MCP 服务实例在创建时绑定一个 namespace，工具 schema 不暴露 namespace 切换参数。进程内 Python API 继续使用 Pydantic contract，并在 gRPC service 边界与 Protobuf 相互转换。

```text
Python SDK ──gRPC──────────────┐
                              │
HTTP/JSON ──gRPC-Gateway──────┼── gRPC service ── application service ── namespace runtime
                              │
MCP stdio ──fixed namespace───┘
```

## 契约和兼容性

公开包名为 `a_memorix.api.v1`。字段号一经发布不得复用；删除字段必须保留编号和名称。破坏性变更进入新的 Protobuf package 和 HTTP 路径版本。Buf lint、生成结果、Python 客户端、直接 gRPC 与 HTTP/JSON 契约测试共同构成当前门禁。

所有 transport 共用应用层行为：

- `RequestContext` 携带 namespace、请求 ID、追踪 ID、调用方和幂等键
- HTTP `Idempotency-Key` 与消息体中的幂等键冲突时返回 `invalid_argument`
- gRPC 使用标准 status code，并附加类型化 `google.rpc.Status` detail
- HTTP 网关将同一 detail 转换为稳定的 `ErrorEnvelope`
- 结果和错误中的 request ID、trace ID 保持一致

## 认证与隔离

管理操作需要至少32字符的服务管理员令牌，默认从 `A_MEMORIX_ADMIN_TOKEN` 读取。namespace API Key 由管理员创建，只授权一个 namespace，控制库仅保存 SHA-256 摘要。HTTP 的 `Authorization` 请求头由网关转交为 gRPC metadata。

匿名模式只能在 gRPC 服务上显式启用，且该服务只能监听回环地址。HTTP 网关不单独判断凭据，它把认证 metadata 交给 gRPC 服务，避免形成第二套鉴权规则。默认网关后端使用适合同机部署的明文 gRPC；远程部署必须补充 TLS 边界，不能直接使用默认参数。

## 当前范围

v1 覆盖 namespace 生命周期、namespace 健康、API Key 生命周期、文本写入和记忆检索。Episode、画像、摘要等已有内核能力尚未进入公开 Protobuf API，它们需要先完成领域语义通用化。

`max_storage_bytes` 当前执行应用层写入准入：在持有 namespace 写锁时，根据现有占用和请求大小预留至少4096字节。它能在写入前阻止明显超额，但不是文件系统级精确硬配额。

## 影响

同一功能只需要演进一份 Protobuf 网络 contract，HTTP 客户端仍能获得常规 JSON 接口。代价是发布流程需要 Buf、Python 生成物和 Go 网关构建链，gRPC-Gateway 也需要作为独立进程或容器分发。OCI 镜像、跨平台网关二进制、备份恢复和完整 CLI 留在阶段5完成。
