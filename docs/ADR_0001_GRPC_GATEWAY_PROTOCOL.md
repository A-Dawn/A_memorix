# ADR 0001：统一网络协议采用 gRPC 与 gRPC-Gateway

- 状态：已接受
- 日期：2026-08-09

## 背景

A_memorix 需要同时服务 Python 程序、跨语言 Agent、普通 HTTP 客户端和 MCP 客户端。若分别维护 gRPC、JSON-RPC 和手写 HTTP API，字段、错误码、认证规则和幂等行为会逐渐出现差异。

## 决策

Protobuf 是唯一网络 IDL。核心服务使用 gRPC，HTTP/JSON 由 gRPC-Gateway 根据 `google.api.http` 注解转码，OpenAPI 也从同一份 Protobuf 生成。项目不维护第二套 JSON-RPC API，也不把 ConnectRPC 作为核心协议。

MCP 属于 Agent 工具接入层，不是新的业务 IDL。每个 MCP 服务实例在创建时绑定一个 Namespace，工具 Schema 不提供 Namespace 切换参数。进程内 Python API 继续使用 Pydantic model，并在进入 gRPC service 时与 Protobuf 相互转换。

```text
Python SDK ──gRPC──────────────┐
                              │
HTTP/JSON ──gRPC-Gateway──────┼── gRPC service ── application service ── Namespace Runtime
                              │
MCP stdio ──fixed Namespace───┘
```

## API 兼容性

公开包名为 `a_memorix.api.v1`。字段号一经发布不得复用；删除字段必须保留编号和名称。破坏性变更进入新的 Protobuf package 和 HTTP 路径版本。每次变更都要通过 Buf lint、代码生成、Python 客户端测试，以及 gRPC 与 HTTP/JSON 的 API 一致性测试。

所有 transport 共用以下应用层行为：

- `RequestContext` 携带 Namespace、请求 ID、追踪 ID、调用方和幂等键
- HTTP `Idempotency-Key` 与消息体中的幂等键冲突时返回 `invalid_argument`
- gRPC 使用标准 status code，并附加类型化 `google.rpc.Status` detail
- HTTP 网关将同一 detail 转换为稳定的 `ErrorEnvelope`
- 结果和错误中的 request ID、trace ID 保持一致

## 认证与隔离

管理操作需要至少32字符的服务管理员令牌，默认从 `A_MEMORIX_ADMIN_TOKEN` 读取。Namespace API Key 由管理员创建，只授权一个 Namespace，管理库仅保存 SHA-256 摘要。HTTP 的 `Authorization` 请求头由网关转交为 gRPC metadata。

匿名模式只能在 gRPC 服务上明确启用，且该服务只能监听回环地址。HTTP 网关不单独判断凭据，它把认证 metadata 交给 gRPC 服务，避免形成第二套鉴权规则。默认网关后端使用适合同机部署的明文 gRPC；远程部署必须启用 TLS，不能直接使用默认参数。

## 当前范围

v1 覆盖 Namespace 生命周期、配置与可用功能查询、API Key 生命周期、单条和批量文本写入、记忆检索、直接读取、单条删除，以及按来源删除 Job。Episode、画像、摘要等已有功能尚未进入公开 Protobuf API，它们的输入、输出和生命周期需要先完成通用化。

写入完成后会持久化幂等记录，并校验同一幂等键对应的请求摘要。正常完成的请求可以在服务重启后重放原响应。元数据、向量库和图存储不属于同一个事务，因此进程在业务写入与幂等记录写入之间异常退出时，重试依靠 external ID 去重，不保证多个存储之间全局 exactly-once。

`max_storage_bytes` 当前在应用层检查写入容量：持有 Namespace 写锁时，根据现有占用和请求大小预留至少4096字节。它能在写入前阻止明显超额，但不是文件系统级精确硬配额。

## 影响

同一功能只需要维护一份 Protobuf API，HTTP 客户端仍能获得常规 JSON 接口。代价是发布流程需要运行 Buf、生成 Python 代码并构建 Go 网关，gRPC-Gateway 也需要作为独立进程或容器发布。OCI 镜像、跨平台网关程序、备份恢复和完整 CLI 留在阶段5完成。
