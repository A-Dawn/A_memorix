# 阶段4：统一协议层完成记录

日期：2026-08-09

## 完成范围

阶段4建立了第一版通用应用接口和统一网络协议。Protobuf 是唯一网络 IDL，gRPC 是核心 RPC，HTTP/JSON 由 gRPC-Gateway 根据同一份 `google.api.http` 注解映射。项目没有引入第二套 JSON-RPC 或 ConnectRPC contract。

本阶段交付内容包括：

- `a_memorix.api.v1` namespace、认证、文本写入和记忆检索 contract
- Buf lint、锁定版本的 Python/Go 代码生成和 OpenAPI 生成
- 基于 `grpc.aio` 的服务端、类型化错误详情和 Python 客户端
- gRPC-Gateway HTTP/JSON 网关及稳定错误 envelope
- 管理员令牌、只保存摘要的 namespace API Key、撤销和过期处理
- 类型化 `AMemorixEngine.ingest_text`、`search_memory` 和写入准入
- 固定 namespace MCP 工具适配器
- 直接 gRPC、HTTP/JSON 和 MCP 的真实客户端契约测试

## 隔离与安全边界

管理员令牌用于 namespace 与 API Key 生命周期操作。namespace API Key 只允许访问所属 namespace，不能列出其他 namespace，也不能执行管理操作。密钥明文只在创建响应中出现，控制数据库保存 SHA-256 摘要。

gRPC 服务默认只监听回环地址，并要求至少32字符的管理员令牌。匿名模式需要显式启用，且不能监听非回环地址。HTTP 网关转交认证 metadata，不维护第二套身份数据库。默认网关到 gRPC 的链路只适合同机部署，远程部署仍需 TLS 边界。

MCP 实例在创建时绑定 namespace，工具 schema 中没有 `namespace_id`。不同 namespace 需要创建不同实例。当前只支持进程内和 stdio 作为受支持的公开运行方式。

## 兼容性约束

Protobuf 字段号不得复用。删除字段时保留字段号和名称，破坏性变更进入新的 package 与 HTTP 路径版本。显式的 `limit=0` 与进程内 contract 一样返回 `invalid_argument`，未设置时才使用默认值5。

`max_storage_bytes` 当前是应用层写入准入，不是精确文件系统硬配额。写入在 namespace 锁内根据现有占用和请求体大小预留至少4096字节，重复 external ID 不会因配额预检破坏幂等重放。

## 验证结果

- Python 全量测试：652 passed、2 skipped
- 协议测试：5 passed，覆盖 API Key、gRPC、HTTP/JSON 和 MCP
- Ruff：通过
- Buf lint、build、generate：通过
- Go test：通过
- Wheel、sdist：构建成功
- 基础 Wheel 在干净 Python 3.12 环境安装并导入成功，未安装 `grpc` 和 `mcp`

两项跳过项是需要 `A_MEMORIX_RUN_LARGE_MIGRATION_TEST=1` 的大规模迁移压测，与协议层无关。

## 延后事项

Episode、画像、摘要等功能尚未进入 v1 Protobuf，它们需要先完成领域语义通用化。完整 CLI、跨平台网关二进制、OCI 镜像、TLS 部署配置、备份恢复、指标和结构化日志属于阶段5。
