# 阶段4.1：通用应用 API 补全记录

日期：2026-08-09

## 完成范围

阶段4.1补全了通用版在接入其他 Agent 前必须具备的应用层功能，并继续让每个 Namespace 独立管理数据、配置、鉴权和并发请求。

- Namespace 管理库 Schema 升级到 v3，旧库通过增量迁移保留 Namespace 和 API Key 数据
- 每个 Namespace 保存自己的非敏感配置、Provider 引用、功能开关和配置版本
- 配置只能在 Namespace 停用时更新，支持 `expected_config_version` 乐观并发控制
- Runtime factory 接收完整 `NamespaceInfo`，由 Agent 程序解析 `secret_ref` 并提供实际 Provider
- 单条写入和批量写入使用持久化幂等记录，默认保留24小时
- 增加批量写入、按内部 ID 或 external ID 读取、单条软删除和按来源删除
- 按来源删除使用持久化 Job，提供提交、查询、分页列表和取消待执行任务
- Namespace、API Key 和 Job 列表使用最大100条的游标分页
- gRPC、gRPC-Gateway HTTP/JSON、Python SDK 和固定 Namespace MCP 提供相同功能
- Buf 兼容性检查保证现有 v1 字段号、服务和 HTTP 映射没有破坏性变化

Episode、人物画像和摘要仍保留在核心代码中，没有直接加入通用 v1 API。它们需要先摆脱 MaiBot 语义，形成可解释的通用输入、输出和生命周期。

## 配置与可用功能查询

Namespace 管理库只保存 Provider ID、模型 ID 和 `secret_ref`，不保存实际密钥。Agent 程序通过 `NamespaceHostPortFactory` 接收 Namespace 配置，再把引用解析为 `EmbeddingProvider`、`LLMProvider`、`IdentityResolver` 和 `MessageSource`。

配置更新依次执行停用、更新和启用。这样 Runtime 不会在请求中途切换模型或身份来源。`GetNamespaceCapabilities` 返回实际可用的 Provider、操作、搜索模式、降级状态和不可用功能，调用方不需要根据 Agent 类型猜测功能。

## 幂等处理

幂等记录的键由 Namespace、操作名和调用方幂等键组成，并保存规范化请求摘要。同一键与同一请求会重放已完成响应，同一键用于不同请求会返回 `conflict`。请求 ID、追踪 ID、调用凭据和幂等键本身不进入摘要。

A_memorix 的元数据、向量库和图存储不属于同一个事务。若进程在业务写入完成后、幂等记录标记完成前退出，重试会依靠 external ID 的确定性映射避免重复写入，但响应可能由 `stored_ids` 变为 `skipped_ids`。已经完成并记录的请求可以在进程重启后准确重放。当前保证是单节点、单 Namespace 写者下的至少一次执行和业务去重，不保证多个存储之间全局 exactly-once。

## Job 生命周期

首个 Job 类型是 `delete_by_source`。任务状态为 `pending`、`running`、`succeeded`、`failed`、`cancelled`。只有待执行任务可以取消；服务退出会等待本进程任务完成。若进程非正常退出，下一次启动会把遗留的待执行或运行中任务标记为可重试失败，避免任务永久停留在运行状态。

## 验证范围

- Namespace 管理库 v2 到 v3 的兼容迁移
- 跨 Engine 重启的幂等响应重放和同键异请求冲突
- 配置停用更新、版本冲突和 Runtime 配置可见性
- 批量写入、直接读取、单条删除和来源删除 Job
- gRPC 与 HTTP/JSON 新接口一致性
- 固定 Namespace MCP 工具不提供 Namespace 切换参数
- Buf lint、兼容性检查和代码生成，Go test、Ruff 和全量 Python 测试

备份恢复、可观测性、TLS 部署、OCI 镜像和精确文件系统配额仍属于阶段5。
