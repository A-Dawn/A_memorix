# 阶段4.1：通用应用契约补全记录

日期：2026-08-09

## 完成范围

阶段4.1补全了通用版在接入其他 Agent 前必须具备的应用层能力，并继续以 namespace 作为独立的数据、配置、鉴权和并发边界。

- 控制库 schema 升级到v3，旧库通过增量迁移保留 namespace 和 API Key 数据
- namespace 持久化非敏感配置、Provider 引用、功能开关和独立配置版本
- 配置只能在 namespace 停用时更新，支持 `expected_config_version` 乐观并发控制
- 运行时工厂接收完整 `NamespaceInfo`，由宿主解析 `secret_ref` 并提供实际 Provider
- 单条写入和批量写入使用持久化幂等账本，默认保留24小时
- 增加批量写入、按内部 ID 或 external ID 读取、单条软删除和按来源删除
- 按来源删除使用持久化 Job，提供提交、查询、分页列表和取消待执行任务
- namespace、API Key 和 Job 列表使用最大100条的游标分页
- gRPC、gRPC-Gateway HTTP/JSON、Python SDK 和固定 namespace MCP 暴露一致能力
- Buf breaking gate 保证现有 v1 字段号、服务和 HTTP 映射没有破坏性变化

Episode、人物画像和摘要仍保留在内核中，没有直接发布为通用 v1 contract。它们需要先摆脱 MaiBot 语义，形成可解释的通用输入、输出和生命周期。

## 配置与能力发现

控制库只保存 Provider ID、模型 ID 和 `secret_ref`，不保存实际密钥。宿主通过 `NamespaceHostPortFactory` 接收 namespace 配置，再把引用解析为 `EmbeddingProvider`、`LLMProvider`、`IdentityResolver` 和 `MessageSource`。

配置更新采用停用、更新、启用的生命周期。这样运行时不会在请求中途热切换模型或身份来源。`GetNamespaceCapabilities` 返回实际运行时通道、可用操作、搜索模式、降级状态和不可用能力，调用方不需要根据 Agent 类型猜测功能。

## 幂等边界

账本键由 namespace、操作名和调用方幂等键组成，并保存规范化请求摘要。同一键与同一请求会重放已完成响应，同一键用于不同请求会返回 `conflict`。请求 ID、追踪 ID、调用凭据和幂等键本身不进入摘要。

A_memorix 的元数据、向量库和图存储不是一个分布式事务。若进程在业务写入完成后、账本标记完成前退出，重试会依靠 external ID 的确定性映射避免重复写入，但响应可能由 `stored_ids` 变为 `skipped_ids`。账本已经完成的请求可跨进程重启精确重放。当前保证是单节点、单 namespace 写者下的可靠至少一次执行和业务去重，不宣称跨存储全局 exactly-once。

## Job 生命周期

首个 Job 类型是 `delete_by_source`。任务状态为 `pending`、`running`、`succeeded`、`failed`、`cancelled`。只有待执行任务可以取消；服务退出会等待本进程任务完成。若进程非正常退出，下一次启动会把遗留的待执行或运行中任务标记为可重试失败，避免任务永久停留在运行状态。

## 验证范围

- 控制库v2到v3兼容迁移
- 跨 Engine 重启的幂等响应重放和同键异请求冲突
- 配置停用更新、版本冲突和运行时配置可见性
- 批量写入、直接读取、单条删除和来源删除 Job
- gRPC 与 HTTP/JSON 新接口一致性
- 固定 namespace MCP 工具不暴露 namespace 切换参数
- Buf lint、breaking、生成，Go test，Ruff和全量 Python 测试

备份恢复、可观测性、TLS 部署、OCI 镜像和精确文件系统配额仍属于阶段5。
