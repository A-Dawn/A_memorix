# 阶段3 Host Port 与 Namespace Runtime 实施记录

记录日期：2026-08-09
实施仓库：`D:/Dev/rdev/A_memorix`
实施分支：`refactor/generic-v2`

## 已确认规则

- `AMemorixEngine` 是多 Namespace 的主要进程内入口。
- `NamespaceRuntimeRegistry` 属于内部生命周期实现，不进入顶层公开 API。
- `SDKMemoryKernel` 保留为单 Namespace 底层入口，当前阶段不破坏已有调用方式。
- Namespace ID 是不可变的小写 ASCII 标识，格式为 `[a-z0-9][a-z0-9._-]{0,127}`。
- 删除后的数据默认隔离7天，隔离期内允许恢复，原 ID 继续保留。
- Namespace 拥有独立的数据、资源和生命周期，不与 Agent 强制一一绑定。

## 请求与错误格式

`RequestContext` 统一携带 `namespace_id`、`agent_id`、`principal_id`、`conversation_id`、`user_id`、`group_id`、`request_id`、`trace_id` 和 `idempotency_key`。除 Namespace ID 外的业务身份字段可以为空，request ID 和 trace ID 会自动生成。

Namespace 管理请求、状态、配额、健康状态和资源用量使用 Pydantic model。公开异常包含稳定错误码、请求关联信息、可重试标记和结构化 details。HTTP、MCP 或 RPC 可以直接映射这些信息，不需要解析内部异常文本。

## Namespace 管理与物理隔离

Namespace 管理库位于 `control/namespaces.db`，只保存公开 ID、内部 storage key、状态、时间、版本、配额和清理时间。记忆正文、索引和任务数据不会进入管理库。

公开 ID 从不参与目录拼接。每次创建都会生成随机 UUID storage key，数据位于 `namespaces/<storage_key>`，删除后移动到 `quarantine/<storage_key>`。Runtime 打开目录前会验证路径边界并拒绝符号链接。

生命周期包含 `creating`、`active`、`inactive`、`quarantined` 和 `purging`。创建、隔离、恢复和清理都先写入可重放状态，再执行对应文件操作。服务启动时会继续完成被进程退出打断的目录移动，不会通过猜测重新创建已经丢失的活动数据目录。

## Runtime Registry

同一 Namespace 的并发首次请求共享一个初始化任务。成功初始化后，请求通过引用计数访问 Runtime；停用、删除、LRU 和服务关闭必须等待活动请求归零，不能提前释放 SQLite、向量、图或文件锁。

Runtime 实例池默认最多保留8个活动实例，每个 Namespace 默认允许64个并发请求，也可以通过 `NamespaceQuota.max_concurrent_requests` 减少上限。容量满时只关闭没有活动请求且最久未使用的 Runtime。所有 Runtime 都在使用时，新 Namespace 请求返回可重试的 `capability_unavailable`，不会越过上限创建实例。

`NamespaceQuota.max_storage_bytes` 当前用于统计资源占用和标记降级状态，不作为严格的写入上限。阶段4建立统一 application service 后，写请求会在该层预先检查容量；现阶段底层 `SDKMemoryKernel` 仍允许直接调用，Runtime 实例池无法可靠判断一次调用属于读还是写。

某个 Namespace 初始化失败时，错误只记录在该 Namespace 的健康状态中，其他 Namespace 仍可创建、加载和访问。关闭失败同样不会丢弃 Runtime 引用或关闭管理库，调用方可以重试关闭，避免遗留无法管理的活动写者锁。

## Host Port

现有 Embedding、LLM、身份解析和消息来源接口可以通过 `NamespaceHostPorts` 按 Namespace 创建。阶段3增加了 Namespace 生命周期管理使用的 `Clock` 接口。配置和功能实现由 Agent 程序提供，不保存到 Namespace 管理库，也不通过全局插件对象传回核心。

Authorization、Telemetry 和 Logger 暂未增加没有调用方的抽象接口。Authorization 要结合阶段4的认证、授权和 Namespace 路由确定最小方法；Telemetry 和 Logger 也要以实际 application service 事件为依据。当前核心继续使用包级标准日志，这些延期不会形成 Agent 类型依赖。

Episode、画像和摘要仍作为内部功能保留。本阶段只保证它们使用 Namespace 独立的存储和 Host Port，不把现有长期对话假设固定为公开 API。

## 验证范围

阶段3新增测试覆盖：

- Namespace ID 与所有管理入口使用同一校验规则
- 控制库中的公开 ID 与物理 storage key 分离
- 删除、隔离、恢复、到期清理和 ID 保留
- 目录移动中断后的启动恢复
- 并发初始化去重和 Namespace 请求配额
- 活动请求期间禁止 LRU、停用和关闭
- Runtime 关闭失败后的引用保留与重试
- 单个 Namespace 初始化失败不影响其他 Namespace
- 两个真实 `SDKMemoryKernel` 使用相同 external ID、用户 ID 和文本时分别完成写入

最终验证结果：

- `ruff check src/a_memorix tests` 通过。
- `pytest -q` 结果为647项通过、2项按环境变量默认跳过、0项失败。
- `python -m build` 成功生成 Wheel 和 sdist。
- 从构建后的 Wheel 在仓库外导入 `AMemorixEngine`、`RequestContext` 和 `SDKMemoryKernel` 成功。

阶段4需要建立统一 application service，以及不依赖传输协议的写入、检索和管理 API，再分别接入 HTTP、固定 Namespace MCP 和 RPC。RPC 采用 gRPC 还是 JSON-RPC 仍需通过 ADR 决策。
