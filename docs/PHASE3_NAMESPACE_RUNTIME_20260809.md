# 阶段3 Host Port 与 Namespace Runtime 实施记录

记录日期：2026-08-09
实施仓库：`D:/Dev/rdev/A_memorix`
实施分支：`refactor/generic-v2`

## 已确认契约

- `AMemorixEngine` 是多 namespace 的主要进程内入口。
- `NamespaceRuntimeRegistry` 属于内部生命周期实现，不进入顶层公开 API。
- `SDKMemoryKernel` 保留为单 namespace 低层入口，当前阶段不破坏已有调用方式。
- namespace ID 是不可变的小写 ASCII 标识，格式为 `[a-z0-9][a-z0-9._-]{0,127}`。
- 删除后的数据默认隔离7天，隔离期内允许恢复，原 ID 继续保留。
- namespace 是独立的数据、资源和生命周期边界，不与 Agent 强制一一绑定。

## 请求与错误契约

`RequestContext` 统一携带 `namespace_id`、`agent_id`、`principal_id`、`conversation_id`、`user_id`、`group_id`、`request_id`、`trace_id` 和 `idempotency_key`。除 namespace ID 外的业务身份字段可以为空，request ID 和 trace ID 会自动生成。

namespace 管理请求、状态、配额、健康状态和资源用量使用 Pydantic 模型。公共异常包含稳定错误码、请求关联信息、可重试标记和结构化 details。协议层可以直接把这些信息映射到 HTTP、MCP 或 RPC，不需要解析内部异常文本。

## 控制面与物理隔离

控制库位于 `control/namespaces.db`，只保存公开 ID、内部 storage key、状态、时间、版本、配额和清理时间。记忆正文、索引和任务数据不会进入控制库。

公开 ID 从不参与目录拼接。每次创建都会生成随机 UUID storage key，数据位于 `namespaces/<storage_key>`，删除后移动到 `quarantine/<storage_key>`。运行时打开目录前会验证路径边界并拒绝符号链接。

生命周期包含 `creating`、`active`、`inactive`、`quarantined` 和 `purging`。创建、隔离、恢复和清理都先写入可重放状态，再执行对应文件操作。服务启动时会继续完成被进程退出打断的目录移动，不会通过猜测重新创建已经丢失的活动数据目录。

## Runtime Registry

同一 namespace 的并发首次请求共享一个初始化任务。成功初始化后，请求通过租约计数访问 runtime；停用、删除、LRU 和服务关闭必须等待活动租约归零，不能提前释放 SQLite、向量、图或文件锁。

注册表默认最多保留8个活动 runtime，每个 namespace 默认允许64个并发请求，也可以通过 `NamespaceQuota.max_concurrent_requests` 收紧。容量满时只关闭没有活动请求且最久未使用的 runtime。所有 runtime 都在使用时，新 namespace 请求返回可重试的 `capability_unavailable`，不会越过上限创建实例。

`NamespaceQuota.max_storage_bytes` 当前用于资源观测和健康降级，不宣称已经形成硬写入配额。阶段4建立统一 application service 后，写请求会在该层执行容量预检；现阶段低层 `SDKMemoryKernel` 仍允许直接调用，注册表无法可靠判断一次租约属于读还是写。

某个 namespace 初始化失败时，错误只记录在该 namespace 的健康状态中，其他 namespace 仍可创建、加载和访问。关闭失败同样不会丢弃 runtime 引用或关闭控制库，调用方可以重试关闭，避免遗留无法管理的活动写者锁。

## Host Port

现有 Embedding、LLM、身份解析和消息来源接口可以通过 `NamespaceHostPorts` 按 namespace 创建。阶段3增加了控制面使用的 `Clock` 接口。配置和能力对象由宿主工厂提供，不持久化到控制数据库，也不通过全局插件对象回流核心。

Authorization、Telemetry 和宿主 Logger 暂未建立空协议。Authorization 要结合阶段4的认证、授权和 namespace 路由确定最小方法；Telemetry 和 Logger 也要以实际 application service 事件为依据。当前内核继续使用包级标准日志，这些延期不会形成 Agent 类型依赖。

Episode、画像和摘要的领域语义仍按已确认决定保留为内部能力。本阶段只保证它们使用 namespace 独立的存储和 Host Port，不把现有长期对话假设冻结成稳定公共协议。

## 验证范围

阶段3新增测试覆盖：

- namespace ID 与所有管理入口使用同一校验规则
- 控制库中的公开 ID 与物理 storage key 分离
- 删除、隔离、恢复、到期清理和 ID 保留
- 目录移动中断后的启动恢复
- 并发初始化去重和 namespace 请求配额
- 活动请求期间禁止 LRU、停用和关闭
- runtime 关闭失败后的引用保留与重试
- 单 namespace 初始化失败不影响其他 namespace
- 两个真实 `SDKMemoryKernel` 使用相同 external ID、用户 ID 和文本时分别完成写入

最终验证结果：

- `ruff check src/a_memorix tests` 通过。
- `pytest -q` 结果为647项通过、2项按环境变量默认跳过、0项失败。
- `python -m build` 成功生成 Wheel 和 sdist。
- 从构建后的 Wheel 在仓库外导入 `AMemorixEngine`、`RequestContext` 和 `SDKMemoryKernel` 成功。

阶段4需要建立统一 application service 和协议无关的写入、检索、管理 contract，再分别接入 HTTP、固定 namespace MCP 和 RPC。RPC 采用 gRPC 还是 JSON-RPC 仍需通过 ADR 决策。
