# A_memorix 通用架构与分发计划

状态：执行中
基线日期：2026-08-05
目标版本：2.0.0 预发布系列

执行进度：

| 阶段 | 状态 | 记录 |
| --- | --- | --- |
| 阶段0：冻结和归档 | 已完成 | `docs/PHASE0_BASELINE_20260805.md` |
| 阶段1：修复影响发布的问题 | 已完成 | `docs/PHASE1_RELEASE_BLOCKERS_20260805.md` |
| 阶段2：建立 A_memorix 2.0 包结构 | 已完成 | `docs/PHASE2_GENERIC_MAINLINE_20260805.md` |
| 阶段2.1：移除 MaiBot Runtime 依赖 | 已完成 | `docs/PHASE2_1_RUNTIME_BOUNDARIES_20260808.md` |
| 阶段3：Host Port 与 Namespace Runtime | 已完成 | `docs/PHASE3_NAMESPACE_RUNTIME_20260809.md` |
| 阶段4：统一协议层 | 已完成 | `docs/PHASE4_UNIFIED_PROTOCOL_20260809.md` |
| 阶段4.1：通用应用 API 补全 | 已完成 | `docs/PHASE4_1_GENERIC_APPLICATION_20260809.md` |
| 阶段5：打包和运维能力 | 已完成 | `docs/PHASE5_PACKAGING_AND_OPERATIONS_20260809.md` |
| 阶段6：官方 Agent 分支 | 独立分支推进 | 不在通用代码分支实施 |
| 阶段7：扩展生态 | 已完成 | `docs/PHASE7_ADAPTER_ECOSYSTEM_20260809.md` |
| 阶段8：2.0 发布 | 执行中 | `docs/PHASE8_ALPHA_RELEASE_GATE_20260809.md` |

## 1. 目标

A_memorix 将从面向 MaiBot 的长期记忆实现，演进为可以被不同 Agent 使用的通用记忆服务。

通用版本由 `main` 分支维护。它通过 MCP、HTTP 和 RPC 提供功能，不直接依赖任何具体 Agent 的内部模块。官方支持的 Agent 使用独立集成分支维护接入代码，社区 Adapter 通过单独的扩展仓库登记、验证和分发。

本计划不要求立即重构 MaiBot。MaiBot 当前实现作为行为基线和代码来源保留，除数据正确性修复外，暂不为抽取通用层调整其接口、运行方式或性能结构。

## 2. 已确认决策

| 编号 | 决策 |
| --- | --- |
| D-01 | `main` 维护基础通用版，不包含具体 Agent 的内部实现 |
| D-02 | 通用版通过 MCP、HTTP、RPC 和 Python SDK 提供一致的功能 |
| D-03 | 每个 `Namespace` 使用独立的数据、资源和生命周期 |
| D-04 | 官方 Agent 定制通过项目集成分支维护 |
| D-05 | 社区 Adapter 通过独立扩展仓库维护列表并接受自动检查 |
| D-06 | 默认许可证为 AGPL-3.0-only，协议变更或其他许可安排需发送邮件至 contact@luminarc.tech 申请 |
| D-07 | 暂不为了通用化改造 MaiBot，只允许必要的正确性修复 |
| D-08 | 旧版 1.x 插件不作为新架构的兼容目标 |

## 3. 当前基线

### 3.1 代码状态

当前存在四层不同的实现状态：

1. `A_memorix/main` 停留在 2026-03-07，对应 1.0.1 插件结构。
2. 本地跟踪的 `origin/MaiBot_branch` 比 `main` 多 11 个提交。
3. 本地 `MaiBot_branch` 又比远端跟踪分支多 2 个提交。
4. `MaiBot/src/A_memorix` 继续演进到 2026-08-05，相对本地 `MaiBot_branch` 仍有 38 个正式文件发生变化。

MaiBot 内嵌版本约有123个 Python 源文件、5.8万行代码。现有功能包括双路检索、稀疏检索、图召回、Episode、人物画像、反馈纠错、生命周期维护、事务失败恢复、图数据恢复、双向量池、导入和调优管理。

### 3.2 测试状态

现有 MaiBot 已提交测试中，与 A_memorix 相关的 66 个测试文件共运行 725 项：

- 719 项通过
- 3 项跳过
- 3 项失败
- Ruff 检查通过

失败集中在真实存储删除、恢复和重启流程。启动阶段可能在实际 embedding 模型尚未确认前，根据配置推导指纹，把完整的 V2 向量判定为不匹配并隔离。该问题会影响发布，必须在通用版继承相关实现前修复。

### 3.3 独立发布缺口

当前 A_memorix 独立仓库缺少以下基础设施：

- `pyproject.toml` 和标准 Python 构建配置
- 可执行的源码测试
- 持续集成工作流
- Wheel、sdist 和容器发布流程
- 明确的协议版本和兼容矩阵
- 与实际 Schema 一致的版本说明

代码中的元数据 Schema 已达到21，README、CHANGELOG 和包版本仍保留较早说明。通用版发布前必须统一软件版本、协议版本、存储 Schema 和 Adapter 兼容版本。

## 4. 包含与不包含的内容

### 4.1 通用版包含

- 记忆写入、去重、检索和排序
- SQLite 权威元数据
- 向量存储和稀疏检索
- 关系图、Episode、人物画像和反馈纠错
- 生命周期、删除、恢复、迁移和故障诊断
- Namespace 管理和 Runtime 调度
- MCP、HTTP、RPC 和 Python SDK
- 配置模型、CLI、监控指标和结构化错误
- 通用单元测试、集成测试、迁移测试和性能基准

### 4.2 通用版不包含

- MaiBot 的配置管理器、日志模块和 WebUI Router
- MaiBot 聊天流、用户、群组和消息数据库模型
- MaiBot 的模型配置选择逻辑
- 具体 Agent 的 Plugin Manifest、启动 hook 和 UI
- 社区 Adapter 的实现代码
- 旧版 slash 命令和旧插件 Web 页面

### 4.3 MaiBot 暂缓范围

本阶段不在 MaiBot 中进行以下改造：

- 不替换现有 Host Service
- 不改为远程 HTTP、MCP 或 RPC 调用
- 不为了通用接口调整 WebUI
- 不为了包结构重新组织现有导入
- 不进行与正确性无关的大规模重构

向量持久化、迁移、删除恢复等数据正确性问题不在暂缓范围内。此类修复先在通用实现完成验证，必要时以最小补丁回补 MaiBot。

## 5. 目标架构

```text
src/a_memorix/
  domain/
    memory/
    episode/
    profile/
    lifecycle/
  application/
    ingest/
    search/
    administration/
    jobs/
  ports/
    embedding.py
    llm.py
    identity.py
    message_source.py
    logger.py
    auth.py
  storage/
    metadata/
    vector/
    graph/
    migrations/
  runtime/
    engine.py
    namespace_registry.py
    lifecycle.py
    health.py
  contracts/
    context.py
    requests.py
    responses.py
    errors.py
  transports/
    http/
    mcp/
    rpc/
  sdk/
  cli/
```

### 5.1 模块依赖规则

`domain`、`storage` 和 `retrieval` 不得导入具体 Agent 模块，也不得依赖 HTTP、MCP 或 RPC 框架。

`application` 负责组织用例，通过 `ports` 调用 embedding、LLM、消息来源、身份解析和日志接口。

`transports` 只处理鉴权、参数解析、协议转换、流式输出和错误映射。所有协议必须调用同一套 application service，不能复制检索、写入或管理逻辑。

`runtime` 管理单个 Namespace 的 MemoryEngine、后台任务、存储句柄和健康状态。

### 5.2 Host Port

需要从当前 `src.*` 直接依赖中抽取以下接口：

- `EmbeddingProvider`
- `LLMProvider`
- `MessageSource`
- `IdentityResolver`
- `AuthorizationProvider`
- `TelemetrySink`
- `LoggerProvider`
- `Clock`

第一阶段只抽取当前代码真实需要的最小方法，不建立覆盖所有未来 Agent 的大接口。

## 6. Namespace 隔离

### 6.1 初始存储模型

2.0 初始版本采用物理隔离。每个 Namespace 拥有独立的数据目录、数据库、向量索引、图快照、导入任务和写锁。

```text
data/
  control/
    namespaces.db
  namespaces/
    <storage_key>/
      metadata/
      vectors/
      graph/
      imports/
      artifacts/
      runtime.lock
```

`namespaces.db` 是 Namespace 管理库，只保存公开 Namespace ID、内部 storage key、状态、创建时间、版本、配额和最近活动时间。业务记忆不进入这个数据库。

### 6.2 路径安全

- 用户提供的 Namespace ID 不直接拼接文件路径。
- storage key 使用内部 UUID 或稳定哈希。
- 所有解析后的路径必须位于 Namespace 根目录内。
- 不允许通过绝对路径、`..`、符号链接或路径别名访问其他 Namespace。
- 导入、备份和恢复同样不能跨 Namespace 操作。

### 6.3 Runtime 实例池

`NamespaceRuntimeRegistry` 负责：

- 显式创建、启用、停用和删除 Namespace
- 按需加载 MemoryEngine
- 对并发初始化进行去重
- 维护最大活跃 Namespace 数量
- 对空闲 Runtime 执行 LRU 关闭
- 等待后台任务退出后释放 SQLite、Faiss 和文件锁
- 防止单个 Namespace 的启动失败影响其他 Namespace
- 提供 Namespace 级健康状态和资源使用量

读操作不能隐式创建 Namespace。只有显式创建接口和经过授权的首次写入策略可以创建 Namespace，默认采用显式创建。

### 6.4 请求上下文

所有公开请求统一携带：

```text
namespace_id
agent_id
principal_id
conversation_id
user_id
group_id
request_id
trace_id
idempotency_key
```

除 `namespace_id` 外，其余字段可按用例设置为可选。`idempotency_key`、外部记忆 ID、删除操作 ID 和后台任务 ID 都必须在对应 Namespace 内保持唯一。

### 6.5 多进程限制

初始版本采用单服务进程，每个 Namespace 只允许一个活动写者。HTTP 服务不得通过简单增加多个 worker 的方式共享同一 Namespace 数据目录。

需要横向扩展时，通过 Namespace 分片把不同 Namespace 分配到不同服务实例。单个 Namespace 的多写者支持不进入 2.0 初始范围。

## 7. 公开 API

### 7.1 应用服务

通用版至少提供以下稳定用例：

- `create_namespace`
- `get_namespace`
- `delete_namespace`
- `ingest_text`
- `ingest_summary`
- `search_memory`
- `get_person_profile`
- `maintain_memory`
- `memory_stats`
- `graph_admin`
- `source_admin`
- `episode_admin`
- `profile_admin`
- `import_admin`
- `runtime_admin`
- `backup_namespace`
- `restore_namespace`

管理类长任务使用统一 Job 模型，返回 `job_id`，通过查询或流式事件获取进度。HTTP、MCP 和 RPC 不得为同一任务定义不同状态机。

### 7.2 类型模型

公开 API 不再使用没有字段限制的 `Dict[str, Any]` 作为主要数据类型。请求、响应、错误、分页、任务状态和健康状态使用明确的 Pydantic model。

统一错误码至少包含：

- `invalid_argument`
- `unauthorized`
- `forbidden`
- `namespace_not_found`
- `not_found`
- `conflict`
- `integrity_error`
- `migration_required`
- `capability_unavailable`
- `timeout`
- `cancelled`
- `internal_error`

错误响应必须保留 `request_id`、`trace_id`、可重试标记和结构化 details，不向客户端返回未经处理的内部调用栈。

### 7.3 版本维度

以下版本分别管理，不能继续共用一个模糊版本号：

| 版本 | 用途 |
| --- | --- |
| 软件版本 | Wheel、容器和源码发布版本 |
| API 版本 | HTTP、MCP、RPC 公开行为版本 |
| 存储 Schema | SQLite、向量、图和数据格式版本 |
| Adapter Protocol 版本 | Agent Adapter 与 Host Port 兼容版本 |

软件版本遵循 SemVer。API v1 在 2.x 软件周期内保持向后兼容，破坏性变更进入 API v2。

## 8. 协议层

### 8.1 HTTP

HTTP 路径建议采用：

```text
/v1/namespaces
/v1/namespaces/{namespace_id}
/v1/namespaces/{namespace_id}/memories:ingest
/v1/namespaces/{namespace_id}/memories:search
/v1/namespaces/{namespace_id}/profiles/{person_id}
/v1/namespaces/{namespace_id}/jobs/{job_id}
/v1/namespaces/{namespace_id}/health
```

Namespace 出现在路径中，便于鉴权、限流、审核和日志聚合。写入接口支持 `Idempotency-Key`，请求体中的键与 Header 冲突时直接拒绝。

服务默认只监听回环地址。开放到远程网络时必须配置认证，并建议由反向代理提供 TLS。

### 8.2 MCP

MCP 工具名称与 application service 保持一致。每个服务实例在启动时绑定固定 Namespace，工具参数中不再重复传递 `namespace_id`。需要服务多个 Namespace 时，应分别创建实例，不能在同一 MCP 会话中切换。

### 8.3 RPC

RPC 层采用 Protobuf 和 gRPC 描述请求、响应及错误。HTTP/JSON 由 gRPC-Gateway 根据同一份 IDL 映射，不建立独立 JSON-RPC API。RPC 不得绕过 Namespace 鉴权、幂等处理和 Job 状态机。

### 8.4 Python SDK

SDK 同时支持进程内调用和远程客户端：

```python
engine = AMemorixEngine(...)
await engine.search_memory(request)

client = AMemorixClient(target=..., api_key=...)
await client.search_memory(request)
```

进程内和远程客户端使用相同的数据类型和行为规则，但不共享网络层实现。

## 9. 分支与仓库规则

### 9.1 主仓库分支

建议分支结构：

```text
main
refactor/generic-v2
integration/maibot
integration/<official-agent>
legacy/plugin-v1
release/<version>
```

迁移期间在 `refactor/generic-v2` 开发。通过发布前检查后再让 `main` 切换到通用版。旧 `main` 在切换前创建不可变标签和 `legacy/plugin-v1` 分支。

### 9.2 集成分支约束

官方集成分支只允许修改：

- `integrations/<agent>/`
- Agent 专用配置和 Manifest
- Agent 专用启动程序
- Agent 集成测试
- Agent 发布文档

CI 比较集成分支和 `main` 的通用目录。若 `domain`、`storage`、`retrieval`、`application` 或公开 API 出现额外修改，检查直接失败。通用修复必须先进入 `main`，再合并到集成分支。

### 9.3 MaiBot 过渡

MaiBot 当前内嵌代码先保留不动。通用版开发完成后，再创建新的 `integration/maibot`：

- 复用现有配置映射和 Host Service 行为
- 保持当前 WebUI 和数据目录兼容
- 不把 MaiBot 的 `src.*` 导入带回通用核心
- 只在集成分支运行 MaiBot 特有测试

现有 `MaiBot_branch` 在新分支通过验收后停止开发，并保留只读历史。

## 10. 扩展仓库

建议创建独立仓库 `A_memorix-extensions`。它保存 Adapter 列表、规范和测试配置，不集中托管社区 Adapter 源码。

### 10.1 Adapter Manifest

Adapter Manifest 采用 [Adapter Protocol v1](ADAPTER_PROTOCOL_V1.md)。Manifest Schema、运行方式、权限声明和版本校验由主包提供，远程与进程内示例分别位于 `docs/examples/adapter-remote.toml` 和 `docs/examples/adapter-in-process.toml`。

### 10.2 扩展仓库检查

- Manifest Schema 校验
- ID、包名和入口点唯一性检查
- 依赖和核心版本范围检查
- 干净环境安装测试
- Adapter API 一致性测试
- Namespace 隔离测试
- 权限和外部网络访问声明
- 许可证字段检查
- 已知恶意包和依赖风险检查

扩展分为官方、社区已验证、社区未验证三种状态。扩展仓库只记录验证结果，不替第三方作者承担运行安全责任。

### 10.3 社区代码归属

社区作者保留自己的代码仓库、发布节奏和许可证责任。扩展仓库接受 Manifest 和验证配置，避免主仓库因为大量第三方依赖产生供应链和维护压力。

## 11. 许可与贡献规则

主仓库默认使用 AGPL-3.0-only。任何协议变更或其他许可安排都必须发送邮件至 `contact@luminarc.tech` 申请，未经书面批准不产生例外。

发布前增加：

- `CONTRIBUTING.md`
- 源文件 SPDX 标识策略
- 第三方依赖许可证列表
- 许可申请处理时限与补充材料要求

如果主仓库接受外部贡献，需要在开放贡献前确认贡献授权方式是否支持项目后续授予单独许可安排。此处应经过专业法务复核。

第三方扩展的许可证由扩展作者决定。A_memorix 的许可例外不自动覆盖第三方扩展。

## 12. 打包与分发

### 12.1 Python 包

初始包建议：

```text
pip install a-memorix
pip install a-memorix[vector]
pip install a-memorix[rpc]
pip install a-memorix[mcp]
pip install a-memorix[lpmm]
pip install a-memorix[all]
```

第一版以 Python 3.12 为正式支持基线。只有在目标依赖、Faiss 和完整测试矩阵通过后，才扩大 Python 版本范围。

依赖按用途拆分：

- 基础：Pydantic、SQLite相关运行能力、NumPy
- 向量：Faiss
- 图与检索：SciPy、jieba
- 导入：pandas、pyarrow、networkx、rich、tenacity
- RPC：gRPC Python、Protobuf 和标准状态详情
- HTTP：由独立 Go gRPC-Gateway 从 Protobuf 注解生成，不进入 Python 基础依赖
- MCP：MCP Python SDK，只用于固定 Namespace 服务

已移除独立 Web 服务后仍残留的依赖不得继续进入基础安装集合。

### 12.2 CLI

```text
a-memorix serve
a-memorix mcp
a-memorix namespace create
a-memorix namespace list
a-memorix doctor
a-memorix migrate
a-memorix backup
a-memorix restore
```

所有 CLI 支持结构化 JSON 输出，便于 Agent、部署脚本和管理程序调用。

### 12.3 容器

发布官方 OCI 镜像，至少提供：

- 固定版本标签和不可变 digest
- 非 root 用户运行
- `/data` 持久化卷
- 健康检查
- 回环监听和显式远程监听配置
- 优雅停机和 Namespace Runtime 关闭
- 镜像依赖列表和校验信息

### 12.4 发布内容

每个稳定版本发布：

- Wheel
- sdist
- OCI 镜像
- 源码归档
- SHA-256 校验文件
- 迁移说明
- API 兼容说明
- 存储 Schema 兼容说明

## 13. 配置

配置来源优先级建议固定为：

```text
命令行参数 > 环境变量 > 配置文件 > 默认值
```

配置分为：

- 服务级配置：监听地址、认证、数据根目录、Runtime 上限
- Namespace 默认配置：embedding、检索、生命周期
- Namespace 覆盖配置：单个 Namespace 的明确差异
- Agent Adapter 配置：只存在于集成分支或扩展包

配置模型必须能输出脱敏结果。密钥不写入普通配置导出、诊断报告或日志。

## 14. 测试与发布前检查

### 14.1 测试分类

- Domain 单元测试
- Storage 真实文件测试
- Host Port 接口测试
- Application 用例测试
- Namespace 隔离测试
- HTTP、MCP、RPC 协议一致性测试
- 历史 Schema 迁移测试
- 崩溃、重启、删除和恢复故障注入测试
- 安装和 CLI smoke test
- 性能与检索质量基准

### 14.2 必须迁移的测试

MaiBot 中与以下能力有关的测试迁入 A_memorix：

- MetadataStore 和 schema 迁移
- VectorStore 原子性、指纹、压实和恢复
- GraphStore 持久化和图数据更新规则
- Episode 完整覆盖与任务状态
- 生命周期和删除恢复
- 检索范围、阈值和图传播
- 人物画像事实记录
- 导入任务和取消行为

MaiBot 只保留以下测试：

- `MemoryService` 调用封装
- MaiBot 配置映射
- 聊天流和身份解析
- WebUI Router 和页面
- MaiBot 端到端写入与检索

### 14.3 CI 矩阵

初始持续集成至少覆盖：

- Windows、Linux
- Python 3.12
- 有 Faiss、无 Faiss降级模式
- HTTP、MCP 和 RPC 可选依赖组合
- Wheel 构建后安装测试
- 旧 Schema 到当前 Schema 的迁移

### 14.4 发布要求

进入 RC 前必须满足：

- 单元和集成测试零失败
- Ruff、类型检查和构建检查通过
- 3 个现有向量重启失败用例恢复通过
- Namespace 路径穿越和越权测试通过
- MCP、HTTP、RPC 对同一请求返回一致结果并产生相同状态变化
- Wheel 和容器在干净环境可启动
- 至少完成一次旧数据备份、迁移、验证和恢复演练
- 文档版本、软件版本、API 版本和 Schema 版本一致

## 15. 实施阶段

### 阶段 0：冻结与留档

工作内容：

- 记录两个仓库当前提交和工作区状态
- 为旧 1.x 主线创建归档标签和维护分支
- 为 MaiBot 当前实现创建可追溯快照
- 暂停在 MaiBot 内直接开发通用功能

退出条件：旧代码、当前 MaiBot 实现和后续通用开发都有明确引用，不依赖未提交工作区恢复。

### 阶段 1：修复影响发布的问题

工作内容：

- 区分向量二进制损坏和 embedding 空间不兼容
- 在实际模型未观测时禁止隔离完整 V2 向量
- 增加候选模型回退后重启测试
- 增加模型顺序变化、指纹缺失和显式重建测试
- 重新执行当前 A_memorix 测试集

退出条件：现有 3 个真实存储失败消失，向量不兼容不会被误报为存储损坏。

### 阶段 2：建立 A_memorix 2.0 包结构

工作内容：

- 创建 `refactor/generic-v2`
- 引入标准 `src/` 包结构和 `pyproject.toml`
- 迁入通用测试
- 统一包名为 `a_memorix`
- 统一显式数据目录和配置注入
- 移除通用核心中的 `src.*`、MaiBot 配置和聊天流依赖

退出条件：核心测试可以在不安装、不导入 MaiBot 的干净环境运行。

### 阶段 2.1：移除 MaiBot Runtime 依赖

工作内容：

- 用明确的 `RuntimeServices` interface 替代 `plugin_instance`、全局 Runtime 实例池和动态 MaiBot 对象探测
- 移除通用管理接口中的 MaiBot 数据迁移任务及来源前缀
- 迁入可复用的 LPMM 格式转换器和真实存储测试
- 对照 MaiBot 测试快照记录未迁测试的通用化归属
- 在干净虚拟环境中安装完整 Wheel 依赖并验证公开 API

退出条件：通用源码和测试不含 MaiBot 专属执行路径，核心服务之间只通过明确接口协作，构建出的 Wheel 可在干净环境安装和导入。

### 阶段 3：Host Port 与 Namespace Runtime

工作内容：

- 提取最小 Host Port
- 建立 RequestContext 和明确的请求、响应类型
- 实现 Namespace 管理库
- 实现物理目录隔离
- 实现 NamespaceRuntimeRegistry 和 LRU 关闭
- 增加 Namespace 级锁、健康状态和资源上限

退出条件：两个 Namespace 使用相同 external ID、用户 ID 和文本时仍完全隔离，任一 Namespace 故障不影响其他 Namespace。

### 阶段 4：统一协议层

工作内容：

- 以 Protobuf 建立唯一网络 IDL
- 实现 gRPC v1 服务、认证和类型化错误
- 通过 gRPC-Gateway 提供 HTTP/JSON 并生成 OpenAPI
- 实现固定 Namespace MCP 工具
- 实现 Python gRPC SDK
- 建立 gRPC、HTTP/JSON 和 MCP API 一致性测试

退出条件：同一 application request 通过不同协议调用时，状态变化、错误码和返回结果一致。

### 阶段 4.1：通用应用 API 补全

工作内容：

- 实现 Namespace 持久化配置、配置版本和可用功能查询
- 实现跨重启幂等记录、请求摘要冲突检测和批量写入
- 增加直接读取、单条删除和按来源删除
- 建立持久化 Job 状态机，先承载按来源删除
- 为 Namespace、API Key 和 Job 列表增加游标分页
- 为 gRPC、HTTP/JSON、Python SDK 和固定 Namespace MCP 提供相同功能
- 把 Protobuf 兼容性检查加入 CI

退出条件：其他 Agent 可以只依赖公开 API 完成 Namespace 配置、可用功能查询、基础记忆管理和长任务跟踪，不需要导入核心私有服务。

### 阶段 5：打包和运维能力

工作内容：

- 拆分基础依赖和 extras
- 实现 CLI
- 构建 Wheel、sdist 和 OCI 镜像
- 实现 Namespace 备份与恢复
- 补充诊断、指标和结构化日志

退出条件：全新环境可以通过 pip 或容器启动服务，创建 Namespace，写入、检索、备份并恢复数据。

阶段5已完成离线备份格式、由服务管理的备份文件、统一 CLI、严格配置、健康检查、结构化日志、指标、Trace、TLS、OCI 镜像、Compose 和发布 workflow。实现与测试范围见 `docs/PHASE5_PACKAGING_AND_OPERATIONS_20260809.md`。

### 阶段 6：官方 Agent 分支

工作内容：

- 创建 `integration/maibot`
- 制定集成分支目录白名单
- 检查集成分支是否修改通用核心
- 复用现有 MaiBot 行为，不进行额外架构优化
- 为后续官方 Agent 提供适配模板

退出条件：集成分支可以持续合并 `main`，且通用核心没有项目私有修改。

### 阶段 7：扩展生态

工作内容：

- 创建 `A_memorix-extensions`
- 发布 Adapter Protocol 和 Adapter 示例
- 实现 Manifest Schema、验证工具和 CI
- 记录官方、社区已验证、社区未验证三种状态
- 完善许可和安全说明

退出条件：社区作者可以在独立仓库完成 Adapter，并通过提交 Manifest 进入扩展列表。

主仓库侧 Adapter Protocol v1、Manifest Schema、校验命令、示例和[扩展列表规范](EXTENSIONS_REGISTRY_SPEC.md)已经完成。[A_memorix-extensions](https://github.com/A-Dawn/A_memorix-extensions) 已建立 Manifest 列表、审核状态记录和独立 CI，社区作者可以从自己的代码仓库提交 Manifest，进入社区未验证列表。授予社区已验证状态所需的安装、API、双 Namespace 和供应链深入检查继续作为扩展仓库迭代项，不阻塞阶段7退出条件。

### 阶段 8：2.0 发布

发布顺序：

```text
2.0.0a1 -> 2.0.0a2 -> 2.0.0b1 -> 2.0.0rc1 -> 2.0.0
```

Alpha 验证接口和 Namespace model，Beta 固定公开 API，RC 只接受影响发布的问题修复。

## 16. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 项目分支再次修改核心 | CI 路径保护，通用修复必须先进入 `main` |
| Namespace 数量增加导致资源耗尽 | 按需加载、LRU 关闭、活跃上限、配额和指标 |
| 向量模型变化导致数据不可用 | 指纹状态机、显式重建、禁止把不兼容当成损坏 |
| 三种协议行为不一致 | 统一 application service 和跨协议 API 一致性测试 |
| 历史数据迁移破坏存储 | 预检、备份、幂等迁移、后置校验和恢复演练 |
| 依赖过重影响安装 | extras 拆分、Wheel smoke test、容器分发 |
| 社区扩展引入供应链风险 | 扩展仓库不托管代码、安装测试、权限声明和审核状态展示 |
| 许可例外与外部贡献冲突 | 发布贡献规则前进行专业法务复核 |

## 17. 暂待确认事项

以下事项不阻塞核心抽取，但必须在对应阶段前形成 ADR：

- 远程对象存储保留策略和凭据接入方式
- 是否提供嵌入式无服务模式的长期兼容承诺
- 官方扩展签名和撤回机制

RPC、HTTP 映射、认证方案和写入容量检查已经由阶段4及 [ADR 0001](ADR_0001_GRPC_GATEWAY_PROTOCOL.md) 确定。精确文件系统硬配额仍属于后续运维功能。

## 18. 第一批工作

- [x] 固化旧主线和 MaiBot 当前实现快照
- [x] 修复 3 个向量重启失败
- [x] 创建通用开发分支
- [x] 增加 `pyproject.toml`
- [x] 迁入通用测试
- [x] 建立 Host Port
- [x] 建立 RequestContext
- [x] 实现 Namespace 管理库和物理隔离
- [x] 建立 NamespaceRuntimeRegistry
- [x] 以 Protobuf、gRPC 和 gRPC-Gateway 实现 HTTP/RPC v1
- [x] 实现固定 Namespace MCP 服务
- [x] 构建 Python gRPC SDK 和跨协议 API 一致性测试
- [x] 实现 Namespace 离线备份、分块传输和恢复
- [x] 构建完整 CLI、发布 Wheel/sdist 和 OCI 镜像
- [ ] 建立官方集成分支约束
- [x] 创建扩展仓库规范
- [x] 创建独立扩展仓库、审核状态记录和校验 CI
- [x] 完成 2.0.0 Alpha 发布前检查
