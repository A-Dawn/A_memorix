# A_memorix 通用架构与分发计划

状态：执行中
基线日期：2026-08-05
目标版本：2.0.0 预发布系列

执行进度：

| 阶段 | 状态 | 记录 |
| --- | --- | --- |
| 阶段0：冻结和归档 | 已完成 | `docs/PHASE0_BASELINE_20260805.md` |
| 阶段1：修复发布阻断项 | 已完成 | `docs/PHASE1_RELEASE_BLOCKERS_20260805.md` |
| 阶段2：建立通用开发主线 | 待开始 | - |

## 1. 目标

A_memorix 将从面向 MaiBot 的长期记忆实现，演进为可以被不同 Agent 使用的通用记忆服务。

通用版本由 `main` 分支维护。它通过 MCP、HTTP 和 RPC 等协议提供能力，不直接依赖任何具体 Agent 的内部模块。官方支持的 Agent 使用独立集成分支维护适配代码，社区适配器通过单独的扩展仓库登记、验证和分发。

本计划不要求立即重构 MaiBot。MaiBot 当前实现作为行为基线和代码来源保留，除数据正确性修复外，暂不为抽取通用层调整其接口、运行方式或性能结构。

## 2. 已确认决策

| 编号 | 决策 |
| --- | --- |
| D-01 | `main` 维护基础通用版，不承载具体 Agent 的宿主实现 |
| D-02 | 通用版通过 MCP、HTTP、RPC 和 Python SDK 暴露统一能力 |
| D-03 | 所有数据访问以 `namespace` 作为强制隔离边界 |
| D-04 | 官方 Agent 定制通过项目集成分支维护 |
| D-05 | 社区适配器通过独立扩展仓库维护索引和质量门禁 |
| D-06 | 默认许可证为 AGPL-3.0，其他许可安排需通过邮件申请 |
| D-07 | 暂不为了通用化改造 MaiBot，只允许必要的正确性修复 |
| D-08 | 旧版 1.x 插件不作为新架构的兼容目标 |

## 3. 当前基线

### 3.1 代码状态

当前存在四层不同的实现状态：

1. `A_memorix/main` 停留在 2026-03-07，对应 1.0.1 插件结构。
2. 本地跟踪的 `origin/MaiBot_branch` 比 `main` 多 11 个提交。
3. 本地 `MaiBot_branch` 又比远端跟踪分支多 2 个提交。
4. `MaiBot/src/A_memorix` 继续演进到 2026-08-05，相对本地 `MaiBot_branch` 仍有 38 个正式文件发生变化。

MaiBot 内嵌版本约有 123 个 Python 源文件、5.8 万行代码。现有核心能力包括双路检索、稀疏检索、图召回、Episode、人物画像、反馈纠错、生命周期维护、事务与投影恢复、双向量池、导入和调优管理。

### 3.2 测试状态

现有 MaiBot 已提交测试中，与 A_memorix 相关的 66 个测试文件共运行 725 项：

- 719 项通过
- 3 项跳过
- 3 项失败
- Ruff 检查通过

失败集中在真实存储删除、恢复和重启闭环。启动阶段可能在实际 embedding 模型尚未观测前，根据配置推导指纹，把完整的 V2 向量判定为不匹配并隔离。该问题属于发布阻断项，必须在通用版继承相关实现前修复。

### 3.3 独立发布缺口

当前 A_memorix 独立仓库缺少以下基础设施：

- `pyproject.toml` 和标准 Python 构建配置
- 可执行的源码测试
- 持续集成工作流
- Wheel、sdist 和容器发布流程
- 明确的协议版本和兼容矩阵
- 与实际 schema 一致的版本说明

代码中的元数据 schema 已达到 21，README、CHANGELOG 和包版本仍保留较早口径。通用版发布前必须统一软件版本、协议版本、存储 schema 和适配器兼容版本。

## 4. 范围边界

### 4.1 通用版包含

- 记忆写入、去重、检索和排序
- SQLite 权威元数据
- 向量存储和稀疏检索
- 关系图、Episode、人物画像和反馈纠错
- 生命周期、删除、恢复、迁移和故障诊断
- namespace 管理和运行时调度
- MCP、HTTP、RPC 和 Python SDK
- 配置模型、CLI、监控指标和结构化错误
- 通用单元测试、集成测试、迁移测试和性能基准

### 4.2 通用版不包含

- MaiBot 的配置管理器、日志模块和 WebUI Router
- MaiBot 聊天流、用户、群组和消息数据库模型
- MaiBot 的模型配置选择逻辑
- 具体 Agent 的插件清单、启动钩子和 UI
- 社区适配器的实现代码
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

### 5.1 分层约束

`domain`、`storage` 和 `retrieval` 不得导入具体 Agent 模块，也不得依赖 HTTP、MCP 或 RPC 框架。

`application` 负责用例编排，通过 `ports` 调用 embedding、LLM、消息来源、身份解析和日志能力。

`transports` 只处理鉴权、参数解析、协议转换、流式输出和错误映射。所有协议必须调用同一套 application service，不能复制检索、写入或管理逻辑。

`runtime` 管理单个 namespace 的 MemoryEngine、后台任务、存储句柄和健康状态。

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

2.0 初始版本采用物理隔离。每个 namespace 拥有独立的数据目录、数据库、向量索引、图快照、导入任务和写锁。

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

`namespaces.db` 只保存控制面信息，包括公开 namespace ID、内部 storage key、状态、创建时间、版本、配额和最近活动时间。业务记忆不进入控制数据库。

### 6.2 路径安全

- 用户提供的 namespace ID 不直接拼接文件路径。
- storage key 使用内部 UUID 或稳定哈希。
- 所有解析后的路径必须位于 namespace 根目录内。
- 不允许通过绝对路径、`..`、符号链接或路径别名跨越 namespace。
- 导入、备份和恢复同样受 namespace 边界约束。

### 6.3 运行时注册表

`NamespaceRuntimeRegistry` 负责：

- 显式创建、启用、停用和删除 namespace
- 按需加载 MemoryEngine
- 对并发初始化进行去重
- 维护最大活跃 namespace 数量
- 对空闲运行时执行 LRU 关闭
- 等待后台任务退出后释放 SQLite、Faiss 和文件锁
- 隔离单个 namespace 的启动失败
- 暴露 namespace 级健康状态和资源使用量

读操作不能隐式创建 namespace。只有显式创建接口和经过授权的首次写入策略可以创建 namespace，默认采用显式创建。

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

除 `namespace_id` 外，其余字段可按用例设置为可选。`idempotency_key`、外部记忆 ID、删除操作 ID 和后台任务 ID 都必须在 namespace 内计算唯一性。

### 6.5 多进程边界

初始版本采用单服务进程、每个 namespace 单活动写者。HTTP 服务不得通过简单增加多 worker 的方式共享同一 namespace 数据目录。

需要横向扩展时，通过 namespace 分片把不同 namespace 分配到不同服务实例。单个 namespace 的多写者支持不进入 2.0 初始范围。

## 7. 公共契约

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

边界层不再使用无约束的 `Dict[str, Any]` 作为主接口。请求、响应、错误、分页、任务状态和健康状态使用明确的 Pydantic 模型。

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

错误响应必须保留 `request_id`、`trace_id`、可重试标记和结构化 details，不向客户端返回未经处理的内部堆栈。

### 7.3 版本维度

以下版本分别管理，不能继续共用一个模糊版本号：

| 版本 | 用途 |
| --- | --- |
| 软件版本 | Wheel、容器和源码发布版本 |
| API 版本 | HTTP、MCP、RPC 公共语义版本 |
| 存储 schema | SQLite、向量、图和 generation 格式版本 |
| 适配器协议版本 | Agent Adapter 与 Host Port 兼容版本 |

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

namespace 出现在路径中，便于鉴权、限流、审计和日志聚合。写入接口支持 `Idempotency-Key`，请求体中的键与 Header 冲突时直接拒绝。

服务默认只监听回环地址。开放到远程网络时必须配置认证，并建议由反向代理提供 TLS。

### 8.2 MCP

MCP 工具名称与 application service 保持一致。namespace 可以采用两种模式：

- 服务启动时绑定固定 namespace，工具参数中不再重复传递。
- 多 namespace 服务把 `namespace_id` 设为所有工具的必填参数。

初始实现优先支持固定 namespace 模式，降低 Agent 误访问其他 namespace 的风险。多 namespace MCP 需要显式鉴权后再开放。

### 8.3 RPC

RPC 层使用稳定 IDL 描述请求、响应、错误和流式任务事件。具体采用 gRPC 还是 JSON-RPC，在公共契约冻结后通过 ADR 确认。

无论选择哪种实现，RPC 都不得绕过 namespace 鉴权、幂等和 Job 状态机。

### 8.4 Python SDK

SDK 同时支持进程内调用和远程客户端：

```python
engine = AMemorixEngine(...)
await engine.search_memory(request)

client = AMemorixClient(base_url=..., api_key=...)
await client.search_memory(request)
```

进程内和远程客户端共享 contracts，但不共享网络层实现。

## 9. 分支与仓库治理

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

迁移期间在 `refactor/generic-v2` 开发。通过发布门禁后再让 `main` 切换到通用版。旧 `main` 在切换前创建不可变标签和 `legacy/plugin-v1` 分支。

### 9.2 集成分支约束

官方集成分支只允许修改：

- `integrations/<agent>/`
- Agent 专用配置和清单
- Agent 专用启动包装
- Agent 集成测试
- Agent 发布文档

CI 比较集成分支和 `main` 的通用目录。若 `domain`、`storage`、`retrieval`、`application` 或公共 contracts 出现额外修改，检查直接失败。通用修复必须先进入 `main`，再合并到集成分支。

### 9.3 MaiBot 过渡

MaiBot 当前内嵌代码先保留不动。通用版开发完成后，再创建新的 `integration/maibot`：

- 复用现有配置映射和 Host Service 行为
- 保持当前 WebUI 和数据目录兼容
- 不把 MaiBot 的 `src.*` 导入带回通用核心
- 只在集成分支运行 MaiBot 特有测试

现有 `MaiBot_branch` 在新分支通过验收后停止开发，并保留只读历史。

## 10. 扩展仓库

建议创建独立仓库 `A_memorix-extensions`。它是适配器注册表、规范和测试入口，不直接集中托管全部社区实现。

### 10.1 扩展清单

```toml
id = "community.example-agent"
name = "Example Agent Adapter"
version = "1.0.0"
package = "a-memorix-example-agent"
entrypoint = "example_agent.adapter:create_adapter"
core_version = ">=2.0,<3.0"
adapter_protocol = "1"
transports = ["in_process", "http"]
license = "AGPL-3.0-only"
source = "https://example.com/repository"
```

### 10.2 注册表门禁

- 清单 schema 校验
- ID、包名和入口点唯一性检查
- 依赖和核心版本范围检查
- 干净环境安装测试
- Adapter Contract Test
- namespace 隔离测试
- 权限和外部网络访问声明
- 许可证字段检查
- 已知恶意包和依赖风险检查

扩展分为官方、已验证社区、未验证社区三个等级。注册表只表达验证状态，不替第三方作者承担运行安全保证。

### 10.3 社区代码归属

社区作者保留自己的代码仓库、发布节奏和许可证责任。扩展仓库接受清单和验证配置，避免主仓库因为大量第三方依赖产生供应链和维护压力。

## 11. 许可与贡献治理

主仓库默认使用 AGPL-3.0。需要其他许可安排的个人或组织通过项目指定邮箱申请，未经书面确认不产生例外。

发布前增加：

- `LICENSING.md`
- `CONTRIBUTING.md`
- 源文件 SPDX 标识策略
- 第三方依赖许可证清单
- 许可申请联系方式

如果主仓库接受外部贡献，需要在开放贡献前确认贡献授权方式是否支持项目后续授予单独许可安排。此处应经过专业法务复核。

第三方扩展的许可证由扩展作者决定。A_memorix 的许可例外不自动覆盖第三方扩展。

## 12. 打包与分发

### 12.1 Python 包

初始包建议：

```text
pip install a-memorix
pip install a-memorix[vector]
pip install a-memorix[server]
pip install a-memorix[mcp]
pip install a-memorix[imports]
pip install a-memorix[all]
```

第一版以 Python 3.12 为正式支持基线。只有在目标依赖、Faiss 和完整测试矩阵通过后，才扩大 Python 版本范围。

依赖按用途拆分：

- 基础：Pydantic、SQLite相关运行能力、NumPy
- 向量：Faiss
- 图与检索：SciPy、jieba
- 导入：pandas、pyarrow、networkx、rich、tenacity
- HTTP：FastAPI、Uvicorn、认证依赖
- MCP：选定的 MCP SDK
- RPC：选定的 RPC 实现

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

所有 CLI 支持结构化 JSON 输出，便于 Agent、部署脚本和控制面调用。

### 12.3 容器

发布官方 OCI 镜像，至少提供：

- 固定版本标签和不可变 digest
- 非 root 用户运行
- `/data` 持久化卷
- 健康检查
- 回环监听和显式远程监听配置
- 优雅停机和 namespace 运行时关闭
- 镜像依赖清单和校验信息

### 12.4 发布产物

每个稳定版本发布：

- Wheel
- sdist
- OCI 镜像
- 源码归档
- SHA-256 校验文件
- 迁移说明
- API 兼容说明
- 存储 schema 兼容说明

## 13. 配置

配置来源优先级建议固定为：

```text
命令行参数 > 环境变量 > 配置文件 > 默认值
```

配置分为：

- 服务级配置：监听地址、认证、数据根目录、运行时上限
- namespace 默认配置：embedding、检索、生命周期
- namespace 覆盖配置：单个 namespace 的显式差异
- Agent Adapter 配置：只存在于集成分支或扩展包

配置模型必须能输出脱敏结果。密钥不写入普通配置导出、诊断报告或日志。

## 14. 测试与发布门禁

### 14.1 测试分层

- Domain 单元测试
- Storage 真实文件测试
- Host Port 契约测试
- Application 用例测试
- Namespace 隔离测试
- HTTP、MCP、RPC 协议一致性测试
- 历史 schema 迁移测试
- 崩溃、重启、删除和恢复故障注入测试
- 安装和 CLI smoke test
- 性能与检索质量基准

### 14.2 必须迁移的测试

MaiBot 中与以下能力有关的测试迁入 A_memorix：

- MetadataStore 和 schema 迁移
- VectorStore 原子性、指纹、压实和恢复
- GraphStore 持久化和投影协议
- Episode 完整覆盖与任务状态
- 生命周期和删除恢复
- 检索范围、阈值和图传播
- 人物画像事实账本
- 导入任务和取消语义

MaiBot 只保留以下测试：

- `MemoryService` 调用包装
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
- 旧 schema 到当前 schema 的迁移

### 14.4 发布门禁

进入 RC 前必须满足：

- 单元和集成测试零失败
- Ruff、类型检查和构建检查通过
- 3 个现有向量重启失败用例恢复通过
- namespace 路径穿越和越权测试通过
- MCP、HTTP、RPC 对同一请求的语义结果一致
- Wheel 和容器在干净环境可启动
- 至少完成一次旧数据备份、迁移、验证和恢复演练
- 文档版本、软件版本、API 版本和 schema 版本一致

## 15. 实施阶段

### 阶段 0：冻结与留档

工作内容：

- 记录两个仓库当前提交和工作区状态
- 为旧 1.x 主线创建归档标签和维护分支
- 为 MaiBot 当前实现创建可追溯快照
- 暂停在 MaiBot 内直接开发通用功能

退出条件：旧代码、当前 MaiBot 实现和后续通用开发都有明确引用，不依赖未提交工作区恢复。

### 阶段 1：修复发布阻断项

工作内容：

- 区分向量二进制损坏和 embedding 空间不兼容
- 在实际模型未观测时禁止隔离完整 V2 向量
- 增加候选模型回退后重启测试
- 增加模型顺序变化、指纹缺失和显式重建测试
- 重新执行当前 A_memorix 测试集

退出条件：现有 3 个真实存储失败消失，向量不兼容不会被误报为存储损坏。

### 阶段 2：建立通用开发主线

工作内容：

- 创建 `refactor/generic-v2`
- 引入标准 `src/` 包结构和 `pyproject.toml`
- 迁入通用测试
- 统一包名为 `a_memorix`
- 统一显式数据目录和配置注入
- 移除通用核心中的 `src.*`、MaiBot配置和聊天流依赖

退出条件：核心测试可以在不安装、不导入 MaiBot 的干净环境运行。

### 阶段 3：Host Port 与 Namespace Runtime

工作内容：

- 提取最小 Host Port
- 建立 RequestContext 和类型化 contracts
- 实现 namespace 控制数据库
- 实现物理目录隔离
- 实现 NamespaceRuntimeRegistry 和 LRU 关闭
- 增加 namespace 级锁、健康状态和资源上限

退出条件：两个 namespace 使用相同 external ID、用户 ID 和文本时仍完全隔离，任一 namespace 故障不影响其他 namespace。

### 阶段 4：统一协议层

工作内容：

- 实现 HTTP v1
- 实现 MCP 工具适配
- 通过 ADR 选择 RPC 具体实现并实现 IDL
- 实现 Python SDK
- 建立跨协议契约测试

退出条件：同一 application request 通过不同协议调用时，状态变化、错误码和结果语义一致。

### 阶段 5：打包和运维能力

工作内容：

- 拆分基础依赖和 extras
- 实现 CLI
- 构建 Wheel、sdist 和 OCI 镜像
- 实现 namespace 备份与恢复
- 补充诊断、指标和结构化日志

退出条件：全新环境可以通过 pip 或容器启动服务，创建 namespace，写入、检索、备份并恢复数据。

### 阶段 6：官方 Agent 分支

工作内容：

- 创建 `integration/maibot`
- 制定集成分支目录白名单
- 建立核心漂移检查
- 复用现有 MaiBot 行为，不进行额外架构优化
- 为后续官方 Agent 提供适配模板

退出条件：集成分支可以持续合并 `main`，且通用核心没有项目私有修改。

### 阶段 7：扩展生态

工作内容：

- 创建 `A_memorix-extensions`
- 发布 Adapter Protocol 和示例适配器
- 实现清单 schema、验证工具和 CI
- 建立官方、已验证社区、未验证社区分级
- 完善许可和安全说明

退出条件：社区作者可以在独立仓库完成适配器，并通过提交清单进入扩展索引。

### 阶段 8：2.0 发布

发布顺序：

```text
2.0.0a1 -> 2.0.0a2 -> 2.0.0b1 -> 2.0.0rc1 -> 2.0.0
```

Alpha 验证接口和 namespace 模型，Beta 冻结公共 contracts，RC 只接受发布阻断修复。

## 16. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 项目分支再次修改核心 | CI 路径保护，通用修复必须先进入 `main` |
| namespace 数量增加导致资源耗尽 | 按需加载、LRU关闭、活跃上限、配额和指标 |
| 向量模型变化导致数据不可用 | 指纹状态机、显式重建、禁止把不兼容当成损坏 |
| 三种协议语义漂移 | 统一 application service 和跨协议契约测试 |
| 历史数据迁移破坏存储 | 预检、备份、幂等迁移、后置校验和恢复演练 |
| 依赖过重影响安装 | extras 拆分、Wheel smoke test、容器分发 |
| 社区扩展引入供应链风险 | 注册表不托管代码、安装测试、权限声明和分级展示 |
| 许可例外与外部贡献冲突 | 发布贡献规则前进行专业法务复核 |

## 17. 暂待确认事项

以下事项不阻塞核心抽取，但必须在对应阶段前形成 ADR：

- RPC 采用 gRPC、JSON-RPC 或其他实现
- HTTP 的正式认证方案
- namespace 配额模型
- 备份的一致性协议和远程对象存储支持
- 是否提供嵌入式无服务模式的长期兼容承诺
- 官方扩展签名和撤回机制
- 许可申请邮箱和处理流程

## 18. 第一批工作清单

- [ ] 固化旧主线和 MaiBot 当前实现快照
- [ ] 修复 3 个向量重启失败
- [ ] 创建通用开发分支
- [ ] 增加 `pyproject.toml`
- [ ] 迁入通用测试
- [ ] 建立 Host Port
- [ ] 建立 RequestContext
- [ ] 实现 namespace 控制面和物理隔离
- [ ] 建立 NamespaceRuntimeRegistry
- [ ] 实现 HTTP v1
- [ ] 实现 MCP 适配
- [ ] 确定并实现 RPC
- [ ] 构建 Python SDK、CLI 和容器
- [ ] 建立官方集成分支约束
- [ ] 创建扩展仓库规范
- [ ] 完成 2.0.0 Alpha 发布门禁
