# A_memorix

A_memorix 是面向 AI Agent 的长期记忆内核。2.x 主线正在从 MaiBot 插件演进为独立 Python 包，目标是通过 namespace 隔离的运行时和统一协议，为不同 Agent 提供可组合的写入、检索、图关系、时序证据和记忆维护能力。

当前版本为 `2.0.0a1`。这一版本已经完成通用核心迁移、宿主依赖解耦和 namespace Runtime，HTTP、MCP、RPC 和稳定外部协议仍在后续阶段实现。现阶段适合参与内核开发、适配器验证和数据兼容性测试，不应视为稳定服务版本。

## 当前能力

- 段落、实体、关系和时间元数据存储
- 向量、稀疏和图关系检索
- 单池、双池向量运行时及故障恢复
- Episode、画像、摘要和记忆生命周期的现有内核能力
- 显式数据目录、异步生命周期和单目录写者保护
- 可注入的 Embedding、LLM、身份解析和消息来源接口
- SQLite namespace 控制面、独立物理目录和7天可恢复隔离
- 并发初始化去重、活跃 runtime 上限、请求配额和 LRU 关闭

Episode、画像和摘要已经与 MaiBot 的模块、配置和数据库类型解耦，但它们的领域语义还需要进一步通用化。MaiBot 数据迁移等项目特定行为不会成为通用稳定 API。

## 安装

开发环境要求 Python 3.12 或更高版本。

```powershell
python -m pip install -e ".[test,vector]"
```

分发包名为 `a-memorix`，Python 导入名为 `a_memorix`。正式发布后可通过 `pip install a-memorix` 安装基础包；FAISS 支持位于 `vector` extra，LPMM Parquet 转换支持位于 `lpmm` extra。

## Python 入口

`AMemorixEngine` 是多 namespace 的主要进程内入口。宿主可按 namespace 返回不同的 Provider：

```python
from a_memorix import (
    AMemorixEngine,
    CreateNamespaceRequest,
    NamespaceHostPorts,
    RequestContext,
)

def host_ports(namespace):
    return NamespaceHostPorts(
        embedding_provider=providers[namespace.namespace_id],
    )

engine = AMemorixEngine(
    data_dir="./data",
    host_port_factory=host_ports,
)

async with engine:
    await engine.create_namespace(
        CreateNamespaceRequest(namespace_id="agent-prod")
    )
    context = RequestContext(
        namespace_id="agent-prod",
        agent_id="assistant",
        user_id="user-1",
    )
    async with engine.runtime(context) as memory:
        await memory.ingest_text(
            external_id="document:1",
            source_type="document",
            text="A_memorix stores long-term evidence.",
            user_id=context.user_id or "",
        )
```

`namespace_id` 是不可变的小写 ASCII 标识，格式为 `[a-z0-9][a-z0-9._-]{0,127}`。它不会直接成为目录名，控制库会为每个 namespace 生成内部 UUID。每个 namespace 拥有独立的元数据、向量、图、导入任务和写者锁。

`SDKMemoryKernel(data_dir=...)` 继续作为单 namespace 低层入口保留。现有调用方可以渐进迁移，但新建的多 Agent 宿主应从 `AMemorixEngine` 开始。

删除 namespace 时，运行时会先停止，数据目录会移入隔离区并默认保留7天。隔离期内可恢复，原 ID 不能被重新创建；到期清理或管理员显式清理后才会释放 ID。

## 宿主接口

| 接口 | 用途 | 是否必需 |
| --- | --- | --- |
| `EmbeddingProvider` | 批量生成向量并报告已观测指纹 | 向量能力必需 |
| `LLMProvider` | Episode、画像、摘要等模型任务 | 对应能力必需 |
| `IdentityResolver` | 把外部身份映射为稳定 person ID | 画像能力可选 |
| `MessageSource` | 按会话和时间范围读取消息 | 摘要、反馈修正可选 |
| `Clock` | 为控制面生命周期提供可替换时间源 | 测试和定制调度可选 |

这些接口只表达 A_memorix 已经需要的数据，不暴露宿主的配置对象、数据库模型和内部服务。`NamespaceHostPorts` 用于一次性传递某个 namespace 的可选能力。鉴权、遥测和协议日志接口会在统一协议层出现真实调用方后再确定，避免提前冻结空接口。

## 分支与分发

- 通用基础版最终由 `main` 维护。
- 官方 Agent 适配使用独立集成分支，不把项目私有代码写回通用核心。
- 社区适配器将在独立的 `A_memorix-extensions` 仓库登记、验证和分发。
- 1.x MaiBot 插件历史保留在 `legacy-v1.0.1` 标签和 `legacy/plugin-v1` 分支。

完整路线见 [通用架构与分发计划](docs/GENERIC_ARCHITECTURE_AND_DISTRIBUTION_PLAN.md)。

## 开发验证

```powershell
pytest -q
python -m build
```

两项大规模格式迁移压测默认跳过，可设置 `A_MEMORIX_RUN_LARGE_MIGRATION_TEST=1` 单独执行。

## 许可证

项目默认采用 [GNU AGPL-3.0-only](LICENSE)。任何协议变更或其他许可安排都必须发送邮件至 `contact@luminarc.tech` 申请，只有收到书面批准后才产生例外。具体规则见 [LICENSING.md](LICENSING.md)。

外部贡献规则和社区扩展治理将在开放对应仓库前单独发布。
