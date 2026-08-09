# A_memorix

A_memorix 是面向 AI Agent 的长期记忆内核。2.x 主线正在从 MaiBot 插件演进为独立 Python 包，目标是通过 namespace 隔离的运行时和统一协议，为不同 Agent 提供可组合的写入、检索、图关系、时序证据和记忆维护能力。

当前版本为 `2.0.0a1`。这一版本完成了通用核心迁移和宿主依赖解耦，但 namespace Runtime、HTTP、MCP、RPC 和稳定外部协议仍在后续阶段实现。现阶段适合参与内核开发、适配器验证和数据兼容性测试，不应视为稳定服务版本。

## 当前能力

- 段落、实体、关系和时间元数据存储
- 向量、稀疏和图关系检索
- 单池、双池向量运行时及故障恢复
- Episode、画像、摘要和记忆生命周期的现有内核能力
- 显式数据目录、异步生命周期和单目录写者保护
- 可注入的 Embedding、LLM、身份解析和消息来源接口

Episode、画像和摘要已经与 MaiBot 的模块、配置和数据库类型解耦，但它们的领域语义还需要进一步通用化。MaiBot 数据迁移等项目特定行为不会成为通用稳定 API。

## 安装

开发环境要求 Python 3.12 或更高版本。

```powershell
python -m pip install -e ".[test,vector]"
```

分发包名为 `a-memorix`，Python 导入名为 `a_memorix`。正式发布后可通过 `pip install a-memorix` 安装基础包；FAISS 支持位于 `vector` extra，LPMM Parquet 转换支持位于 `lpmm` extra。

## Python 入口

顶层包只公开运行内核、检索请求和宿主需要实现的接口：

```python
from a_memorix import (
    EmbeddingProvider,
    IdentityResolver,
    KernelSearchRequest,
    LLMProvider,
    MessageSource,
    SDKMemoryKernel,
)
```

最小生命周期如下。`provider` 需要实现 `EmbeddingProvider`；没有 Embedding Provider 时，内核会按配置进入稀疏或仅元数据降级路径。

```python
kernel = SDKMemoryKernel(
    data_dir="./data/example",
    embedding_provider=provider,
)

await kernel.initialize()
try:
    await kernel.ingest_text(
        external_id="document:1",
        source_type="document",
        text="A_memorix stores long-term evidence.",
    )
    result = await kernel.search_memory(
        KernelSearchRequest(query="long-term evidence", limit=5)
    )
finally:
    await kernel.shutdown()
```

`data_dir` 是运行时边界的一部分，必须显式提供。当前每个内核实例对应一个数据目录；完整的 namespace 注册表、资源限额和跨 namespace 故障隔离属于阶段3。

## 宿主接口

| 接口 | 用途 | 是否必需 |
| --- | --- | --- |
| `EmbeddingProvider` | 批量生成向量并报告已观测指纹 | 向量能力必需 |
| `LLMProvider` | Episode、画像、摘要等模型任务 | 对应能力必需 |
| `IdentityResolver` | 把外部身份映射为稳定 person ID | 画像能力可选 |
| `MessageSource` | 按会话和时间范围读取消息 | 摘要、反馈修正可选 |

这些接口只表达 A_memorix 需要的数据，不暴露宿主的配置对象、数据库模型和内部服务。

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
