# A_memorix

A_memorix 是面向 AI Agent 的长期记忆内核。2.x 主线正在从 MaiBot 插件演进为独立 Python 包，目标是通过 namespace 隔离的运行时和统一协议，为不同 Agent 提供可组合的写入、检索、图关系、时序证据和记忆维护能力。

当前版本为 `2.0.0a1`。这一版本已经完成通用核心迁移、宿主依赖解耦、namespace Runtime 和第一版统一协议层。Protobuf 是唯一网络 IDL，服务原生暴露 gRPC，HTTP/JSON 由 gRPC-Gateway 按同一份注解生成。MCP 以固定 namespace 工具适配器接入。现阶段适合参与内核开发、适配器验证和数据兼容性测试，不应视为稳定服务版本。

## 当前能力

- 段落、实体、关系和时间元数据存储
- 向量、稀疏和图关系检索
- 单池、双池向量运行时及故障恢复
- Episode、画像、摘要和记忆生命周期的现有内核能力
- 显式数据目录、异步生命周期和单目录写者保护
- 可注入的 Embedding、LLM、身份解析和消息来源接口
- SQLite namespace 控制面、独立物理目录和7天可恢复隔离
- 并发初始化去重、活跃 runtime 上限、请求配额和 LRU 关闭
- 类型化写入、检索、namespace 管理和 API Key 应用接口
- gRPC、gRPC-Gateway HTTP/JSON、Python 客户端和固定 namespace MCP 适配
- 版本化 namespace 离线备份、分块传输、完整性校验和新 namespace 恢复

Episode、画像和摘要已经与 MaiBot 的模块、配置和数据库类型解耦，但它们的领域语义还需要进一步通用化。MaiBot 数据迁移等项目特定行为不会成为通用稳定 API。

## 安装

开发环境要求 Python 3.12 或更高版本。

```powershell
python -m pip install -e ".[test,vector]"
```

分发包名为 `a-memorix`，Python 导入名为 `a_memorix`。正式发布后可通过 `pip install a-memorix` 安装基础包；FAISS 支持位于 `vector` extra，gRPC 位于 `rpc` extra，MCP 位于 `mcp` extra，LPMM Parquet 转换支持位于 `lpmm` extra。

## Python 入口

`AMemorixEngine` 是多 namespace 的主要进程内入口。宿主可按 namespace 返回不同的 Provider：

```python
from a_memorix import (
    AMemorixEngine,
    CreateNamespaceRequest,
    IngestTextRequest,
    NamespaceHostPorts,
    RequestContext,
    SearchMemoryRequest,
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
    await engine.ingest_text(
        IngestTextRequest(
            context=context,
            external_id="document:1",
            source_type="document",
            text="A_memorix stores long-term evidence.",
        )
    )
    result = await engine.search_memory(
        SearchMemoryRequest(context=context, query="long-term evidence")
    )
```

`namespace_id` 是不可变的小写 ASCII 标识，格式为 `[a-z0-9][a-z0-9._-]{0,127}`。它不会直接成为目录名，控制库会为每个 namespace 生成内部 UUID。每个 namespace 拥有独立的元数据、向量、图、导入任务和写者锁。

`SDKMemoryKernel(data_dir=...)` 继续作为单 namespace 低层入口保留。现有调用方可以渐进迁移，但新建的多 Agent 宿主应从 `AMemorixEngine` 开始。

删除 namespace 时，运行时会先停止，数据目录会移入隔离区并默认保留7天。隔离期内可恢复，原 ID 不能被重新创建；到期清理或管理员显式清理后才会释放 ID。

## gRPC 与 HTTP/JSON

安装协议依赖并启动 gRPC 服务：

```powershell
python -m pip install -e ".[rpc]"
$env:A_MEMORIX_ADMIN_TOKEN = "replace-with-at-least-32-random-characters"
python -m a_memorix.server --data-dir ./data
```

服务默认监听 `127.0.0.1:50051`。未设置至少32字符的管理员令牌时，服务拒绝启动。开发环境可以显式传入 `--allow-unauthenticated`，但此模式只允许回环地址。

HTTP/JSON 不是第二套 API。`proto/a_memorix/api/v1` 中的 `google.api.http` 注解同时生成 gRPC-Gateway 路由和 OpenAPI 文档：

```powershell
$env:GOPROXY = "https://proxy.golang.org,direct"
go run ./cmd/a-memorix-gateway --listen 127.0.0.1:8080 --grpc-target 127.0.0.1:50051
```

客户端通过 `Authorization: Bearer <token>` 访问。管理员令牌用于 namespace 和密钥管理；namespace API Key 只能访问自己所属的 namespace，密钥明文只在创建时返回，控制库仅保存 SHA-256 摘要。网关会把认证、请求 ID、追踪 ID 和幂等键转交给 gRPC 服务。v1 已提供 namespace 配置与能力发现、单条和批量写入、检索、直接读取、单条删除，以及按来源删除 Job。

当前网关到 gRPC 的连接面向同机回环部署，使用明文连接。远程部署需要在反向代理处终止 TLS，或扩展网关的后端 TLS 配置，不能直接暴露默认监听方式。

Python 客户端使用生成的 gRPC stub：

```python
from a_memorix import AMemorixClient
from a_memorix.api.v1 import namespace_pb2

async with AMemorixClient("127.0.0.1:50051", api_key=admin_token) as client:
    response = await client.create_namespace(
        namespace_pb2.CreateNamespaceRequest(namespace_id="agent-prod")
    )
```

协议决策和错误语义见 [统一协议 ADR](docs/ADR_0001_GRPC_GATEWAY_PROTOCOL.md)，生成的 HTTP 描述位于 [OpenAPI v1](docs/openapi/a_memorix_v1.swagger.json)。

## Namespace备份

备份属于管理员控制面。namespace 必须先停用，恢复时必须指定一个尚不存在的新 ID，恢复结果保持停用状态：

```python
from a_memorix import RestoreNamespaceBackupRequest

await engine.disable_namespace("agent-prod")
backup = await engine.create_namespace_backup("agent-prod")
restored = await engine.restore_namespace_from_backup(
    RestoreNamespaceBackupRequest(
        backup_id=backup.backup_id,
        target_namespace_id="agent-prod-restored",
    )
)
```

`.amxbackup`归档包含 namespace 数据、配额和非敏感配置，不包含 API Key、Job、幂等记录或实际 Provider 密钥。gRPC 和 HTTP/JSON 提供最大1 MiB的分块上传、下载接口，归档和内部文件都使用 SHA-256 校验。格式、一致性边界和故障恢复规则见[阶段5.1记录](docs/PHASE5_1_NAMESPACE_BACKUP_20260809.md)。

## MCP 适配

MCP 服务在创建时绑定一个 namespace，工具参数不接受 `namespace_id`，因此同一 MCP 会话无法切换租户：

```python
from a_memorix import AMemorixEngine, create_fixed_namespace_mcp

server = create_fixed_namespace_mcp(
    AMemorixEngine(data_dir="./data"),
    "agent-prod",
    create_namespace=True,
)
server.run(transport="stdio")
```

当前支持的公开运行方式是进程内或 stdio。MCP 工具覆盖写入、批量写入、检索、直接读取、删除和 Job 查询，所有工具固定使用创建服务时绑定的 namespace。MCP 适配器本身不提供远程认证，不应把返回的服务对象直接公开为未鉴权的 HTTP 服务。

## 宿主接口

| 接口 | 用途 | 是否必需 |
| --- | --- | --- |
| `EmbeddingProvider` | 批量生成向量并报告已观测指纹 | 向量能力必需 |
| `LLMProvider` | Episode、画像、摘要等模型任务 | 对应能力必需 |
| `IdentityResolver` | 把外部身份映射为稳定 person ID | 画像能力可选 |
| `MessageSource` | 按会话和时间范围读取消息 | 摘要、反馈修正可选 |
| `Clock` | 为控制面生命周期提供可替换时间源 | 测试和定制调度可选 |

这些接口只表达 A_memorix 已经需要的数据，不暴露宿主的配置对象、数据库模型和内部服务。`NamespaceHostPorts` 用于一次性传递某个 namespace 的可选能力。协议鉴权位于 gRPC 适配层，不会污染 Host Port；遥测和宿主日志接口仍等待真实调用方出现后再确定。

## 分支与分发

- 通用基础版最终由 `main` 维护。
- 官方 Agent 适配使用独立集成分支，不把项目私有代码写回通用核心。
- 社区适配器将在独立的 `A_memorix-extensions` 仓库登记、验证和分发。
- 1.x MaiBot 插件历史保留在 `legacy-v1.0.1` 标签和 `legacy/plugin-v1` 分支。

完整路线见 [通用架构与分发计划](docs/GENERIC_ARCHITECTURE_AND_DISTRIBUTION_PLAN.md)。

## 开发验证

```powershell
pytest -q
buf lint
buf generate
go test ./...
python -m build
```

两项大规模格式迁移压测默认跳过，可设置 `A_MEMORIX_RUN_LARGE_MIGRATION_TEST=1` 单独执行。

## 许可证

项目默认采用 [GNU AGPL-3.0-only](LICENSE)。任何协议变更或其他许可安排都必须发送邮件至 `contact@luminarc.tech` 申请，只有收到书面批准后才产生例外。具体规则见 [LICENSING.md](LICENSING.md)。

外部贡献规则和社区扩展治理将在开放对应仓库前单独发布。
