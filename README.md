# A_memorix

A_memorix 是面向 AI Agent 的长期记忆服务。2.x 正在从 MaiBot 插件演进为独立 Python 包，通过彼此独立的 Namespace 和统一协议，为不同 Agent 提供可组合的写入、检索、图关系、时序证据和记忆维护能力。

当前版本为 `2.0.0a2`。这一版本已经完成通用核心迁移、MaiBot 依赖移除、Namespace Runtime、统一协议和基础部署支持。Protobuf 是唯一网络 IDL，服务原生提供 gRPC，HTTP/JSON 由 gRPC-Gateway 根据同一份注解生成。每个 MCP 服务固定绑定一个 Namespace。现阶段适合参与核心开发、Adapter 验证和数据兼容性测试，不应视为稳定服务版本。

## 当前功能

- 段落、实体、关系和时间元数据存储
- 向量、稀疏和图关系检索
- 单池、双池向量存储及故障恢复
- Episode、画像、摘要和记忆生命周期等现有功能
- 显式数据目录、异步生命周期和单目录写者保护
- 可注入的 Embedding、LLM、身份解析和消息来源接口
- SQLite Namespace 管理库、独立物理目录和7天可恢复删除
- 并发初始化去重、活跃 Runtime 上限、请求配额和 LRU 关闭
- 类型明确的写入、检索、Namespace 管理和 API Key 接口
- gRPC、gRPC-Gateway HTTP/JSON、Python 客户端和固定 Namespace MCP 服务
- 带版本的 Namespace 离线备份、分块传输、完整性校验和新 Namespace 恢复
- 统一 CLI、标准健康检查、Prometheus 指标、OTLP Trace 和结构化日志
- Python 服务镜像、Go 网关镜像、Compose 部署和版本标签发布流水线

Episode、画像和摘要已经移除对 MaiBot 模块、配置和数据库类型的依赖，但具体输入、输出和生命周期还需要进一步通用化。MaiBot 数据迁移等项目特定行为不会成为稳定的公开 API。

## 安装

开发环境要求 Python 3.12 或更高版本。

```powershell
python -m pip install -e ".[test,vector]"
```

分发包名为 `a-memorix`，Python 导入名为 `a_memorix`。正式发布后可通过 `pip install a-memorix` 安装基础包；FAISS 支持位于 `vector` extra，gRPC 位于 `rpc` extra，MCP 位于 `mcp` extra，可观测性位于 `observability` extra，LPMM Parquet 转换支持位于 `lpmm` extra。运行独立服务的常用安装方式是：

```powershell
python -m pip install "a-memorix[rpc,vector,observability]"
```

## Python 入口

`AMemorixEngine` 是多 Namespace 的主要进程内入口。Agent 程序可为不同 Namespace 返回不同的 Provider：

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

`namespace_id` 是不可变的小写 ASCII 标识，格式为 `[a-z0-9][a-z0-9._-]{0,127}`。它不会直接成为目录名，Namespace 管理库会为每个 Namespace 生成内部 UUID。每个 Namespace 拥有独立的元数据、向量、图、导入任务和写者锁。

`SDKMemoryKernel(data_dir=...)` 继续作为单 Namespace 的底层入口保留。现有调用方可以逐步迁移，但新建的多 Agent 程序应从 `AMemorixEngine` 开始。

### Relation extraction

普通 `ingest_text` 默认会在 LLM 可用时于段落落库后异步分析正文，并把抽取出的实体、关系写入 graph。没有 LLM 时，继承默认策略的请求会保留基础记忆写入，只有显式要求关系抽取才返回能力错误。关系向量、自动关系抽取、双向量池和 metadata-only 降级写入均默认开启；调用方仍可在 Namespace 或单次请求中明确关闭相应能力：

```python
from a_memorix import (
    NamespaceConfig,
    ProviderReference,
    RelationExtractionConfig,
    RelationExtractionMode,
)

config = NamespaceConfig(
    llm=ProviderReference(provider_id="host-llm", model_id="memory-model"),
    relation_extraction=RelationExtractionConfig(
        enabled=True,
        default_enabled=True,
        profile="agent-memory-v1",
        max_chunk_chars=8_000,
        chunk_overlap_chars=500,
    ),
)

request = IngestTextRequest(
    context=context,
    source_type="conversation",
    text="Alice works at Lumina.",
    relation_extraction=RelationExtractionMode.INHERIT,
)
response = await engine.ingest_text(request)
job = await engine.get_job(context.namespace_id, response.relation_extraction_job_id)
```

段落会先完成持久化，relation extraction 随后作为可查询的后台 Job 执行。`general-v1` 不限定 predicate，适合未知文本；`agent-memory-v1` 使用稳定的跨 session 记忆 predicate，并归一常见近义关系。Namespace 也可以提供自己的实体类型和 predicate 列表。长文本按配置分块，结果归并去重后再写入 graph。重复提交已存在的 `external_id` 也能发起补抽取，成功结果由 paragraph 上的 profile、prompt 和模型指纹保证幂等。

删除 Namespace 时，对应 Runtime 会先停止，数据目录会移入隔离区并默认保留7天。保留期内可以恢复，原 ID 不能重新创建；到期清理或管理员主动清理后才会释放 ID。

## gRPC 与 HTTP/JSON

安装协议依赖并启动 gRPC 服务：

```powershell
python -m pip install -e ".[rpc,observability]"
$env:A_MEMORIX_ADMIN_TOKEN = "replace-with-at-least-32-random-characters"
a-memorix serve --data-dir ./data
```

服务默认监听 `127.0.0.1:50051`。未设置至少32字符的管理员令牌时，服务拒绝启动。开发环境可以明确传入 `--allow-unauthenticated`，但此模式只允许回环地址。

HTTP/JSON 不是第二套 API。`proto/a_memorix/api/v1` 中的 `google.api.http` 注解同时生成 gRPC-Gateway 路由和 OpenAPI 文档：

```powershell
$env:GOPROXY = "https://proxy.golang.org,direct"
go run ./cmd/a-memorix-gateway --listen 127.0.0.1:8080 --grpc-target 127.0.0.1:50051
```

客户端通过 `Authorization: Bearer <token>` 访问。管理员令牌用于 Namespace 和密钥管理；Namespace API Key 只能访问所属的 Namespace，密钥明文只在创建时返回，管理库仅保存 SHA-256 摘要。网关会把认证、请求 ID、追踪 ID 和幂等键转交给 gRPC 服务。v1 已提供 Namespace 配置与可用功能查询、单条和批量写入、检索、直接读取、单条删除，以及按来源删除 Job。

网关支持面向客户端的 HTTPS、mTLS，以及连接 gRPC 后端时的 TLS、mTLS。默认仍是同机或容器网络内的明文连接，不能按默认参数直接开放到不可信网络。后端 TLS 使用 `--grpc-ca`、`--grpc-server-name`、`--grpc-client-cert` 和 `--grpc-client-key`，HTTPS 使用 `--tls-cert`、`--tls-key` 和 `--tls-client-ca`。

Python 客户端使用生成的 gRPC stub：

```python
from a_memorix import AMemorixClient
from a_memorix.api.v1 import namespace_pb2

async with AMemorixClient("127.0.0.1:50051", api_key=admin_token) as client:
    response = await client.create_namespace(
        namespace_pb2.CreateNamespaceRequest(namespace_id="agent-prod")
    )
```

协议决策和错误格式见 [统一协议 ADR](docs/ADR_0001_GRPC_GATEWAY_PROTOCOL.md)，生成的 HTTP 描述位于 [OpenAPI v1](docs/openapi/a_memorix_v1.swagger.json)。

## CLI 与配置

`a-memorix serve` 只启动 Python gRPC 服务。其他管理命令都是远程客户端，只调用公开 gRPC API，不会直接修改数据目录。一个不依赖外部 Embedding Provider 的最小工作流如下：

```powershell
a-memorix namespace create agent-prod --allow-metadata-only-write --sparse-retrieval
a-memorix memory ingest agent-prod --source-type document --external-id document:1 --text "A_memorix stores isolated memories."
a-memorix memory search agent-prod --query "isolated memories"
a-memorix doctor --health-only
```

远程地址、令牌文件和 TLS 参数放在命令组与子命令之间，例如 `a-memorix namespace --target memory.example:50051 --token-file ./admin-token list`。管理命令输出 JSON，错误输出到 stderr，并使用稳定错误码。

配置采用 TOML，完整示例位于 [deploy/a-memorix.example.toml](deploy/a-memorix.example.toml)。优先级固定为命令行参数、环境变量、`--config` 或 `A_MEMORIX_CONFIG` 指定的文件、内置默认值。管理员令牌和 API Key 不写入 TOML，可通过 `A_MEMORIX_ADMIN_TOKEN`、`A_MEMORIX_API_KEY` 或只读令牌文件提供。执行 `a-memorix config` 可以查看不包含密钥值的生效配置。

## 可观测性

gRPC 服务注册标准 `grpc.health.v1.Health`。`a-memorix doctor` 和网关 `/healthz` 都使用该状态，网关只有在后端服务返回 `SERVING` 时才健康。日志默认输出 JSON，包含 RPC 方法、状态和耗时。

设置 `A_MEMORIX_METRICS_PORT=9464` 后会提供 Prometheus 指标，包括请求数、处理耗时和活跃请求。设置 `A_MEMORIX_OTLP_ENDPOINT` 后会通过 OTLP/gRPC 导出 Trace，可用 `A_MEMORIX_OTLP_INSECURE` 和 `A_MEMORIX_TRACE_SAMPLE_RATIO` 调整传输与采样。指标端口默认不启用，示例配置默认仅监听回环地址。

## 容器部署

仓库提供独立的 Python 服务镜像和 Go 网关镜像。设置管理员令牌后可一条命令启动：

```powershell
$env:A_MEMORIX_ADMIN_TOKEN = "replace-with-at-least-32-random-characters"
docker compose up --build -d
```

Compose 默认映射 gRPC `50051`、HTTP/JSON `8080` 和 Prometheus `9464`，数据保存在命名卷 `a-memorix-data`。两个容器都以非 root 用户运行、移除 Linux capabilities 并使用只读根文件系统。生产环境应通过端口绑定、防火墙或 TLS 配置限制管理接口，不应把管理员令牌写入镜像或提交到仓库。

## Namespace 备份

备份只能由管理员操作。Namespace 必须先停用，恢复时必须指定一个尚不存在的新 ID，恢复结果保持停用状态：

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

`.amxbackup` 归档包含 Namespace 数据、配额和非敏感配置，不包含 API Key、Job、幂等记录或实际 Provider 密钥。gRPC 和 HTTP/JSON 提供最大1 MiB的分块上传、下载接口，归档和内部文件都使用 SHA-256 校验。格式、数据一致性和故障恢复规则见[阶段5.1记录](docs/PHASE5_1_NAMESPACE_BACKUP_20260809.md)。

## MCP 适配

MCP 服务在创建时绑定一个 Namespace，工具参数不接受 `namespace_id`，因此同一 MCP 会话无法切换 Namespace：

```python
from a_memorix import AMemorixEngine, create_fixed_namespace_mcp

server = create_fixed_namespace_mcp(
    AMemorixEngine(data_dir="./data"),
    "agent-prod",
    create_namespace=True,
)
server.run(transport="stdio")
```

当前支持进程内调用和 stdio。MCP 工具覆盖写入、批量写入、检索、直接读取、删除和 Job 查询，所有工具固定使用创建服务时绑定的 Namespace。MCP 服务本身不提供远程认证，不应把返回的服务对象直接公开为未鉴权的 HTTP 服务。

## Agent 接入接口

| 接口 | 用途 | 是否必需 |
| --- | --- | --- |
| `EmbeddingProvider` | 批量生成向量并报告实际模型指纹 | 使用向量功能时必需 |
| `LLMProvider` | Episode、画像、摘要等模型任务 | 使用相应功能时必需 |
| `IdentityResolver` | 把外部身份映射为稳定 person ID | 使用画像功能时可选 |
| `MessageSource` | 按会话和时间范围读取消息 | 摘要、反馈修正可选 |
| `Clock` | 为 Namespace 生命周期管理提供可替换时间源 | 测试和定制调度可选 |

这些接口只接收 A_memorix 实际需要的数据，不接触 Agent 程序的配置对象、数据库 model 和内部服务。`NamespaceHostPorts` 用于一次性传递某个 Namespace 可使用的 Provider。认证、网络请求日志和指标由 gRPC 层处理，不进入 Host Port。业务侧 Telemetry 接口会在出现实际调用方后再确定。

## Adapter Manifest

社区 Adapter 使用同一份 Manifest Schema 声明运行方式、核心版本范围、transport、Host Port 和权限。远程 Adapter 只能使用 gRPC、gRPC-Gateway HTTP/JSON 或固定 Namespace MCP；进程内 Adapter 只能依赖顶层公开 Python API，不能导入 `a_memorix.core`。

```powershell
a-memorix adapter validate docs/examples/adapter-remote.toml
a-memorix --pretty adapter schema
```

权限字段供部署者和扩展列表核对最小权限，不替代 API Key、容器沙箱、文件系统 ACL 或网络策略。完整规则见 [Adapter Protocol v1](docs/ADAPTER_PROTOCOL_V1.md) 和 [扩展列表规范](docs/EXTENSIONS_REGISTRY_SPEC.md)。

## 分支与分发

- 通用基础版由 `main` 维护。
- 官方 Agent 适配使用独立集成分支，不把项目私有代码写回通用核心。
- 社区 Adapter 在独立的 [A_memorix-extensions](https://github.com/A-Dawn/A_memorix-extensions) 仓库登记、验证和分发。
- 1.x MaiBot 插件历史保留在 `legacy-v1.0.1` 标签和 `legacy/plugin-v1` 分支。

完整路线见 [通用架构与分发计划](docs/GENERIC_ARCHITECTURE_AND_DISTRIBUTION_PLAN.md)。

## 公共量化评测

LongMemEval-S Cleaned 用于衡量跨 session 与长上下文记忆，SWE-bench Lite 用于衡量 issue-to-source-file 的代码库检索。数据、模型凭据、cache 和结果均只保存在本地。评测 summary 会固定 case、数据、模型、运行参数和环境，`a-memorix-eval compare` 用于检查 candidate 相对 baseline 的质量与性能变化。当前完整基线包含470个 LongMemEval case 和300个 SWE-bench Lite case，均无失败，具体结果和复跑命令见[公共量化评测](docs/EVALUATION.md)。

## 开发验证

```powershell
pytest -q
buf lint
buf generate
go test ./...
python -m build
twine check dist/*
```

两项大规模格式迁移压测默认跳过，可设置 `A_MEMORIX_RUN_LARGE_MIGRATION_TEST=1` 单独执行。

## 许可证

项目默认采用 [GNU AGPL-3.0-only](LICENSE)。任何协议变更或其他许可安排都必须发送邮件至 `contact@luminarc.tech` 申请，只有收到书面批准后才产生例外。具体规则见 [LICENSING.md](LICENSING.md)。

外部贡献采用非独占的 [A_memorix CLA](CLA.md)。贡献者保留代码版权，项目获得维护、公开发布和提供其他许可所需的授权。贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

安全漏洞和疑似恶意扩展请通过`security@luminarc.tech`私密报告，具体要求见[安全策略](SECURITY.md)。

社区扩展仓库已经支持 Manifest、审核状态和第一批自动检查；授予已验证状态所需的安装、API、Namespace 隔离和供应链检查仍需继续建设。
