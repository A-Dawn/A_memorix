# A_memorix Adapter Protocol v1

状态：Alpha  
协议版本：`1`  
Manifest Schema 版本：`1`

## 目标与边界

Adapter Protocol 规定 Agent 适配器如何声明兼容范围、运行形态、通信方式和所需权限。它不规定具体 Agent 的消息模型，也不把第三方代码加载进 A_memorix 服务。Episode、画像和摘要等尚未通用化的领域语义不属于协议 v1。

适配器代码由适配器作者维护。通用主仓库只提供公开 Python API、Protobuf API、manifest 模型和校验工具。社区索引负责登记、测试与展示，不复制第三方源码，也不替作者发布软件包。

## 运行形态

`runtime = "remote"` 表示适配器只能通过公开协议使用独立的 A_memorix 服务。允许的 transport 是 `grpc`、`http_json` 和 `mcp`。gRPC 是核心网络协议，HTTP/JSON 由同一份 Protobuf 通过 gRPC-Gateway 映射，不能为适配器增加另一套 JSON-RPC 语义。MCP 继续遵守固定 namespace 绑定和现有运行边界。

`runtime = "in_process"` 表示适配器与 Agent 运行在同一 Python 进程，可以使用 `a_memorix` 顶层公开 API、`AMemorixEngine` 和 Host Ports。它只能声明 `in_process` transport，必须提供可安装的 Python 包名和 `module:callable` 形式的入口点。入口点由 Agent 宿主管理，协议 v1只保证它可以被导入，不冻结所有 Agent 共用的构造参数或生命周期 ABI。

进程内适配器不得导入 `a_memorix.core`、读写控制数据库或依赖内核私有对象。需要进入公共 API 的能力必须先在主仓库形成稳定 contract。

## Namespace 隔离

每个适配器实例必须绑定一个明确的 namespace。远程适配器使用仅属于该 namespace 的 API Key；需要管理 namespace、API Key 或备份的操作必须单独声明，并使用管理员凭据。凭据、服务地址和实际 namespace ID属于部署配置，不能写入 manifest。

适配器不得根据消息内容动态切换 namespace，也不能把一个 namespace 的缓存、幂等键、检索结果或备份句柄复用于另一个 namespace。批量处理仍需为每个请求传递同一个已绑定的 `RequestContext.namespace_id`。跨 namespace 聚合不属于协议 v1。

## Manifest

Manifest 使用 TOML。字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | Manifest Schema 版本，当前固定为1 |
| `id` | 反向域名风格的稳定 ID |
| `name` | 展示名称 |
| `version` | SemVer 2.0适配器版本 |
| `runtime` | `in_process` 或 `remote` |
| `package` | 可选分发包名，进程内适配器必填 |
| `entrypoint` | Python `module:callable`，仅进程内适配器使用 |
| `core_version` | 支持的 A_memorix PEP 440版本范围 |
| `adapter_protocol` | Adapter Protocol 版本，当前固定为`1` |
| `transports` | 使用的通信方式 |
| `host_ports` | 进程内适配器提供的 Host Port |
| `license` | SPDX 许可证表达式 |
| `source` | 不含凭据的 HTTPS 源码地址 |
| `permissions` | API、网络、文件系统、环境变量和子进程声明 |

权限字段必须显式出现，空权限使用空数组或 `false`。`permissions.api` 至少声明一项实际使用的公开能力。网络 origin 必须使用 `http`、`https`、`grpc`、`grpcs`，不接受路径、凭据和通配符；`a-memorix` 是部署时注入的服务地址占位符。环境变量只填写变量名，不能填写值。

Manifest 权限是审计声明，不是沙箱。部署者仍需使用 namespace API Key、容器权限、文件系统 ACL 和网络策略执行最小权限。

## 版本兼容

Manifest Schema 与 Adapter Protocol 独立演进。新增可选字段可以保持同一 Schema 主版本；删除字段、改变字段语义或收紧既有合法值需要提升 Schema 版本。Adapter Protocol 只有在可观察行为或隔离约束不兼容时才提升版本。

适配器版本发布后不可原地替换 manifest。兼容范围变化也必须发布新的适配器版本。Alpha 阶段允许协议补充，进入 Beta 后协议 v1字段和行为冻结。

## 校验

```powershell
a-memorix adapter validate docs/examples/adapter-remote.toml
a-memorix --pretty adapter schema
```

`validate` 默认检查当前安装的 A_memorix 版本，也可以通过 `--core-version`验证其他目标版本。命令成功时输出规范化 JSON，结构错误、权限冲突或版本不兼容时返回退出码2。

结构校验通过不代表扩展已经获得官方或社区验证。信任等级由扩展索引在独立元数据中维护，不能由适配器自行声明。
