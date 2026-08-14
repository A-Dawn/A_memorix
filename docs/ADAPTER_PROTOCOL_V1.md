# A_memorix Adapter Protocol v1

状态：Alpha  
协议版本：`1`  
Manifest Schema 版本：`1`

## 目标与范围

Adapter Protocol 规定 Agent Adapter 如何声明兼容范围、运行方式、通信方式和所需权限。它不规定具体 Agent 的消息 model，也不把第三方代码加载进 A_memorix 服务。Episode、画像和摘要等尚未通用化的功能不属于协议 v1。

Adapter 代码由作者维护。主仓库只提供公开 Python API、Protobuf API、Manifest 模型和校验工具。社区扩展仓库负责登记、测试与展示，不复制第三方源码，也不替作者发布软件包。

## 运行方式

`runtime = "remote"` 表示 Adapter 只能通过公开 API 使用独立的 A_memorix 服务。允许的 transport 是 `grpc`、`http_json` 和 `mcp`。gRPC 是核心网络协议，HTTP/JSON 由同一份 Protobuf 通过 gRPC-Gateway 映射，不能为 Adapter 增加另一套 JSON-RPC 语义。每个 MCP 服务继续固定绑定一个 Namespace。

`runtime = "in_process"` 表示 Adapter 与 Agent 运行在同一 Python 进程，可以使用 `a_memorix` 顶层公开 API、`AMemorixEngine` 和 Host Ports。它只能声明 `in_process` transport，必须提供可安装的 Python 包名和 `module:callable` 形式的入口点。入口点由 Agent 程序管理，协议 v1只保证它可以被导入，不固定所有 Agent 共用的构造参数或生命周期 ABI。

进程内 Adapter 不得导入 `a_memorix.core`、读写 Namespace 管理库或依赖核心私有对象。需要开放的功能必须先加入主仓库的公开 API。

## Namespace 隔离

每个 Adapter 实例必须绑定一个明确的 Namespace。远程 Adapter 使用仅属于该 Namespace 的 API Key；需要管理 Namespace、API Key 或备份的操作必须单独声明，并使用管理员凭据。凭据、服务地址和实际 Namespace ID 属于部署配置，不能写入 Manifest。

Adapter 不得根据消息内容动态切换 Namespace，也不能把一个 Namespace 的缓存、幂等键、检索结果或备份句柄复用于另一个 Namespace。批量处理仍需为每个请求传递同一个已绑定的 `RequestContext.namespace_id`。跨 Namespace 聚合不属于协议 v1。

## Manifest

Manifest 使用 TOML。字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | Manifest Schema 版本，当前固定为1 |
| `id` | 反向域名风格的稳定 ID |
| `name` | 展示名称 |
| `version` | Adapter 的 SemVer 2.0 版本 |
| `runtime` | `in_process` 或 `remote` |
| `package` | 可选 Python 包名，进程内 Adapter 必填 |
| `entrypoint` | Python `module:callable`，仅进程内 Adapter 使用 |
| `core_version` | 支持的 A_memorix PEP 440版本范围 |
| `adapter_protocol` | Adapter Protocol 版本，当前固定为`1` |
| `transports` | 使用的通信方式 |
| `host_ports` | 进程内 Adapter 使用的 Host Port |
| `license` | SPDX 许可证表达式 |
| `source` | 不含凭据的 HTTPS 源码地址 |
| `permissions` | API、网络、文件系统、环境变量和子进程声明 |

权限字段必须明确出现，空权限使用空数组或 `false`。`permissions.api` 至少声明一项实际使用的公开 API。网络 origin 必须使用 `http`、`https`、`grpc`、`grpcs`，不接受路径、凭据和通配符。`a-memorix`、`embedding-provider`、`llm-provider` 分别表示部署时注入的记忆服务、向量服务和文本生成服务地址。环境变量只填写变量名，不能填写值。

Manifest 权限用于审核，不等同于沙箱。部署者仍需使用 Namespace API Key、容器权限、文件系统 ACL 和网络策略执行最小权限。

## 版本兼容

Manifest Schema 与 Adapter Protocol 分别管理版本。新增可选字段可以保持同一 Schema 主版本；删除字段、改变字段含义或限制既有合法值时，需要提升 Schema 版本。只有可观察行为或隔离规则出现不兼容变化时，才提升 Adapter Protocol 版本。

Adapter 版本发布后不可原地替换 Manifest。兼容范围变化也必须发布新的 Adapter 版本。Alpha 阶段允许补充协议，进入 Beta 后固定协议 v1 的字段和行为。

## 校验

```powershell
a-memorix adapter validate docs/examples/adapter-remote.toml
a-memorix --pretty adapter schema
```

`validate` 默认检查当前安装的 A_memorix 版本，也可以通过 `--core-version`验证其他目标版本。命令成功时输出规范化 JSON，结构错误、权限冲突或版本不兼容时返回退出码2。

结构校验通过不代表扩展已经获得官方或社区验证。审核状态由扩展仓库单独记录，不能由 Adapter 自行声明。
