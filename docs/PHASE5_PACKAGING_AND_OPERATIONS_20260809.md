# 阶段5：打包和运维能力记录

日期：2026-08-09
版本：2.0.0a2
状态：已完成

## 目标与范围

阶段5把已经完成的通用核心、Namespace 管理和统一协议整理成可安装、可启动、可诊断、可发布的服务。核心协议仍是 Protobuf 定义的 gRPC，HTTP/JSON 只由 gRPC-Gateway 映射。管理工具不得绕过 RPC 直接修改服务数据目录。

Episode、画像、摘要等功能的通用语义不在本阶段固定。它们继续保留在核心代码中，等待后续阶段设计与 Agent 无关的公开 API。

## 统一命令行

项目提供单一 `a-memorix` 入口：

| 命令 | 作用 | 数据访问方式 |
| --- | --- | --- |
| `serve` | 启动 Python gRPC 服务 | 拥有服务数据目录 |
| `mcp` | 启动固定 Namespace MCP | 拥有指定本地数据目录 |
| `namespace` | 创建、查询、停用、恢复和清理 Namespace | gRPC |
| `api-key` | 创建、列出和撤销 Namespace API Key | gRPC |
| `memory` | 写入、检索、读取和删除记忆 | gRPC |
| `backup` | 创建、分块下载、上传、恢复和删除备份 | gRPC |
| `doctor` | 检查标准服务健康状态和本地配置 | gRPC与只读本地诊断 |
| `config` | 输出不含密钥值的生效配置 | 只读配置 |

备份上传和下载固定使用最大1 MiB分块。下载先写入同目录临时文件，完成 SHA-256 校验后再原子替换目标文件；上传失败会请求服务端终止临时会话。

## 配置规则

配置模型采用严格 TOML schema，未知字段、越界端口和不完整 TLS 配置会在启动前失败。覆盖优先级是命令行、环境变量、显式 TOML、默认值。相对路径以 TOML 文件所在目录为基准，命令行和环境变量路径按当前进程环境解释。

密钥值不进入配置 model。管理员令牌和 Namespace API Key 通过环境变量或只读文件加载，`config` 与 `doctor` 的输出只包含令牌文件路径，不包含令牌内容。参考配置位于 `deploy/a-memorix.example.toml`。

## 健康、安全与可观测性

Python 服务注册标准 `grpc.health.v1.Health`，总服务和五个公开 v1 服务在启动完成后进入 `SERVING`，优雅关闭前切换到非服务状态。Go 网关的 `/healthz` 会调用后端健康服务，不把进程存活误当成整体可用。

Python gRPC 服务和客户端支持 TLS、mTLS、自定义 CA 与 server name。Go 网关分别支持后端 gRPC TLS、mTLS和面向外部客户端的 HTTPS、mTLS。未鉴权服务仍被限制为回环地址。

日志默认是单行 JSON，记录 UTC 时间、级别、logger、RPC 方法、gRPC 状态和处理耗时。可选 Prometheus endpoint 提供请求计数、耗时直方图和活跃请求数；可选 OTLP/gRPC 导出 RPC Trace。监控相关依赖保留在独立 extra 中，不增加基础安装的依赖数量。

## 发布内容

- `a-memorix` Wheel 与 sdist
- 包含 RPC、向量和可观测性依赖的非 root Python 服务镜像
- 静态编译、非 root 的 Go gRPC-Gateway 镜像
- 同时启动服务和网关的 `compose.yaml`
- Linux、macOS、Windows 的 amd64 网关二进制，以及 Linux、macOS 的 arm64二进制
- PyPI Trusted Publishing、GHCR 和 GitHub Release 自动发布流程

服务镜像的数据目录是`/data`，网关镜像不持有业务数据。Compose 使用命名卷持久化数据，两个容器都启用只读根文件系统、移除 Linux capabilities，并通过后端健康依赖控制启动顺序。

## 验收结果

阶段退出条件由自动化测试覆盖：从 CLI 创建 Namespace、写入和检索元数据记忆、停用 Namespace、创建备份、分块下载、重新上传并恢复为新 Namespace。测试同时覆盖配置优先级、未知字段拒绝、JSON 日志、标准 gRPC 健康检查、Prometheus 指标、Python 双向 TLS 和网关后端健康探测。

本机已完成 Wheel 与 sdist 构建、Compose 配置展开、667项 Python 测试和完整 Go 测试，2项可选大规模迁移压测按设计跳过。Docker Desktop 守护进程在验收时未运行，因此本机未实际构建镜像；CI 对两个 Dockerfile 分别执行无推送构建，版本标签流程负责发布镜像。该环境限制不改变源码和流水线的阶段完成状态，但首次远程 CI 仍应作为镜像基础层兼容性的最终确认。

## 后续阶段

阶段6开始建立官方 Agent 集成分支。该阶段应复用当前公开 API、CLI 和镜像，不把 MaiBot 配置对象或数据库类型带回 main，也不借集成工作重新设计已经稳定的通用接口。
