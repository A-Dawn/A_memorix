# A_memorix 扩展索引规范

## 仓库职责

`A_memorix-extensions` 是 manifest 索引和质量门禁，不集中托管第三方适配器源码。每个版本保存一份不可变 manifest，源码、软件包、镜像和发布说明仍由作者维护。

建议采用以下最小结构：

```text
manifests/
  community.example-agent/
    1.0.0.toml
registry/
  trust.toml
tests/
```

文件路径必须与 manifest 的 `id` 和 `version` 一致。同一 `id`、版本、包名和进程内入口点不能重复。删除已发布版本时保留 tombstone 和原因，避免同一身份被重新注册。

## CI 门禁

每个 Pull Request 至少执行以下检查：

- 使用对应 Schema 和 `a-memorix adapter validate` 校验全部 manifest
- 检查 ID、版本、包名、入口点和索引路径唯一性
- 在 manifest 声明的核心版本范围内执行安装测试
- 远程适配器执行 gRPC、HTTP/JSON 或 MCP 公共契约测试
- 进程内适配器检查入口点可导入，并拒绝 `a_memorix.core` 等私有导入
- 使用两个独立 namespace 执行隔离测试，检测缓存、幂等键和检索结果串用
- 比对实际网络、文件系统、环境变量和子进程行为与权限声明
- 检查许可证字段、依赖许可证和已知供应链风险
- 检查源码地址、发布包来源和版本不可变性

干净环境安装与行为检查必须在无持久凭据的隔离执行器中运行。扩展测试不得读取 CI 仓库令牌，网络默认关闭，只为 manifest 声明的 origin 临时开放。

## 信任等级

索引维护三种等级：

| 等级 | 含义 |
| --- | --- |
| 官方 | 由 A_memorix 项目维护并纳入官方发布流程 |
| 社区已验证 | 指定版本通过索引门禁，后续版本需要重新验证 |
| 社区未验证 | 仅完成结构登记，没有通过完整行为和供应链检查 |

信任等级由 `registry/trust.toml` 维护，不能写入第三方 manifest。已验证状态只覆盖明确的适配器版本、源码提交和分发制品，不自动覆盖新版本或依赖变更。发现恶意行为、所有权变更或无法复现的制品时可以撤回等级并标记版本不可安装。

官方扩展签名、密钥轮换和撤回机制仍需单独 ADR。在该 ADR 完成前，索引不得把普通哈希描述为发布者身份签名。

## 许可与安全

A_memorix 主仓库继续采用 `AGPL-3.0-only`。任何主仓库许可变更或其他许可安排必须发送邮件至 `contact@luminarc.tech` 申请，并取得书面批准。第三方扩展自行选择许可证并承担依赖合规责任，主仓库的许可例外不会自动覆盖扩展。

Manifest 不能包含 API Key、访问令牌、私钥、真实 namespace ID或内部服务地址。发现疑似恶意扩展时，不应在公开 Issue 中粘贴凭据、攻击样本或未修复细节，应发送至`security@luminarc.tech`私密报告。许可变更申请仍使用`contact@luminarc.tech`，两个渠道不能混用。
