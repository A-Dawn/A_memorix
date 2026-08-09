# 贡献指南

## 适用范围

`main` 维护不依赖具体 Agent 的通用版本。新的 Agent 接入代码应放在对应集成分支或独立 Adapter 仓库，不应把 Agent 的配置类型、数据库 model 或内部服务加入通用核心。

安全漏洞、凭据泄露和供应链问题请发送至 `security@luminarc.tech`，不要通过公开 Issue 或 Pull Request 披露未修复细节。

## 开发流程

1. 从最新 `main` 创建独立分支。
2. 保持修改范围明确，新增或改变行为时补充相应测试。
3. 提交前运行与修改范围相称的检查。完整检查包括：

```powershell
ruff check .
mypy
pytest -q
buf lint
buf generate
go test ./...
python -m build
twine check dist/*
```

4. 通过 Pull Request 合入 `main`，并等待所有必需 CI 检查完成。

生成的 Protobuf、OpenAPI 或 Go 文件发生变化时，应同时提交生成结果。不要提交 API Key、访问令牌、私钥、真实部署地址或包含用户数据的测试文件。

## Contributor License Agreement

外部贡献者需要接受 [A_memorix Contributor License Agreement](CLA.md)。这是非独占授权，不是版权转让：贡献者保留代码版权和自行使用、修改、分发、授权的权利；项目获得维护、发布和提供单独许可所需的永久授权。

项目只记录 GitHub 用户名、关联邮箱、CLA 版本、签署时间和相关仓库或 Pull Request，不要求个人住址、证件、国籍或电话号码。由雇主或其他机构持有版权时，提交者需要确认已经获得授权，或者由该机构的授权代表接受 CLA。

CLA 更新后只对接受新版本之后的贡献生效。已经根据旧版本授予的许可继续有效。

## 许可

公开版本继续采用 `AGPL-3.0-only`。其他许可安排必须发送邮件至 `contact@luminarc.tech` 申请，收到书面批准后才生效。具体规则见 [LICENSING.md](LICENSING.md)。
