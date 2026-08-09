# 阶段8：Alpha 发布门禁记录

日期：2026-08-09  
候选版本：2.0.0a2  
状态：代码与构建门禁已通过，主线切换和对外发布待批准

## 已通过门禁

通用分支`refactor/generic-v2`已在 GitHub Actions 完成 Linux、Windows、Faiss、无 Faiss降级、Python 3.12、Ruff、mypy、679项 Python 测试、Protobuf生成、Buf lint、Go测试、Wheel、sdist、twine检查以及服务、网关双镜像构建。2项可选大规模迁移压测按设计跳过。

首轮 Windows 全量任务发现 LPMM 转换器错误文本受系统代码页影响。转换器现已为路径越界、输出非空、向量维度错误和无可用向量增加稳定的 ASCII 错误标识，保留原有中文说明。修正后的完整远端矩阵通过。

[A_memorix-extensions](https://github.com/A-Dawn/A_memorix-extensions)也已通过独立远端 CI，包括从固定主仓库提交安装校验器、Ruff、注册表校验、测试和 JSON Schema 输出验证。

## 发布保护

发布工作流要求 Git 标签与`pyproject.toml`及包内`__version__`完全一致，并要求 CHANGELOG 存在对应版本标题。标签目标必须已经进入`main`提交历史，不能直接从开发分支绕过主线发布。

CI 在迁移期间同时监听`main`和`refactor/generic-v2`。通用版进入`main`后，可以移除开发分支 push 触发，仅保留`main`、Pull Request和手动触发。

## 待批准事项

当前 GitHub 默认分支`main`仍是1.x旧版，`refactor/generic-v2`尚未合入。现有默认分支规则只阻止删除和非快进更新，没有要求 Pull Request或状态检查。切换前需要确定合并方式与保护规则。

仓库尚未建立`release`环境。实际发布前还需要在 GitHub建立该环境，并在 PyPI为`A-Dawn/A_memorix`的`release.yml`配置 Trusted Publisher。发布`v2.0.0a2`会对外推送 PyPI包、两个 GHCR镜像、多平台网关二进制和 GitHub Release，因此必须在获得明确批准后执行。

Buf 官方 setup action 的最新 v1版本仍声明 Node.js 20。GitHub Runner当前会强制以 Node.js 24兼容模式执行，任务已经通过但保留上游弃用提示；仓库已向该 action传递只读 GitHub token，避免匿名 API 限流。
