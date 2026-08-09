# 阶段8：Alpha 发布门禁记录

日期：2026-08-09  
候选版本：2.0.0a2  
状态：Alpha 发布门禁已通过，PyPI 外部配置和对外发布待完成

## 已通过门禁

通用分支`refactor/generic-v2`完成验证后已无冲突 fast-forward 进入`main`。默认分支在 GitHub Actions 完成 Linux、Windows、Faiss、无 Faiss 降级、Python 3.12、Ruff、mypy、679项 Python 测试、Protobuf 生成、Buf lint、Go测试、Wheel、sdist、twine检查以及服务、网关双镜像构建。2项可选大规模迁移压测按设计跳过。

首轮 Windows 全量任务发现 LPMM 转换器错误文本受系统代码页影响。转换器现已为路径越界、输出非空、向量维度错误和无可用向量增加稳定的 ASCII 错误标识，保留原有中文说明。修正后的完整远端矩阵通过。

[A_memorix-extensions](https://github.com/A-Dawn/A_memorix-extensions)也已通过独立远端 CI，包括从固定主仓库提交安装校验器、Ruff、注册表校验、测试和 JSON Schema 输出验证。

## 发布保护

发布工作流要求 Git 标签与`pyproject.toml`及包内`__version__`完全一致，并要求 CHANGELOG 存在对应版本标题。标签目标必须已经进入`main`提交历史，不能直接从开发分支绕过主线发布。

CI只监听`main`、Pull Request和手动触发。默认分支要求所有修改通过 squash Pull Request合入，并通过 Python、协议、Linux无 Faiss、Windows有或无 Faiss以及两个容器构建共7项检查。规则同时禁止删除和强制推送。

`v*`发布标签受独立规则保护，创建后不能删除或改写。GitHub 的`release`环境已经建立，发布工作流仍要求标签提交属于`main`历史。

## 发布前外部配置

仍需在 PyPI 为`A-Dawn/A_memorix`的`release.yml`和`release`环境配置 Trusted Publisher。发布`v2.0.0a2`会对外推送 PyPI 包、两个 GHCR 镜像、多平台网关二进制和 GitHub Release，因此配置完成前不创建标签。

Buf 官方 setup action 的最新 v1版本仍声明 Node.js 20。GitHub Runner当前会强制以 Node.js 24兼容模式执行，任务已经通过但保留上游弃用提示；仓库已向该 action传递只读 GitHub token，避免匿名 API 限流。
