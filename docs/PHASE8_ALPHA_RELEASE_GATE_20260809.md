# 阶段8：Alpha 发布前检查记录

日期：2026-08-09  
候选版本：2.0.0a2  
状态：暂停发布，公共量化基线已经完成

## 2026-08-10 状态修正

现有 CI 能证明功能、协议、构建和数据安全路径通过，公共量化基线用于衡量通用化改造后的检索质量与性能。`v2.0.0a2` 继续暂停发布，不创建 tag，也不触发 PyPI、GHCR 或 GitHub Release。LongMemEval-S Cleaned 与 SWE-bench Lite 的本地基线已经完成，数据版本、Embedding 配置、指标和默认回归范围也已固定。

公共评测工具已经接入 LongMemEval-S Cleaned 500题和 SWE-bench Lite 300题，固定文件大小、SHA-256、来源与许可。评测只使用 Embedding，不调用 LLM；凭据、下载数据、repo cache、临时 Namespace 和结果均留在本地忽略目录。

LongMemEval session 粒度热 cache 基线完成470个可评分 case，失败0。整体 MRR 为0.932080、`nDCG@10` 为0.926632、`Recall-any@10` 为0.995745、`Recall-fraction@10` 为0.977163；总耗时中位数618毫秒、P95 714毫秒。分类型 MRR 最低的是 single-session-preference 的0.780和 temporal-reasoning 的0.890，后续优化需要重点观察这两类。

SWE-bench Lite 热 cache 基线完成全部300个 case，失败0。整体 MRR 为0.689214、`nDCG@10` 为0.735735、`Recall-any@10` 为0.876667；总耗时中位数5.417秒、P95 20.417秒，检索 P95 为292毫秒。两轮完整基线均为单进程、未续跑、`cache_misses=0`、远程请求0，Embedding 预热参数为每批16条、最多3个并发请求。发布仍按当前决定暂停。

真实运行发现 Windows 会短暂锁住刚写完的图快照或归档解压目录，使目录替换报 `WinError 5`。图存储和 SWE-bench repository cache 现以有限重试处理短锁，并有回归测试。评测输出目录增加了跨进程锁，避免两个 resume 进程同时改写结果和 Namespace。Git 无法访问时，repository cache 会从 GitHub codeload 下载固定 commit 的归档；Embedding 服务短时过载时，每个批次最多尝试5次并逐步延长等待。

批量写入路径现将向量和图持久化延后到批次末尾；无限配额不再对每条记忆重复扫描目录；ASCII 源码写入 FTS 时不再经过 Jieba；向量恢复只在目标 ID 确实已删除时刷新写缓冲。受控 Astropy case 的写入从77.595秒降到6.002秒，排名和 MRR 不变，提升约12.93倍。

summary 现在记录评测 schema 版本、精确 case ID、数据和模型指纹、预热参数、环境版本、质量指标与阶段耗时。`a-memorix-eval compare` 会拒绝不可比的 summary，并检查质量下降和热 cache 性能比例。后续工作转为针对低分类型改进检索，并用相同条件复跑 candidate；当前基线只覆盖 Embedding 检索，不代表 LLM 答案正确率。

## 已通过的功能检查

开发分支 `refactor/generic-v2` 完成验证后，以 fast-forward 方式合入 `main`，过程没有冲突。GitHub Actions 已在默认分支完成 Linux、Windows、Faiss、无 Faiss 模式、Python 3.12、Ruff、mypy、679项 Python 测试、Protobuf 生成、Buf lint、Go 测试、Wheel、sdist、twine 检查以及服务、网关两个镜像的构建。2项可选大规模迁移压测按设计跳过。

首轮 Windows 全量任务发现 LPMM 转换器错误文本受系统代码页影响。转换器现已为路径越界、输出非空、向量维度错误和无可用向量增加稳定的 ASCII 错误标识，保留原有中文说明。修正后的完整远端矩阵通过。

[A_memorix-extensions](https://github.com/A-Dawn/A_memorix-extensions) 也已通过独立远端 CI，包括从主仓库指定 commit 安装校验器、Ruff、扩展列表校验、测试和 JSON Schema 输出验证。

## 发布保护

发布工作流要求 Git 标签与`pyproject.toml`及包内`__version__`完全一致，并要求 CHANGELOG 存在对应版本标题。标签目标必须已经进入`main`提交历史，不能直接从开发分支绕过主线发布。

CI 只监听 `main`、Pull Request 和手动触发。默认分支要求所有修改通过 squash Pull Request 合入，并通过 Python、Protobuf 与 Go、Linux 无 Faiss、Windows 有或无 Faiss以及两个容器构建共7项检查。规则同时禁止删除和强制推送。

`v*` 发布标签受独立规则保护，创建后不能删除或改写。GitHub 的 `release` environment 已经建立，发布 workflow 仍要求标签指向 `main` 历史中的 commit。

## 发布前外部配置

仍需在 PyPI 为`A-Dawn/A_memorix`的`release.yml`和`release`环境配置 Trusted Publisher。发布`v2.0.0a2`会对外推送 PyPI 包、两个 GHCR 镜像、多平台网关二进制和 GitHub Release，因此配置完成前不创建标签。

Buf 官方 setup action 的最新 v1版本仍声明 Node.js 20。GitHub Runner当前会强制以 Node.js 24兼容模式执行，任务已经通过但保留上游弃用提示；仓库已向该 action传递只读 GitHub token，避免匿名 API 限流。
