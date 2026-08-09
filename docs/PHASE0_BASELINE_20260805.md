# 阶段0基线与归档记录

记录日期：2026-08-05
记录目的：固定通用化改造开始前的代码来源、旧版维护基线和 MaiBot 内嵌实现快照。

## 归档引用

### A_memorix

| 项目 | 值 |
| --- | --- |
| 仓库 | `D:/Dev/rdev/A_memorix` |
| 分支 | `main` |
| 改造前 HEAD | `b9cefdf603c292bee73507bb03b92443f1a362f1` |
| 归档标签 | `legacy-v1.0.1` |
| 维护分支 | `legacy/plugin-v1` |
| 远端 | `origin/main` |

`legacy-v1.0.1` 和 `legacy/plugin-v1` 都指向改造前的 `main` HEAD。该引用代表旧版独立插件结构，不应再接收通用架构改动。

### MaiBot

| 项目 | 值 |
| --- | --- |
| 仓库 | `D:/Dev/rdev/MaiBot` |
| 分支 | `dev` |
| 快照 HEAD | `c43d354c2ab7f8a0d0326a538573d7ca8bd2f07f` |
| 快照标签 | `snapshot/a-memorix-phase0-20260805` |
| 远端跟踪 | `origin/dev` |
| 快照提交 | `test: 稳定迁移与后台复制测试` |

快照标签固定 MaiBot 当前提交版本，特别包括 `src/A_memorix` 当前已提交实现。标签不包含 MaiBot 工作区的未提交改动。

## 工作区状态

### A_memorix

归档前工作区状态：

- 分支：`main`
- 工作区仅有未跟踪的 `docs/` 目录
- 该目录来自本次通用架构计划和阶段基线记录
- 没有已修改或已删除的旧版源码文件

阶段记录提交后，`main` 只新增计划与基线文档，不改变旧版运行代码。

### MaiBot

快照时工作区状态：

- 分支：`dev`
- 工作区共有60条状态记录
- 根目录存在配置、文档、测试数据、实验脚本和截图等未提交改动
- `src/A_memorix` 相对快照 HEAD 没有已修改的已跟踪文件
- `src/A_memorix` 有7个未跟踪分析文档

MaiBot 工作区中的未提交文件属于既有工作，不在本阶段清理、回滚、暂存或提交范围内。`snapshot/a-memorix-phase0-20260805` 只固定提交对象，后续如需恢复未提交研究材料，应以原工作区和用户自行保存的内容为准。

## MaiBot 子树未跟踪文件

以下文件在快照时位于 `src/A_memorix` 下，但尚未进入 Git提交：

- `docs/a_memorix_pytest_migration_logic_audit_20260729.md`
- `docs/episode_segmentation_coverage_and_discard_protocol_issue_20260729.md`
- `docs/episode_segmentation_token_timeout_analysis_20260729.md`
- `docs/graph_adjacency_npz_corruption_20260729.md`
- `docs/hash_entity_pollution_analysis_20260729.md`
- `docs/person_profile_refresh_stagnation_analysis_20260729.md`
- `docs/query_memory_import_retrieval_issue_20260729.md`

实际状态统计为7个未跟踪文件；上面列出的文件以快照时工作区扫描结果为准。若后续要把其中的分析材料纳入历史，应单独决定归属，不应直接混入通用核心。

## 阶段0边界

从本记录提交开始：

1. A_memorix 的通用架构工作在新开发分支进行，稳定后再切换 `main`。
2. `legacy/plugin-v1` 只接受旧版插件必要的安全或构建维护。
3. MaiBot 暂不进行为了抽取通用层的接口、运行方式和性能重构。
4. MaiBot 只对向量重启、删除恢复等数据正确性问题保留最小修复通道。
5. 通用实现、公开 API、Namespace Runtime 和协议代码不得继续直接写入 MaiBot 工作树。
6. MaiBot 当前实现作为行为参考，通用测试迁移后再建立 `integration/maibot`。

## 复核命令

```bash
# A_memorix 旧版归档
git show legacy-v1.0.1
git log legacy/plugin-v1 -1

# MaiBot 内嵌实现快照
git -C ../MaiBot show snapshot/a-memorix-phase0-20260805:src/A_memorix/__init__.py
git -C ../MaiBot status --short

# 当前实现来源
git -C ../MaiBot log -1 -- src/A_memorix
git -C ../MaiBot diff --name-only snapshot/a-memorix-phase0-20260805 -- src/A_memorix
```

阶段0完成后，下一阶段从修复向量指纹重启失败开始，再在独立开发分支中整理 Host Port 和 Namespace Runtime。
