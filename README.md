# A_Memorix

**轻量化的长期记忆与认知增强插件** (v1.0.1)

> 消えていかない感覚 , まだまだ足りてないみたい !

知识图谱、Episode 情景记忆、人物画像、时间演化，统一在一个插件里完成，ALL IN ONE。

A_Memorix 不只是“把内容存下来”。它把文本、关系和时间证据组织成一个可长期运行的记忆系统：能检索、能巩固、能形成画像，也能随着时间强化、衰减、保护和恢复。

如果你只想要一个最小向量库，它不是最轻的方案。  
如果你想让 MaiBot 拥有可长期维护的记忆层，它就是为此而生的！

## 快速导航

- [快速入门](QUICK_START.md)
- [配置参数详解](CONFIG_REFERENCE.md)
- [导入指南与最佳实践](IMPORT_GUIDE.md)
- [更新日志](CHANGELOG.md)
- [1.0.0 发布总结](RELEASE_SUMMARY_1.0.0.md)

> [!NOTE]
> 当前 `main` 分支和 `dev` 分支均已更新到 `v1.0.1`。
> 如果想继续使用更早版本，请访问分支 <https://github.com/A-Dawn/A_memorix/tree/v0.7.0-LTSC> 获取。

> [!IMPORTANT]
> `v1.0.0` 是硬切升级。
> 新主线不再在启动时自动迁移旧配置、旧向量或旧 schema。
>
> 升级前请先备份data文件夹，然后执行：
>
> ```bash
> python plugins/A_memorix/scripts/release_vnext_migrate.py preflight --strict
> python plugins/A_memorix/scripts/release_vnext_migrate.py migrate --verify-after
> python plugins/A_memorix/scripts/release_vnext_migrate.py verify --strict
> ```

## 为什么需要它

普通 memory 更像“可搜索存档”。

但长期运行的 Agent 通常还缺这些能力：

- 记忆如何随着时间变化
- 事件如何从零散证据变成可复用情景
- 人物如何从长期证据中形成稳定画像
- 不同召回方式如何被统一到一个入口

A_Memorix 的目标就是把这些层补齐，同时保持可控、可观察、可维护的特性。

## 30 秒了解它

### 它做什么

- 把文本与事实写入长期记忆
- 把证据巩固成 `Episode`
- 把长期行为抽象成 `Profile`
- 把检索、时间、情景与演化放进同一套系统

## v1.0.1 修复亮点

- **图谱页不再无声空白**：修复 `/api/graph` 在图文件已存在但运行时尚未装载时返回空图的问题，WebUI 会自动补加载持久化图数据。
- **问题数据集的大图加载恢复可用**：优化 `exclude_leaf=true` 过滤路径，修复因大图过滤过慢而表现为“没有图节点”的问题。
- **真实调优链路不再因 `RLock` 崩溃**：修复调优任务构建运行时配置时误深拷贝注入实例，避免 `cannot pickle '_thread.RLock' object`。
- **调优前置检查更明确**：如果真实 embedding 输出维度与当前向量库不一致，调优会退化为回退检索并显著变慢；建议先执行 `python plugins/A_memorix/scripts/runtime_self_check.py --json`。

## v1.0.0 更新亮点

- **Episode 成为正式能力**：新增情景记忆落库、重建和检索链路
- **Aggregate 查询上线**：`search / time / episode` 可并发执行并统一汇总
- **运行时模块化**：生命周期、后台任务、请求路由、检索运行时从 `plugin.py` 拆分出去
- **升级路径硬切**：旧配置、旧向量、旧 schema 不再在运行时隐式迁移
- **运维能力补齐**：新增离线迁移、自检脚本、Web 调优中心

## 记忆模型

| 层 | 作用 | 不做什么 |
| --- | --- | --- |
| `Paragraph` | 保存原始证据、时间元数据与来源 | 不直接当作高层结论 |
| `Relation` | 组织稳定事实与实体关系 | 不靠暴力覆盖维护一致性 |
| `Episode` | 把同源、同时间窗口内容巩固成情景记忆 | 不直接替代图谱边 |
| `Profile` | 抽象长期行为与人物特征 | 不反写污染底层事实 |

这也是 A_Memorix 和普通“只负责召回”的 memory 工具最核心的差异。

## 核心能力

### 检索

- **双路检索**：关系图谱 + 向量语义并行召回
- **时序检索**：按事件时间窗口过滤并支持时间回退
- **聚合检索**：统一汇总 `search / time / episode`
- **稀疏增强**：`FTS5 + BM25`，embedding 弱或不可用时自动补位
- **图辅助回退**：低分结果可触发路径与关系补召回

### 记忆

- **关系写入状态机**：支持 `none / pending / ready / failed`
- **Episode 情景记忆**：按来源与时间窗口进行语义切分与巩固
- **人物画像**：按用户与会话维护长期证据抽象
- **记忆演化**：支持强化、衰减、冻结、保护、恢复

### 运维

- **Web Import 导入中心**：上传、粘贴、本地扫描、OpenIE、LPMM 转换、时序回填、MaiBot 迁移
- **检索调优中心**：支持 Web 侧参数调优与任务化评估
- **运行时自检**：直接验证 embedding 真实输出维度
- **离线迁移脚本**：迁移前检查、执行迁移、迁移后校验

## 快速开始

建议优先走：**离线导入 -> 聊天验证 -> Web 维护**。

### 1. 安装依赖

在 MaiBot 主程序使用的同一个虚拟环境中执行：

```bash
pip install -r requirements.txt --upgrade
pip install -r plugins/A_memorix/requirements.txt --upgrade
```

更完整的 `venv / uv / conda / Docker` 启动方式见 [QUICK_START.md](QUICK_START.md)。

### 2. 启用插件

编辑 `plugins/A_memorix/config.toml`：

```toml
[plugin]
enabled = true
```

### 3. 批量导入文本

将 `.txt` 文件放入：

```text
plugins/A_memorix/data/raw/
```

然后执行：

```bash
python plugins/A_memorix/scripts/process_knowledge.py
```

常用参数：

```bash
python plugins/A_memorix/scripts/process_knowledge.py --force
python plugins/A_memorix/scripts/process_knowledge.py --chat-log
python plugins/A_memorix/scripts/process_knowledge.py --chat-log --chat-reference-time "2026/02/12 10:30"
```

文本策略、示例与 Web Import 细节见 [IMPORT_GUIDE.md](IMPORT_GUIDE.md)。

### 4. 验证是否生效

```text
/query stats
```

如果能看到段落、实体、关系或 Episode 统计，说明插件已经工作。

## 常用命令

| 命令 | 作用 | 示例 |
| --- | --- | --- |
| `/import` | 导入文本、段落、关系、文件、JSON | `/import text 人工智能是...` |
| `/query search` | 语义检索 | `/query search 项目复盘` |
| `/query time` | 时序检索 | `/query time q=会议 from=2026/03/01 to=2026/03/06` |
| `/query episode` | 情景记忆检索 | `/query episode q=项目复盘 top_k=5` |
| `/query aggregate` | 聚合检索 | `/query aggregate q=项目复盘 mix=true` |
| `/query entity` | 实体查询 | `/query entity A_Memorix` |
| `/query relation` | 关系查询 | `/query relation ARC|完成了|发布` |
| `/query person` | 人物画像查询 | `/query person ARC` |
| `/memory` | 记忆强化、保护、恢复、状态查看 | `/memory status` |
| `/visualize` | 启动可视化 Web 面板 | `/visualize` |

最常用的 `/memory` 命令：

- `/memory status`
- `/memory protect <query> [hours]`
- `/memory reinforce <query>`
- `/memory restore <hash>`

## Web 界面

运行 `/visualize` 后，可访问：

- 图谱编辑器：`http://localhost:8082`
- 导入中心：`http://localhost:8082/import`
- 调优中心：`http://localhost:8082/tuning`

Web 侧主要解决三件事：

- **看结构**：浏览图谱、实体、关系和来源
- **管导入**：统一处理文件、JSON、扫描和迁移任务
- **调系统**：查看自检状态与检索调优任务

## 常用脚本

| 脚本 | 用途 |
| --- | --- |
| `process_knowledge.py` | 批量导入原始文本 |
| `import_lpmm_json.py` | 导入 OpenIE JSON |
| `convert_lpmm.py` | 转换 LPMM 存储数据 |
| `backfill_temporal_metadata.py` | 回填历史时间字段 |
| `audit_vector_consistency.py` | 审计关系向量一致性 |
| `backfill_relation_vectors.py` | 回填缺失关系向量 |
| `rebuild_episodes.py` | 按来源重建 Episode |
| `release_vnext_migrate.py` | 升级前检查、迁移与校验 |
| `runtime_self_check.py` | 运行时自检 |

示例：

```bash
python plugins/A_memorix/scripts/import_lpmm_json.py <path_to_json_file_or_dir>
python plugins/A_memorix/scripts/convert_lpmm.py -i <lpmm_data_dir> -o <output_data_dir> --dim 384
python plugins/A_memorix/scripts/backfill_temporal_metadata.py --dry-run
python plugins/A_memorix/scripts/rebuild_episodes.py --all --wait
```

## 配置重点

完整配置请看 [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)，这里先看最常改的几组：

### 存储与嵌入

- `storage.data_dir`
- `embedding.dimension`
- `embedding.quantization_type`
- `embedding.retry.*`

> `v1.0.0` 下 `embedding.quantization_type` 只支持 `int8 / SQ8`。

> 如需使用 Web 调优中心，建议先通过 `runtime_self_check.py` 确认 `embedding.dimension` 与真实编码输出一致，否则调优结果会受到回退链路和高延迟影响。

### 检索

- `retrieval.alpha`
- `retrieval.top_k_relations`
- `retrieval.top_k_paragraphs`
- `retrieval.top_k_final`
- `retrieval.search.*`
- `retrieval.aggregate.*`
- `retrieval.sparse.*`

### 路由与人物画像

- `routing.search_owner`
- `routing.tool_search_mode`
- `person_profile.enabled`
- `person_profile.opt_in_required`

### Episode 与记忆演化

- `episode.enabled`
- `episode.query_enabled`
- `episode.generation_enabled`
- `memory.half_life_hours`
- `memory.enable_auto_reinforce`
- `memory.prune_threshold`

## 适合什么场景

- 长期陪伴型 Agent
- 角色扮演 / 世界观 / 设定集
- 项目对话与时间线管理
- 需要人物画像或事件巩固的 Agent 记忆系统
- 希望在 MaiBot 内完成导入、检索、运维、调优，获得一个真正的记忆闭环

## Troubleshooting

### 一键包环境下 FTS5 不可用

部分运行环境的 SQLite 没有启用 `FTS5`，会导致稀疏检索路径不可用。

常见现象：

- 日志出现 `FTS5 schema 创建失败`

可以在 `config.toml` 中关闭稀疏检索：

```toml
[retrieval.sparse]
enabled = false
```

关闭后系统仍可使用向量检索与图检索，只是不会使用 `FTS5 + BM25` 路径。

## 独立运行

A_Memorix 不依赖旧 LPMM 的运行时实现，使用自己独立的数据目录与存储结构：

- 存储路径：`plugins/A_memorix/data/`
- 数据格式：`NPZ / PKL / SQLite`
- 升级方式：显式迁移与校验
- 运行方式：模块化插件，可独立维护

## 许可证

本项目采用 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0)。

## 贡献说明

当前不接受 PR，只接受 issue。

**作者**: `A_Dawn`
