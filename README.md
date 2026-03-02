# A_Memorix

**轻量级知识图谱插件** - 基于双路检索 + 人物画像的独立记忆增强系统 (v0.6.1)

> 消えていかない感覚 , まだまだ足りてないみたい !

> [!WARNING]
> **重要提示**：v0.2.0 版本由于底层存储架构重构（引入 SciPy 稀疏矩阵与 Faiss SQ8 量化），**与 v0.1.3 及早期版本的导入数据不完全兼容**。
> 升级后，虽然系统会尝试自动迁移部分数据，但为确保知识图谱的检索精度和完整性，强烈建议在升级后使用 `process_knowledge.py` 脚本重新导入原始文本。

> [!NOTE]
> **v0.6.1 热修复（WebUI 配置接口序列化兼容）**：
> 1. 修复 `A_Memorix` 在 WebUI 插件配置接口中的 `tomlkit` 节点序列化问题；
> 2. 仅影响 `/api/webui/plugins/config/{plugin_id}` 及其 schema 路由；
> 3. 全局 `/api/webui/config/*` 接口行为保持不变。

> [!NOTE]
> **v0.6.0 导入能力与WebUI增强**：
> 1. 新增 Web Import 导入中心（`/import`），支持上传/粘贴/本地扫描/LPMM OpenIE/LPMM转换/时序回填/MaiBot 迁移；
> 2. 导入状态细化到任务/文件/分块级，支持取消与“失败项重试（分块优先 + 文件回退）”；
> 3. 导入期间写操作保护、删除后 manifest 同步失效、导入文档弹窗与中文状态展示已对齐。


## 📑 文档索引

- [⚡ 快速入门（5分钟上手）](QUICK_START.md)
- [📘 配置参数详解（config.toml）](CONFIG_REFERENCE.md)
- [📗 导入指南与最佳实践](IMPORT_GUIDE.md)
- [📝 更新日志](CHANGELOG.md)

---

## ✨ 特性

- **🧠 双路检索** - 关系图谱 + 向量语义并行检索，结合 Personalized PageRank 智能排序。
- **⏱️ 时序检索（v0.5.0）** - 支持 `time/hybrid` 模式，按事件时间区间命中并可回退 `created_at`。
- **👤 人物画像（v0.5.0）** - 支持人物画像快照、别名解析、手工覆盖、按会话 opt-in 注入控制。
- **📥 Web Import 导入中心（v0.6.0）** - 提供统一导入控制台（`/import`），支持上传/粘贴/路径扫描/LPMM 迁移与任务级可观测。
- **🧩 稀疏检索增强（FTS5 + BM25）** - embedding 不可用或召回偏弱时自动走 sparse，支持 `jieba/mixed/char_2gram` 分词与 `ngram` 倒排回退。
- **🧬 生物学记忆 (V5)** - 模拟人类记忆的**衰减 (Decay)**、**强化 (Reinforce)** 与 **结构化重组 (Prune)** 机制，实现记忆的动态生命周期管理。
- **🔄 智能回退** - 当直接检索结果弱时，自动触发多跳路径搜索，增强间接关系召回。
- **🛡️ 网络鲁棒性** - 内置指数退避重试机制，支持自定义嵌入请求的重试策略，从容应对网络波动。
- **📊 知识图谱可视化** - 全新 Glassmorphism 风格 Web 编辑器，支持基于 **PageRank** 的信息密度筛选、记忆溯源管理及全量图谱探索。
- **📝 对话自动总结** - 自动总结历史聊天记录并提取知识，支持定时触发和人设深度整合。
- **🎯 智能分类** - 兼容并自动识别结构化/叙事性/事实性知识，采用差异化处理策略。
- **💾 高效存储** - SciPy 稀疏矩阵存储图结构，Faiss SQ8 向量量化节省 75%+ 空间。
- **🔌 完全独立** - 不依赖原 LPMM 系统，拥有独立的数据格式和存储路径。
- **🤖 LLM 集成** - 提供 Tool 和 Action 组件，支持 LLM 自主调用知识库。

---

## 📦 安装

### 方式一：一键安装（推荐）

如果主程序支持插件依赖管理，插件启用时会自动尝试安装 `python_dependencies`。

### 方式二：手动安装

在主程序根目录下进入虚拟环境后，在插件目录下运行：

```bash
pip install -r requirements.txt
```

**核心依赖：**

- `numpy`, `scipy` (计算与矩阵)
- `faiss-cpu` (向量检索)
- `rich` (终端可视化)
- `tenacity` (重试机制)
- `nest-asyncio` (环境兼容)
- `jieba` (稀疏检索中文分词；未安装时自动回退 char n-gram)
- `networkx`, `pyarrow`, `pandas` (LPMM 转换链路)
- `fastapi`, `uvicorn`, `pydantic` (可视化服务器)
- `python-multipart` (Web 上传解析)

---

## 🧯 问题解决（Troubleshooting）

### 一键包环境下 FTS5 排序失效

已知在部分一键包运行时环境中，SQLite 未启用 `FTS5`，会导致稀疏检索（`FTS5 + BM25`）路径不可用，表现为相关排序增强失效。

常见现象：

- 日志出现 `FTS5 schema 创建失败（可能不支持 FTS5）（no moduled named 'FTS5'）`

说明：

- `fts5` 不是可单独安装的 PyPI 包，通常由 Python 运行时内置的 SQLite 编译选项决定是否可用。
- 当前该一键包环境下暂无通用修复方案。

建议处理（直接修改对应配置）：

在 `config.toml` 中关闭稀疏检索增强，避免反复触发无效 FTS 初始化：

```toml
[retrieval.sparse]
enabled = false
```

关闭后系统仍可使用向量/图检索链路，但不再使用 `FTS5 + BM25` 稀疏排序增强。

---

## 🚀 快速开始

A_Memorix 提供多种方式管理知识库，建议优先选择 **自动化脚本** 进行初始化，配合 **可视化编辑器** 进行日常维护。

### 1. 自动化批量导入 (`process_knowledge.py`)

> 📖 **详细指南**：关于各类文本的格式要求、策略选择及各类示例，请务必阅读 [**导入指南与最佳实践**](IMPORT_GUIDE.md)。

适用于从大量历史文档快速构建知识库。脚本会自动调用 LLM 提取实体和关系。

**文件要求：**

- **格式**：仅支持 `.txt` 平面文本。
- **内容**：支持**自由形式的自然语言文本**。无需特定标记或结构，脚本会调用 LLM 自动分析其中的实体与关系。
- **编码**：必须使用 `UTF-8` 编码。
- **路径**：文件需放入 `plugins/A_memorix/data/raw/` 目录。

**操作步骤：**

1. 将 `.txt` 格式的原始文档放入 `plugins/A_memorix/data/raw/` 目录。
2. 运行脚本（请确定你运行脚本的环境已经安装了依赖）：
   ```bash
   python plugins/A_memorix/scripts/process_knowledge.py
   ```

**支持参数：**

- `--force`: 强制重新导入已处理过的文件。
- `--clear-manifest`: 清空导入历史记录并重新扫描。
- `--type <type>`: 指定内容类型（`structured`, `narrative`, `factual`）。
- `--chat-log`: 聊天记录模式。默认按 `narrative` 处理，并使用 LLM 语义理解提取 `event_time/event_time_start/event_time_end`（可解析相对时间）。
- `--chat-reference-time <datetime>`: 聊天记录模式的相对时间参考点（如 `2026/02/12 10:30`）；不传则使用当前本地时间。

### 1.1 迁移 LPMM 数据 (`import_lpmm_json.py`)

如果你有符合 LPMM 规范的 OpenIE JSON 数据，可以使用此脚本将其转换为 A_Memorix 格式并导入：

```bash
python plugins/A_memorix/scripts/import_lpmm_json.py <path_to_json_file_or_dir>
```

**参数：**

- `path`: JSON 文件路径或包含 `*-openie.json` 的目录。
- `--force`: 强制重新导入。

### 1.2 转换 LPMM 存储文件 (`convert_lpmm.py`)  [新增]

如果你有 LPMM 导出的 parquet/graph 文件，可使用该脚本直接转换为 A_Memorix 存储结构：

```bash
python plugins/A_memorix/scripts/convert_lpmm.py -i <lpmm_data_dir> -o <output_data_dir> --dim 384
```

**说明：**

- 输入目录支持 `paragraph.parquet`、`entity.parquet` 以及 `rag-graph.graphml/graph.graphml/graph_structure.pkl`。
- 当前版本优先保证 ID 与元数据一致性，关系向量不做直接导入（避免检索反查不一致）。

### 1.3 回填旧数据时序字段 (`backfill_temporal_metadata.py`)

当历史段落缺失 `event_time/event_time_start/event_time_end` 时，可使用脚本按 `created_at` 回填，提升 `time/hybrid` 检索命中率：

```bash
python plugins/A_memorix/scripts/backfill_temporal_metadata.py --dry-run
python plugins/A_memorix/scripts/backfill_temporal_metadata.py --limit 50000
```

默认回填策略：`event_time=created_at`、`time_granularity=day`、`time_confidence=0.2`。

### 2. 指令交互

在聊天窗口中直接输入以下一级命令进行操作：

| 命令         | 模式                                             | 说明                | 示例                         |
| ------------ | ------------------------------------------------ | ------------------- | ---------------------------- |
| `/import`    | `text`, `paragraph`, `relation`, `file`, `json`  | 导入知识            | `/import text 人工智能是...` |
| `/query`     | `search(s)`, `time(t)`, `entity(e)`, `relation(r)`, `stats` | 查询知识 | `/query t q=项目进展 from=2025/01/01 to=2025/01/31` |
| `/delete`    | `paragraph`, `entity`, `clear`                   | 删除知识            | `/delete paragraph <hash>`   |
| `/memory`    | `status`, `protect`, `reinforce`, `restore`      | 记忆系统维护 (V5)   | `/memory status`             |
| `/person_profile` | `on`, `off`, `status`                         | 人物画像注入开关（按会话+用户） | `/person_profile on` |
| `/visualize` | -                                                | 启动可视化 Web 面板 | `/visualize`                 |

#### 🧠 记忆系统维护 (`/memory`)

- **查看状态**: `/memory status` - 显示活跃/冷冻/保护记忆数量及系统参数。
- **保护记忆**: `/memory protect <query> [hours]` - 保护相关记忆不被衰减/修剪。
  - 不填时间 = **永久锁定 (Pin)**
  - 填时间 = **临时保护 (TTL)** (e.g., `/memory protect 昨天的会议 24`)
- **手动强化**: `/memory reinforce <query>` - 手动触发检索强化（绕过冷却时间），提升记忆权重。
- **恢复记忆**: `/memory restore <hash>` - 从回收站恢复误删记忆（仅当节点存在时有效）。

#### 📂 导入知识 (`/import`)

- **文本（自动提取）**：`/import text 知识内容...`
- **单个段落**：`/import paragraph 段落内容...`
- **关系 (主|谓|宾)**：`/import relation Apple|founded|Steve Jobs`
- **文件 (.txt, .md, .json)**：`/import file ./my_notes.txt`
- **JSON 结构化**：`/import json {"paragraphs": [...], "entities": [...], "relations": [...]}`

#### 🔍 查询知识 (`/query`)

- **全文检索**：`/query search <query>` (缩写: `/query s`) - 支持智能回退到路径搜索。
- **时序检索**：`/query time <k=v参数>` (缩写: `/query t`) - 支持 `q/query`、`from/start`、`to/end`、`person`、`source`、`top_k`。
  - 时间格式仅支持：`YYYY/MM/DD` 或 `YYYY/MM/DD HH:mm`。
  - 日期格式会自动展开：`from -> 00:00`，`to -> 23:59`。
- **实体查询**：`/query entity <name>` (缩写: `/query e`)
- **关系查询**：`/query relation <spec>` (缩写: `/query r`) - 支持自然语言或 `S|P|O` 格式。
- **统计信息**：`/query stats`
- **人物画像**：`/query person <id|别名>`（简写：`/query p`）

#### 🗑️ 删除与维护

- **按 Hash 删除段落**：`/delete paragraph <hash>`
- **删除特定实体**：`/delete entity <name>`
- **清空数据库**：`/delete clear` (慎用！)

### 3. 可视化编辑 (推荐)

运行 `/visualize` 命令后，访问 `http://localhost:8082` 即可进入图形化编辑器。支持：

- 节点/关联的实时增删改查。
- **显著性视图**: 通过底部的“Dock 栏” -> “视图配置”，调整信息密度滑块，查看从核心骨干到全量细节的不同层级图谱。
- **记忆溯源**: 通过“记忆溯源”面板，按导入文件（来源）批量管理和删除记忆。
- **知识字典**: 浏览所有实体与关系的列表视图。

### 4. 核心配置说明 (`config.toml`)

你可以通过修改 `config.toml` 来定制插件行为。v0.2.0 版本提供了更细粒度的控制。
完整逐项说明请查看：[📘 配置参数详解（config.toml）](CONFIG_REFERENCE.md)。

#### 💾 存储与嵌入 `[storage] & [embedding]`

- **`storage.data_dir`**: 数据存储路径（默认为插件内 `data` 目录）。
- **`embedding.quantization_type`**: 向量量化模式 (`int8` 推荐, `float32`, `pq`)。
- **`embedding.dimension`**: 向量维度（默认 1024）。
- **`embedding.retry.max_attempts`**: 最大重试次数 (默认 10)。
- **`embedding.retry.max_wait_seconds`**: 最大等待时间 (默认 30)。

#### ⚙️ 检索与排序 `[retrieval]`

- **`alpha`**: 双路检索融合权重 (0.0=仅关系, 1.0=仅段落, 0.5=平衡)。
- **`enable_ppr`**: 是否启用 Personalized PageRank 算法优化排序。
- **`top_k_relations` / `top_k_paragraphs`**: 分别控制单路检索召回数量。
- **`relation_semantic_fallback`**: 是否允许关系检索回退到语义搜索。
- **`search.smart_fallback.enabled`**: search 低分时是否启用路径回退（默认 `true`）。
- **`search.smart_fallback.threshold`**: search 路径回退触发阈值（默认 `0.6`）。
- **`search.safe_content_dedup.enabled`**: 统一链路安全去重开关（默认 `true`）。
- **`time.skip_threshold_when_query_empty`**: time 且 query 为空时跳过阈值过滤（默认 `true`）。
- **`sparse.mode`**: 稀疏检索模式（`auto/fallback_only/hybrid`），默认 `auto`。
- **`sparse.tokenizer_mode`**: 分词模式（`jieba/mixed/char_2gram`）。
- **`sparse.enable_ngram_fallback_index`**: FTS miss 时启用 ngram 倒排回退（默认开）。
- **`sparse.enable_relation_sparse_fallback`**: 关系通道稀疏回退独立开关（默认开）。
- **`fusion.method`**: 融合方法（默认 `weighted_rrf`，支持向量+BM25 候选融合）。
- **`fusion.vector_weight + fusion.bm25_weight`**: 若和不为 1，会自动归一化。

#### 🔀 检索路由 `[routing]`

- **`search_owner`**: `search/time` 主责入口（`action|tool|dual`），默认 `action`。
- **`tool_search_mode`**: Tool 在 `search/time` 的模式（`forward|disabled`），默认 `forward`；`legacy` 为兼容别名并按 `forward` 处理。
- **`enable_request_dedup`**: 启用短时请求去重，抑制 Action+Tool 同轮重复检索。
- **`request_dedup_ttl_seconds`**: 去重 TTL，默认 `2` 秒。

#### 👤 人物画像 `[person_profile]`

- **`enabled`**: 人物画像模块总开关。
- **`opt_in_required`**: 是否要求显式开启注入（默认 `true`）。
- **`default_injection_enabled`**: 无显式开关记录时的默认注入状态。
- **`profile_ttl_minutes`**: 画像快照 TTL。
- **`refresh_interval_minutes`**: 定时刷新周期。
- **`active_window_hours`**: 仅刷新活跃窗口内人物。
- **`max_refresh_per_cycle`**: 每轮最大刷新人数。
- **`top_k_evidence`**: 画像构建证据上限。

#### 🧩 稀疏检索行为说明

- `sparse.mode=auto` 下，满足任一条件会触发段落 sparse：embedding 不可用、向量结果为空、向量最高分 `< 0.45`。
- 段落 sparse 回退链路：`FTS5 BM25 -> ngram 倒排 -> (可选)LIKE 扫描`。
- 关系 sparse 是否参与由 `sparse.enable_relation_sparse_fallback` 控制。
- 首次加载 sparse 可能触发 FTS/倒排回填，冷启动时延会高于常态。

#### 🧬 记忆系统 (V5) `[memory]`

- **`half_life_hours`**: 记忆半衰期（小时）。图谱连接权重每经过一个半衰期会衰减 50%。(默认 24.0)
- **`enable_auto_reinforce`**: 是否开启检索强化。开启后，被搜索命中的记忆会自动恢复活跃并增加权重。(默认 true)
- **`prune_threshold`**: 冻结/修剪阈值。权重低于此值的非保护记忆将被冻结。(默认 0.1)
- **`freeze_duration_hours`**: 冷冻期时长。记忆冻结超过此时长后将被移入回收站。(默认 24.0)

#### 🎯 动态阈值 `[threshold]`

- **`min_threshold`**: 硬性最小相似度阈值 (默认 0.3)。
- **`enable_auto_adjust`**: 是否启用动态阈值调整（基于结果分布）。
- **`std_multiplier`**: 异常值过滤的标准差倍数。

#### 🧠 自动化功能 `[summarization] & [schedule]`

- **`summarization.enabled`**: 开启对话自动总结。
- **`summarization.model_name`**: 支持 `auto` / 任务名 / 模型名；也支持数组与选择器（如 `["utils:model1", "utils:model2", "replyer"]`）。
- **`schedule.import_times`**: 定时自动导入时间点列表 (e.g., `["04:00"]`).

#### 🛡️ 聊天流过滤 `[filter]`

- **`mode`**: `whitelist` (白名单) 或 `blacklist` (黑名单)。
- **`chats`**: 目标列表。支持 `group:123`(群), `user:456`(私聊), `stream:hash`(流ID) 或纯数字 ID(兼容)。

#### 🖥️ 可视化与调试 `[web] & [advanced]`

- **`web.port`**: 可视化界面端口 (默认 8082)。
- **`advanced.debug`**: 开启详细调试日志。

---

## 🏗️ 架构设计

### 目录结构

```
plugins/A_memorix/
├── core/                     # 核心引擎
│   ├── storage/              # 向量、图、元数据存储（NPZ, PKL, SQLite）
│   ├── embedding/            # 嵌入生成（调用主程序 API）
│   ├── retrieval/            # 双路检索与排序 (PPR 算法)
│   └── utils/                # 文本规范化与哈希工具
├── scripts/                  # 自动化脚本
│   └── process_knowledge.py  # 批量导入工具
├── components/               # 插件组件
│   ├── commands/             # 指令集 (/import, /query, etc.)
│   ├── tools/                # LLM 外部工具 (knowledge_query)
│   └── actions/              # 自动行为 (knowledge_search)
├── server.py                 # FastAPI 可视化服务器后端
├── data/                     # 独立数据目录（存储于插件文件夹内）
└── config.toml               # 插件配置文件
```

---

## 🔒 独立性声明

A_Memorix 是**完全独立**的知识管理系统，与原 LPMM 在技术实现上有本质区别：

| 维度         | 原 LPMM               | A_Memorix                         |
| ------------ | --------------------- | --------------------------------- |
| **后端引擎** | 基于对象/字典的图算法 | 基于 SciPy 稀疏矩阵的线性代数计算 |
| **向量格式** | float32 (高内存消耗)  | Faiss SQ8 量化 (极致内存压缩)     |
| **存储路径** | 全局 `data/` 目录     | 隔离的 `plugins/A_memorix/data/`  |
| **依赖关系** | 与主程序逻辑混杂      | 模块化解耦，可独立升级            |
| **数据格式** | JSON/SQLite           | NPZ/PKL/SQLite                    |

---

## 📜 许可证

本项目采用 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0) 许可证。

## 贡献声明

本项目目前不接受任何PR，只接受issue，如有相关问题请提交issue或联系ARC

**作者**: A_Dawn
