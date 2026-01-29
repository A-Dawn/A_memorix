# A_Memorix

**轻量级知识图谱插件** - 基于双路检索的完全独立的记忆增强系统 (v0.2.3)

> 消えていかない感覚 , まだまだ足りてないみたい !

> [!WARNING]
> **重要提示**：v0.2.0 版本由于底层存储架构重构（引入 SciPy 稀疏矩阵与 Faiss SQ8 量化），**与 v0.1.3 及早期版本的导入数据不完全兼容**。
> 升级后，虽然系统会尝试自动迁移部分数据，但为确保知识图谱的检索精度和完整性，强烈建议在升级后使用 `process_knowledge.py` 脚本重新导入原始文本。

---

## ✨ 特性

- **🧠 双路检索** - 关系图谱 + 向量语义并行检索，结合 Personalized PageRank 智能排序。
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
- `fastapi`, `uvicorn`, `pydantic` (可视化服务器)

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

### 1.1 迁移 LPMM 数据 (`import_lpmm_json.py`)

如果你有符合 LPMM 规范的 OpenIE JSON 数据，可以使用此脚本将其转换为 A_Memorix 格式并导入：

```bash
python plugins/A_memorix/scripts/import_lpmm_json.py <path_to_json_file_or_dir>
```

**参数：**

- `path`: JSON 文件路径或包含 `*-openie.json` 的目录。
- `--force`: 强制重新导入。

### 2. 指令交互

在聊天窗口中直接输入以下一级命令进行操作：

| 命令         | 模式                                             | 说明                | 示例                         |
| ------------ | ------------------------------------------------ | ------------------- | ---------------------------- |
| `/import`    | `text`, `paragraph`, `relation`, `file`, `json`  | 导入知识            | `/import text 人工智能是...` |
| `/query`     | `search(s)`, `entity(e)`, `relation(r)`, `stats` | 查询知识            | `/query s 什么是AI?`         |
| `/delete`    | `paragraph`, `entity`, `clear`                   | 删除知识            | `/delete paragraph <hash>`   |
| `/visualize` | -                                                | 启动可视化 Web 面板 | `/visualize`                 |

#### 📂 导入知识 (`/import`)

- **文本（自动提取）**：`/import text 知识内容...`
- **单个段落**：`/import paragraph 段落内容...`
- **关系 (主|谓|宾)**：`/import relation Apple|founded|Steve Jobs`
- **文件 (.txt, .md, .json)**：`/import file ./my_notes.txt`
- **JSON 结构化**：`/import json {"paragraphs": [...], "entities": [...], "relations": [...]}`

#### 🔍 查询知识 (`/query`)

- **全文检索**：`/query search <query>` (缩写: `/query s`) - 支持智能回退到路径搜索。
- **实体查询**：`/query entity <name>` (缩写: `/query e`)
- **关系查询**：`/query relation <spec>` (缩写: `/query r`) - 支持自然语言或 `S|P|O` 格式。
- **统计信息**：`/query stats`

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

#### 🎯 动态阈值 `[threshold]`

- **`min_threshold`**: 硬性最小相似度阈值 (默认 0.3)。
- **`enable_auto_adjust`**: 是否启用动态阈值调整（基于结果分布）。
- **`std_multiplier`**: 异常值过滤的标准差倍数。

#### 🧠 自动化功能 `[summarization] & [schedule]`

- **`summarization.enabled`**: 开启对话自动总结。
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
