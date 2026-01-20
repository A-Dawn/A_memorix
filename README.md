# A_Memorix

**轻量级知识图谱插件** - 基于双路检索的完全独立的记忆增强系统

> 消えていかない感覚 , まだまだ足りてないみたい !
---

## ✨ 特性

- **🧠 双路检索** - 关系图谱 + 向量语义并行检索，结合 Personalized PageRank 智能排序。
- **📊 知识图谱可视化** - 内置 Web 可视化编辑器，支持节点/边的增删改查。
- **📝 对话自动总结** - 自动总结历史聊天记录并提取知识，支持定时触发和人设深度整合。
- **🎯 智能分类** - 兼容并自动识别结构化/叙事性/事实性知识，采用差异化处理策略。
- **💾 高效存储** - SciPy 稀疏矩阵存储图结构，int8 量化向量节省 75% 空间。
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
- `nest-asyncio` (环境兼容)
- `fastapi`, `uvicorn`, `pydantic` (可视化服务器)

---

## 🚀 快速开始

A_Memorix 提供多种方式管理知识库，建议优先选择 **自动化脚本** 进行初始化，配合 **可视化编辑器** 进行日常维护。

### 1. 自动化批量导入 (`process_knowledge.py`)

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

### 2. 指令交互

在聊天窗口中直接输入以下一级命令进行操作：

| 命令 | 模式 | 说明 | 示例 |
|------|------|------|------|
| `/import` | `text`, `paragraph`, `relation`, `file`, `json` | 导入知识 | `/import text 人工智能是...` |
| `/query` | `search(s)`, `entity(e)`, `relation(r)`, `stats` | 查询知识 | `/query s 什么是AI?` |
| `/delete` | `paragraph`, `entity`, `clear` | 删除知识 | `/delete paragraph <hash>` |
| `/visualize` | - | 启动可视化 Web 面板 | `/visualize` |

#### 📂 导入知识 (`/import`)
- **文本（自动提取）**：`/import text 知识内容...`
- **单个段落**：`/import paragraph 段落内容...`
- **关系 (主|谓|宾)**：`/import relation Apple|founded|Steve Jobs`
- **文件 (.txt, .md, .json)**：`/import file ./my_notes.txt`
- **JSON 结构化**：`/import json {"paragraphs": [...], "entities": [...], "relations": [...]}`

#### 🔍 查询知识 (`/query`)
- **全文检索**：`/query search <query>` (缩写: `/query s`)
- **实体查询**：`/query entity <name>` (缩写: `/query e`)
- **关系查询**：`/query relation <spec>` (缩写: `/query r`)
- **统计信息**：`/query stats`

#### 🗑️ 删除与维护
- **按 Hash 删除段落**：`/delete paragraph <hash>`
- **删除特定实体**：`/delete entity <name>`
- **清空数据库**：`/delete clear` (慎用！)

### 3. 可视化编辑 (推荐)

运行 `/visualize` 命令后，访问 `http://localhost:8082` 即可进入图形化编辑器。支持：
- 节点/关联的实时增删改查。
- 知识网络拓扑结构展示。

### 4. 核心配置说明 (`config.toml`)

你可以通过修改 `config.toml` 来定制插件行为。

#### 🛡️ 聊天流过滤 `[filter]`
控制哪些聊天流可以访问/写入知识库：
- `enabled`: 是否启用过滤。
- `mode`: `whitelist` (仅允许列表内) 或 `blacklist` (禁止列表内)。
- `chats`: 列表，包含 `group_id`, `user_id` 或 `stream_id`。

#### 🧠 对话自动总结 `[summarization]`
- `enabled`: 是否允许对话自动转知识。
- `include_personality`: 总结时是否参考机器人人格。
- `default_knowledge_type`: 默认导入类型（`narrative`, `factual`, `structured`）。

#### 🕒 定时导入 `[schedule]`
- `enabled`: 是否启用定时任务。
- `import_times`: 自动执行时间点，如 `["04:00", "16:00"]`。

#### ⚙️ 检索与排序 `[retrieval]`
- `enable_ppr`: 是否启用 Personalized PageRank 智能排序。
- `alpha`: 段落与关系的融合权重 (0-1)。
- `min_threshold`: 结果过滤的最小相似度分数。


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

| 维度 | 原 LPMM | A_Memorix |
|------|---------|-----------|
| **后端引擎** | 基于对象/字典的图算法 | 基于 SciPy 稀疏矩阵的线性代数计算 |
| **向量格式** | float32 (高内存消耗) | int8 量化 (极致内存压缩) |
| **存储路径** | 全局 `data/` 目录 | 隔离的 `plugins/A_memorix/data/` |
| **依赖关系** | 与主程序逻辑混杂 | 模块化解耦，可独立升级 |
| **数据格式** | JSON/SQLite | NPZ/PKL/SQLite |

---

## 📜 许可证

本项目采用 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0) 许可证。

## 贡献声明

本项目目前不接受任何PR，只接受issue，如有相关问题请提交issue或联系ARC

**作者**: A_Dawn
