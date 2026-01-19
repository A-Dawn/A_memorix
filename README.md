# A_Memorix

**轻量级知识图谱插件** - 基于双路检索的完全独立的记忆增强系统

---

## ✨ 特性

- **🧠 双路检索** - 关系图谱 + 向量语义并行检索，结合 Personalized PageRank 智能排序。
- **📊 知识图谱可视化** - 内置 Web 可视化编辑器，支持节点/边的增删改查。
- **🎯 智能分类** - 自动识别结构化/叙事性/事实性知识，采用差异化处理策略。
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

### 命令行交互（不建议）

A_Memorix 提供以下一级命令（直接输入命令即可触发）：

| 命令 | 说明 | 示例 |
|------|------|------|
| `/import` | 导入知识 | `/import text 人工智能是...` |
| `/query` | 查询知识 | `/query search 什么是AI?` |
| `/delete` | 删除知识 | `/delete paragraph <hash>` |
| `/visualize` | 启动可视化 | `/visualize` |
| `/debug_server` | 调试服务器 | `/debug_server` |

### 1. 自动批量导入 (`process_knowledge.py`)

适用于从大量历史文档快速构建知识库。脚本会自动调用 LLM 提取实体和关系。

**文件要求：**
- **格式**：仅支持 `.txt` 平面文本。
- **内容**：支持**自由形式的自然语言文本**。无需特定标记或结构，脚本会调用 LLM 自动分析其中的实体与关系。
- **编码**：必须使用 `UTF-8` 编码。
- **路径**：文件需放入 `plugins/A_memorix/data/raw/` 目录。

**操作步骤：**
1. 将 `.txt` 格式的原始文档放入 `plugins/A_memorix/data/raw/` 目录。
2. 在主程序根目录下运行：
   ```bash
   python plugins/A_memorix/scripts/process_knowledge.py
   ```

**支持参数：**
- `--force`: 强制重新导入已处理过的文件。
- `--clear-manifest`: 清空导入历史记录并重新扫描。
- `--type <type>`: 强制指定内容类型（如：`structured`, `narrative`, `factual`）。

### 2. 导入知识 (`/import`)

支持多种导入模式（默认为 `text`）：

```bash
# 导入文本（自动分段并提取实体关系）
/import text 北京是中国的首都，拥有3000多年建城史。

# 导入单个段落
/import paragraph 机器学习是人工智能的一个重要子领域。

# 导入关系 (格式: 主|谓|宾)
/import relation Apple|founded|Steve Jobs

# 从文件导入 (.txt, .md, .json)
/import file ./my_notes.txt

**文件格式要求：**
- **TXT/MD**：普通文本，建议按段落分块（空行分隔）。
- **JSON**：需符合以下结构（支持可选字段）：
  ```json
  {
    "paragraphs": ["段落内容1", "段落内容2"],
    "entities": ["实体1", "实体2"],
    "relations": [
      {"subject": "主体", "predicate": "谓词", "object": "客体"}
    ]
  }
  ```
```

### 2. 查询知识 (`/query`)

支持多种查询模式（默认为 `search`）：

```bash
# 双路检索（段落 + 关系）
/query search 北京的历史

# 查询特定实体
/query entity 北京

# 查询统计信息
/query stats

# 查看详细帮助
/query help
```

### 3. 删除知识 (`/delete`)

```bash
# 按 hash 删除段落
/delete paragraph a1b2c3d4...

# 删除特定实体
/delete entity 北京

# 清空整个知识库（需谨慎！）
/delete clear
```

### 4. 可视化编辑（建议！）

```bash
# 启动可视化 Web 服务器
/visualize
```

服务器启动后访问 `http://localhost:8082` 即可图形化操作。

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

**作者**: A_Dawn
