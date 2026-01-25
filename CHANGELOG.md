# 更新日志 (Changelog)

## [0.2.1] - 2026-01-26

本次更新重点在于 **可视化交互的全方位重构** 以及 **底层鲁棒性的进一步增强**。

### 🎨 可视化与交互重构

#### WebUI (Glassmorphism)

- **全新视觉设计**: 采用深色磨砂玻璃 (Glassmorphism) 风格，配合动态渐变背景。
- **Dock 菜单栏**: 底部新增 macOS 风格 Dock 栏，聚合所有常用功能。
- **显著性视图 (Saliency View)**: 基于 **PageRank** 算法的“信息密度”滑块，支持以此过滤叶子节点，仅展示核心骨干或全量细节。
- **功能面板**:
  - **❓ 操作指南**: 内置交互说明与特性介绍。
  - **🔍 悬浮搜索**: 支持按拼音/ID 实时过滤节点。
  - **📂 记忆溯源**: 支持按源文件批量查看和删除记忆数据。
  - **📖 内容字典**: 列表化展示所有实体与关系，支持排序与筛选。

### 🛠️ 稳定性与工程改进

#### EmbeddingAPI

- **鲁棒性增强**: 引入 `tenacity` 实现指数退避重试机制。
- **错误处理**: 失败时返回 `NaN` 向量而非零向量，允许上层逻辑安全跳过。

#### MetadataStore

- **自动修复**: 自动检测并修复 `vector_index` 列错位（文件名误存）的历史数据问题。
- **数据统计**: 新增 `get_all_sources` 接口支持来源统计。

#### 脚本与工具

- **用户体验**: 引入 `rich` 库优化终端输出进度条与状态显示。
- **接口开放**: `process_knowledge.py` 新增 `import_json_data` 供外部调用。
- **LPMM 迁移**: 新增 `import_lpmm_json.py`，支持导入符合 LPMM 规范的 OpenIE JSON 数据。

#### plugin.py

- **版本号**: `0.2.0` → `0.2.1`

---

## [0.2.0] - 2026-01-22

> [!CAUTION]
> **不完全兼容变更**：v0.2.0 版本重构了底层存储架构。由于数据结构的重大调整，**旧版本的导入数据无法在新版本中完全无损兼容**。
> 虽然部分组件支持自动迁移，但为确保数据一致性和检索质量，**强烈建议在升级后重新使用 `process_knowledge.py` 导入原始数据**。

本次更新为**重大版本升级**，包含向量存储架构重写、检索逻辑强化及多项稳定性改进。

### 🚀 核心架构重写

#### VectorStore: SQ8 量化 + Append-Only 存储

- **全新存储格式**: 从 `.npy` 迁移至 `vectors.bin`（float16 增量追加）和 `vectors_ids.bin`，大幅减少内存占用。
- **原生 SQ8 量化**: 使用 Faiss `IndexScalarQuantizer(QT_8bit)`，替代手动 int8 量化逻辑。
- **L2 Normalization 强制化**: 所有向量在存储和检索时统一执行 L2 归一化，确保 Inner Product 等价于 Cosine 相似度。
- **Fallback 索引机制**: 新增 `IndexFlatIP` 回退索引，在 SQ8 训练完成前提供检索能力，避免冷启动无结果问题。
- **Reservoir Sampling 训练采样**: 使用蓄水池采样收集训练数据（上限 10k），保证小数据集和流式导入场景下的训练样本多样性。
- **线程安全**: 新增 `threading.RLock` 保护并发读写操作。
- **自动迁移**: 支持从旧版 `.npy` 格式自动迁移至新 `.bin` 格式。

### ✨ 检索功能增强

#### KnowledgeQueryTool: 智能回退与多跳路径搜索

- **Smart Fallback (智能回退)**: 当向量检索置信度低于阈值 (默认 0.6) 时，自动尝试提取查询中的实体进行多跳路径搜索（`_path_search`），增强对间接关系的召回能力。
- **结果去重 (`_deduplicate_results`)**: 新增基于内容相似度的安全去重逻辑，防止冗余结果污染 LLM 上下文，同时确保至少保留一条结果。
- **语义关系检索 (`_semantic_search_relation`)**: 支持自然语言查询关系（无需 `S|P|O` 格式），内部使用 `REL_ONLY` 策略进行向量检索。
- **路径搜索 (`_path_search`)**: 新增 `GraphStore.find_paths` 调用，支持查找两个实体间的间接连接路径（最大深度 3，最多 5 条路径）。
- **Clean Output**: LLM 上下文中不再包含原始相似度分数，避免模型偏见。

#### DualPathRetriever: 并发控制与调试模式

- **PPR 并发限制 (`ppr_concurrency_limit`)**: 新增 Semaphore 控制 PageRank 计算并发数，防止 CPU 峰值过载。
- **Debug 模式**: 新增 `debug` 配置项，启用时打印检索结果原文到日志。
- **Entity-Pivot 关系检索**: 优化 `_retrieve_relations_only` 策略，通过检索实体后扩展其关联关系，替代直接检索关系向量。

### ⚙️ 配置与 Schema 扩展

#### plugin.py

- **版本号**: `0.1.3` → `0.2.0`
- **默认配置版本**: `config_version` 默认值更新为 `2.0.0`
- **新增配置项**:
  - `retrieval.relation_semantic_fallback` (bool): 是否启用关系查询的语义回退。
  - `retrieval.relation_fallback_min_score` (float): 语义回退的最小相似度阈值。
- **相对路径支持**: `storage.data_dir` 现在支持相对路径（相对于插件目录），默认值改为 `./data`。
- **全局实例获取**: 新增 `A_MemorixPlugin.get_global_instance()` 静态方法，供组件可靠获取插件实例。

#### config.toml / \_manifest.json

- **新增 `ppr_concurrency_limit`**: 控制 PPR 算法并发数。
- **新增训练阈值配置**: `embedding.min_train_threshold` 控制触发 SQ8 训练的最小样本数。

### 🛠️ 稳定性与工程改进

#### GraphStore

- **`find_paths` 方法**: 新增多跳路径查找功能，支持 BFS 搜索指定深度内的实体间路径。
- **`find_node` 方法**: 新增大小写不敏感的节点查找。

#### MetadataStore

- **Schema 迁移**: 自动添加缺失的 `is_permanent`, `last_accessed`, `access_count` 字段。

#### 脚本与工具

- **新增脚本**:
  - `scripts/diagnose_relations_source.py`: 诊断关系溯源问题。
  - `scripts/verify_search_robustness.py`: 验证检索鲁棒性。
  - `scripts/run_stress_test.py`, `stress_test_data.py`: 压力测试套件。
  - `scripts/migrate_canonicalization.py`, `migrate_paragraph_relations.py`: 数据迁移工具。
- **目录整理**: 将大量旧版测试脚本移动至 `deprecated/` 目录。

### 🗑️ 移除与废弃

- 废弃 `vectors.npy` 存储格式（自动迁移至 `.bin`）。

---

## [0.1.3] - 上一个稳定版本

- 初始发布，包含基础双路检索功能。
- 手动 Int8 向量量化。
- 基于 `.npy` 的向量存储。
